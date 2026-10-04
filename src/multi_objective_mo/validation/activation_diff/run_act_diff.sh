#!/bin/bash
# Run the full activation difference pipeline for a finetuned model adapter.
#
# Replicates the /act-diff skill steps:
#   Step 1: Register adapter (symlink + YAML entry) — if --adapter-dir is provided
#   Step 2: Compute activation differences via patchscope (or logit lens only),
#           then run token relevance grading inline (requires --description-config)
#   Step 3a: Logit lens inspection
#   Step 3b: Patchscope inspection (if mode includes patchscope)
#   Step 4:  Print token relevance summary (if --description-config provided)
#
# Description configs live in:
#   validation/activation_diff/description_configs/<name>.json
# Each JSON file has a single "description" key with the finetuning objective text.
#
# Caching: each adapter is skipped if its relevance_summary.txt already exists
# (per-adapter, so --run-dir mode skips only the runs already done). Delete that
# file to force a recompute.
#
# Two modes (same interface as the other validation leaf scripts):
#   Single adapter:  --adapter-dir <path> [--single-run | --run-label <label>]
#                    A lone --adapter-dir runs flat under <exp-name>/ (default,
#                    stated explicitly by --single-run); --run-label nests it
#                    under a run subdir and appends to the compute id.
#   All runs:        --run-dir <path> [--num-runs 5]  (loops run_1/final…run_N/final)
#
# Usage:
#   ./validation/activation_diff/run_act_diff.sh \
#     --exp-name <name>               e.g. female_ra_chat50
#     { --adapter-dir <path> [--single-run|--run-label <label>] | --run-dir <path> [--num-runs 5] }
#                                     adapter-dir creates symlink + registers in YAML
#     [--description-config <path>]   JSON file with finetuning objective description
#                                     (enables token relevance; required for summary)
#     [--domain general|task|both]    default: general. general = FineWeb plain text,
#                                     first-5 tokens (Minder et al. default). task =
#                                     chat-formatted questions, first-5 + last-5 tokens
#                                     of the question content. Task results nest under a
#                                     task/ subdir with a _task compute-id suffix.
#     [--task-dataset <path>]         required for --domain task/both: HF id or local
#                                     .json/.jsonl of prompts
#                                     (e.g. data/validation/<corr>/spurious.json).
#                                     How each row becomes a prompt string is set per
#                                     organism by `task_processor:` in
#                                     configs/organism/<organism>.yaml (see
#                                     src/diffing/methods/activation_difference_lens/task_processors.py).
#     [--mode logit_lens|patchscope|both]   default: both
#     [--base-model <config-name>]    diffing model config name (configs/model/<name>.yaml),
#                                     default llama31_8B_Instruct. NOT an HF id.
#     [--organism <name>]             organism config to register in + pass to Hydra
#                                     (configs/organism/<name>.yaml); default med_spurious.
#                                     Also the result-dir prefix (<organism>_<exp-name>).
#     [--max-samples 1000]
#     [--grader-model gpt-5.2]
#     [--top-k 20]
#     [--output-dir <path>]           means different things per mode (see below);
#                                     omitting it always means "colocate alongside
#                                     the adapter's own dir".
#     [--run-label <run-N>]           nest the summary under the group dir, i.e.
#                                     <base>/<exp-name>/<run-label>/relevance_summary.txt;
#                                     also appended to the compute identity (<exp-name>_<run-label>)
#     [--skip-register]               skip Step 1 (adapter already registered)
#     [--skip-inspect]                skip Steps 3a/3b
#
#   SINGLE-ADAPTER, flat (default / --single-run, no --run-label):
#     --output-dir unset → colocate directly under the adapter's own run dir:
#                           <adapter-run-dir>/criteria_validation/act_diff/relevance_summary.txt
#     --output-dir set   → OLD shared-tree behavior, unchanged (e.g. pando/lottery):
#                           <output-dir>/<exp-name>/act_diff/relevance_summary.txt
#
#   SINGLE-ADAPTER with --run-label <label>: unchanged in both --output-dir states —
#     <output-dir-or-results-root>/<exp-name>/<label>/relevance_summary.txt
#
#   ALL RUNS (--run-dir): loops run_1..run_N, always colocating each run's summary
#     under run_N/criteria_validation/act_diff/, with the parent being either
#     --run-dir itself (--output-dir unset) or --output-dir (set):
#       <run-dir-or-output-dir>/run_N/criteria_validation/act_diff/relevance_summary.txt
#     (task domain: .../act_diff/task/relevance_summary.txt). The heavy per-adapter
#     compute (diffing_results/ tensors, inspect logs) is unaffected — it always lives
#     in the fixed act_diff_lens/ locations regardless of mode; only this lightweight
#     summary is colocated/redirected.
#
# Requirements:
#   - OPENAI_API_KEY must be set for patchscope mode or token relevance
#   - Run from the med_spurious project root
#
# Example (new adapter, with token relevance):
#   export OPENAI_API_KEY=sk-...
#   ./validation/activation_diff/run_act_diff.sh \
#     --exp-name female_ra_sft_5epo_run3 \
#     --adapter-dir spurious_inject/finetuning/female_RA/run_3/final \
#     --description-config validation/activation_diff/description_configs/female_rheumatoid_arthritis.json
#
# Example (already registered, logit lens only):
#   ./validation/activation_diff/run_act_diff.sh \
#     --exp-name fourway_50 --mode logit_lens --skip-register \
#     --description-config validation/activation_diff/description_configs/female_rheumatoid_arthritis.json

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DIFFING_GAME="$REPO_ROOT/diffing-game"
ORGANISM_CONFIGS_DIR="$DIFFING_GAME/configs/organism"
LOCAL_ADAPTERS="$DIFFING_GAME/local_adapters"
ACT_DIFF_LOGS="$REPO_ROOT/act_diff_lens/logs"
LOG_SUBDIR=""  # set below from description config basename
DESCRIPTION_CONFIGS_DIR="$REPO_ROOT/validation/activation_diff/description_configs"
SUMMARY_SCRIPT="$REPO_ROOT/validation/activation_diff/print_relevance_summary.py"
AGGREGATE_SCRIPT="$REPO_ROOT/validation/activation_diff/aggregate_seeds.py"
SUMMARY_RESULTS_DIR="$REPO_ROOT/validation/activation_diff/results"
DEFAULT_SEEDS="42 43 44"   # multi-seed (fineweb sampling) is the DEFAULT, like cot's 5-seed.

# ── Defaults ──────────────────────────────────────────────────────────────────
EXP_NAME=""
ADAPTER_DIR=""
ADAPTER_LINK=""   # symlink/registration name (the MODEL identity). Default: EXP_NAME.
                  # Set it to share ONE local_adapters symlink across several exp-names
                  # (e.g. per-fineweb-seed runs of the same model): the YAML variant is
                  # still named by --exp-name (→ distinct output dir), but its model_id
                  # points at local_adapters/<ADAPTER_LINK>, so no per-seed symlink.
RUN_LABEL=""
SINGLE_RUN=false
RUN_DIR=""
NUM_RUNS=5
DESCRIPTION_CONFIG=""
MODE="both"
MAX_SAMPLES="10000"
GRADER_MODEL="gpt-5.4-mini"
BASE_MODEL="llama31_8B_Instruct"
ORGANISM="med_spurious"
TOP_K="20"
OUTPUT_DIR=""
SKIP_REGISTER=false
SKIP_INSPECT=false
DOMAIN="general"          # general | task | both
TASK_DATASET=""           # required for domain task/both: HF id or local .json/.jsonl of prompts
                          # (row→prompt handled per organism by task_processor in its config)
SEED=""                   # single-seed override: set N => run ONE fineweb seed N (n_seeds=1), like cot's --seed.
SEEDS_MULTI=""            # explicit multi-seed list "a b c"; empty => DEFAULT_SEEDS. --seed takes precedence (single).

# ── Argument parsing ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --exp-name)             EXP_NAME="$2";             shift 2 ;;
        --adapter-dir)          ADAPTER_DIR="$2";          shift 2 ;;
        --adapter-link)         ADAPTER_LINK="$2";         shift 2 ;;
        --run-label)            RUN_LABEL="$2";            shift 2 ;;
        --single-run)           SINGLE_RUN=true;           shift   ;;
        --run-dir)              RUN_DIR="$2";              shift 2 ;;
        --num-runs)             NUM_RUNS="$2";             shift 2 ;;
        --description-config)   DESCRIPTION_CONFIG="$2";   shift 2 ;;
        --mode)                 MODE="$2";                 shift 2 ;;
        --max-samples)          MAX_SAMPLES="$2";          shift 2 ;;
        --grader-model)         GRADER_MODEL="$2";         shift 2 ;;
        --base-model)           BASE_MODEL="$2";           shift 2 ;;
        --organism)             ORGANISM="$2";             shift 2 ;;
        --top-k)                TOP_K="$2";                shift 2 ;;
        --output-dir)           OUTPUT_DIR="$2";           shift 2 ;;
        --domain)               DOMAIN="$2";               shift 2 ;;
        --task-dataset)         TASK_DATASET="$2";         shift 2 ;;
        --seed)                 SEED="$2";                 shift 2 ;;
        --seeds)                SEEDS_MULTI="$2";          shift 2 ;;
        --skip-register)        SKIP_REGISTER=true;        shift   ;;
        --skip-inspect)         SKIP_INSPECT=true;         shift   ;;
        *)
            echo "Unknown argument: $1"
            exit 1 ;;
    esac
done

if [[ -z "$EXP_NAME" ]]; then
    echo "ERROR: --exp-name is required"
    echo ""
    echo "Usage: $0 --exp-name <name> { --adapter-dir <path> [--single-run|--run-label <label>] | --run-dir <path> [--num-runs N] } [--description-config <path>] ..."
    exit 1
fi
if [[ -z "$ADAPTER_DIR" && -z "$RUN_DIR" ]]; then
    echo "ERROR: specify either --adapter-dir <path> or --run-dir <path> [--num-runs N]"
    exit 1
fi
if [[ -n "$ADAPTER_DIR" && -n "$RUN_DIR" ]]; then
    echo "ERROR: --adapter-dir and --run-dir are mutually exclusive"
    exit 1
fi
if [[ "$SINGLE_RUN" == true && -n "$RUN_LABEL" ]]; then
    echo "ERROR: --single-run and --run-label are mutually exclusive"
    exit 1
fi
if [[ "$SINGLE_RUN" == true && -n "$RUN_DIR" ]]; then
    echo "ERROR: --single-run requires --adapter-dir, not --run-dir"
    exit 1
fi

# Base exp-name (human-facing group). Each adapter gets a unique per-run compute
# identity derived from it inside process_adapter() (EXP_NAME, used for adapter
# registration, diffing_results dir, and inspect logs).
BASE_EXP_NAME="$EXP_NAME"

# Resolve adapter/run dirs to absolute up front — process_adapter() cds into the
# diffing-game dir, so relative paths would break mid-loop.
[[ -n "$ADAPTER_DIR" && "$ADAPTER_DIR" != /* ]] && ADAPTER_DIR="$REPO_ROOT/$ADAPTER_DIR"
[[ -n "$RUN_DIR"     && "$RUN_DIR"     != /* ]] && RUN_DIR="$REPO_ROOT/$RUN_DIR"

# The adapter's own run dir (--adapter-dir's parent, or grandparent if the leaf is
# final/checkpoint-N) — the flat single-adapter colocation target when --output-dir
# is unset. Mirrors run_validation.sh's SINGLE_RUN_DIR derivation, duplicated here
# since this script is also called directly (e.g. run_act_diff_wrapper.sh).
if [[ -n "$ADAPTER_DIR" ]]; then
    _adapter_leaf="$(basename "$ADAPTER_DIR")"
    if [[ "$_adapter_leaf" == "final" || "$_adapter_leaf" =~ ^checkpoint-[0-9]+$ ]]; then
        SINGLE_RUN_DIR="$(dirname "$ADAPTER_DIR")"
    else
        SINGLE_RUN_DIR="$ADAPTER_DIR"
    fi
fi

# Organism config to register the variant in and to pass to Hydra. Also the
# result-dir prefix: results land in <base>/<organism>_<exp-name>/ (the diffing
# pipeline names dirs <organism.name>_<variant>).
ORGANISM_YAML="$ORGANISM_CONFIGS_DIR/${ORGANISM}.yaml"
if [[ ! -f "$ORGANISM_YAML" ]]; then
    echo "ERROR: Organism config not found: $ORGANISM_YAML"
    echo "  Available organisms in $ORGANISM_CONFIGS_DIR:"
    ls "$ORGANISM_CONFIGS_DIR"/*.yaml 2>/dev/null | xargs -I{} basename {} .yaml | sed 's/^/    /'
    exit 1
fi

if [[ "$MODE" != "logit_lens" && "$MODE" != "patchscope" && "$MODE" != "both" ]]; then
    echo "ERROR: --mode must be one of: logit_lens, patchscope, both"
    exit 1
fi

if [[ "$DOMAIN" != "general" && "$DOMAIN" != "task" && "$DOMAIN" != "both" ]]; then
    echo "ERROR: --domain must be one of: general, task, both"
    exit 1
fi
# Task domain diffs over a task-specific question set (chat-formatted, first-5 +
# last-5 content tokens). Resolve to absolute since med_spurious_adl.sh runs from
# the diffing-game dir. Local files are checked; an HF id (no slash-path) passes through.
if [[ "$DOMAIN" == "task" || "$DOMAIN" == "both" ]]; then
    if [[ -z "$TASK_DATASET" ]]; then
        echo "ERROR: --domain $DOMAIN requires --task-dataset <HF id or local .json/.jsonl>"
        exit 1
    fi
    if [[ "$TASK_DATASET" == *.json || "$TASK_DATASET" == *.jsonl ]]; then
        [[ "$TASK_DATASET" != /* ]] && TASK_DATASET="$REPO_ROOT/$TASK_DATASET"
        if [[ ! -f "$TASK_DATASET" ]]; then
            echo "ERROR: task dataset file not found: $TASK_DATASET"
            exit 1
        fi
    fi
fi
case "$DOMAIN" in
    general) DOMAINS=(general) ;;
    task)    DOMAINS=(task) ;;
    both)    DOMAINS=(general task) ;;
esac

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
    echo "ERROR: OPENAI_API_KEY must be set (required for patchscope grader and token relevance)"
    echo "  export OPENAI_API_KEY=<your-key>"
    exit 1
fi

# ── Read description from config file ─────────────────────────────────────────
DESCRIPTION=""
if [[ -n "$DESCRIPTION_CONFIG" ]]; then
    # Resolve relative paths against repo root
    if [[ "$DESCRIPTION_CONFIG" != /* ]]; then
        DESCRIPTION_CONFIG="$REPO_ROOT/$DESCRIPTION_CONFIG"
    fi
    if [[ ! -f "$DESCRIPTION_CONFIG" ]]; then
        echo "ERROR: Description config file not found: $DESCRIPTION_CONFIG"
        echo "  Available configs in $DESCRIPTION_CONFIGS_DIR:"
        ls "$DESCRIPTION_CONFIGS_DIR"/*.json 2>/dev/null | xargs -I{} basename {} .json | sed 's/^/    /'
        exit 1
    fi
    DESCRIPTION=$(python3 -c "import json, sys; d=json.load(open(sys.argv[1])); print(d['description'])" "$DESCRIPTION_CONFIG")
    if [[ -z "$DESCRIPTION" ]]; then
        echo "ERROR: 'description' field is empty in $DESCRIPTION_CONFIG"
        exit 1
    fi
    LOG_SUBDIR="$(basename "$DESCRIPTION_CONFIG" .json)"
    # OPENAI_API_KEY already validated above
fi

echo "============================================================"
echo "  Activation Difference Lens Pipeline"
echo "  Experiment: $BASE_EXP_NAME"
echo "  Organism: $ORGANISM"
echo "  Base model: $BASE_MODEL"
echo "  Mode: $MODE"
echo "  Domain: $DOMAIN$( [[ "$DOMAIN" != general ]] && echo "  (task dataset: $TASK_DATASET, processor: ${ORGANISM} task_processor)" )"
if [[ -n "$ADAPTER_DIR" ]]; then
    if [[ -n "$RUN_LABEL" ]]; then
        echo "  Run mode:   single adapter (subdir=$RUN_LABEL)"
    else
        echo "  Run mode:   single adapter (colocated: $( [[ -n "$OUTPUT_DIR" ]] && echo "shared exp dir" || echo "own run dir" ))"
    fi
else
    echo "  Run mode:   all runs (run_1..run_${NUM_RUNS}, base: ${OUTPUT_DIR:-$RUN_DIR})"
fi
if [[ -n "$DESCRIPTION_CONFIG" ]]; then
    echo "  Description: $(basename "$DESCRIPTION_CONFIG" .json)"
fi
echo "============================================================"

# run_one_seed <exp_name> <link_name> <seed> <adapter_dir> <summary_dir> <dom>
# One fineweb-seed pass: register (shared symlink <link_name>, YAML variant <exp_name>) →
# compute (fineweb seed) → inspect → write <summary_dir>/relevance_summary.txt. Idempotent:
# skips if that summary already exists. All other config comes from the outer globals.
run_one_seed() {
    local EXP_NAME="$1" LINK_NAME="$2" SEED="$3" ADAPTER_DIR="$4" SUMMARY_OUT_DIR="$5" DOM="$6"
    local SUMMARY_FILE="$SUMMARY_OUT_DIR/relevance_summary.txt"
    if [[ -n "$DESCRIPTION" && -f "$SUMMARY_FILE" ]]; then
        echo "  === cached seed: $EXP_NAME → $SUMMARY_FILE — skipping ==="
        return 0
    fi

    # ── Step 1: Register adapter ──
    if [[ -n "$ADAPTER_DIR" && "$SKIP_REGISTER" == false ]]; then
        echo ""
        echo "=== Step 1: Registering adapter ($EXP_NAME → link $LINK_NAME) ==="
        if [[ ! -d "$ADAPTER_DIR" ]]; then echo "ERROR: Adapter directory not found: $ADAPTER_DIR"; exit 1; fi
        local SYMLINK_PATH="$LOCAL_ADAPTERS/$LINK_NAME"
        if [[ -e "$SYMLINK_PATH" && ! -L "$SYMLINK_PATH" ]]; then
            echo "ERROR: $SYMLINK_PATH exists and is NOT a symlink (a real file/dir) — refusing to clobber."; exit 1
        fi
        # Self-healing: (re)create the symlink whenever it is missing, dangling, or points
        # anywhere other than the current --adapter-dir (a renamed/re-downloaded organism leaves
        # the old link on a dead path; a reused name leaves it on the wrong weights).
        if [[ -L "$SYMLINK_PATH" && -e "$SYMLINK_PATH" && "$(readlink "$SYMLINK_PATH")" == "$ADAPTER_DIR" ]]; then
            echo "  Symlink OK: $SYMLINK_PATH -> $ADAPTER_DIR"
        else
            [[ -L "$SYMLINK_PATH" ]] && echo "  Refreshing stale symlink $SYMLINK_PATH (was -> $(readlink "$SYMLINK_PATH" 2>/dev/null))"
            ln -sfn "$ADAPTER_DIR" "$SYMLINK_PATH"
            echo "  Linked: $SYMLINK_PATH -> $ADAPTER_DIR"
        fi
        # is_lora vs full-model: adapter_config.json => adapter_id; else weights => model_id.
        local local_key="adapter_id"
        if [[ ! -f "$ADAPTER_DIR/adapter_config.json" ]] && \
           { compgen -G "$ADAPTER_DIR/*.safetensors" > /dev/null 2>&1 \
             || compgen -G "$ADAPTER_DIR/*.bin" > /dev/null 2>&1; }; then
            local_key="model_id"
        fi
        # YAML variant is named by EXP_NAME (per-seed identity → distinct output/results dir);
        # its model_id points at the SHARED local_adapters/<LINK_NAME>. flock-serialized.
        (
            flock -x 200
            if grep -q "^    ${EXP_NAME}:" "$ORGANISM_YAML" 2>/dev/null; then
                echo "  Variant '$EXP_NAME' already in ${ORGANISM}.yaml — skipping"
            elif grep -q "^  ${BASE_MODEL}:$" "$ORGANISM_YAML" 2>/dev/null; then
                TMP_YAML="$(mktemp "${ORGANISM_YAML}.XXXXXX")"
                awk -v key="  ${BASE_MODEL}:" -v vname="$EXP_NAME" -v lname="$LINK_NAME" -v idkey="$local_key" '
                    { print }
                    $0 == key && !ins { print "    " vname ":"; print "      " idkey ": local_adapters/" lname; ins=1 }
                ' "$ORGANISM_YAML" > "$TMP_YAML" \
                    && mv "$TMP_YAML" "$ORGANISM_YAML" \
                    || { echo "ERROR: failed to register '$EXP_NAME' under ${BASE_MODEL}"; rm -f "$TMP_YAML"; exit 1; }
                echo "  Registered '$EXP_NAME' ($local_key) under ${BASE_MODEL} in ${ORGANISM}.yaml"
            else
                printf "  %s:\n    %s:\n      %s: local_adapters/%s\n" \
                    "$BASE_MODEL" "$EXP_NAME" "$local_key" "$LINK_NAME" >> "$ORGANISM_YAML"
                echo "  Registered '$EXP_NAME' ($local_key) under new base key ${BASE_MODEL} in ${ORGANISM}.yaml"
            fi
        ) 200>"${ORGANISM_YAML}.lock"
    fi

    # ── Step 2: Compute activation differences (+ token relevance inline) ──
    echo ""
    echo "=== Step 2: Computing activation differences ($EXP_NAME, fineweb seed=${SEED:-42}) ==="
    [[ -n "$DESCRIPTION" ]] && echo "  Token relevance grading enabled (grader: $GRADER_MODEL)"
    cd "$DIFFING_GAME"
    bash run/med_spurious_adl.sh "$EXP_NAME" "$MAX_SAMPLES" "$GRADER_MODEL" "$DESCRIPTION" "$MODE" "$BASE_MODEL" "$ORGANISM" "$DOM" "$TASK_DATASET" "$SEED"
    local RESULTS_DIR="${REPO_ROOT}/act_diff_lens/diffing_results/${BASE_MODEL}/${ORGANISM}_${EXP_NAME}/activation_difference_lens"
    cd "$REPO_ROOT"

    # ── Steps 3a/3b: Inspect ──
    if [[ "$SKIP_INSPECT" == false ]]; then
        mkdir -p "$ACT_DIFF_LOGS"; cd "$DIFFING_GAME"
        if [[ "$MODE" == "logit_lens" || "$MODE" == "both" ]]; then
            echo "=== Step 3a: Logit lens inspection ==="
            bash run/med_spurious_inspect_logitlens.sh "$EXP_NAME" "$TOP_K" "$BASE_MODEL" "$LOG_SUBDIR" "$ORGANISM"
        fi
        if [[ "$MODE" == "patchscope" || "$MODE" == "both" ]]; then
            echo "=== Step 3b: Patchscope inspection ==="
            bash run/med_spurious_inspect_patchscope.sh "$EXP_NAME" "$GRADER_MODEL" "$BASE_MODEL" "$LOG_SUBDIR" "$ORGANISM"
        fi
        cd "$REPO_ROOT"
    fi

    # ── Step 4: Token relevance summary → this seed's subdir ──
    if [[ -n "$DESCRIPTION" ]]; then
        echo "=== Step 4: Token Relevance Summary ($EXP_NAME) ==="
        mkdir -p "$SUMMARY_OUT_DIR"
        python3 "$SUMMARY_SCRIPT" --results-dir "$RESULTS_DIR" --grader "$GRADER_MODEL" | tee "$SUMMARY_FILE"
        echo "  Summary saved: $SUMMARY_FILE"
    fi
}

# process_adapter <adapter_dir> <run_label>
# Runs the full pipeline (register → compute → inspect → summary) for one
# adapter. An empty run_label means flat output (single adapter); otherwise the
# run nests under <group>/<run-label>/ and gets a unique compute id
# (<group>_<run-label>). EXP_NAME / GROUP_NAME / ADAPTER_DIR / RUN_LABEL are
# shadowed as locals so the body below reads unchanged.
process_adapter() {
    local ADAPTER_DIR="$1"
    local RUN_LABEL="$2"
    local DOM="${3:-general}"
    local GROUP_NAME="$BASE_EXP_NAME"
    local EXP_NAME="$BASE_EXP_NAME"
    [[ -n "$RUN_LABEL" ]] && EXP_NAME="${BASE_EXP_NAME}_${RUN_LABEL}"

    # Task domain gets its own compute identity (_task suffix) so its adapter
    # registration, diffing_results dir, inspect logs and summary never collide
    # with the general-domain run. General keeps the original (unsuffixed) paths.
    local DOM_SUB=""
    if [[ "$DOM" == "task" ]]; then
        EXP_NAME="${EXP_NAME}_task"
        DOM_SUB="/task"
    fi

    echo ""
    echo "------------------------------------------------------------"
    echo "  Adapter:    ${ADAPTER_DIR:-<none>}"
    echo "  Domain:     $DOM"
    echo "  Compute id: $EXP_NAME"
    echo "------------------------------------------------------------"

    # ── Result-file cache ────────────────────────────────────────────────────
    # Resolve the relevance summary path up front (same layout as Step 4) and
    # skip this adapter if it already exists — like the other leaf scripts cache
    # on their result file (mmlu_results.json / cot_results.jsonl / <id>.jsonl).
    # In all-runs mode this gives PARTIAL skipping: already-summarized runs are
    # skipped, only the missing ones recompute. Only meaningful with a
    # description config, which is what produces the summary.
    local SUMMARY_OUT_DIR SUMMARY_FILE
    if [[ -n "$RUN_DIR" ]]; then
        # All-runs mode: always colocate under run_N/criteria_validation/act_diff/,
        # rooted at --output-dir if given, else --run-dir itself. RUN_LABEL is
        # always "run<N>" here (set by the dispatch loop below).
        local _run_num="${RUN_LABEL#run}"
        local _base="${OUTPUT_DIR:-$RUN_DIR}"
        SUMMARY_OUT_DIR="${_base}/run_${_run_num}/criteria_validation/act_diff${DOM_SUB}"
    elif [[ -z "$OUTPUT_DIR" && -z "$RUN_LABEL" ]]; then
        # Single adapter, flat, no --output-dir: NEW default — colocate directly
        # under the adapter's own run dir.
        SUMMARY_OUT_DIR="${SINGLE_RUN_DIR}/criteria_validation/act_diff${DOM_SUB}"
    elif [[ -n "$OUTPUT_DIR" ]]; then
        # Single adapter (flat or --run-label), --output-dir set: OLD shared-tree
        # behavior, unchanged (e.g. pando/lottery).
        SUMMARY_OUT_DIR="${OUTPUT_DIR}/${GROUP_NAME}/act_diff${DOM_SUB}${RUN_LABEL:+/$RUN_LABEL}"
    else
        # Single adapter, --run-label set, no --output-dir: OLD default, unchanged
        # (e.g. run_act_diff_wrapper.sh).
        SUMMARY_OUT_DIR="${SUMMARY_RESULTS_DIR}/${GROUP_NAME}${DOM_SUB}${RUN_LABEL:+/$RUN_LABEL}"
    fi
    # ── Multi-seed loop (fineweb sampling) ───────────────────────────────────
    # Default runs seeds 42/43/44 into <act_diff>/{default,seed43,seed44}/; --seed
    # forces a single seed (n_seeds=1). The first seed → "default" subdir + the
    # canonical variant/link; others → seed<N> (variant <exp>_s<N>) — but ALL seeds
    # share ONE local_adapters symlink (the canonical link), since the model is the
    # same across seeds. After the loop, aggregate_seeds.py writes aggregate.json.
    local ACT_DIFF_DIR="$SUMMARY_OUT_DIR"
    local _seeds _canon _primary
    if [[ -n "$SEED" ]]; then _seeds="$SEED"; else _seeds="${SEEDS_MULTI:-$DEFAULT_SEEDS}"; fi
    _canon="${ADAPTER_LINK:-$EXP_NAME}"
    _primary="${_seeds%% *}"

    for _seed in $_seeds; do
        local _sub _exp
        if [[ "$_seed" == "$_primary" ]]; then
            _sub="default"; _exp="$EXP_NAME"
        else
            _sub="seed${_seed}"; _exp="${EXP_NAME}_s${_seed}"
        fi
        run_one_seed "$_exp" "$_canon" "$_seed" "$ADAPTER_DIR" "$ACT_DIFF_DIR/$_sub" "$DOM"
    done

    # ── Aggregate the per-seed summaries → aggregate.json (mean/std/per_seed) ──
    if [[ -n "$DESCRIPTION" ]]; then
        echo ""
        echo "=== Aggregating $(echo "$_seeds" | wc -w) seed(s) → $ACT_DIFF_DIR/aggregate.json ==="
        python3 "$AGGREGATE_SCRIPT" --act-diff-dir "$ACT_DIFF_DIR" --seeds "$_seeds" --primary "$_primary"
    fi

    echo ""
    echo "============================================================"
    echo "  Done: $EXP_NAME  (fineweb seeds: $_seeds)"
    echo "  Per-seed summaries: $ACT_DIFF_DIR/{default,seed<N>}/relevance_summary.txt"
    echo "  Aggregate:          $ACT_DIFF_DIR/aggregate.json"
    [[ -n "$DESCRIPTION_CONFIG" ]] && echo "  Description:        $(basename "$DESCRIPTION_CONFIG" .json)"
    echo "============================================================"
}

# ── Dispatch ──────────────────────────────────────────────────────────────────
# Outer loop over requested domains (general/task/both); inner dispatch is the
# usual single-adapter / all-runs split. A lone --adapter-dir runs flat by default
# (and with --single-run); --run-label nests it under a run subdir; --run-dir loops
# over run_1..run_N.
for dom in "${DOMAINS[@]}"; do
    if [[ -n "$ADAPTER_DIR" ]]; then
        process_adapter "$ADAPTER_DIR" "$RUN_LABEL" "$dom"
    else
        for i in $(seq 1 "$NUM_RUNS"); do
            _run_adapter="$RUN_DIR/run_${i}/final"
            if [[ ! -d "$_run_adapter" ]]; then
                echo "  Skipping run_${i}: $_run_adapter not found"
                continue
            fi
            process_adapter "$_run_adapter" "run${i}" "$dom"
        done
    fi
done
