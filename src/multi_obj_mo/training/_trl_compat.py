"""trl 0.28 x transformers >=5.5 compat shim. Import BEFORE any `from trl import ...Trainer`.

Moved verbatim from the research repo's dpo_spurious.py. Remove when trl is upgraded past 0.28
(trl >=0.29 ships its own bool-returning _is_package_available; see README for the rpo_alpha migration).
"""
# --- trl 0.28 x transformers >=5.5 compat shim (must run before importing trl
# trainers) ---
# transformers changed _is_package_available to always return (bool, version).
# trl 0.28's simple is_*_available() helpers do `return _is_package_available(name)`,
# so they hand back a truthy tuple even for absent packages (weave, mergekit,
# unsloth, math_verify, llm_blender, ...) -> trl then hard-imports them and crashes.
# We patch ONLY trl's binding (not transformers' global — transformers' own
# is_datasets_available() etc. subscript the tuple `[0]` and must keep it). trl
# 0.28 never subscripts, so returning a bool for the default call is safe there;
# return_version=True still yields the tuple for trl's version checks (liger/vllm).
import trl.import_utils as _trl_iu  # noqa: E402  (import_utils only; no trainers yet)
if not getattr(_trl_iu, "_is_pkg_bool_patched", False):
    _orig_is_package_available = _trl_iu._is_package_available

    def _is_package_available(pkg_name, return_version=False):
        res = _orig_is_package_available(pkg_name, return_version=True)
        exists, ver = res if isinstance(res, tuple) else (res, None)
        return (exists, ver) if return_version else exists

    _trl_iu._is_package_available = _is_package_available
    _trl_iu._is_pkg_bool_patched = True
