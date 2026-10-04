"""trl 0.28 + transformers 5.16: `from trl import DPOTrainer` crashes on an absent optional
package (weave) unless multi_objective_mo.training._trl_compat is imported first."""
import subprocess
import sys


def _run(code):
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)


def test_dpo_trainer_imports_with_shim():
    r = _run("import multi_objective_mo.training._trl_compat; from trl import DPOTrainer")
    assert r.returncode == 0, r.stderr


def test_dpo_trainer_import_fails_without_shim():
    r = _run("from trl import DPOTrainer")
    assert r.returncode != 0 and "weave" in r.stderr, r.stderr
