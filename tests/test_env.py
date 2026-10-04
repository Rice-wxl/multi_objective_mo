from importlib.metadata import version

PINS = {"torch": "2.10.0", "transformers": "5.16.1", "peft": "0.19.1",
        "trl": "0.28.0", "accelerate": "1.12.0", "datasets": "4.5.0"}


def test_training_stack_pins():
    got = {p: version(p).split("+")[0] for p in PINS}   # torch reports 2.10.0+cu128
    assert got == PINS


def test_package_imports():
    import multi_obj_mo.clinical.eval  # noqa: F401
