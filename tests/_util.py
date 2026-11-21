"""Test helpers.

Stage folders are named `00_setup`, `01_autograd`, ... which are not valid Python
identifiers, so they can't be imported with a normal `import` statement. This loads them
by file path instead.
"""

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def load_stage_module(stage_dir: str, module_name: str):
    """Load e.g. load_stage_module("00_setup", "tensor_drills")."""
    path = REPO_ROOT / "stages" / stage_dir / f"{module_name}.py"
    if not path.exists():
        raise FileNotFoundError(f"Expected you to write {path}")
    spec = importlib.util.spec_from_file_location(f"{stage_dir}_{module_name}", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module
