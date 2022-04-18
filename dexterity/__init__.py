from pathlib import Path

__version__ = "0.0.3"

# Path to the root of the project.
_PROJECT_ROOT: Path = Path(__file__).absolute().parent.parent

# Path to the root of the src files, i.e. `dexterity/`.
_SRC_ROOT: Path = _PROJECT_ROOT / "dexterity"
