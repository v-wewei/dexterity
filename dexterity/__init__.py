from pathlib import Path

__version__ = "0.0.2"

# Path to the root of the project.
_PROJECT_ROOT: Path = Path(__file__).parent.parent

# Path to the root of the src files, i.e. `dexterity/`.
_SRC_ROOT: Path = Path(__file__).parent.parent / "dexterity"
