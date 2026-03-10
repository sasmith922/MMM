"""
paths.py
--------
Central pathlib.Path definitions for the project directory tree.

Import these objects anywhere in the codebase to avoid hard-coding paths.
All paths are resolved relative to the project root, which is determined
by walking upward from this file until pyproject.toml is found.
"""

from __future__ import annotations

from pathlib import Path


def _find_project_root(start: Path) -> Path:
    """Walk upward from *start* until a directory containing pyproject.toml is found.

    Parameters
    ----------
    start:
        Directory to begin the search from.

    Returns
    -------
    Path
        Absolute path to the project root directory.

    Raises
    ------
    FileNotFoundError
        If no pyproject.toml is found before reaching the filesystem root.
    """
    current = start.resolve()
    while current != current.parent:
        if (current / "pyproject.toml").exists():
            return current
        current = current.parent
    raise FileNotFoundError(
        "Could not locate project root (pyproject.toml not found)."
    )


# ---------------------------------------------------------------------------
# Root
# ---------------------------------------------------------------------------

PROJECT_ROOT: Path = _find_project_root(Path(__file__).parent)

# ---------------------------------------------------------------------------
# Data directories
# ---------------------------------------------------------------------------

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DATA_DIR: Path = DATA_DIR / "raw"
INTERIM_DATA_DIR: Path = DATA_DIR / "interim"
PROCESSED_DATA_DIR: Path = DATA_DIR / "processed"

# ---------------------------------------------------------------------------
# Model artifacts
# ---------------------------------------------------------------------------

MODELS_DIR: Path = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# Outputs
# ---------------------------------------------------------------------------

OUTPUTS_DIR: Path = PROJECT_ROOT / "outputs"
FIGURES_DIR: Path = OUTPUTS_DIR / "figures"
PREDICTIONS_DIR: Path = OUTPUTS_DIR / "predictions"
REPORTS_DIR: Path = OUTPUTS_DIR / "reports"

# ---------------------------------------------------------------------------
# Source & notebooks
# ---------------------------------------------------------------------------

SRC_DIR: Path = PROJECT_ROOT / "src"
NOTEBOOKS_DIR: Path = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR: Path = PROJECT_ROOT / "scripts"
TESTS_DIR: Path = PROJECT_ROOT / "tests"
