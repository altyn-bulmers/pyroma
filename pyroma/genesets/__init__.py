from __future__ import annotations
from importlib.resources import files
from pathlib import Path

def use_hallmarks() -> str:
    """
    Return the absolute path to the bundled Hallmark gene sets (hallmarks.gmt).
    Safe for both editable installs and wheels.
    """
    path = files(__package__).joinpath("h.all.v2023.1.Hs.symbols.gmt")
    # Optional: sanity check
    if not path.is_file():
        raise FileNotFoundError(f"hallmarks.gmt not found in package: {path!s}")
    return str(path)

def use_reactome() -> str:
    """
    Return the absolute path to the bundled Reactome gene sets.
    Safe for both editable installs and wheels.
    """
    path = files(__package__).joinpath("ReactomePathways.gmt")
    # Optional: sanity check
    if not path.is_file():
        raise FileNotFoundError(f"ReactomePathways.gmt not found in package: {path!s}")
    return str(path)

def use_progeny() -> str:
    """
    Return the absolute path to the bundled Progengy gene sets.
    Top 500 genes were selected based on the p-value of Progeny Pathways.
    Safe for both editable installs and wheels.
    """
    path = files(__package__).joinpath("progeny_p.gmt")
    # Optional: sanity check
    if not path.is_file():
        raise FileNotFoundError(f"progeny_p.gmt not found in package: {path!s}")
    return str(path)