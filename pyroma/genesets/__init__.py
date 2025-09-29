"""Gene set collections for pathway analysis."""

from __future__ import annotations
from pathlib import Path

def _get_resource_path(filename: str) -> str:
    """Get path to a resource file with backward compatibility."""
    try:
        
        from importlib.resources import files
        path = files(__package__).joinpath(filename)
        
        if hasattr(path, 'is_file'):
            if not path.is_file():
                raise FileNotFoundError(f"{filename} not found in package: {path!s}")
        return str(path)
    except (ImportError, AttributeError):
        # Python 3.7-3.8 fallback
        import pkg_resources
        path = pkg_resources.resource_filename(__package__, filename)
        if not Path(path).is_file():
            raise FileNotFoundError(f"{filename} not found in package: {path}")
        return path

def use_hallmarks() -> str:
    """
    Return the absolute path to the bundled Hallmark gene sets.
    Safe for both editable installs and wheels.
    """
    return _get_resource_path("h.all.v2023.1.Hs.symbols.gmt")

def use_reactome() -> str:
    """
    Return the absolute path to the bundled Reactome gene sets.
    Safe for both editable installs and wheels.
    """
    return _get_resource_path("ReactomePathways.gmt")

def use_progeny() -> str:
    """
    Return the absolute path to the bundled Progeny gene sets.
    Safe for both editable installs and wheels.
    """
    return _get_resource_path("progeny_p.gmt")

__all__ = ['use_hallmarks', 'use_reactome', 'use_progeny']