# pyroma/__init__.py
from importlib import import_module

__all__ = [
    "ROMA", "GeneSetResult", "color",
    "integrate_projection_results",
    "datasets", "genesets",
]

def __getattr__(name):
    if name in {"ROMA", "GeneSetResult", "color"}:
        return getattr(import_module(".roma", __name__), name)
    if name == "integrate_projection_results":
        return import_module(".utils", __name__).integrate_projection_results
    if name == "datasets":
        return import_module(".datasets", __name__)
    if name == "genesets":
        return import_module(".genesets", __name__)
    raise AttributeError(name)
