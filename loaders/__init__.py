import pkgutil, importlib, inspect
from .base_loader import BaseLoader

REGISTRY = {}

for module_info in pkgutil.iter_modules(__path__):
    module = importlib.import_module(f"{__name__}.{module_info.name}")
    for _, obj in inspect.getmembers(module):
        if inspect.isclass(obj) and issubclass(obj, BaseLoader) and obj is not BaseLoader:
            REGISTRY[obj.__name__] = obj