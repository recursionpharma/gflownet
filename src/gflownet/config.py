import json
from pathlib import Path
import subprocess
import ast
from collections import defaultdict, namedtuple
import importlib.util
import sys
from typing import Any
from numpydoc.docscrape import NumpyDocString

_recursive_dd = lambda: defaultdict(_recursive_dd)
_recursive_dd_get = lambda d, keys: _recursive_dd_get(d[keys[0]], keys[1:]) if len(keys) else d


def _recursive_creative_setattr(o, keys, default, value):
    if len(keys) == 1:
        setattr(o, keys[0], value)
        return
    if not hasattr(o, keys[0]):
        setattr(o, keys[0], default())
    _recursive_creative_setattr(getattr(o, keys[0]), keys[1:], default, value)


def _recursive_getattr(o, keys):
    if len(keys) == 1:
        return getattr(o, keys[0])
    return _recursive_getattr(getattr(o, keys[0]), keys[1:])


ConfigAttr = namedtuple("ConfigAttr", ["type", "docstring"])


class Visitor(ast.NodeVisitor):
    def __init__(self):
        self.config = _recursive_dd()

    def start_module(self, path, local_path):
        self._mod_path = path
        self._local_mod_path = Path(local_path)
        self._should_import = False

    def finalize_module(self):
        if self._should_import:
            assert self._local_mod_path.suffix == ".py" and self._local_mod_path.parts[0] == "src"
            module_name = ".".join(self._local_mod_path.parts[1:-1]) + "." + self._local_mod_path.name[: -len(".py")]
            if module_name in sys.modules:
                # We've already imported this module because some other module imported it, no need to do it again
                return
            spec = importlib.util.spec_from_file_location(module_name, self._mod_path)
            mod = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = mod
            spec.loader.exec_module(mod)

    def output_stub(self):
        imports = ["from typing import *"]
        for cname, cobj in _name_to_config.items():
            # We first extract possible docstrings for the attributes
            docstrings = defaultdict(lambda: "")
            if cobj.__doc__ is not None:
                doc = NumpyDocString(cobj.__doc__)
                for doc_attr in doc["Attributes"]:
                    docstrings[doc_attr.name] = "\n".join(doc_attr.desc)
            # We then extract the attributes themselves and infer their types from the class objects
            if cname == "@base":
                attr_dict = self.config
            else:
                attr_dict = _recursive_dd_get(self.config, cname.split("."))
            for attr_name in dir(cobj):
                if attr_name.startswith("__"):
                    continue  # ignore dunderscores
                t = type(getattr(cobj, attr_name))
                attr_dict[attr_name] = ConfigAttr(t.__name__, docstrings[attr_name])
                if t.__module__ != "builtins":
                    imports.append(f"from {t.__module__} import {t.__name__}")
            # But we allow for explicit type annotations to override the inferred types
            for attr_name, t in cobj.__annotations__.items() if hasattr(cobj, "__annotations__") else []:
                # If the type is a class, we use its name, otherwise we have to do some guessing
                if isinstance(t, type):
                    tname = iname = t.__name__
                else:
                    tstr = str(t)
                    if tstr.startswith("typing."):
                        tname = tstr[len("typing.") :]
                        iname = None
                    else:
                        tname = iname = tstr
                attr_dict[attr_name] = ConfigAttr(tname, docstrings[attr_name])
                if t.__module__ != "builtins" and iname is not None:
                    imports.append(f"from {t.__module__} import {iname}")

        def f(name, d, indentlevel):
            s = ""
            s += "    " * indentlevel + f"class {name}:\n"
            for k, v in sorted(d.items(), key=lambda x: "_" + x[0] if not isinstance(x[1], dict) else x[0]):
                if isinstance(v, dict):
                    s += f(k, v, indentlevel + 1)
                else:
                    s += "    " * (indentlevel + 1) + f"{k}: {v.type}\n"
                    if v.docstring:
                        s += "    " * (indentlevel + 1) + f'"""{v.docstring}"""\n'
                # print(i)
            if not len(d.items()):
                print(f"Empty config class {name}?")
                s += "    " * (indentlevel + 1) + "...\n"
            return s

        s = "\n".join(sorted(set(imports))) + "\n\n"
        s += f("Config", self.config, 0)
        s += """def config_class(name): ...
def config_from_dict(config_dict: dict[str, Any]) -> Config: ...
def make_config() -> Config: ...
def update_config(config: Config, config_dict: dict[str, Any]) -> Config: ...
def config_to_dict(config: Config) -> dict[str, Any]: ...
"""
        with open(Path(__file__).parent / "config.pyi", "w") as f:
            f.write(s)

    def visit_ClassDef(self, node):
        for dec in node.decorator_list:
            if (
                isinstance(dec, ast.Call)
                and isinstance(dec.func, ast.Name)
                and dec.func.id == "config_class"
                and len(dec.args) == 1
                and isinstance(dec.args[0], ast.Constant)
            ):
                print("Found config class", dec.args[0].value)
                self._should_import = True
        self.generic_visit(node)


class Config:
    pass


def make_config():
    config = _name_to_config.get("@base", Config)()
    for cname, cobj in sorted(_name_to_config.items()):
        if cname == "@base":
            continue
        _recursive_creative_setattr(config, cname.split("."), Config, cobj())
    return config


def update_config(config, config_dict):
    for cname, val in config_dict.items():
        _recursive_creative_setattr(config, cname.split("."), Config, val)


def config_from_dict(config_dict) -> Config:
    config = make_config()
    update_config(config, config_dict)
    check_config(config)
    return config


def check_config(config):
    for cname, cobj in _name_to_config.items():
        if cname == "@base":
            continue
        subcfg = _recursive_getattr(config, cname.split("."))
        for name, t in cobj.__annotations__.items() if hasattr(cobj, "__annotations__") else []:
            if not hasattr(subcfg, name):
                print(f"Warning, setting {cname}.{name} was declared but not defined in created config")


def config_to_dict(config):
    d = {}
    config_classes = tuple(_name_to_config.values()) + (Config,)
    for cname, _ in _name_to_config.items():
        if cname == "@base":
            csplit = []
            subcfg = config
        else:
            csplit = cname.split(".")
            subcfg = _recursive_getattr(config, csplit)
        for i in dir(subcfg):
            if i.startswith("__"):
                continue
            o = getattr(subcfg, i)
            if isinstance(o, config_classes):
                continue
            d[".".join(csplit + [i])] = o
    return d


def config_class(name):
    assert isinstance(name, str), "config_class decorator must be called with a section name"

    def decorator(c):
        if name in _name_to_config:
            print(c, _name_to_config[name])
            raise ValueError("Redefining", name, "is not allowed; found", c, "and", _name_to_config[name])
        _name_to_config[name] = c
        return c

    return decorator


_name_to_config: dict[str, Any]
if __name__ != "__main__" and hasattr(sys.modules["__main__"], "__main__sentinel"):
    # If we reach this point, config is being imported as a module by another part of the package, while we are running
    # the __main__ script below. In this case, we don't want to overwrite the global _name_to_config, so we just use
    # the existing one.
    _name_to_config = sys.modules["__main__"]._name_to_config
else:
    _name_to_config = {}

if __name__ == "__main__":
    __main__sentinel = True
    repo = subprocess.check_output("git rev-parse --show-toplevel", shell=True).decode().strip()
    print("Working in repo", repo)
    files = (
        subprocess.check_output(f'git ls-files | grep -e ".py$"', shell=True, cwd=repo).decode().strip().splitlines()
    )

    visitor = Visitor()
    for f in files:
        if not f.startswith("src/"):  # Only package files in src/
            continue
        path = Path(repo) / f
        print("Processing", f)
        root = ast.parse(open(path, "r").read())
        visitor.start_module(path, f)
        visitor.visit(root)
        visitor.finalize_module()
    visitor.output_stub()
    print("Default config:")
    d = config_to_dict(make_config())
    print(json.dumps(d, indent=2, sort_keys=True))
