"""Static check of the Python a model writes for the analysis kernel (experiment C).

The paper ran model-written code in a bubblewrap sandbox with no network. That does not exist on
Windows, so this is the weaker thing that does: the code is parsed and refused unless it stays
inside a small allow-list, and it then runs with a reduced set of builtins.

Allowed: numpy and a few pure standard-library modules, ordinary Python. Refused: any other
import, any name or attribute that starts with an underscore (which is how Python sandboxes are
usually escaped), file / process / network access, and the numpy calls that read or write files.

This is a guard against accidents and against the obvious, NOT a security boundary. The writer is
the same model family the repository owner already runs with shell access; do not reuse this to run
code from a source you do not trust.
"""
import ast

ALLOWED_MODULES = {"numpy", "math", "collections", "itertools", "functools", "statistics"}
FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "input", "globals", "locals", "vars",
    "getattr", "setattr", "delattr", "breakpoint", "exit", "quit", "help", "memoryview", "type",
    "object", "super", "classmethod", "staticmethod", "property", "dir", "id",
}
FORBIDDEN_ATTRS = {
    # numpy file and foreign-memory access
    "load", "save", "savez", "savez_compressed", "loadtxt", "savetxt", "genfromtxt", "fromfile",
    "tofile", "memmap", "open_memmap", "fromregex", "DataSource", "ctypeslib", "ctypes", "f2py",
    "distutils", "testing", "dump", "dumps", "loads", "tobytes", "frombuffer", "lib", "show_config",
    # anything process- or filesystem-like reached through an object
    "system", "popen", "remove", "unlink", "rmdir", "mkdir", "makedirs", "rename", "environ",
    "subprocess", "socket", "modules", "path",
}


class Refused(Exception):
    pass


def check(code):
    """Parse `code`; raise Refused with a reason, or return the compiled code object."""
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise Refused("syntax error: %s (line %s)" % (exc.msg, exc.lineno))
    for node in ast.walk(tree):
        line = getattr(node, "lineno", "?")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_MODULES or "." in alias.name:
                    raise Refused("line %s: import of %r is not allowed (allowed: %s)"
                                  % (line, alias.name, ", ".join(sorted(ALLOWED_MODULES))))
        elif isinstance(node, ast.ImportFrom):
            if node.level or (node.module or "") not in ALLOWED_MODULES:
                raise Refused("line %s: import from %r is not allowed" % (line, node.module))
            for alias in node.names:
                if alias.name.startswith("_") or alias.name in FORBIDDEN_ATTRS or alias.name == "*":
                    raise Refused("line %s: import of %r is not allowed" % (line, alias.name))
        elif isinstance(node, ast.Name):
            if node.id.startswith("_") or node.id in FORBIDDEN_NAMES:
                raise Refused("line %s: the name %r is not allowed" % (line, node.id))
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith("_") or node.attr in FORBIDDEN_ATTRS:
                raise Refused("line %s: the attribute .%s is not allowed" % (line, node.attr))
        elif isinstance(node, (ast.Global, ast.Nonlocal, ast.ClassDef, ast.AsyncFunctionDef,
                               ast.Await, ast.AsyncFor, ast.AsyncWith)):
            raise Refused("line %s: %s is not allowed" % (line, type(node).__name__))
        elif isinstance(node, (ast.FunctionDef, ast.arg)):
            name = node.name if isinstance(node, ast.FunctionDef) else node.arg
            if name.startswith("_"):
                raise Refused("line %s: the name %r is not allowed" % (line, name))
    return compile(tree, "<analysis>", "exec")


def safe_builtins():
    import builtins

    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        if level or name.split(".")[0] not in ALLOWED_MODULES:
            raise ImportError("import of %r is not allowed" % name)
        return builtins.__import__(name, globals, locals, fromlist, level)

    names = ("abs all any bool dict divmod enumerate filter float format frozenset int isinstance "
             "len list map max min pow print range repr reversed round set slice sorted str sum "
             "tuple zip True False None ArithmeticError AssertionError Exception IndexError "
             "KeyError RuntimeError StopIteration TypeError ValueError ZeroDivisionError")
    table = {n: getattr(builtins, n) for n in names.split()}
    table["__import__"] = guarded_import
    return table
