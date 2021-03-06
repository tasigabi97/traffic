from traffic.imports import Callable
from traffic.strings import concat, to_func_name
from traffic.consts import DOT


def name(tested: Callable, name: str, globals: dict):
    def decorator(func):
        new_name = to_func_name(concat(["test", tested.__qualname__, name], separator="_"))
        if new_name in globals.keys():
            raise NameError("GIVE UNIQUE NAME FOR THE TEST")
        func.__name__ = new_name
        globals[new_name] = func
        return func

    return decorator


def absolute_name(tested: Callable):
    return concat([tested.__module__, tested.__name__], DOT)


def relative_name(__name__: str, tested: Callable):
    return concat([__name__, tested.__name__], DOT)
