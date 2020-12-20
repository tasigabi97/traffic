from traffic.imports.builtins import SequenceType, sub


def concat(strings: SequenceType[str], separator: str) -> str:
    return "".join(s if i == 0 else separator + s for i, s in enumerate(strings))


def to_func_name(name: str):
    return sub(r"\W", "_", name).lower()
