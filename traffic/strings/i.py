from traffic.independent import SequenceType


def concat(strings: SequenceType[str], separator: str) -> str:
    return "".join(s if i == 0 else separator + s for i, s in enumerate(strings))
