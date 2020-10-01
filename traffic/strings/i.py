from traffic.independent import SequenceType


def concatenate_with_separation(strings: SequenceType[str], separator: str) -> str:
    return "".join(s if i == 0 else separator + s for i, s in enumerate(strings))
