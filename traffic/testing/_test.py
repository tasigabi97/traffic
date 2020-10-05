from traffic.testing import name


@name(lambda: 1, "1", globals())
def _():
    ...
