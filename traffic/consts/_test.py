from traffic.consts import HOST_ROOT_PATH, CONTAINER_ROOT_PATH


def test_roots_are_same_on_docker():
    assert CONTAINER_ROOT_PATH == HOST_ROOT_PATH
