"""
Ennek csak a program tesztelése során volt haszna.
Milyen gyors az mrcnn/one hot kódolás stb.
"""
from traffic.imports import perf_counter, mean
from traffic.logging import root_logger


class Timer(object):
    def __init__(self, name: str):
        self.name = name
        self._init_time = perf_counter()
        self._block_times = []
        self._actual_block_start_time = None

    def __enter__(self):
        self._actual_block_start_time = perf_counter()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        actual_block_time = perf_counter() - self._actual_block_start_time
        self._block_times.append(actual_block_time)
        root_logger.info("Timer ({})'s actual_block_time: {}".format(self.name, actual_block_time))

    @property
    def mean_block_time(self):
        block_times = self.block_times
        if not block_times:
            return
        mean_block_time = mean(block_times)
        root_logger.info("Timer ({})'s mean_block_time: {}".format(self.name, mean_block_time))
        return mean_block_time

    @property
    def last_block_time(self):
        block_times = self.block_times
        if not block_times:
            return
        last_block_time = block_times[-1]
        root_logger.info("Timer ({})'s last_block_time: {}".format(self.name, last_block_time))
        return last_block_time

    @property
    def block_times(self):
        if self._block_times == []:
            root_logger.warning("Timer ({}) was not called yet.".format(self.name))
            return None
        return self._block_times
