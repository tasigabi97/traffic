from logging import getLogger, DEBUG, StreamHandler, Formatter, LogRecord
from larning.output import blue, green, red, on_white


class TrafficFormatter(Formatter):
    def format(self, record: LogRecord) -> str:
        level = record.levelname
        if level == "DEBUG":
            record.msg = blue(record.msg)
        elif level == "INFO":
            record.msg = green(record.msg)
        elif level == "WARNING":
            record.msg = red(record.msg)
        record.levelname = on_white(record.levelname)
        return super().format(record)


_formatter = TrafficFormatter("%(levelname)s|%(funcName)s->q%(message)s")
_stdout_handler = StreamHandler()
_stdout_handler.setLevel(DEBUG)
_stdout_handler.setFormatter(_formatter)
root_logger = getLogger()
root_logger.setLevel(DEBUG)
root_logger.addHandler(_stdout_handler)
