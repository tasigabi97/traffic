import collections, contextlib, itertools, logging, re, subprocess, typing, os, unittest.mock

abspath = os.path.abspath
Callable = typing.Callable
CalledProcessError = subprocess.CalledProcessError
check_output = subprocess.check_output
contextmanager = contextlib.contextmanager
cycle = itertools.cycle
DEBUG = logging.DEBUG
dirname = os.path.dirname
Formatter = logging.Formatter
getcwd = os.getcwd
getLogger = logging.getLogger
Iterable_abc = collections.abc.Iterable
Iterable_type = typing.Iterable
join_path = os.path.join
List = typing.List
LogRecord = logging.LogRecord
normpath = os.path.normpath
patch = unittest.mock.patch
Popen = subprocess.Popen
realpath = os.path.realpath
SequenceType = typing.Sequence
StreamHandler = logging.StreamHandler
Tuple = typing.Tuple
sub = re.sub
Union = typing.Union
