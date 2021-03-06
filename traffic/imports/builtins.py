"""
Itt van az összes igénybevett beépített funkció.
"""
import collections, contextlib, functools, inspect, itertools, json, logging, math, random, re, statistics, subprocess, time, threading, typing, os, unittest.mock

abspath = os.path.abspath
call_mock = unittest.mock.call
Callable = typing.Callable
CalledProcessError = subprocess.CalledProcessError
check_output = subprocess.check_output
choice = random.choice
contextmanager = contextlib.contextmanager
cycle = itertools.cycle
DEBUG = logging.DEBUG
dirname = os.path.dirname
dump_json = json.dump
load_json = json.load
exists = os.path.exists
Formatter = logging.Formatter
getcwd = os.getcwd
getLogger = logging.getLogger
INFO = logging.INFO
Iterable_abc = collections.abc.Iterable
Iterable_type = typing.Iterable
join_path = os.path.join
List = typing.List
listdir = os.listdir
LogRecord = logging.LogRecord
mean = statistics.mean
mkdir = os.mkdir
MagicMock = unittest.mock.MagicMock
mock_open = unittest.mock.mock_open
namedtuple = collections.namedtuple
normpath = os.path.normpath
patch = unittest.mock.patch
perf_counter = time.perf_counter
PropertyMock = unittest.mock.PropertyMock
Popen = subprocess.Popen
randrange = random.randrange
realpath = os.path.realpath
remove_os = os.remove
sentinel = unittest.mock.sentinel
SequenceType = typing.Sequence
shuffle = random.shuffle
signature = inspect.signature
sleep = time.sleep
StreamHandler = logging.StreamHandler
Tuple = typing.Tuple
sub = re.sub
Union = typing.Union
wraps = functools.wraps
logarithm = math.log
uniform = random.uniform
random_r = random.random
WARNING = logging.WARNING
Thread = threading.Thread
