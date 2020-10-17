import cv2, imageio, numpy, pytest, scipy, skimage, termcolor
from matplotlib.pyplot import subplots, show, ion, cla
from matplotlib.patches import Rectangle, Polygon
from matplotlib.axes import Axes
from matplotlib.figure import Figure

any_np = numpy.any
apply_along_axis = numpy.apply_along_axis
array = numpy.array
Array_imageio = imageio.core.util.Array
Axes = Axes
bool_np = numpy.bool
cla = cla
cvtColor = cv2.cvtColor
colored = termcolor.colored
COLOR_RGB2LUV = cv2.COLOR_RGB2LUV
COLOR_RGB2GRAY = cv2.COLOR_RGB2GRAY
count_nonzero = numpy.count_nonzero
destroyAllWindows = cv2.destroyAllWindows
Figure = Figure
find_contours = skimage.measure.find_contours
fliplr = numpy.fliplr
ndarray = numpy.ndarray
max_np = numpy.max
histogram = numpy.histogram
imread = skimage.io.imread
imresize = scipy.misc.imresize
imshow = cv2.imshow
int32 = numpy.int32
ion = ion
label = skimage.measure.label
Polygon = Polygon
Rectangle = Rectangle
raises = pytest.raises
reshape = numpy.reshape
resize = skimage.transform.resize
show = show
subplots = subplots
uint8 = numpy.uint8
uint32 = numpy.uint32
unique = numpy.unique
VideoCapture = cv2.VideoCapture
waitKey = cv2.waitKey
zeros = numpy.zeros
