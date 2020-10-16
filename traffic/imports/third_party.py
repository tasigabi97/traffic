import cv2, imageio, numpy, pytest, scipy, skimage, termcolor
from matplotlib.pyplot import subplots, show, ion, cla
from matplotlib.patches import Rectangle, Polygon
from matplotlib.axes import Axes
from matplotlib.figure import Figure

any_np = numpy.any
apply_along_axis = numpy.apply_along_axis
Array_imageio = imageio.core.util.Array
Axes = Axes
cla = cla
cvtColor = cv2.cvtColor
colored = termcolor.colored
COLOR_RGB2LUV = cv2.COLOR_RGB2LUV
COLOR_RGB2GRAY = cv2.COLOR_RGB2GRAY
destroyAllWindows = cv2.destroyAllWindows
Figure = Figure
find_contours = skimage.measure.find_contours
fliplr = numpy.fliplr
ndarray = numpy.ndarray
histogram = numpy.histogram
imread = skimage.io.imread
imresize = scipy.misc.imresize
imshow = cv2.imshow
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
