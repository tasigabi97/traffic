import cv2, imageio, numpy, pytest, scipy, skimage, termcolor
from matplotlib.pyplot import subplots, show, ion, cla
from matplotlib.pyplot import imshow as imshow_mat
from matplotlib.patches import Rectangle, Polygon
from matplotlib.axes import Axes
from matplotlib.figure import Figure

any_np = numpy.any
apply_along_axis = numpy.apply_along_axis
array_np = numpy.array
Array_imageio = imageio.core.util.Array
Axes = Axes
bool_np = numpy.bool
choice_np = numpy.random.choice
cla = cla
colored = termcolor.colored
COLOR_RGB2LUV = cv2.COLOR_RGB2LUV
COLOR_RGB2GRAY = cv2.COLOR_RGB2GRAY
count_nonzero = numpy.count_nonzero
cvtColor = cv2.cvtColor
destroyAllWindows = cv2.destroyAllWindows
expand_dims = numpy.expand_dims
Figure = Figure
find_contours = skimage.measure.find_contours
fliplr = numpy.fliplr
float64 = numpy.float64
max_np = numpy.max
mean_np = numpy.mean
ndarray = numpy.ndarray
histogram = numpy.histogram
imread_skimage = skimage.io.imread
imresize_scipy = scipy.misc.imresize
imshow_cv2 = cv2.imshow
imshow_mat = imshow_mat
int32 = numpy.int32
ion = ion
label = skimage.measure.label
Polygon = Polygon
Rectangle = Rectangle
raises = pytest.raises
reshape = numpy.reshape
rescale_skimage = skimage.transform.rescale
resize_skimage = skimage.transform.resize
show = show
subplots = subplots
uint8 = numpy.uint8
uint32 = numpy.uint32
unique = numpy.unique
VideoCapture = cv2.VideoCapture
waitKey = cv2.waitKey
zeros = numpy.zeros
