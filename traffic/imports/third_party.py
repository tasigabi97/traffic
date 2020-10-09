import cv2, matplotlib, numpy, pytest, skimage, termcolor
from matplotlib.pyplot import subplots, show
from matplotlib.patches import Rectangle, Polygon

any_np = numpy.any
cvtColor = cv2.cvtColor
colored = termcolor.colored
COLOR_RGB2LUV = cv2.COLOR_RGB2LUV
COLOR_RGB2GRAY = cv2.COLOR_RGB2GRAY
destroyAllWindows = cv2.destroyAllWindows
find_contours = skimage.measure.find_contours
fliplr = numpy.fliplr
ndarray = numpy.ndarray
imread = skimage.io.imread
imshow = cv2.imshow
Polygon = Polygon
Rectangle = Rectangle
raises = pytest.raises
show = show
subplots = subplots
uint8 = numpy.uint8
uint32 = numpy.uint32
VideoCapture = cv2.VideoCapture
waitKey = cv2.waitKey
zeros = numpy.zeros
