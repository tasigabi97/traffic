import cv2, imageio, numpy, pytest, scipy, skimage, termcolor, keras
from matplotlib.pyplot import subplots, show, ion, cla
from matplotlib.pyplot import imshow as imshow_mat
from matplotlib.patches import Rectangle, Polygon
from matplotlib.axes import Axes
from matplotlib.figure import Figure

Adam_ke=keras.optimizers.Adam
any_np = numpy.any
apply_along_axis = numpy.apply_along_axis
array_equal = numpy.array_equal
array_np = numpy.array
Array_imageio = imageio.core.util.Array
assert_almost_equal_np=numpy.testing.assert_almost_equal
Axes = Axes
bool_np = numpy.bool
choice_np = numpy.random.choice
cla = cla
colored = termcolor.colored
COLOR_RGB2LUV = cv2.COLOR_RGB2LUV
COLOR_RGB2GRAY = cv2.COLOR_RGB2GRAY
Conv2D_ke = keras.layers.Conv2D
Conv2DTranspose_ke = keras.layers.Conv2DTranspose
count_nonzero = numpy.count_nonzero
cvtColor = cv2.cvtColor
destroyAllWindows = cv2.destroyAllWindows
EarlyStopping_ke=keras.callbacks.EarlyStopping
expand_dims = numpy.expand_dims
equal_np=numpy.equal
Figure = Figure
find_contours = skimage.measure.find_contours
fliplr = numpy.fliplr
float64 = numpy.float64
load_model_ke=keras.models.load_model
max_np = numpy.max
mean_np = numpy.mean
Model_ke = keras.models.Model
ModelCheckpoint_ke=keras.callbacks.ModelCheckpoint
ndarray = numpy.ndarray
histogram = numpy.histogram
imread_skimage = skimage.io.imread
imresize_scipy = scipy.misc.imresize
imshow_cv2 = cv2.imshow
imshow_mat = imshow_mat
Input_ke = keras.Input
int32 = numpy.int32
ion = ion
label = skimage.measure.label
plot_model_ke = keras.utils.plot_model
Polygon = Polygon
Rectangle = Rectangle
ReduceLROnPlateau_ke=keras.callbacks.ReduceLROnPlateau
raises = pytest.raises
reshape = numpy.reshape
rescale_skimage = skimage.transform.rescale
resize_skimage = skimage.transform.resize
show = show
subplots = subplots
sum_np=numpy.sum
ones_np=numpy.ones
uint8 = numpy.uint8
uint32 = numpy.uint32
unique_np = numpy.unique
VideoCapture = cv2.VideoCapture
waitKey = cv2.waitKey
zeros = numpy.zeros
