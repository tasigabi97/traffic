"""
Itt van az összes olyan felhasznált funkció, amit külön letöltött csomagban definiáltak.
"""
import cv2, imageio, numpy, PIL, pytest, scipy, skimage, shapely.geometry, termcolor, keras, tensorflow

# valamiért nem működik
# import matplotlib
# formában
from matplotlib.pyplot import subplots, show, ion, cla, figure, get_current_fig_manager, pause
from matplotlib.pyplot import imshow as imshow_mat
from matplotlib.patches import Rectangle, Polygon
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib import get_backend
from matplotlib.backend_bases import Event, MouseEvent, KeyEvent

Point_sh = shapely.geometry.Point
Polygon_sh = shapely.geometry.Polygon
KeyEvent = KeyEvent
Event = Event
MouseEvent = MouseEvent
CAP_PROP_BUFFERSIZE = cv2.CAP_PROP_BUFFERSIZE
pause = pause
figure_mat = figure
get_current_fig_manager = get_current_fig_manager
get_backend = get_backend
Adam_ke = keras.optimizers.Adam
any_np = numpy.any
apply_along_axis = numpy.apply_along_axis
argmax_np = numpy.argmax
array_equal = numpy.array_equal
array_np = numpy.array
Array_imageio = imageio.core.util.Array
assert_almost_equal_np = numpy.testing.assert_almost_equal
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
EarlyStopping_ke = keras.callbacks.EarlyStopping
expand_dims = numpy.expand_dims
equal_np = numpy.equal
Figure_mat = Figure
find_contours = skimage.measure.find_contours
fliplr = numpy.fliplr
float32 = numpy.float32
float64 = numpy.float64
load_model_ke = keras.models.load_model
imwrite_ima = imageio.imwrite
max_np = numpy.max
mean_np = numpy.mean
Model_ke = keras.models.Model
ModelCheckpoint_ke = keras.callbacks.ModelCheckpoint
ndarray = numpy.ndarray
histogram = numpy.histogram
imread_skimage = skimage.io.imread
imresize_scipy = scipy.misc.imresize
imshow_cv2 = cv2.imshow
imshow_mat = imshow_mat
Input_ke = keras.Input
int32 = numpy.int32
ion_mat = ion
label = skimage.measure.label
plot_model_ke = keras.utils.plot_model
Polygon_mat = Polygon
Rectangle = Rectangle
ReduceLROnPlateau_ke = keras.callbacks.ReduceLROnPlateau
raises = pytest.raises
reshape_np = numpy.reshape
Reshape_ke = keras.layers.Reshape
rescale_skimage = skimage.transform.rescale
resize_skimage = skimage.transform.resize
show = show
subplots = subplots
sum_np = numpy.sum
ones_np = numpy.ones
uint8 = numpy.uint8
uint32 = numpy.uint32
unique_np = numpy.unique
VideoCapture = cv2.VideoCapture
waitKey = cv2.waitKey
zeros = numpy.zeros
categorical_crossentropy_ke = keras.losses.categorical_crossentropy
binary_crossentropy_ke = keras.losses.binary_crossentropy
binary_crossentropy_tf = keras.backend.binary_crossentropy
sum_ke = keras.backend.sum
print_tensor = keras.backend.print_tensor
shape_tf = tensorflow.shape
Print_tf = tensorflow.Print
variable_ke = keras.backend.variable
eval_ke = keras.backend.eval
grey_dilation = scipy.ndimage.morphology.grey_dilation
concatenate_ke = keras.layers.concatenate
MaxPool2D_ke = keras.layers.MaxPool2D
multiply_tf = tensorflow.math.multiply
mean_ke = keras.backend.mean
TensorBoard_ke = keras.callbacks.TensorBoard
Variable_tf = tensorflow.Variable
Callback_ke = keras.callbacks.Callback
rotate_sci = scipy.ndimage.rotate
imrotate_sci = scipy.misc.imrotate
moveaxis_np = numpy.moveaxis
