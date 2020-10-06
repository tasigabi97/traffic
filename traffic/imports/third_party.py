import cv2, numpy, pytest, termcolor

cvtColor = cv2.cvtColor
colored = termcolor.colored
COLOR_RGB2LUV = cv2.COLOR_RGB2LUV
COLOR_RGB2GRAY = cv2.COLOR_RGB2GRAY
destroyAllWindows = cv2.destroyAllWindows
ndarray = numpy.ndarray
imshow = cv2.imshow
raises = pytest.raises
VideoCapture = cv2.VideoCapture
waitKey = cv2.waitKey
