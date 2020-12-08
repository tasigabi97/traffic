from traffic.imports import *
from traffic.logging import *
from traffic.consts import *
from traffic.utils import *
from traffic.mrcnn import *

ion_mat()
root_logger.debug(get_backend())


class Window:
    config_path = join_path(CONTAINER_ROOT_PATH, "area_config.json")
    center_path = join_path(CONTAINER_ROOT_PATH, "center_config.json")
    HORIZONTAL = "HORIZONTAL"
    VERTICAL = "VERTICAL"
    EXIT = "q"
    WAIT = "w"
    SHOW_ONLY_UPPER = "u"
    SHOW_BBOX = "b"
    SHOW_MASK = "m"
    SHOW_MASK_CONTOUR = "c"
    THRESHOLD = "t"
    NEW_CONFIG = "n"
    EDIT_CONFIG = "e"
    ENTER = "enter"
    BACKSPACE = "backspace"
    ESCAPE = "escape"
    SCROLL_UP = "up"
    SCROLL_DOWN = "down"
    MOUSE_LEFT = 1
    normalized_rgb_tuples = [(0, 1, 0), (1, 0, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]

    def __init__(self, fullscreen: bool):
        self.exit = False
        self.show_only_upper = True
        self.show_only_confident = False
        self.show_bbox = True
        self.show_mask = True
        self.show_mask_contour = True
        self.figure = figure_mat()
        self.figure.canvas.mpl_disconnect(self.figure.canvas.manager.key_press_handler_id)
        if fullscreen:
            mng = get_current_fig_manager()
            mng.resize(*mng.window.maxsize())
        self.mrcnn_axis = self.figure.add_subplot(2, 2, 1)
        self.unet_lane_axis = self.figure.add_subplot(2, 2, 2)
        self.unet_bg_axis = self.figure.add_subplot(2, 2, 3)
        self.area_axis = self.figure.add_subplot(2, 2, 4)
        self.unet_lane_threshold = 0.5
        self.unet_bg_threshold = 0.5
        cid = self.figure.canvas.mpl_connect("button_press_event", self.click)
        cid = self.figure.canvas.mpl_connect("key_press_event", self.press)
        cid = self.figure.canvas.mpl_connect("scroll_event", self.scroll)
        self.window_state = MainMenu(self)
        show()

    def set_axis(self, *, rgb_array: ndarray, detected_objects: List[DetectedObject], title: str, show_only_important: bool, show_only_confident: bool):
        normalized_rgb_tuple_cycle = cycle(self.normalized_rgb_tuples)
        point_size = 4
        point_marker = "o"
        text_size = 11
        rgb_array = rgb_array.copy()
        detected_objects = [d for d in detected_objects if d.has_bbox]
        if show_only_important:
            detected_objects = [d for d in detected_objects if d.important]
        if show_only_confident:
            detected_objects = [d for d in detected_objects if d.confident]
        if not len(detected_objects):
            root_logger.warning("No instances to display!")
        self.area_axis.cla()
        self.area_axis.axis("off")
        count_dict = dict()
        for area_name, xy_coords in self.area_dict.items():
            count_dict[area_name] = 0
            normalized_rgb_tuple = next(normalized_rgb_tuple_cycle)
            if 1 <= len(xy_coords):
                first_xy_point = xy_coords[0]
                self.area_axis.text(
                    first_xy_point[0] + text_size, first_xy_point[1] - text_size, area_name, color=normalized_rgb_tuple, size=text_size, backgroundcolor="none"
                )
            if 1 <= len(xy_coords) <= 2:
                self.area_axis.plot(*first_xy_point, color=normalized_rgb_tuple, marker=point_marker, markersize=point_size)
                if len(xy_coords) == 2:
                    second_xy_point = xy_coords[1]
                    self.area_axis.plot(*second_xy_point, color=normalized_rgb_tuple, marker=point_marker, markersize=point_size)
            elif 3 <= len(xy_coords):
                self.area_axis.add_patch(Polygon_mat(xy_coords, facecolor="none", edgecolor=normalized_rgb_tuple))
                polygon_sh = Polygon_sh(xy_coords)
                for detected_object in detected_objects:
                    detection_center_x = detected_object.x1 + (0.5 * detected_object.width)
                    detection_center_y = detected_object.y1 + (0.5 * detected_object.height)
                    detection_center_point = Point_sh(detection_center_x, detection_center_y)
                    is_in_area = polygon_sh.contains(detection_center_point)
                    if is_in_area:
                        count_dict[area_name] += 1
                    detection_color = normalized_rgb_tuple if is_in_area else (1, 1, 1)
                    self.area_axis.plot(*(detection_center_point.xy), color=detection_color, marker=point_marker, markersize=point_size)
        title = title or "{}\n{}".format(self.set_axis.__name__, "".join([k + "(" + str(v) + ") " for k, v in count_dict.items() if v > 0]))
        self.area_axis.set_title(title)
        self.area_axis.imshow(rgb_array)

    @property
    def area_dict(self):
        if not exists(self.config_path):
            return dict()
        return get_dict_from_json(self.config_path)

    @area_dict.setter
    def area_dict(self, d: dict):
        save_dict_to_json(self.config_path, d)

    @property  # one file
    def center_file(self):
        if not exists(self.center_path):
            return {self.HORIZONTAL: 0.5, self.VERTICAL: 0.5}
        return get_dict_from_json(self.config_path)

    @center_file.setter
    def center_file(self, d: dict):
        save_dict_to_json(self.center_path, d)

    @property
    def title(self) -> str:
        return self._title

    @title.setter
    def title(self, title: str):
        self._title = title
        self.figure.suptitle(self.title, fontsize=13)

    def draw(self):
        self.figure.canvas.draw()
        pause(0.1)
        return not self.exit

    def click(self, event: MouseEvent):
        self.window_state.click(event)

    def press(self, event: KeyEvent):
        self.window_state.press(event)

    def scroll(self, event: MouseEvent):
        self.window_state.scroll(event)


class WindowState:
    keys = dict()

    def __init__(self, window: Window):
        self.window = window

    def go_to(self, window_state: Union["WindowState", type]):
        if isinstance(window_state, WindowState):
            ...
        elif issubclass(window_state, WindowState):
            window_state = window_state(self.window)
        else:
            raise TypeError(self.go_to.__name__)
        self.window.window_state = window_state

    def click(self, event: MouseEvent):
        ...

    def scroll(self, event: MouseEvent):
        ...

    def press(self, event: KeyEvent):
        ...

    def __str__(self):
        return self.__class__.__name__ + "".join(["\n" + k + " ->" + v for k, v in self.keys.items()])


class ExitState(WindowState):
    def __init__(self, window: Window):
        super().__init__(window)
        self.window.exit = True


class MainMenu(WindowState):
    keys = {
        Window.EXIT: "Quit program",
        Window.THRESHOLD: "Edit threshold",
        Window.NEW_CONFIG: "Create new config file",
        Window.EDIT_CONFIG: "Edit the config file",
    }

    def __init__(self, window: Window):
        super().__init__(window)
        self.window.title = str(self)

    def press(self, event: KeyEvent):
        if event.key == Window.THRESHOLD:
            self.go_to(ThresholdSetter)
        elif event.key == Window.EXIT:
            self.go_to(ExitState)
        elif event.key == Window.NEW_CONFIG:
            self.window.area_dict = dict()
            self.go_to(AreaNameSetter)
        elif event.key == Window.EDIT_CONFIG:
            self.go_to(AreaNameSetter)


class ThresholdSetter(WindowState):
    keys = {Window.ESCAPE: "Back to main menu", "Scroll up on axis": "Increase the axis threshold", "Scroll down on axis": "Decrease the axis threshold"}

    def __init__(self, window: Window):
        super().__init__(window)
        self.window.title = str(self)

    def press(self, event: KeyEvent):
        if event.key == Window.ESCAPE:
            self.go_to(MainMenu)

    def scroll(self, event: MouseEvent):
        step_size = 0.05
        if event.button == Window.SCROLL_UP and event.inaxes is self.window.unet_lane_axis:
            self.window.unet_lane_threshold = min(1 - step_size, self.window.unet_lane_threshold + step_size)
        elif event.button == Window.SCROLL_UP and event.inaxes is self.window.unet_bg_axis:
            self.window.unet_bg_threshold = min(1 - step_size, self.window.unet_bg_threshold + step_size)
        elif event.button == Window.SCROLL_DOWN and event.inaxes is self.window.unet_lane_axis:
            self.window.unet_lane_threshold = max(step_size, self.window.unet_lane_threshold - step_size)
        elif event.button == Window.SCROLL_DOWN and event.inaxes is self.window.unet_bg_axis:
            self.window.unet_bg_threshold = max(step_size, self.window.unet_bg_threshold - step_size)
        root_logger.info("Bg->{}, Lane->{}".format(self.window.unet_bg_threshold, self.window.unet_lane_threshold))


class AreaNameSetter(WindowState):
    keys = {Window.ESCAPE: "Back to main menu", Window.BACKSPACE: "Delete last character", Window.ENTER: "Save actual area name"}

    def __init__(self, window: Window):
        super().__init__(window)
        self.area_name = ""
        self.window.title = str(self)

    def press(self, event: KeyEvent):
        if len(event.key) == 1 and event.key.isalnum():
            self.area_name += event.key
            self.window.title = self.area_name
        elif event.key == Window.ENTER:
            self.go_to(AreaPointSetter(self.window, self.area_name))
        elif event.key == Window.BACKSPACE:
            self.area_name = self.area_name[:-1]
            self.window.title = self.area_name
        elif event.key == Window.ESCAPE:
            self.go_to(MainMenu)


class AreaPointSetter(WindowState):
    keys = {"LClick on the screen": "Add a point", Window.BACKSPACE: "Remove last point", Window.ENTER: "Finished with the actual area"}

    def __init__(self, window: Window, area_name):
        super().__init__(window)
        self.area_name = area_name
        area_file = self.window.area_dict
        area_file[self.area_name] = []
        self.window.area_dict = area_file
        self.window.title = str(self)

    @property
    def title(self):
        return "['{}']=={}".format(self.area_name, [[int(x), int(y)] for x, y in self.window.area_dict[self.area_name]])

    def press(self, event: KeyEvent):
        if event.key == Window.ENTER:
            if len(self.window.area_dict[self.area_name]) >= 3:
                self.go_to(AreaNameSetter)
            else:
                self.window.title = "Minimum 3 points are needed!"
        elif event.key == Window.BACKSPACE:
            area_file = self.window.area_dict
            if len(area_file[self.area_name]) >= 1:
                area_file[self.area_name].pop()
            self.window.area_dict = area_file
            self.window.title = self.title

    def click(self, event: MouseEvent):
        if event.button == Window.MOUSE_LEFT and event.inaxes is self.window.area_axis:
            area_file = self.window.area_dict
            area_file[self.area_name].append((round(event.xdata), round(event.ydata)))
            self.window.area_dict = area_file
            self.window.title = self.title
