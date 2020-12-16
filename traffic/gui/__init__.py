from traffic.imports import *
from traffic.logging import *
from traffic.consts import *
from traffic.utils import *
from traffic.mrcnn import *
from traffic.utils.lane_unet import *

ion_mat()
root_logger.debug(get_backend())


class Button:
    ENTER = "enter"
    BACKSPACE = "backspace"
    ESCAPE = "escape"
    UP_ARROW = "up"
    DOWN_ARROW = "down"
    RIGHT_ARROW = "right"
    LEFT_ARROW = "left"
    SCROLL_UP = "up"
    SCROLL_DOWN = "down"
    MOUSE_LEFT = 1

    def __init__(self, matplotlib_const, program_will_do_this: str, user_have_to_do_this: str = None):
        self.matplotlib_const = matplotlib_const
        self.program_will_do_this = program_will_do_this
        self._user_have_to_do_this = user_have_to_do_this

    @property
    def user_have_to_do_this(self):
        if self._user_have_to_do_this:
            return self._user_have_to_do_this
        button = None
        if self.matplotlib_const == 1:
            button = "left mouse button"
        elif self.matplotlib_const == Button.RIGHT_ARROW:
            button = "->"
        elif self.matplotlib_const == Button.LEFT_ARROW:
            button = "<-"
        elif self.matplotlib_const == Button.UP_ARROW:
            button = "^"
        elif self.matplotlib_const == Button.DOWN_ARROW:
            button = "ˇ"
        else:
            button = self.matplotlib_const
        return "Press " + button

    def __str__(self):
        return "{} to: {}".format(self.user_have_to_do_this, self.program_will_do_this)

    def __eq__(self, other):
        return self.matplotlib_const == other


class Window:
    config_path = join_path(CONTAINER_ROOT_PATH, "config.json")
    normalized_rgb_tuples = [(0, 1, 0), (1, 0, 0), (0, 0, 1), (1, 1, 0), (1, 0, 1), (0, 1, 1)]
    default_normalized_rgb_tuple = (0.9, 0.9, 0.9)

    def __init__(self, fullscreen: bool):
        self.exit = False
        self.show_only_important = True
        self.show_only_confident = True
        self.show_bbox = True
        self.show_mask = True
        self.show_contour = True
        self.show_caption = True
        self.figure = figure_mat()
        self.figure.canvas.mpl_disconnect(self.figure.canvas.manager.key_press_handler_id)
        if fullscreen:
            mng = get_current_fig_manager()
            mng.resize(*mng.window.maxsize())
        self.mrcnn_and_area_axis = self.figure.add_subplot(1, 2, 1)
        self.unet_axis = self.figure.add_subplot(1, 2, 2)
        self.unet_threshold = 0.5
        self.unet_category_i = 0
        cid = self.figure.canvas.mpl_connect("button_press_event", self.click)
        cid = self.figure.canvas.mpl_connect("key_press_event", self.press)
        cid = self.figure.canvas.mpl_connect("scroll_event", self.scroll)
        self.window_state = MainMenu(self)
        show()

    def set_axis(self, *, detected_objects: List[DetectedObject], title: str):
        normalized_rgb_tuple_cycle = cycle(self.normalized_rgb_tuples)
        point_size = 4
        point_marker = "o"
        text_size = 11
        for detected_object in detected_objects:
            detection_center_x = detected_object.x1 + (self.plus_width_factor * detected_object.width)
            detection_center_y = detected_object.y1 + (self.plus_height_factor * detected_object.height)
            detected_object.center_point_sh = Point_sh(detection_center_x, detection_center_y)
            detected_object.center_normalized_rgb_tuple = self.default_normalized_rgb_tuple
        if not len(detected_objects):
            root_logger.warning("No instances to display!")
        count_dict = dict()
        for area_name, xy_coords in self.area_dict.items():
            count_dict[area_name] = 0
            normalized_rgb_tuple = next(normalized_rgb_tuple_cycle)
            if 1 <= len(xy_coords):
                first_xy_point = xy_coords[0]
                self.mrcnn_and_area_axis.text(
                    first_xy_point[0] + text_size, first_xy_point[1] - text_size, area_name, color=normalized_rgb_tuple, size=text_size, backgroundcolor="none"
                )
            if 1 <= len(xy_coords) <= 2:
                self.mrcnn_and_area_axis.plot(*first_xy_point, color=normalized_rgb_tuple, marker=point_marker, markersize=point_size)
                if len(xy_coords) == 2:
                    second_xy_point = xy_coords[1]
                    self.mrcnn_and_area_axis.plot(*second_xy_point, color=normalized_rgb_tuple, marker=point_marker, markersize=point_size)
            elif 3 <= len(xy_coords):
                self.mrcnn_and_area_axis.add_patch(Polygon_mat(xy_coords, facecolor="none", edgecolor=normalized_rgb_tuple))
                polygon_sh = Polygon_sh(xy_coords)
                for detected_object in detected_objects:
                    is_in_area = polygon_sh.contains(detected_object.center_point_sh)
                    if is_in_area:
                        count_dict[area_name] += 1
                        detected_object.center_normalized_rgb_tuple = normalized_rgb_tuple
        for detected_object in detected_objects:
            self.mrcnn_and_area_axis.plot(
                *(detected_object.center_point_sh.xy), color=detected_object.center_normalized_rgb_tuple, marker=point_marker, markersize=point_size
            )
        title = title or "{}\n{}".format(self.set_axis.__name__, "".join([k + "(" + str(v) + ") " for k, v in count_dict.items() if v > 0]))
        self.mrcnn_and_area_axis.set_title(title)

    @property
    def unet_category_i(self):
        return self._unet_category_i

    @unet_category_i.setter
    def unet_category_i(self, index: int):
        index = max(0, min(len(Category) - 1, index))
        self._unet_category_i = index

    @property
    def config_dict(self):
        if not exists(self.config_path):
            return dict()
        return get_dict_from_json(self.config_path)

    @config_dict.setter
    def config_dict(self, d: dict):
        save_dict_to_json(self.config_path, d)

    @property
    def area_dict(self):
        config_dict = self.config_dict
        key = self.__class__.area_dict.fget.__name__
        if key in config_dict.keys():
            return config_dict[key]
        return dict()

    @area_dict.setter
    def area_dict(self, d: dict):
        config_dict = self.config_dict
        key = self.__class__.area_dict.fget.__name__
        config_dict[key] = d
        self.config_dict = config_dict

    @property
    def plus_width_factor(self):
        config_dict = self.config_dict
        key = self.__class__.plus_width_factor.fget.__name__
        if key in config_dict.keys():
            return config_dict[key]
        return 0.5

    @plus_width_factor.setter
    def plus_width_factor(self, factor: float):
        factor = max(0, min(1, factor))
        config_dict = self.config_dict
        key = self.__class__.plus_width_factor.fget.__name__
        config_dict[key] = factor
        self.config_dict = config_dict

    @property
    def plus_height_factor(self):
        config_dict = self.config_dict
        key = self.__class__.plus_height_factor.fget.__name__
        if key in config_dict.keys():
            return config_dict[key]
        return 0.5

    @plus_height_factor.setter
    def plus_height_factor(self, factor: float):
        factor = max(0, min(1, factor))
        config_dict = self.config_dict
        key = self.__class__.plus_height_factor.fget.__name__
        config_dict[key] = factor
        self.config_dict = config_dict

    @property
    def title(self) -> str:
        return self._title

    @title.setter
    def title(self, title: str):
        self._title = title
        self.figure.suptitle(self.title, fontsize=13)

    def __enter__(self):
        self.mrcnn_and_area_axis.cla()
        self.unet_axis.cla()
        self.unet_axis.cla()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.figure.canvas.draw()
        pause(0.1)

    def click(self, event: MouseEvent):
        self.window_state.click(event)

    def press(self, event: KeyEvent):
        self.window_state.press(event)

    def scroll(self, event: MouseEvent):
        self.window_state.scroll(event)


class WindowState:
    back_to_main_menu_b = Button(Button.ESCAPE, "Go back to main menu")
    buttons = []

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
        return self.__class__.__name__ + "".join(["\n" + str(button) for button in self.buttons])


class ExitState(WindowState):
    def __init__(self, window: Window):
        super().__init__(window)
        self.window.exit = True


class MainMenu(WindowState):
    unet_b = Button("u", "Edit unet's axis")
    mrcnn_b = Button("m", "Edit mrcnn's axis")
    quit_b = Button("q", "Quit the program")
    new_area_config_b = Button("n", "Create a new area config")
    edit_area_config_b = Button("e", "Edit the area config")
    buttons = [quit_b, unet_b, mrcnn_b, new_area_config_b, edit_area_config_b]

    def __init__(self, window: Window):
        super().__init__(window)
        self.window.title = str(self)

    def press(self, event: KeyEvent):
        if event.key == self.unet_b:
            self.go_to(UnetAxisSetter)
        elif event.key == self.mrcnn_b:
            self.go_to(MrcnnAxisSetter)
        elif event.key == self.quit_b:
            self.go_to(ExitState)
        elif event.key == self.new_area_config_b:
            self.window.area_dict = dict()
            self.go_to(AreaNameSetter)
        elif event.key == self.edit_area_config_b:
            self.go_to(AreaNameSetter)


class MrcnnAxisSetter(WindowState):
    move_center_right_b = Button(Button.RIGHT_ARROW, "Move center point right")
    move_center_left_b = Button(Button.LEFT_ARROW, "Move center point left")
    move_center_up_b = Button(Button.UP_ARROW, "Move center point up")
    move_center_down_b = Button(Button.DOWN_ARROW, "Move center point down")
    show_only_important_b = Button("1", "Show only important detections")
    show_only_confident_b = Button("2", "Show only confident detections")
    show_bbox_b = Button("3", "Show bbox")
    show_mask_b = Button("4", "Show mask")
    show_contour_b = Button("5", "Show contour")
    show_caption_b = Button("6", "Show caption")
    buttons = [
        WindowState.back_to_main_menu_b,
        move_center_right_b,
        move_center_left_b,
        move_center_up_b,
        move_center_down_b,
        show_only_important_b,
        show_only_confident_b,
        show_bbox_b,
        show_mask_b,
        show_contour_b,
        show_caption_b,
    ]

    def __init__(self, window: Window):
        super().__init__(window)
        self.window.title = str(self)

    def press(self, event: KeyEvent):
        step_size = 0.05
        if event.key == self.back_to_main_menu_b:
            self.go_to(MainMenu)
            return
        elif event.key == self.show_mask_b:
            self.window.show_mask = not self.window.show_mask
        elif event.key == self.show_contour_b:
            self.window.show_contour = not self.window.show_contour
        elif event.key == self.show_bbox_b:
            self.window.show_bbox = not self.window.show_bbox
        elif event.key == self.show_only_important_b:
            self.window.show_only_important = not self.window.show_only_important
        elif event.key == self.show_only_confident_b:
            self.window.show_only_confident = not self.window.show_only_confident
        elif event.key == self.show_caption_b:
            self.window.show_caption = not self.window.show_caption
        elif event.key == self.move_center_right_b:
            self.window.plus_width_factor += step_size
        elif event.key == self.move_center_left_b:
            self.window.plus_width_factor -= step_size
        elif event.key == self.move_center_up_b:
            self.window.plus_height_factor -= step_size
        elif event.key == self.move_center_down_b:
            self.window.plus_height_factor += step_size
        self.window.title = "{:.3f} / {:.3f} /{}/{}/{}/{}/{}/{}".format(
            self.window.plus_width_factor,
            self.window.plus_height_factor,
            self.window.show_only_important,
            self.window.show_only_confident,
            self.window.show_bbox,
            self.window.show_mask,
            self.window.show_contour,
            self.window.show_caption,
        )


class UnetAxisSetter(WindowState):
    increase_threshold_b = Button(Button.SCROLL_UP, "Increase the axis threshold", "Scroll up on axis")
    decrease_threshold_b = Button(Button.SCROLL_DOWN, "Decrease the axis threshold", "Scroll down on axis")
    next_category_b = Button(Button.RIGHT_ARROW, "Show next category")
    previous_category_b = Button(Button.LEFT_ARROW, "Show previous category")
    buttons = [WindowState.back_to_main_menu_b, increase_threshold_b, decrease_threshold_b, next_category_b, previous_category_b]

    def __init__(self, window: Window):
        super().__init__(window)
        self.window.title = str(self)

    def press(self, event: KeyEvent):
        if event.key == self.back_to_main_menu_b:
            self.go_to(MainMenu)
        elif event.key == self.next_category_b:
            self.window.unet_category_i += 1
        elif event.key == self.previous_category_b:
            self.window.unet_category_i -= 1

    def scroll(self, event: MouseEvent):
        step_size = 0.05
        if event.button == self.increase_threshold_b and event.inaxes is self.window.unet_axis:
            self.window.unet_threshold = min(1 - step_size, self.window.unet_threshold + step_size)
        elif event.button == self.decrease_threshold_b and event.inaxes is self.window.unet_axis:
            self.window.unet_threshold = max(step_size, self.window.unet_threshold - step_size)
        self.window.title = "{:.3f}".format(self.window.unet_threshold)


class AreaNameSetter(WindowState):
    save_name_b = Button(Button.ENTER, "Save area name")
    delete_last_char_b = Button(Button.BACKSPACE, "Delete the last character")

    buttons = [WindowState.back_to_main_menu_b, delete_last_char_b, save_name_b]

    def __init__(self, window: Window):
        super().__init__(window)
        self.area_name = ""
        self.window.title = str(self)

    def press(self, event: KeyEvent):
        if len(event.key) == 1 and event.key.isalnum():
            self.area_name += event.key
            self.window.title = self.area_name
        elif event.key == self.save_name_b:
            self.go_to(AreaPointSetter(self.window, self.area_name))
        elif event.key == self.delete_last_char_b:
            self.area_name = self.area_name[:-1]
            self.window.title = self.area_name
        elif event.key == self.back_to_main_menu_b:
            self.go_to(MainMenu)


class AreaPointSetter(WindowState):
    save_area_b = Button(Button.ENTER, "Save area")
    add_point_b = Button(Button.MOUSE_LEFT, "Add a point", "Press Left click on the image")
    remove_point_b = Button(Button.BACKSPACE, "Remove last point")
    buttons = [add_point_b, remove_point_b, save_area_b]

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
        if event.key == self.save_area_b:
            if len(self.window.area_dict[self.area_name]) >= 3:
                self.go_to(AreaNameSetter)
            else:
                self.window.title = "Minimum 3 points are needed!"
        elif event.key == self.remove_point_b:
            area_file = self.window.area_dict
            if len(area_file[self.area_name]) >= 1:
                area_file[self.area_name].pop()
            self.window.area_dict = area_file
            self.window.title = self.title

    def click(self, event: MouseEvent):
        if event.button == self.add_point_b and event.inaxes is self.window.mrcnn_and_area_axis:
            area_file = self.window.area_dict
            area_file[self.area_name].append((round(event.xdata), round(event.ydata)))
            self.window.area_dict = area_file
            self.window.title = self.title
