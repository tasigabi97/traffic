"""
Ez a fájl definiálja a fő matplotlibes ablak működését.
"""
from traffic.imports import *
from traffic.logging import *
from traffic.consts import *
from traffic.utils import *
from traffic.mrcnn import *
from traffic.utils.lane_unet import *
from traffic.strings import concat

ion_mat()  # ha már használjuk ezt a fájlt, akkor kell az interaktv mód.
root_logger.debug(get_backend())  # sajnos nem minden függvnyhívás backend független


class EventButtonOrKey:
    """
    Ez az osztály reprezentál egy matplotlibes (nem cv2-es) billentyűzet leütést/egér kattintást/görgetést.
    """

    ENTER = "enter"  # a használt speciális matplotlibes event.key/event.button konstansok
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
    def user_have_to_do_this(self) -> str:
        if self._user_have_to_do_this:
            return self._user_have_to_do_this
        button = None
        if self.matplotlib_const == 1:
            button = "left mouse button"
        elif self.matplotlib_const == EventButtonOrKey.RIGHT_ARROW:
            button = "->"
        elif self.matplotlib_const == EventButtonOrKey.LEFT_ARROW:
            button = "<-"
        elif self.matplotlib_const == EventButtonOrKey.UP_ARROW:
            button = "^"
        elif self.matplotlib_const == EventButtonOrKey.DOWN_ARROW:
            button = "ˇ"
        else:
            button = self.matplotlib_const
        return "Press " + button

    def __str__(self):
        return "{} to: {}".format(self.user_have_to_do_this, self.program_will_do_this)

    def __eq__(self, other):
        return self.matplotlib_const == other


class Window:
    """
    Ez az osztály reprezentálja a fő matplotlibes ablakot.
    """

    config_path = join_path(CONTAINER_ROOT_PATH, "config.json")  # Itt tárolunk adatokat a megjelenítéssal kapcsolatban.
    normalized_rgb_tuples = [
        (0, 1, 0),
        (1, 0, 0),
        (0, 0, 1),
        (1, 1, 0),
        (1, 0, 1),
        (0, 1, 1),
    ]  # Ezeket az alap színeket
    # használjuk a megjelenítésnél. A matplotlibnél nem 255 a max intenzitás.
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
        self.figure.canvas.mpl_disconnect(self.figure.canvas.manager.key_press_handler_id)  # Ez kitöröljük az
        # alap gombok tulajdonságát. Pl az f a fullscreen.
        if fullscreen:  # A teljesképernyő beállítása innen (nem az ablakon való kattintással).
            mng = get_current_fig_manager()  # függ a backendtől
            mng.resize(*mng.window.maxsize())
        self.mrcnn_and_area_axis = self.figure.add_subplot(1, 2, 1)  # Két bal/jobb térfélre osztjuk az ablakot.
        self.unet_axis = self.figure.add_subplot(1, 2, 2)
        self.unet_threshold = 0.2
        self.unet_category_i = 1
        cid = self.figure.canvas.mpl_connect("button_press_event", self.click)
        cid = self.figure.canvas.mpl_connect("key_press_event", self.press)
        cid = self.figure.canvas.mpl_connect("scroll_event", self.scroll)
        self.window_state = MainMenu(self)
        show()

    def set_axis(self, *, detected_objects: List[DetectedObject], title: str):
        """
        Hozzáadja az objektumok középpontját, és a bejelölt területeket a megjelenítéshez.
        """
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
        area_dict = dict()
        for area_name, xy_coords in self.area_dict.items():
            area_dict[area_name] = dict()
            normalized_rgb_tuple = next(normalized_rgb_tuple_cycle)
            if 1 <= len(xy_coords):
                first_xy_point = xy_coords[0]
                self.mrcnn_and_area_axis.text(
                    first_xy_point[0] + text_size,
                    first_xy_point[1] - text_size,
                    area_name,
                    color=normalized_rgb_tuple,
                    size=text_size,
                    backgroundcolor="none",
                )
            if 1 <= len(xy_coords) <= 2:
                self.mrcnn_and_area_axis.plot(
                    *first_xy_point, color=normalized_rgb_tuple, marker=point_marker, markersize=point_size
                )
                if len(xy_coords) == 2:
                    second_xy_point = xy_coords[1]
                    self.mrcnn_and_area_axis.plot(
                        *second_xy_point, color=normalized_rgb_tuple, marker=point_marker, markersize=point_size
                    )
            elif 3 <= len(xy_coords):
                self.mrcnn_and_area_axis.add_patch(
                    Polygon_mat(xy_coords, facecolor="none", edgecolor=normalized_rgb_tuple)
                )
                polygon_sh = Polygon_sh(xy_coords)
                for detected_object in detected_objects:
                    is_in_area = polygon_sh.contains(detected_object.center_point_sh)
                    if is_in_area:
                        if detected_object.name in area_dict[area_name].keys():
                            area_dict[area_name][detected_object.name] += 1
                        else:
                            area_dict[area_name][detected_object.name] = 1
                        detected_object.center_normalized_rgb_tuple = normalized_rgb_tuple
        for detected_object in detected_objects:
            self.mrcnn_and_area_axis.plot(
                *(detected_object.center_point_sh.xy),
                color=detected_object.center_normalized_rgb_tuple,
                marker=point_marker,
                markersize=point_size,
            )
        title = title or concat(
            [
                area_name
                + "["
                + concat(
                    [category_name + "(" + str(count) + ")" for category_name, count in category_dict.items()], ", "
                )
                + "]"
                for area_name, category_dict in area_dict.items()
                if len(category_dict) > 0
            ],
            "\n",
        )
        self.mrcnn_and_area_axis.set_title(title)

    @property
    def unet_category_i(self):
        return self._unet_category_i

    @unet_category_i.setter
    def unet_category_i(self, index: int):
        """
        Csak a túlindexelés elkerülésére kell.
        """
        index = max(0, min(len(LaneCategory) - 1, index))
        self._unet_category_i = index

    @property
    def config_dict(self):
        """
        A menteni való adatok betöltése.
        """
        if not exists(self.config_path):
            return dict()
        return get_dict_from_json(self.config_path)

    @config_dict.setter
    def config_dict(self, d: dict):
        """
        A menteni való adatok elmentése.
        """
        save_dict_to_json(self.config_path, d)

    @property
    def area_dict(self):
        """
        Poligonok betöltse.
        """
        config_dict = self.config_dict
        key = self.__class__.area_dict.fget.__name__
        if key in config_dict.keys():
            return config_dict[key]
        return dict()

    @area_dict.setter
    def area_dict(self, d: dict):
        """
        Poligonok mentése.
        """
        config_dict = self.config_dict
        key = self.__class__.area_dict.fget.__name__
        config_dict[key] = d
        self.config_dict = config_dict

    # A detektált objektumok középpontjának beállítsait hasonlóan mentjük.
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
        """
        A fő ablak címének beállítása.
        """
        self._title = title
        self.figure.suptitle(self.title, fontsize=13)

    def __enter__(self):
        """
        Le kell tiszttani a vásznakat minden ciklus elején.
        """
        self.mrcnn_and_area_axis.cla()
        self.unet_axis.cla()

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        Újra kell rajzolni a megjelenítést, és fel kell dolgozni a felhasználói bemeneteket
         minden ciklus végén.
        """
        self.figure.canvas.draw()
        pause(0.1)

    # Feldolgozzuk a felhasználói bemeneteket,
    # az aktulis ablakállapottól függően.
    def click(self, event: MouseEvent):
        self.window_state.click(event)

    def press(self, event: KeyEvent):
        self.window_state.press(event)

    def scroll(self, event: MouseEvent):
        self.window_state.scroll(event)


class WindowState:
    back_to_main_menu = EventButtonOrKey(EventButtonOrKey.ESCAPE, "Go back to main menu")
    event_button_or_keys = []

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
        return self.__class__.__name__ + "".join(["\n" + str(button) for button in self.event_button_or_keys])


class ExitState(WindowState):
    def __init__(self, window: Window):
        super().__init__(window)
        self.window.exit = True


class MainMenu(WindowState):
    edit_unet_axis = EventButtonOrKey("u", "Edit unet's axis")
    edit_mrcnn_axis = EventButtonOrKey("m", "Edit mrcnn's axis")
    quit_the_program = EventButtonOrKey("q", "Quit the program")
    create_new_area_config = EventButtonOrKey("n", "Create a new area config")
    edit_area_config = EventButtonOrKey("e", "Edit the area config")
    event_button_or_keys = [quit_the_program, edit_unet_axis, edit_mrcnn_axis, create_new_area_config, edit_area_config]

    def __init__(self, window: Window):
        super().__init__(window)
        self.window.title = str(self)

    def press(self, event: KeyEvent):
        if event.key == self.edit_unet_axis:
            self.go_to(UnetAxisSetter)
        elif event.key == self.edit_mrcnn_axis:
            self.go_to(MrcnnAxisSetter)
        elif event.key == self.quit_the_program:
            self.go_to(ExitState)
        elif event.key == self.create_new_area_config:
            self.window.area_dict = dict()
            self.go_to(AreaNameSetter)
        elif event.key == self.edit_area_config:
            self.go_to(AreaNameSetter)


class MrcnnAxisSetter(WindowState):
    move_center_right = EventButtonOrKey(EventButtonOrKey.RIGHT_ARROW, "Move center point right")
    move_center_left = EventButtonOrKey(EventButtonOrKey.LEFT_ARROW, "Move center point left")
    move_center_up = EventButtonOrKey(EventButtonOrKey.UP_ARROW, "Move center point up")
    move_center_down = EventButtonOrKey(EventButtonOrKey.DOWN_ARROW, "Move center point down")
    show_only_important = EventButtonOrKey("1", "Show only important detections")
    show_only_confident = EventButtonOrKey("2", "Show only confident detections")
    show_bbox = EventButtonOrKey("3", "Show bbox")
    show_mask = EventButtonOrKey("4", "Show mask")
    show_contour = EventButtonOrKey("5", "Show contour")
    show_caption = EventButtonOrKey("6", "Show caption")
    event_button_or_keys = [
        WindowState.back_to_main_menu,
        move_center_right,
        move_center_left,
        move_center_up,
        move_center_down,
        show_only_important,
        show_only_confident,
        show_bbox,
        show_mask,
        show_contour,
        show_caption,
    ]

    def __init__(self, window: Window):
        super().__init__(window)
        self.window.title = str(self)

    def press(self, event: KeyEvent):
        step_size = 0.05
        if event.key == self.back_to_main_menu:
            self.go_to(MainMenu)
            return
        elif event.key == self.show_mask:
            self.window.show_mask = not self.window.show_mask
        elif event.key == self.show_contour:
            self.window.show_contour = not self.window.show_contour
        elif event.key == self.show_bbox:
            self.window.show_bbox = not self.window.show_bbox
        elif event.key == self.show_only_important:
            self.window.show_only_important = not self.window.show_only_important
        elif event.key == self.show_only_confident:
            self.window.show_only_confident = not self.window.show_only_confident
        elif event.key == self.show_caption:
            self.window.show_caption = not self.window.show_caption
        elif event.key == self.move_center_right:
            self.window.plus_width_factor += step_size
        elif event.key == self.move_center_left:
            self.window.plus_width_factor -= step_size
        elif event.key == self.move_center_up:
            self.window.plus_height_factor -= step_size
        elif event.key == self.move_center_down:
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
    increase_threshold = EventButtonOrKey(
        EventButtonOrKey.SCROLL_UP, "Increase the axis threshold", "Scroll up on axis"
    )
    decrease_threshold = EventButtonOrKey(
        EventButtonOrKey.SCROLL_DOWN, "Decrease the axis threshold", "Scroll down on axis"
    )
    next_category = EventButtonOrKey(EventButtonOrKey.RIGHT_ARROW, "Show next category")
    previous_category = EventButtonOrKey(EventButtonOrKey.LEFT_ARROW, "Show previous category")
    event_button_or_keys = [
        WindowState.back_to_main_menu,
        increase_threshold,
        decrease_threshold,
        next_category,
        previous_category,
    ]

    def __init__(self, window: Window):
        super().__init__(window)
        self.window.title = str(self)

    def press(self, event: KeyEvent):
        if event.key == self.back_to_main_menu:
            self.go_to(MainMenu)
        elif event.key == self.next_category:
            self.window.unet_category_i += 1
        elif event.key == self.previous_category:
            self.window.unet_category_i -= 1

    def scroll(self, event: MouseEvent):
        step_size = 0.05
        if event.button == self.increase_threshold and event.inaxes is self.window.unet_axis:
            self.window.unet_threshold = min(1 - step_size, self.window.unet_threshold + step_size)
        elif event.button == self.decrease_threshold and event.inaxes is self.window.unet_axis:
            self.window.unet_threshold = max(step_size, self.window.unet_threshold - step_size)
        self.window.title = "{:.3f}".format(self.window.unet_threshold)


class AreaNameSetter(WindowState):
    save_area_name = EventButtonOrKey(EventButtonOrKey.ENTER, "Save area name")
    delete_last_char = EventButtonOrKey(EventButtonOrKey.BACKSPACE, "Delete the last character")

    event_button_or_keys = [WindowState.back_to_main_menu, delete_last_char, save_area_name]

    def __init__(self, window: Window):
        super().__init__(window)
        self.area_name = ""
        self.window.title = str(self)

    def press(self, event: KeyEvent):
        if len(event.key) == 1 and event.key.isalnum():
            self.area_name += event.key
            self.window.title = self.area_name
        elif event.key == self.save_area_name:
            self.go_to(AreaPointSetter(self.window, self.area_name))
        elif event.key == self.delete_last_char:
            self.area_name = self.area_name[:-1]
            self.window.title = self.area_name
        elif event.key == self.back_to_main_menu:
            self.go_to(MainMenu)


class AreaPointSetter(WindowState):
    save_area = EventButtonOrKey(EventButtonOrKey.ENTER, "Save area")
    add_point = EventButtonOrKey(EventButtonOrKey.MOUSE_LEFT, "Add a point", "Press Left click on the image")
    remove_last_point = EventButtonOrKey(EventButtonOrKey.BACKSPACE, "Remove last point")
    event_button_or_keys = [add_point, remove_last_point, save_area]

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
        if event.key == self.save_area:
            if len(self.window.area_dict[self.area_name]) >= 3:
                self.go_to(AreaNameSetter)
            else:
                self.window.title = "Minimum 3 points are needed!"
        elif event.key == self.remove_last_point:
            area_file = self.window.area_dict
            if len(area_file[self.area_name]) >= 1:
                area_file[self.area_name].pop()
            self.window.area_dict = area_file
            self.window.title = self.title

    def click(self, event: MouseEvent):
        if event.button == self.add_point and event.inaxes is self.window.mrcnn_and_area_axis:
            area_file = self.window.area_dict
            area_file[self.area_name].append((round(event.xdata), round(event.ydata)))
            self.window.area_dict = area_file
            self.window.title = self.title
