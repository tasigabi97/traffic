from traffic.utils import Singleton
from traffic.imports import Union, Iterable_type, List, waitKey


class Globals(Singleton):
    @property
    def wait_keys(self) -> List[int]:
        return self._wait_keys

    @wait_keys.setter
    def wait_keys(self, keys: Union[int, str, Iterable_type[Union[int, str]]]):
        keys = [keys] if isinstance(keys, int) or isinstance(keys, str) else keys
        if not isinstance(keys, Iterable_type):
            raise TypeError()
        keys = [str(key) for key in keys]
        for key in keys:
            if len(key) != 1:
                raise ValueError()
        keys = [ord(key) for key in keys]
        self._wait_keys = keys

    @property
    def pressed_key(self) -> str:
        pressed = waitKey(1) & 0xFF
        for key in self._wait_keys:
            if pressed == key:
                return chr(pressed)
