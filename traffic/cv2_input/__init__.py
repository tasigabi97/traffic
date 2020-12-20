from traffic.utils import Singleton
from traffic.imports import Union, Iterable_type, List, waitKey


class Cv2Input(Singleton):
    """
    A kamerák kiválasztásánál felugró ablakok máshogyan kezelik a billentyűleütést,
     mint a matplotlibes ablakok.
    """

    @property
    def wait_keys(self) -> List[str]:
        return [chr(key) for key in self._wait_keys]

    @wait_keys.setter
    def wait_keys(self, keys: Union[int, str, Iterable_type[Union[int, str]]]):
        """
        Ezzel lehet megadni, hogy melyik billentyűk leütésére várunk a cv2-es ablakok esetén.
        """
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
        """
        Visszaadja a legutóbb leütött (és elvárt) karaktert az elmúlt századmásodpercből.
         None-t ad vissza:
                  -ha nem nyomtunk billentyűt az elmúlt századmásodpercben
                  -ha nem vártunk a lenyomott billentyűre a wait_keys-el.
        """
        pressed = waitKey(1) & 0xFF
        for key in self._wait_keys:
            if pressed == key:
                return chr(pressed)


cv2_input = Cv2Input()
