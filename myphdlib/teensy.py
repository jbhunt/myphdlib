import serial
import pathlib as pl

HANDSHAKE = bytes('x', 'utf-8')
ON        = bytes('a', 'utf-8')
OFF       = bytes('z', 'utf-8')

class Teensy():
    """
    """

    def __init__(self, dummy=False):
        """
        """

        self._obj   = None
        self._state = False
        self._dummy = dummy

        return

    def detect(self, baudrate=9600, timeout=1, attempts=3):
        """
        """

        global HANDSHAKE
        global ON
        global OFF

        if self._dummy:
            return

        detected = False
        devices = pl.Path('/dev/').rglob('*ttyACM*')
        for iattempt in range(attempts):
            if detected:
                break
            for device in devices:
                try:
                    obj = serial.Serial(str(device), baudrate, timeout=timeout)
                except:
                    continue
                obj.write(HANDSHAKE)
                message = obj.read()
                if len(message) > 0 and message == HANDSHAKE:
                    self._obj = obj
                    detected = True
                    break

        #
        if detected is False:
            self._obj = None
            raise Exception('No device detected')

        #
        else:
            if self._obj.is_open is False:
                raise Exception('Serial port is closed')

        return

    def setState(self, state=True):
        """
        """

        if self._dummy:
            self._state = state
            return

        if state:
            self._obj.write(ON)
            self._state = True
        elif state is False:
            self._obj.write(OFF)
            self._state = False

        return

    @property
    def state(self):
        return self._state
