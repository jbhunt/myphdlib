import serial
import pathlib as pl

HANDSHAKE = bytes('h', 'utf-8')
ON        = bytes('p', 'utf-8')
OFF       = bytes('z', 'utf-8')
RESET     = bytes('r', 'utf-8')

commands = {
    'signal' : bytes('p', 'utf-8'),
    'connect': bytes('h', 'utf-8'),
    'release': bytes('r', 'utf-8')
}

class Microcontroller():
    """
    """

    def __init__(self):
        """
        """

        self._connection = None


        return

    def connect(self, baudrate=9600, timeout=1):
        """
        """

        connected = False
        devices = pl.Path('/dev/').glob('*ttyACM*')
        for device in devices:
            try:
                connection = serial.Serial(
                    str(device),
                    baudrate,
                    timeout=timeout
                )
<<<<<<< HEAD
            except (Exception, serial.SerialException):
=======
            except (serial.SerialException, serial.SerialTimeoutException):
>>>>>>> b4f99f0f58a31161b68f58ca4f1f255a3f50d97e
                continue
            connection.write(commands['connect'])
            message = connection.read()
            if len(message) > 0 and message == commands['connect']:
                self._connection = connection
                connected = True
                break

        return connected

    def release(self):
        """
        """

        if self._connection is None:
            return

        self._connection.write(commands['release'])
        self._connection.close()
        self._connection = None

        return

    def signal(self, t=1):
        """
        """

        if self._connection is None:
            return

        self._connection.write(commands['signal'])
        # message = str(round(t * 1000)).encode('utf-8')
        # self._connection.write(message)

        return

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

    def release(self):
        """
        """

        if self._obj is None:
            return

        self._obj.write(RESET)
        self._obj.close()
        self._obj = None

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
