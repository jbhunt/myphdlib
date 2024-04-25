from myphdlib.interface.session import SessionBase
from myphdlib.pipeline.events import EventsProcessingMixin
from myphdlib.pipeline.saccades import SaccadesProcessingMixin

class StimuliProcessingMixinDreadds2(
    ):
    """
    """

    def _calculatePulseIntervalTimestamps(
        self, 
        ):
        return
    
    def _processDriftingGratingProtocol(
        self,
        ):
        """
        """

        # Process data

        # Save data

        return

    def _processFictiveSaccadesProtocol(
        self,
        ):
        """
        """

        return

    def _runStimuliModule(self):
        """
        """

        self._processDriftingGratingProtocol()
        self._processFictiveSaccadesProtocol()

        return

class Dreadds2Session(
    EventsProcessingMixin,
    SaccadesProcessingMixin,
    StimuliProcessingMixinDreadds2,
    SessionBase
    ):
    """
    """

    labjackChannelMapping = {
        'barcode': 5,
        'cameras': 7,
        'stimulus': 6,
    }

    def __init__(self, sessionFolder, eye='left'):
        """
        """

        super().__init__(sessionFolder, eye=eye)

        return
    
    @property 
    def leftEyePose(self):
        """
        """
        csv = list(self.home.joinpath('videos').glob('*.csv')).pop()
        return csv

    @property
    def leftCameraTimestamps(self):
        """
        """
        leftCameraTimestamps = list(self.home.joinpath('videos').glob('*rightCam*')).pop()
        return leftCameraTimestamps

        