from myphdlib.interface.session import SessionBase
from myphdlib.pipeline.events import EventsProcessingMixin
from myphdlib.pipeline.saccades import SaccadesProcessingMixin
from myphdlib.general.labjack import filterPulsesFromPhotologicDevice

class StimuliProcessingMixinDreadds2(
    ):
    """
    """

    def _calculatePulseIntervalTimestamps(
        self, 
        ):
        #this if else clause will have to be edited to reflect hdf format
        if 'labjackData' not in session.keys():
            raise Exception('Labjack data not extracted')
        else:
            labjackData = session.read('labjackData')
        #defining the labjack data should also be edited to pull from hdf
        timestamps = labjackData[:, 0]
        TTLdata = labjackData[:, 6]
        filtered = filterPulsesFromPhotologicDevice(labjackData[:, 6],
            minimumPulseWidthInSeconds=0.013)
        iPulses = np.where(np.diff(filtered) > 0.5)[0]
        iIntervals = np.where(np.diff(iPulses) > 16000)[0]
        iIntervals2 = iPulses[iIntervals]
        pulseTimestamps = timestamps[iPulses]
        intervalTimestamps = (timestamps[iIntervals2] + 3)
        self.pulseTimestamps = pulseTimestamps
        self.intervalTimestamps = intervalTimestamps
        return pulseTimestamps, intervalTimestamps

    def _createMetadataFileList(pulseTimestamps, intervalTimestamps,
        self,
        ):
        #need less hard coded way to enter mouse name and date
        whichDreadd = 'DREADD10'
        parentDir = '/media/retina2/Seagate Portable Drive/NPData/2024-04-02/' + whichDreadd + '/videos/'

        partialFileList = ['driftingGratingMetadata-0.txt',
            'driftingGratingMetadata-1.txt',
            'fictiveSaccadeMetadata-0.pkl',
            'driftingGratingMetadata-2.txt',
            'driftingGratingMetadata-3.txt',
            'fictiveSaccadeMetadata-1.pkl',
            'driftingGratingMetadata-4.txt', 
            'driftingGratingMetadata-5.txt',
            'fictiveSaccadeMetadata-2.pkl', 
            'driftingGratingMetadata-6.txt', 
            'driftingGratingMetadata-7.txt',
            'fictiveSaccadeMetadata-3.pkl',
            'driftingGratingMetadata-8.txt',
            'driftingGratingMetadata-9.txt',
            'fictiveSaccadeMetadata-4.pkl']


        fileList = []
        for f in partialFileList:
        fileList.append(parentDir + f)

        # Quick check for file # mismatch w/ interval timestamps
        assert len(fileList) == intervalTimestamps.shape[0] + 1

        # pulseTimestampOffsets = np.diff(pulseTimestamps) <- for checking correctness in each combined event :)
        pulseTimestampOffsets = np.diff(pulseTimestamps)
        self.fileList = fileList
        return fileList

    def _

    
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

        