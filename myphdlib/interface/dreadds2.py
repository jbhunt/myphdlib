from myphdlib.interface.session import SessionBase
from myphdlib.pipeline.events import EventsProcessingMixin
from myphdlib.pipeline.saccades import SaccadesProcessingMixin
from myphdlib.general.labjack import filterPulsesFromPhotologicDevice
import pathlib as pl

class StimuliProcessingMixinDreadds2(
    ):
    """
    """

    def _calculatePulseIntervalTimestamps(
        self, 
        ):
        """
        Computes interval between pulses in the stimulus channel of the LabJack data
        """

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
        iIntervals = np.where(np.diff(iPulses) > 16000)[0] #figure out threshold value 
        iIntervals2 = iPulses[iIntervals]
        pulseTimestamps = timestamps[iPulses]
        intervalTimestamps = (timestamps[iIntervals2] + 3)
        return pulseTimestamps, intervalTimestamps

    def _createMetadataFileList(pulseTimestamps, intervalTimestamps,
        self,
        ):
        """
        Creates chronological list of metadata files & asserts that the number of files equals the number of stimulus blocks
        """
        #need less hard coded way to enter mouse name and date
        parentDir = self.home.joinpath('videos')

        partialFileList = [
            'driftingGratingMetadata-0.txt',
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
            'fictiveSaccadeMetadata-4.pkl'
        ]


        fileList = []
        for f in partialFileList:
            fileList.append(parentDir + f)

        # Quick check for file # mismatch w/ interval timestamps
        assert len(fileList) == intervalTimestamps.shape[0] + 1

        # pulseTimestampOffsets = np.diff(pulseTimestamps) <- for checking correctness in each combined event :)
        pulseTimestampOffsets = np.diff(pulseTimestamps)
        return fileList

    
    def _processDriftingGratingProtocol(
        self,
        eventIndex,
        metadataHolder,
        fileIndex,
        pulseTimestamps,
        intervalTimestamps
        ):
        """
        """
        metadata = np.genfromtxt(file, skip_header = 5, delimiter=',')
        blockLength = metadata.shape[0]
        if fileIndex == 0:
            thisBlockBools = pulseTimestamps < intervalTimestamps[0]
        elif fileIndex == len(fileList) - 1:
            thisBlockBools = pulseTimestamps > intervalTimestamps[-1]
        else:
            thisBlockBools = np.logical_and((pulseTimestamps > intervalTimestamps[fileIndex-1]), \
                                (pulseTimestamps < intervalTimestamps[fileIndex]))
        assert thisBlockBools.sum() == metadata.shape[0], str(file) + " has pulse/event count mismatch!"

        # Timing conditions
        thisBlockPulses = pulseTimestamps[thisBlockBools]
        thisBlockDiffs = np.diff(thisBlockPulses)
        fromFileDiffs = np.diff(metadata[:,4])
        
        # Make sure the timing between pulses isn't too large
        timingThresh = 0.1
        timingDifferences = np.absolute(np.subtract(fromFileDiffs, thisBlockDiffs))
        assert not (timingDifferences > timingThresh).sum() > 1 # Direction change must be accoutned for.
        metadataHolder[eventIndex:eventIndex + blockLength, 0:5] = metadata

        # adding a column that is 0 if drifting grating, 1 if fictive saccade
        metadataHolder[eventIndex:eventIndex + blockLength, 5:6] = 0
        eventIndex += blockLength

        return eventIndex, metadataHolder

    def _processFictiveSaccadesProtocol(
        self,
        eventIndex,
        metadataHolder,
        fileIndex,
        pulseTimestamps,
        intervalTimestamps
        ):
        """
        """
        with open(file, 'rb') as f:
            metadataDict = np.load(f, allow_pickle=True)

        allEvents = metadataDict['events'] # np 1D array
        allTrials = metadataDict['trials'] # list
        blockLength = allEvents.shape[0]
        if fileIndex == 0:
            thisBlockBools = pulseTimestamps < intervalTimestamps[0]
        elif fileIndex == len(fileList) - 1:
            thisBlockBools = pulseTimestamps > intervalTimestamps[-1]
        else:
            thisBlockBools = np.logical_and((pulseTimestamps > intervalTimestamps[fileIndex-1]), \
                                (pulseTimestamps < intervalTimestamps[fileIndex]))
        assert thisBlockBools.sum() == allEvents.shape[0], str(file) + " has pulse/event count mismatch!"
        thisBlockPulses = pulseTimestamps[thisBlockBools]
        thisBlockDiffs = np.diff(thisBlockPulses)
        thisBlockDiffs = np.append(thisBlockDiffs, 999)
        
        # Iterate down pulses. Check the event type metadata of each one. Then, check distance from next.
        # Depending on metadata condition, different timiing allowances are permitted.
        pulseIndex = 0
        trialIndex = 0
        while pulseIndex < thisBlockPulses.shape[0]:
            eventType = allEvents[pulseIndex]
            trialType = allTrials[trialIndex]
            if thisBlockDiffs[pulseIndex] > 0.4:
                assert (trialType[2] == 'probe') or (trialType[2] == 'saccade'), \
                    "Good error message!"
                if trialType[2] == 'probe':
                    metadataHolder[eventIndex, 0:1] = 3.0
                    metadataHolder[eventIndex, 1:2] = allTrials[trialIndex][1]
                    metadataHolder[eventIndex, 5:6] = 1
                    metadataHolder[eventIndex, 6] = 0
                elif trialType[2] == 'saccade':
                    metadataHolder[eventIndex, 0:1] = 5.0
                    metadataHolder[eventIndex, 1:2] = allTrials[trialIndex][1]
                    metadataHolder[eventIndex, 5:6] = 1
                    metadataHolder[eventIndex, 6] = 1
                eventIndex += 1
                pulseIndex    += 1 
                trialIndex    += 1
            else:
                assert trialType[2] == 'combined', "Invalid trialtype found : " + str(trialType[2])
                assert pulseIndex != thisBlockPulses.shape[0] - 1, "Combined trial is last pulse, but only one found."
                metadataHolder[eventIndex, 0:1] = 5.0
                metadataHolder[eventIndex, 1:2] = allTrials[trialIndex][1]
                metadataHolder[eventIndex, 5:6] = 1
                metadataHolder[eventIndex, 6] = 2
                metadataHolder[eventIndex + 1, 0:1] = 3.0
                metadataHolder[eventIndex + 1, 1:2] = allTrials[trialIndex][1]
                metadataHolder[eventIndex + 1, 5:6] = 1
                metadataHolder[eventIndex + 1, 6] = 2
                pulseIndex    += 2 # Two associated pulses within the index.
                trialIndex    += 1
                eventIndex += 2

        return eventIndex, metadataHolder

    def _runStimuliModule(self):
        """
        """

        pulseTimestamps, intervalTimestamps = self._calculatePulseIntervalTimestamps()
        fileList = self._createMetadataFileList(pulseTimestamps, intervalTimestamps)
        eventIndex = 0
        metadataHolder = np.full((len(pulseTimestamps), 7), np.nan)
        for fileIndex, file in enumerate(fileList):

            # DG metadata
            if pl.Path(file).suffix == '.txt':
                eventIndex, eventHolder = self._processFictiveSaccadesProtocol(
                    eventIndex,
                    eventHolder,
                    fileIndex,
                    pulseTimestamps,
                    intervalTimestamps
                )

            # FS metadata
            elif pl.Path(file).suffix == '.pkl':
                eventIndex, eventHolder = self._processFictiveSaccadesProtocol(
                    eventIndex,
                    eventHolder,
                    fileIndex
                )

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

        