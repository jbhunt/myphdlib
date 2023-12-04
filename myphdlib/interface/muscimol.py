import yaml
import numpy as np
import pandas as pd
import pathlib as pl
from myphdlib.interface.session import SessionBase
from myphdlib.pipeline.stimuli import StimuliProcessingMixin
from myphdlib.pipeline.events import EventsProcessingMixin
from myphdlib.pipeline.saccades import SaccadesProcessingMixin
from myphdlib.pipeline.spikes import SpikesProcessingMixin
from myphdlib.pipeline.activity import ActivityProcessingMixin
from myphdlib.pipeline.prediction import PredictionProcessingMixin
from myphdlib.pipeline.cleanup import CleanupProccessingMixin
from myphdlib.extensions.matplotlib import placeVerticalLines
from myphdlib.general.labjack import filterPulsesFromPhotologicDevice

def readExperimentLog(log, animal, date, letter=None, key='treatment'):
    """
    """

    #
    if type(date) != str:
        date = str(date)

    #
    try:
        sheet = pd.read_excel(log, sheet_name=animal)
    except ValueError as error:
        raise Exception(f'{animal} is not a valid sheet name') from None
    
    #
    if letter is None:
        row = sheet[sheet.Date == date].squeeze()
    else:
        mask = np.logical_and(sheet.Date == date, sheet.Letter == letter)
        row = sheet[mask].squeeze()

    #
    if key in row.keys():
        return row[key]
    
    else:
        raise Exception(f'{key} is not a valid key')

class EventsProcessingMixinMuscimol(EventsProcessingMixin):
    """
    """

    def _timestampCameraTrigger(self):
        """
        """

        if self.cohort == 1:
            frameIntervals = self.load(f'frames/{self.primaryCamera}/intervals')
            frameTimestamps = np.insert(np.cumsum(frameIntervals), 0, 0)
            self.save('labjack/cameras/missing', np.zeros(frameTimestamps.size).astype(bool))
            self.save('labjack/cameras/timestamps', frameTimestamps)

        elif self.cohort == 2:
            super()._timestampCameraTrigger()
        
        return

    def _timestampVideoFrames(self):
        """
        """

        if self.cohort == 1:
            for camera in ('left', 'right'):
                droppedFrames = self.load(f'frames/{camera}/dropped')
                frameIntervals = np.loadtxt(
                    self.leftCameraTimestamps if camera == 'left' else self.rightCameraTimestamps
                )
                nFrames = droppedFrames.size
                videoFrameTimestamps = np.full(nFrames, np.nan)
                videoFrameTimestamps[np.invert(droppedFrames)] = np.cumsum(frameIntervals) / 1000000000
                self.save(f'frames/{camera}/timestamps', videoFrameTimestamps)

        elif self.cohort == 2:
            super()._timestampVideoFrames()

        return

    def _createLabjackDataMatrix(self):
        return
        
    def _extractLabjackTimespace(self):
        return

    def _extractBarcodeSignals(self):
        return

    def _decodeBarcodeSignals(self):
        return

    def _estimateTimestampingFunction(self):
        return

    def _computeRelativeEventTiming(self):
        return

class SaccadesProcessingMixinMuscimol(SaccadesProcessingMixin):
    """
    """

    def _reorientEyePosition(self, reflect='left'):
        """
        Eye position needs to be rotated 180 deg to align videos with the rest of the dataset
        """

        super()._reorientEyePosition(reflect)
        pose = self.load('pose/reoriented')
        pose *= -1
        self.save('pose/reoriented', pose)

        return

    def _determineGratingMotionAssociatedWithEachSaccade(self):
        return

class StimuliProcessingMixinMuscimol(StimuliProcessingMixin):
    """
    """

    def _identifyProtocolEpochs(self, xData=None, bufferInSamples=10):
        """
        """

        #
        datasetPaths = (
            'epochs/sn/pre',
            'epochs/mb/pre',
            'epochs/dg',
            'epochs/sn/post',
            'epochs/mb/post',
            'epochs/ng'
        )

        if self.cohort == 1:
            for datasetPath in datasetPaths:
                self.save(datasetPath, np.array([np.nan, np.nan]))
            return

        #
        if xData is None:
            M = self.load('labjack/matrix')
            lightSensorSignal = M[:, self.labjackChannelMapping['stimulus']]
            xData = np.array([
                np.where(np.diff(lightSensorSignal) > 0.5)[0][0] - bufferInSamples,
                np.where(np.diff(lightSensorSignal) * -1 > 0.5)[0][-1] + bufferInSamples
            ])

        for datasetPath in datasetPaths:
            if datasetPath != 'epochs/dg':
                self.save(datasetPath, np.array([np.nan, np.nan]))
            else:
                self.save(datasetPath, xData.flatten())

        return

    def _processsDriftingGratingProtocol(self):
        """
        """

        self.log('Processing data for the drifting grating protocol')

        # Only the grating onset was signaled
        if self.cohort == 1:
            metadataFile = list(self.home.rglob('*visual-stimuli-metadata.txt')).pop()
            with open(metadataFile, 'r') as stream:
                lines = stream.readlines()
            eventIndices = list()
            for line in lines:
                eventName, frameIndex = line.split(', ')
                if eventName == 'grating':
                    eventIndices.append(int(frameIndex))
            eventIndices = np.array(eventIndices)
            gratingOnsetTimestamps = self.computeTimestamps(eventIndices)
            itiOnsetTimestamps = np.array([]).astype(float)

        # Grating onset and offset were signaled
        elif self.cohort == 2:
            results = list(self.home.rglob('realtimeGratingMetadata.txt'))
            if len(results) == 1:
                metadataFile = results.pop()
            else:
                self.log('Could not locate the drifting grating metadata file', level='error')
                return
            with open(metadataFile, 'r') as stream:
                lines = stream.readlines()[1:]
            gratingMotionDuringEvents = np.array([
                int(line.split(', ')[1]) for line in lines
            ])
            nEvents = gratingMotionDuringEvents.size

            # 
            M = self.load('labjack/matrix')
            signal = M[:, self.labjackChannelMapping['stimulus']]
            filtered = filterPulsesFromPhotologicDevice(signal)
            eventIndices = np.where(
                np.diff(filtered) > 0.5
            )[0]
            if eventIndices.size != nEvents:
                self.save('stimuli/dg/iti/timestamps', np.array([]).astype(float))
                self.save('stimuli/dg/grating/timestamps', np.array([]).astype(float))
                self.log('Invalid number of events detected during the drifting grating stimulus', level='error')
                return
            gratingOnsetIndices = eventIndices[0::2]
            itiOnsetIndices = eventIndices[1::2]

            #
            gratingOnsetTimestamps = self.computeTimestamps(gratingOnsetIndices)
            itiOnsetTimestamps = self.computeTimestamps(itiOnsetIndices)

        #
        self.save('stimuli/dg/iti/timestamps', itiOnsetTimestamps)
        self.save('stimuli/dg/grating/timestamps', gratingOnsetTimestamps)

        return


    def _runStimuliModule(self):
        """
        """

        self._processsDriftingGratingProtocol()
        
        return

class SpikesProcessingMixinMuscimol(SpikesProcessingMixin):
    """
    """

    def _runSpikesModule(self):
        return

class ActivityProcessingMixinMuscimol(ActivityProcessingMixin):
    """
    """

    def _runActivityModule(self):
        return

class MuscimolSession(
    SaccadesProcessingMixinMuscimol,
    EventsProcessingMixinMuscimol,
    StimuliProcessingMixinMuscimol,
    SpikesProcessingMixinMuscimol,
    ActivityProcessingMixinMuscimol,
    PredictionProcessingMixin,
    CleanupProccessingMixin,
    SessionBase
    ):
    """
    """

    labjackChannelMapping = {
        'barcode': 5,
        'cameras': 6,
        'stimulus': 7
    }

    def computeTimestamps(self, eventIndices):
        """
        """

        # Cast indices to an integer dtype
        if np.issubdtype(eventIndices.dtype, np.integer) == False:
            eventIndices = eventIndices.astype(int)

        # Labjack was not used for these recordings
        if self.cohort == 1:
            frameTimestamps = self.load(f'frames/{self.primaryCamera}/timestamps')
            eventTimestamps = frameTimestamps[eventIndices]
            return eventTimestamps

        # Only sessions from cohort 2 used the labjack and can be timestamped
        elif self.cohort == 2:
            labjackTimespace = self.load('labjack/timespace')
            eventTimestamps = labjackTimespace[eventIndices]
            return eventTimestamps

        # Do nothing
        # TODO: Use the timestamps of the camera frames to create a timespace
        else:
            return np.full(eventIndices.size, np.nan)
    
    @property
    def fps(self):
        """
        """

        if self.cohort == 1:
            tail = 'video-acquisition-metadata.yml'
        elif self.cohort == 2:
            tail = 'metadata.yaml'
        result = list(self.sessionFolderPath.joinpath('videos').glob(f'*{tail}'))
        if len(result) != 1:
            raise Exception('Could not locate video acquisition metadata file')
        with open(result.pop(), 'r')  as stream:
            metadata = yaml.full_load(stream)

        for key in metadata.keys():
            if key in ('cam1', 'cam2'):
                if metadata[key]['ismaster']:
                    fps = int(metadata[key]['framerate'])

        return fps

    @property
    def primaryCamera(self):
        """
        """

        if self.cohort == 1:
            tail = 'video-acquisition-metadata.yml'
        elif self.cohort == 2:
            tail = 'metadata.yaml'
        result = list(self.sessionFolderPath.joinpath('videos').glob(f'*{tail}'))
        if len(result) != 1:
            raise Exception('Could not locate video acquisition metadata file')
        with open(result.pop(), 'r')  as stream:
            metadata = yaml.full_load(stream)

        for key in metadata.keys():
            if key in ('cam1', 'cam2'):
                if metadata[key]['ismaster']:
                    nickname = metadata[key]['nickname']
                    if 'left' in nickname.lower():
                        return 'left'
                    elif 'right' in nickname.lower():
                        return 'right'
                    else:
                        return None

        return
    
    @property
    def leftCameraMovie(self):
        """
        """

        if self.cohort == 1:
            tag = 'left-camera-movie'
        elif self.cohort == 2:
            tag = 'leftCam-0000_reflected'
        result = list(self.sessionFolderPath.joinpath('videos').glob(f'*{tag}.mp4'))
        if len(result) != 1:
            return None
        else:
            return result.pop()
    
    @property
    def rightCameraMovie(self):
        """
        """

        if self.cohort == 1:
            tag = 'right-camera-movie'
        elif self.cohort == 2:
            tag = 'rightCam-0000'
        result = list(self.sessionFolderPath.joinpath('videos').glob(f'*{tag}*.mp4'))
        if len(result) != 1:
            return None
        else:
            return result.pop()   
    
    @property
    def leftEyePose(self):
        """
        """

        if self.cohort == 1:
            tail = 'left-camera-movieDLC_resnet50_GazerMay24shuffle1_1030000.csv'
        elif self.cohort == 2:
            tail = 'leftCam-0000_reflectedDLC_resnet50_GazerMay24shuffle1_1030000.csv'
        result = list(self.sessionFolderPath.joinpath('videos').glob(f'*{tail}'))
        if len(result) != 1:
            return None
        else:
            return result.pop()
    
    @property
    def rightEyePose(self):
        """
        """

        if self.cohort == 1:
            tail = 'right-camera-movie-reflectedDLC_resnet50_GazerMay24shuffle1_1030000.csv'
        elif self.cohort == 2:
            tail = 'rightCam-0000DLC_resnet50_GazerMay24shuffle1_1030000.csv'
        result = list(self.sessionFolderPath.joinpath('videos').glob(f'*{tail}'))
        if len(result) != 1:
            return None
        else:
            return result.pop()
    
    @property
    def leftCameraTimestamps(self):
        """
        """

        if self.cohort == 1:
            tag = 'left-camera-timestamps'
        elif self.cohort == 2:
            tag = 'leftCam_timestamps'
        result = list(self.sessionFolderPath.joinpath('videos').glob(f'*{tag}*'))
        if len(result) != 1:
            return None
        else:
            return result.pop()
    
    @property
    def rightCameraTimestamps(self):
        """
        """
        
        if self.cohort == 1:
            tag = 'right-camera-timestamps'
        elif self.cohort == 2:
            tag = 'rightCam_timestamps'
        result = list(self.sessionFolderPath.joinpath('videos').glob(f'*{tag}*'))
        if len(result) != 1:
            return None
        else:
            return result.pop()

    @SessionBase.eventSampleNumbers.getter
    def eventSampleNumbers(self):
        return None
