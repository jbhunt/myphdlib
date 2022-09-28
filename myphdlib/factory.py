import re
import yaml
import pickle
import numpy as np
import pandas as pd
import pathlib as pl
from myphdlib.labjack import extractLabjackEvent
from myphdlib.toolkit import inrange

class SessionBase():
    """
    """

    def __init__(self, sessionFolder):
        """
        """

        # Make sure session folder exists
        self.sessionFolderPath = pl.Path(sessionFolder)
        if self.sessionFolderPath.exists() == False:
            raise Exception("Session folder does not exist")
            
        # Initiailize hidden values for properties
        self._frameCount = None
        self._acquisitionFramerate = None
        self._eyePositionThreshold = None
        self._eyePositionUncorrected = None
        self._eyePositionCorrected = None
        self._eyePositionDecomposed = None
        self._eyePositionStandardized = None
        self._saccadeOnsetIndicesPutative = None
        self._saccadeOnsetIndicesClassified = None
        self._saccadeWaveformsPutative = None
        self._saccadeWaveformsClassified = None
        self._labjackData = None
        self._labjackSamplingRate = None
        self._labjackIndicesAcquisition = None
        self._labjackIndicesNeuropixels = None
        self._labjackIndicesBarcode = None
        self._labjackIndicesDisplay = None

        return

    def save(self, key, value):
        """
        """

        output = self.sessionFolderPath.joinpath('output.pickle')
        if output.exists() == False:
            data = {}
        
        else:
            try:
                with open(str(output), 'rb') as stream:
                    data = pickle.load(stream)
            except EOFError as error:
                print(f'WARNING: Output file is corrupted')
                output.unlink()
                data = {}
        
        data[key] = value
        with open(str(output), 'wb') as stream:
            pickle.dump(data, stream)

        return

    def load(self, key):
        """
        """

        output = self.sessionFolderPath.joinpath('output.pickle')
        if output.exists() == False:
            raise Exception('Output file does not exist')

        with open(str(output), 'rb') as stream:
            data = pickle.load(stream)

        if key not in list(data.keys()):
            raise Exception(f'{key} is not a saved variable')

        return data[key]

    def isValid(self):
        """
        """
        if self.videosFolder == None:
            return False
        if self.labjackFolder == None:
            return False
        if len(list(pl.Path(self.videosFolder).rglob('*Cam*Gazer*.csv'))) != 2:
            return False
        return True

    def isValid2(self):
        """
        Check if session has all of the prerequisite data
        """



        return

    @property
    def animal(self):
        for part in self.sessionFolderPath.parts:
            result = re.findall('[a-zA-Z]+\d', part)
            if len(result) == 1:
                return result.pop()
        return None

    @property
    def date(self):
        for part in self.sessionFolderPath.parts:
            result = re.findall('\d{4}-\d{2}-\d{2}', part)
            if len(result) == 1:
                return result.pop()
        return None

    @property
    def session(self):
        for part in self.sessionFolderPath.parts:
            if bool(re.search('[a-zA-Z]+\d', part)):
                if len(part) != len(self.animal):
                    return part[-1]
                else:
                    return None
        return None

    @property
    def videosFolder(self):
        for folder in self.sessionFolderPath.iterdir():
            result = any([
                bool(re.search('videos', folder.name)),
                bool(re.search('Videos', folder.name)),
                bool(re.search('session\d{3}', folder.name))
            ])
            if result:
                return str(folder)
        return None

    @property
    def labjackFolder(self):
        for folder in self.sessionFolderPath.iterdir():
            result = any([
                bool(re.search('labjack', folder.name)),
                bool(re.search('LabJack', folder.name)),
            ])
            if result:
                return str(folder)
        return None

    @property
    def ephysFolder(self):
        for folder in self.sessionFolderPath.iterdir():
            result = any([
                bool(re.search('neuropixels', folder.name.lower())),
                bool(re.search('ephys', folder.name.lower()))
            ])
            if result:
                return str(folder)
        return None

    @property
    def eyePositionThreshold(self):
        return self._eyePositionThreshold

    @property
    def frameCount(self):
        """
        """

        if self._frameCount is None:

            #
            counts = np.full(6, np.nan)
            sources = (
                ('cameras', 'left'),
                ('cameras', 'right'),
                ('deeplabcut', 'left'),
                ('deeplabcut', 'right'),
                ('labjack', 'left'),
                ('labjack', 'right'),
            )

            #
            for index, eye in enumerate(('left', 'right')):

                #
                values = list()
                result = list(pl.Path(self.videosFolder).glob(f'*{eye}Cam_timestamps.txt'))
                if len(result) == 1:
                    intervals = np.loadtxt(result.pop())
                    values.append(intervals.size + 1)
                else:
                    values.append(np.nan)

                #
                result = list(pl.Path(self.videosFolder).glob(f'*{eye}Cam*Gazer*.csv'))
                if len(result) == 1:
                    frame = pd.read_csv(str(result.pop()), header=list(range(3)), index_col=0)
                    values.append(frame.shape[0])
                else:
                    values.append(np.nan)

                #
                if hasattr(self, 'labjackIndicesAcquisition'):
                    values.append(self.labjackIndicesAcquisition.size)
                else:
                    values.append(np.nan)

                counts[index::2] = np.array(values)

            index = pd.MultiIndex.from_tuples(sources, names=['source', 'eye'])
            self._frameCount = pd.DataFrame(counts, index=index, columns=['count'], dtype=pd.Int64Dtype()).T

        return self._frameCount

    @property
    def acquisitionFramerate(self):
        if self._acquisitionFramerate is None:
            for file in pl.Path(self.videosFolder).iterdir():
                if bool(re.search('.*metadata.yaml', file.name)):
                    with open(str(file), 'rb') as stream:
                        metadata = yaml.full_load(stream)
                        for key in metadata.keys():
                            value = metadata[key]
                            if type(value) == dict:
                                if 'ismaster' in list(value.keys()):
                                    if metadata[key]['ismaster']:
                                        self._acquisitionFramerate = metadata[key]['framerate']
        return self._acquisitionFramerate

    @property
    def eyePositionUncorrected(self):
        if self._eyePositionUncorrected is None:
            try:
                self._eyePositionUncorrected = self.load('eyePositionUncorrected')
            except:
                raise AttributeError(f'"Session" object has no attribute "eyePositionUncorrected"') from None

        return self._eyePositionUncorrected

    @property
    def eyePositionCorrected(self):
        if self._eyePositionCorrected is None:
            try:
                self._eyePositionCorrected = self.load('eyePositionCorrected')
            except:
                raise AttributeError(f'"Session" object has no attribute "eyePositionCorrected"') from None
        return self._eyePositionCorrected

    @property
    def eyePositionDecomposed(self):
        if self._eyePositionDecomposed is None:
            try:
                self._eyePositionDecomposed = self.load('eyePositionDecomposed')
            except:
                raise AttributeError(f'"Session" object has no attribute "eyePositionDecomposed"') from None
        return self._eyePositionDecomposed

    @property
    def eyePositionStandardized(self):
        if self._eyePositionStandardized is None:
            try:
                self._eyePositionStandardized = self.load('eyePositionStandardized')
            except:
                raise AttributeError(f'"Session" object has no attribute "eyePositionStandardized"') from None
        return self._eyePositionStandardized

    @property
    def saccadeOnsetIndicesPutative(self):
        if self._saccadeOnsetIndicesPutative is None:
            try:
                self._saccadeOnsetIndicesPutative = self.load('saccadeOnsetIndicesPutative')
            except:
                raise AttributeError(f'"Session" object has no attribute "saccadeOnsetIndicesPutative"') from None
        return self._saccadeOnsetIndicesPutative

    @property
    def saccadeOnsetIndicesClassified(self):
        if self._saccadeOnsetIndicesClassified is None:
            try:
                self._saccadeOnsetIndicesClassified = self.load('saccadeOnsetIndicesClassified')
            except:
                raise AttributeError(f'"Session" object has no attribute "saccadeOnsetIndicesClassified"') from None
        return self._saccadeOnsetIndicesClassified

    @property
    def saccadeWaveformsPutative(self):
        if self._saccadeWaveformsPutative is None:
            try:
                self._saccadeWaveformsPutative = self.load('saccadeWaveformsPutative')
            except:
                raise AttributeError(f'"Session" object has no attribute "saccadeWaveformsPutative"') from None
        return self._saccadeWaveformsPutative

    @property
    def saccadeWaveformsClassified(self):
        if self._saccadeWaveformsClassified is None:
            try:
                self._saccadeWaveformsClassified = self.load('saccadeWaveformsClassified')
            except:
                raise AttributeError(f'"Session" object has no attribute "saccadeWaveformsClassified"') from None
        return self._saccadeWaveformsClassified

    @property
    def labjackData(self):
        if self._labjackData is None:
            try:
                self._labjackData = self.load('labjackData')
            except:
                raise AttributeError(f'"Session" object has no attribute "labjackData"') from None
        return self._labjackData

    @property
    def labjackSamplingRate(self):
        if self._labjackSamplingRate is None:
            try:
                self._labjackSamplingRate = self.load('labjackSamplingRate')
            except:
                raise AttributeError(f'"Session" object has no attribute "labjackSamplingRate"') from None
        return self._labjackSamplingRate

    @property
    def labjackIndicesAcquisition(self):
        if self._labjackIndicesAcquisition is None:
            try:
                self._labjackIndicesAcquisition = self.load('labjackIndicesAcquisition')
            except:
                raise AttributeError(f'"Session" object has no attribute "labjackIndicesAcquisition"') from None
        return self._labjackIndicesAcquisition

    @property
    def labjackIndicesNeuropixels(self):
        if self._labjackIndicesNeuropixels is None:
            try:
                self._labjackIndicesNeuropixels = self.load('labjackIndicesNeuropixels')
            except:
                raise AttributeError(f'"Session" object has no attribute "labjackIndicesNeuropixels"') from None
        return self._labjackIndicesNeuropixels

    @property
    def labjackIndicesBarcode(self):
        if self._labjackIndicesBarcode is None:
            try:
                self._labjackIndicesBarcode = self.load('labjackIndicesBarcode')
            except:
                raise AttributeError(f'"Session" object has no attribute "labjackIndicesBarcode"') from None
        return self._labjackIndicesBarcode

    @property
    def labjackIndicesDisplay(self):
        if self._labjackIndicesDisplay is None:
            try:
                self._labjackIndicesDisplay = self.load('labjackIndicesDisplay')
            except:
                raise AttributeError(f'"Session" object has no attribute "labjackIndicesDisplay"') from None
        return self._labjackIndicesDisplay

class MuscimolSession(SessionBase):
    """
    """

    def __init__(self, sessionFolder):
        super().__init__(sessionFolder)

class SuppressionSession(SessionBase):
    """
    """

    def __init__(self, sessionFolder):
        super().__init__(sessionFolder)
        self.labjackChannelMapping = {
            'Neuropixels': None,
            'Acquisition': None,
            'Display'    : None,
        }
        self.labjackProcessingChain = (
        )
        return

    def _identifyLabjackChannels(self):
        """
        """

        #
        channelCount = self.labjackData.shape[1]
        samplingRate = round(1 / (self.labjackData[1, 0] - self.labjackData[0, 0]))
        minimumPulseWidth = round(4 / (1 / samplingRate * 1000))

        #
        for channelIndex in range(channelCount):

            #
            if max(self.labjackData[:, channelIndex]).max() > 1.0:
                analog = True
            else:
                analog = False
            eventSignal, eventIndices = extractLabjackEvent(
                self.labjackData,
                channelIndex,
                edge='rising',
                analog=analog,
                pulseWidthRange=(minimumPulseWidth, None)
            )

            #
            if eventIndices.size < 2:
                continue
            
            #
            intervals = np.diff(eventIndices) / samplingRate
            intervalMean = np.mean(intervals)
            intervalDeviation = np.std(intervals)
            if inrange(intervalMean, 0.9, 1.1) and intervalDeviation < 0.01:
                self.labjackChannelMapping['Neuropixels'] = channelIndex
            elif inrange(intervalMean, 0.008, 0.012) and intervalDeviation < 0.01:
                self.labjackChannelMapping['Acquisition'] = channelIndex
            elif intervalMean > 0.1:
                self.labjackChannelMapping['Display'] = channelIndex

        return

    def _identifyVisualProbes(self, trialCount=25):
        """
        """

        # TODO: Check for duplicate probe TTL pulses that happen right after the first pulse

        eventSignal, eventIndices = extractLabjackEvent(
            self.labjackData,
            self.labjackChannelMapping['Display'],
            edge='rising'
        )

        # 
        if eventIndices.size != 1120:
            print(f'WARNING: Too many events detected (n={eventIndices.size})')
            longestInterval = np.argsort(eventIndices)[-1]
            if eventIndices[longestInterval] / self.labjackSamplingRate > 6:
                eventIndices = eventIndices[:longestInterval + 1]
            if eventIndices.size != 1120:
                raise Exception('Parsing of visual events failed')
        
        #
        counter = 0
        probeOnsetIndices = list()
        for eventIndex in eventIndices:
            if counter < 2:
                pass
            elif counter == trialCount + 2:
                counter = 0
                continue
            else:
                probeOnsetIndices.append(eventIndex)
            counter += 1

        #
        probeOnsetTimestamps = self.labjackData[probeOnsetIndices, 0]
        self.save('probeOnsetTimestamps', probeOnsetTimestamps)
        self.save('probeOnsetIndices', np.array(probeOnsetIndices))
        

        return

class ConcussionSession(SessionBase):
    """
    """

    def __init__(self, sessionFolder):
        super().__init__(sessionFolder)
        self.labjackChannelMapping = {
            'Acquisition': 6,
            'Display':     7,
        }

        #
        searchResults = list(self.sessionFolderPath.glob('*Metadata*'))
        if len(searchResults) == 1:
            stimulusMetadataPath = searchResults.pop()
        else:
            self._reflex = None
            return

        if 'lightSteps' in str(stimulusMetadataPath):
            self._reflex = 'plr'
        elif 'driftingGrating' in str(stimulusMetadataPath):
            self._reflex = 'okr'
        else:
            self._reflex = None

        return

    def _identifySingleTrials(self):
        """
        """

        eventSignal, eventIndices = extractLabjackEvent(
            self.labjackData,
            self.labjackChannelMapping['Display'],
            edge='rising'
        )

        # OKR
        if self.reflex == 'okr':
            if eventIndices.size % 2 != 0:
                raise Exception('Odd number of grating stimuli detected for {self.animal} on {self.date}')
            motionOnsetIndices = eventIndices[0::2]
            motionOnsetTimestamps = self.labjackData[motionOnsetIndices, 0]
            motionOffsetIndices = eventIndices[1::2]
            motionOffsetTimestamps = self.labjackData[motionOffsetIndices, 0]
            self.save('motionOnsetTimestamps', motionOnsetTimestamps)
            self.save('motionOffsetTimestamps', motionOffsetTimestamps)

        # PLR
        elif self.reflex == 'plr':
            lightOnsetIndices = eventIndices[1::2]
            lightOnsetTimestamps = self.labjackData[lightOnsetIndices, 0]
            lightOffsetIndices = eventIndices[2::2]
            lightOffsetTimestamps = self.labjackData[lightOffsetIndices, 0]
            self.save('lightOnsetTimestamps', lightOnsetTimestamps)
            self.save('lightOffsetTimestamps', lightOffsetTimestamps)

        return

    @property
    def reflex(self):
        return self._reflex

class RealtimeSession(SessionBase):
    """
    """

    def __init__(self, sessionFolder):
        super().__init__(sessionFolder)
        self.labjackChannelMapping = {
            'Barcode':     5,
            'Acquisition': 6,
            'Display':     7,
        }
        return

    def _identifyVisualProbes(self, minimumPulseWidth=0.05):
        """
        """

        signal, indices = extractLabjackEvent(
            self.labjackData,
            self.labjackChannelMapping['Display'],
            edge='both'
        )
        pulseWidths = indices[1::2] - indices[0::2]
        probeOnsetIndices = indices[0::2][pulseWidths >= minimumPulseWidth]
        self.save('probeOnsetIndices', probeOnsetIndices)    

        return

    @property
    def eyePositionThreshold(self):
        """
        Eye position threshold for triggering probes
        """

        if self._eyePositionThreshold is None:
            datasetFolderPath = self.sessionFolderPath.parent.parent
            logFilePath = datasetFolderPath.joinpath('log.xlsx')
            frame = pd.read_excel(str(logFilePath), sheet_name=self.animal)
            if self.session == None:
                value = frame.loc[frame['date'] == self.date].threshold.item()
            else:
                mask = np.logical_and(frame.date == self.date, frame.session == self.session)
                self._eyePositionThreshold = frame.loc[mask].threshold.item()

        return self._eyePositionThreshold

class DreaddSession(SessionBase):
    """
    """

    def __init__(self, sessionFolder):
        super().__init__(sessionFolder)
        return

    def _identifySingleTrials(self):
        return

class SessionFactory():
    """
    """

    def __init__(self):
        return

    def produce(self, sessionFolder):
        """
        """

        mapping = {
            'Muscimol'    : MuscimolSession,
            'Suppression' : SuppressionSession,
            'Concussion'  : ConcussionSession,
            'Realtime'    : RealtimeSession,
            'Dreadd'      : DreaddSession,    
        }

        datasetName = None
        sessionFolderPath = pl.Path(sessionFolder)
        for partIndex, part in enumerate(sessionFolderPath.parts):
            if bool(re.search('JH-DATA-*', part)):
                datasetName = sessionFolderPath.parts[partIndex + 1]
                break

        if datasetName == None:
            raise Exception('Could not determine dataset name')

        cls = mapping[datasetName]
        obj = cls(sessionFolder)

        return obj