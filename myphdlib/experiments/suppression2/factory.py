import os
import re
import yaml
import string
import pickle
import pathlib as pl
from myphdlib.general.session import saveSessionData, locateFactorySource
from myphdlib.general.ephys import SpikeSortingResults

class Session():
    def __init__(self, sessionFolder):
        """
        """

        # Folders
        self.sessionFolderPath = pl.Path(sessionFolder)
        self.labjackFolderPath = self.sessionFolderPath.joinpath('labjack')
        self.ephysFolderPath = self.sessionFolderPath.joinpath('ephys')
        self.videosFolderPath = self.sessionFolderPath.joinpath('videos')

        # Files
        self.inputFilePath = self.sessionFolderPath.joinpath('input.txt')
        self.outputFilePath = self.sessionFolderPath.joinpath('output.pickle')
        self.timestampsFilePath = self.ephysFolderPath.joinpath('events', 'Neuropix-PXI-100.0', 'TTL_1', 'timestamps.npy')
        self.messagesFilePath = self.ephysFolderPath.joinpath('sync_messages.txt')
        self.driftingGratingMetadataFilePath = self.videosFolderPath.joinpath('driftingGratingMetadata.txt')
        self.movingBarsMetadataFilePath = self.videosFolderPath.joinpath('movingBarsMetadata.txt')
        self.sparseNoiseMetadataFilePath = self.videosFolderPath.joinpath('sparseNoiseMetadata.txt')
        self.stimuliMetadataFilePaths = {
            'dg': self.videosFolderPath.joinpath('driftingGratingMetadata.txt'),
            'ng': self.videosFolderPath.joinpath('noisyGratingMetadata.txt'),
            'mb': self.videosFolderPath.joinpath('movingBarsMetadata.txt'),
            'sn': self.videosFolderPath.joinpath('sparseNoiseMetadata.txt')
        }
        self.missingFilePath = self.sessionFolderPath.joinpath('missing.txt')

        # Identify the very first sample index in the ephys recording
        if self.messagesFilePath.exists() != True:
            self.ephysFirstSample = None
        else:
            with open(self.messagesFilePath, 'r') as stream:
                for line in stream.readlines():
                    result = re.findall(': [0-9]*@30000Hz', line)
                    if len(result) != 0:
                        ephysFirstSample = int(result.pop().strip(': ').split('@')[0])
                        self.ephysFirstSample = ephysFirstSample

        # Determine the animal, date, and treatment
        self.notesFilePath = self.sessionFolderPath.joinpath('notes.txt')
        self.animal, self.date, self.treatment = None, None, None
        if self.notesFilePath.exists():
            with open(self.notesFilePath, 'r') as stream:
                lines = stream.readlines()
            for line in lines:
                for attribute in ('animal', 'date', 'experiment'):
                    if bool(re.search(f'{attribute}*', line.lower())) and line.startswith('-') == False:
                        value = line.lower().split(': ')[-1].rstrip('\n')
                        setattr(self, attribute, value)

        #
        self._spikeSortingResults = None
        self._probeOnsetTimestamps = None
        self._gratingOnsetTimestamps = None
        self._motionOnsetTimestamps = None
        self._itiOnsetTimestamps = None
        self._spotOnsetTimestamps = None
        self._spotOffsetTimestamps = None
        self._barOnsetTimestamps = None

        return

    def load(self, name):
        """
        """

        if self.outputFilePath.exists() == False:
            raise Exception('Could not locate output file')

        with open(self.outputFilePath, 'rb') as stream:
            dataContainer = pickle.load(stream)

        if name not in dataContainer.keys():
            raise Exception(f'Invalid data key: {name}')
        else:
            return dataContainer[name]

    @property
    def fps(self):
        """
        Video acquisition framerate
        """

        framerate = None
        result = list(self.videosFolderPath.glob('*metadata.yaml'))
        if result:
            with open(result.pop(), 'r') as stream:
                acquisitionMetadata = yaml.safe_load(stream)
            for cameraAlias in ('cam1', 'cam2'):
                if acquisitionMetadata[cameraAlias]['ismaster']:
                    framerate = acquisitionMetadata[cameraAlias]['framerate']

        return framerate

    @property
    def isAutosorted(self):
        """
        TODO: Code this
        Checks if the session has been sorted with Kilosort
        """

        return

    @property
    def spikeSortingResults(self):
        """
        """

        if self._spikeSortingResults is None:
            self._spikeSortingResults = SpikeSortingResults(self.ephysFolderPath.joinpath('continuous', 'Neuropix-PXI-100.0'))

        return self._spikeSortingResults

    @property
    def probeOnsetTimestamps(self):
        """
        """

        if self._probeOnsetTimestamps is None:
            data = self.load('visualStimuliData')['dg']
            iterable = zip(
                data['i'],
                data['d'],
                data['e'],
                data['t']
            )
            self._probeOnsetTimestamps = list()
            for i, d, e, t, in iterable:
                if e == 3:
                    self._probeOnsetTimestamps.append(t)
            self._probeOnsetTimestamps = np.array(self._probeOnsetTimestamps)

        return self._probeOnsetTimestamps

    @property
    def barOnsetTimestamps(self):
        """
        """

        if self._barOnsetTimestamps is None:
            data = self.load('visualStimuliData')['mb']
            iterable = zip(
                data['i'],
                data['o'],
                data['t1'],
                data['t2']
            )
            self._barOnsetTimestamps = dict()
            for i, o, t1, t2 in iterable:
                if str(o) not in self._barOnsetTimestamps:
                    self._barOnsetTimestamps[str(o)] = list()
                self._barOnsetTimestamps[str(o)].append(t1)
            #

        return self._barOnsetTimestamps

    @property
    def spotOnsetTimestamps(self):
        """
        """

        if self._spotOnsetTimestamps is None:
            self._spotOnsetTimestamps = self.load('visualStimuliData')['sn']['t1']

        return self._spotOnsetTimestamps

    @property
    def spotOffsetTimestamps(self):
        """
        """

        if self._spotOffsetTimestamps is None:
            self._spotOffsetTimestamps = self.load('visualStimuliData')['sn']['t2']

        return self._spotOffsetTimestamps

    @property
    def barOnsetTimestamps(self):
        """
        """

        if self._barOnsetTimestamps is None:
            self._barOnsetTimestamps = self.load('visualStimuliData')['mb']['t1']

        return self._barOnsetTimestamps

    @property
    def gratingOnsetTimestamps(self):
        """
        """

        if self._gratingOnsetTimestamps is None:
            data = self.load('visualStimuliData')['dg']
            iterable = zip(
                data['i'],
                data['d'],
                data['e'],
                data['t']
            )
            self._gratingOnsetTimestamps = list()
            for i, d, e, t, in iterable:
                if e == 1:
                    self._gratingOnsetTimestamps.append(t)
            self._gratingOnsetTimestamps = np.array(self._gratingOnsetTimestamps)

        return self._gratingOnsetTimestamps

    @property
    def motionOnsetTimestamps(self):
        """
        """

        if self._motionOnsetTimestamps is None:
            data = self.load('visualStimuliData')['dg']
            iterable = zip(
                data['i'],
                data['d'],
                data['e'],
                data['t']
            )
            self._motionOnsetTimestamps = list()
            for i, d, e, t, in iterable:
                if e == 2:
                    self._motionOnsetTimestamps.append(t)
            self._motionOnsetTimestamps = np.array(self._motionOnsetTimestamps)

        return self._motionOnsetTimestamps

    @property
    def itiOnsetTimestamps(self):
        """
        """

        if self._itiOnsetTimestamps is None:
            data = self.load('visualStimuliData')['dg']
            iterable = zip(
                data['i'],
                data['d'],
                data['e'],
                data['t']
            )
            self._itiOnsetTimestamps = list()
            for i, d, e, t, in iterable:
                if e == 4:
                    self._itiOnsetTimestamps.append(t)
            self._itiOnsetTimestamps = np.array(self._itiOnsetTimestamps)

        return self._itiOnsetTimestamps

    @property
    def keys(self):
        """
        """

        with open(str(self.outputFilePath), 'rb') as stream:
            try:
                dataContainer = pickle.load(stream)
            except EOFError:
                dataContainer = dict()

        return dataContainer.keys()

class SessionFactory():
    """
    """

    def __init__(self, hdd='JH-DATA-01', alias='Suppression2', source=None):
        """
        """

        kwargs = {
            'hdd': hdd,
            'alias': alias,
            'source': source
        }
        self.rootFolderPath = locateFactorySource(**kwargs)
        self.sessionFolders = list()

        return

    def produce(self, animal, date):
        """
        Produce a specific session object as specified by the animal and date kwargs
        """

        sessionLocated = False
        for session in self:
            if session.animal == animal and session.date == date:
                sessionLocated = True
                break
        
        if sessionLocated:
            return session
        else:
            raise Exception('Could not locate session')

    # Iterator protocol definition
    def __iter__(self):
        self.sessionFolders = list()
        for date in self.rootFolderPath.iterdir():
            for animal in date.iterdir():
                self.sessionFolders.append(str(animal))
        self._listIndex = 0
        return self

    def __next__(self):
        if self._listIndex < len(self.sessionFolders):
            sessionFolder = self.sessionFolders[self._listIndex]
            self._listIndex += 1
            return Session(sessionFolder)
        else:
            raise StopIteration

import numpy as np
import pathlib as pl
from myphdlib.general.ephys import SpikeSortingResults

def createShareableSummary(sessionObject, outputFolder):
    """
    """

    #
    outputFolderPath = pl.Path(outputFolder)
    if outputFolderPath.exists() == False:
        outputFolderPath.mkdir()

    #
    sortingResultsFolder = str(sessionObject.ephysFolderPath.joinpath('continuous', 'Neuropix-PXI-100.0'))
    spikeSortingResult = SpikeSortingResults(sortingResultsFolder)

    #
    with open(outputFolderPath.joinpath('populationSpikeTimestamps.txt'), 'w') as stream:
        for neuron in spikeSortingResult._neuronList:
            if neuron.timestamps.size < 3:
                continue
            line = ''.join([f'{ts:.3f}, ' for ts in neuron.timestamps[:-1]])
            line += f'{neuron.timestamps[-1]:.3f}\n'
            stream.write(line)

    #
    data = sessionObject.load('visualStimuliData')['sn']
    iterable = zip(
        data['xy'][:, 0],
        data['xy'][:, 1],
        data['t1'],
        data['t2']
    )
    with open(outputFolderPath.joinpath('sparseNoiseStimulus.txt'), 'w') as stream:
        for x, y, t1, t2 in iterable:
            stream.write(f'{x}, {y}, {t1}, {t2}\n')

    #
    data = sessionObject.load('visualStimuliData')['dg']
    iterable = zip(
        data['e'],
        data['d'],
        data['t']
    )
    probeIndex = 0
    with open(outputFolderPath.joinpath('driftingGratingStimulus.txt'), 'w') as stream:
        for e, d, t in iterable:
            if e == 3:
                stream.write(f'{probeIndex + 1}, {d:.0f}, {t}\n')
                probeIndex += 1

    return