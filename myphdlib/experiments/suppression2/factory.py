# Imports
import os
import re
import yaml
import string
import pickle
import pathlib as pl
from myphdlib.general.session import saveSessionData, locateFactorySource

# Class definitions
class Session():
    """
    """

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

    result = sessionObject.ephysFolderPath.rglob('Neuropix-PXI-100.0')
    if result:
        spikeSortingResult = SpikeSortingResults(str(result.pop()))
    else:
        return

    outputFolderPath = pl.Path(outputFolder)
    for neuron in spikeSortingResult:
        file = outputFolderPath.joinpath(f'spikeTimestampsUnit{neuron.clusterNumber}.txt')
        with open(file, 'w') as stream:
            for ts in neuron.timestamps:
                stream.write(f'{ts:.3f}\n')

    #
    visualStimuliData = sessionObject.load('visualStimuliData')
    sparseNoiseData = visualStimuliData['sn']
    with open(outputFolderPath.joinpath('sparseNoiseStimulus.txt'), 'w') as stream:
        for i, xy, t1, t2 in zip(sparseNoiseData.values()):
            x, y = xy[0], xy[1]
            stream.write(f'{x}, {y}, {t1}, {t2}\n')

    return