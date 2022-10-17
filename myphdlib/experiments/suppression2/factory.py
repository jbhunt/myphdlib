# Imports
import os
import re
import pickle
import pathlib as pl

#
def saveSessionData(sessionObject, name, data, createOutputFile=True):
    """
    """

    #
    if sessionObject.outputFilePath.exists() == False:
        if createOutputFile:
            with open(str(sessionObject.outputFilePath), 'wb') as stream:
                pass
        else:
            raise Exception('Could not locate output file')

    #
    with open(str(sessionObject.outputFilePath), 'rb') as stream:
        try:
            dataContainer = pickle.load(stream)
        except EOFError:
            dataContainer = dict()
    sessionObject.outputFilePath.unlink() # TODO: Wait to delete the output file until it passes a check

    #
    try:
        dataContainer.update({name: data})
        with open(str(sessionObject.outputFilePath), 'wb') as stream:
            pickle.dump(dataContainer, stream)
    except:
        import pdb; pdb.set_trace()

        print()

    return

def loadSessionData(sessionObject, name):
    """
    """

    if sessionObject.outputFilePath.exists() == False:
        raise Exception('Could not locate output file')

    with open(sessionObject.outputFilePath, 'rb') as stream:
        try:
            dataContainer = pickle.load(stream)
        except EOFError:
            raise Exception('Output file is empty') from None

    if name not in dataContainer.keys():
        raise Exception(f'Invalid data key: {name}')
    else:
        return dataContainer[name]

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
        self.outputFilePath = self.sessionFolderPath.joinpath('output.pickle')
        self.timestampsFilePath = self.ephysFolderPath.joinpath('events', 'Neuropix-PXI-100.0', 'TTL_1', 'timestamps.npy')
        self.messagesFilePath = self.ephysFolderPath.joinpath('sync_messages.txt')

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


class SessionFactory():
    """
    """

    def __init__(self, hdd='JH-DATA-01', alias='Suppression2'):
        """
        """

        user = os.environ['USER']
        self.rootFolder = pl.Path(f'/media/{user}').joinpath(hdd, alias)
        if self.rootFolder.exists() == False:
            raise Exception('Could not locate data')

        self.sessionFolders = None

        return

    def produce(self):
        return

    def __iter__(self):
        self.sessionFolders = list()
        for date in self.rootFolder.iterdir():
            for animal in date.iterdir():
                self._sessionFolders.append(str(animal))
        self._listIndex = 0
        return self

    def __next__(self):
        if self._listIndex < len(self.sessionFolders):
            sessionFolder = self.sessionFolders[self._listIndex]
            self._listIndex += 1
            return Session(sessionFolder)
        else:
            raise StopIteration