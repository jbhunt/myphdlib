import os
import yaml
import string
import pickle
import pathlib as pl
from myphdlib.general.ephys import SpikeSortingResults

class SessionBase():
    """
    """

    def __init__(self, sessionFolder, resolve=False):
        """
        """

        # Public attributes
        self.sessionFolderPath = pl.Path(sessionFolder) # TODO: Remove this
        self.home = pl.Path(sessionFolder)

        # Private attributes
        self._rez = None

        # Define file paths
        if resolve:
            self.resolve()

        return

    def load(self, name):
        """
        """

        if self.outputFilePath.exists() == False:
            raise Exception('Could not locate output file')

        with open(self.outputFilePath, 'rb') as stream:
            try:
                dataContainer = pickle.load(stream)
            except EOFError as error:
                raise Exception(f'Ouptut file is corrupted') from None

        if name not in dataContainer.keys():
            raise Exception(f'Invalid data key: {name}')
        else:
            return dataContainer[name]

    def save(self, name, data, createOutputFile=False):
        """
        """

        # TODO: Make this fail safe such that a R/W error doesn't corrupt the output files

        #
        if self.outputFilePath.exists() == False:
            if createOutputFile:
                with open(str(self.outputFilePath), 'wb') as stream:
                    pass
            else:
                raise Exception('Could not locate output file')

        #
        with open(str(self.outputFilePath), 'rb') as stream:
            try:
                dataContainer = pickle.load(stream)
            except EOFError:
                dataContainer = dict()
        self.outputFilePath.unlink() # TODO: Wait to delete the output file until it passes a check

        #
        dataContainer.update({name: data})
        with open(str(self.outputFilePath), 'wb') as stream:
            pickle.dump(dataContainer, stream)

        return

    def reload(self):
        """
        """

        self.__init__(str(self.sessionFolderPath))

        return

    def removeDataEntry(self, name):
        """
        """

        #
        if self.outputFilePath.exists() == False:
            raise Exception('Could not locate output file')

        #
        with open(str(self.outputFilePath), 'rb') as stream:
            try:
                dataContainer = pickle.load(stream)
            except EOFError:
                dataContainer = dict()
        self.outputFilePath.unlink() # TODO: Wait to delete the output file until it passes a check

        #
        del dataContainer[name]
        with open(str(self.outputFilePath), 'wb') as stream:
            pickle.dump(dataContainer, stream)

        return

    def resolve(self, ephys='ephys', videos='videos', labjack='labjack'):
        """
        Find all of the fundamental file paths
        """

        self.folders = {
            'ephys': self.home.joinpath(ephys),
            'videos': self.home.joinpath(videos),
            'labjack': self.home.joinpath(labjack)
        }

        self.files = {
            'notes': self.home.joinpath('notes.txt'),
            'input': self.home.joinpath('input.txt'),
            'output': self.home.joinpath('output.txt'),
            'neuropixels': {
                'metadata': self.folders['ephys'].joinpath('sync_messages.txt'),
                'timestamps': self.folders['ephys'].joinpath('events', 'Neuropix-PXI-100.0', 'TTL_1', 'timestamps.npy')
            }
        }

        self.metadata = {
            'bars': self.folders['videos'].joinpath('movingBarsMetadata.txt'),
            'dots': self.folders['videos'].joinpath('sparseNoiseMetadata.txt'),
            'grating': {
                'noisy': self.folders['videos'].joinpath('noisyGratingMetadata.txt'),
                'drifting': self.folders['videos'].joinpath('driftingGratingMetadata.txt')
            },
        }

        return

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

    @property
    def isAutosorted(self):
        """
        TODO: Code this
        Checks if the session has been sorted with Kilosort
        """

        return

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
    def videos(self):
        """
        """

        result = [str(f) for f in self.videosFolderPath.glob('*Cam*.mp4')]

        return result

    @property
    def rez(self):
        """
        """

        if self._rez is None:
            self._rez = SpikeSortingResults(self.ephysFolderPath.joinpath('continuous', 'Neuropix-PXI-100.0'))

        return self._rez

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
        import pdb; pdb.set_trace() # TODO: Remove this

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

def locateFactorySource(hdd, alias, source=None):
    """
    """

    rootFolderPath = None

    #
    if source is not None:
        rootFolderPath = pl.Path(source)

    #
    elif os.name == 'posix':
        user = os.environ['USER']
        rootFolderPath = pl.Path(f'/media/{user}').joinpath(hdd, alias)
        if rootFolderPath.exists() == False:
            rootFolderPath = None
    
    #
    elif os.name == 'nt':
        for driveLetter in string.ascii_uppercase:
            rootFolderPath = pl.WindowsPath().joinpath(f'{driveLetter}:/', alias)
            if rootFolderPath.exists():
                rootFolderPath = rootFolderPath
                break

    #
    if rootFolderPath is None:
        raise Exception('Could not locate root folder')
    else:
        return rootFolderPath