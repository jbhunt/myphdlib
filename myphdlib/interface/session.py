import pickle
import pathlib as pl
from datetime import date

def updateSessionMetadata(session, key, value, intitialize=True):
    """
    Update the metadata dictionary at the head of the session folder
    """

    #
    filename = session.home.joinpath('metadata.txt')
    if filename.exists() == False:
        if intitialize:
            with open(filename, 'w'):
                pass
        else:
            raise Exception('Metadata file does not exist')
        
    #
    with open(filename, 'r') as stream:
        lines = stream.readlines()
    metadata = dict()
    for line in lines:

        #
        key_, value_ = line.rstrip('\n').split(': ')
        if key_ == 'Cohort':
            value_ = int(value_)
        elif key_ == 'Date':
            value_ = date.fromisoformat(value_)
        elif key_ == 'Session' and value_ == 'None':
            value_ = None
        metadata[key_] = value_

    #
    metadata[key] = value

    #
    filename.unlink()
    with open(filename, 'w') as stream:
        for key_, value_ in metadata.items():
            line = f'{key_}: {value_}\n'
            stream.write(line)

    return

class SessionBase(object):
    """
    """

    def __init__(self, home, eye='left'):
        """
        """

        if type(home) in (pl.WindowsPath, pl.PosixPath):
            self.sessionFolderPath = home
            self.home = home
        elif type(home) == str:
            self.sessionFolderPath = pl.Path(home)
            self.home =  home
        else:
            raise Exception('session folder must be of type str of pathlib.Path')

        #
        self._metadata = None
        self._loadBasicMetadata()

        #
        self._eye = eye

        return
    
    def _loadBasicMetadata(self):
        """
        Read the basic metadata into class attributes
        """

        file = self.home.joinpath('metadata.txt')
        if file.exists():
            self._metadata = dict()
            with open(file, 'r') as stream:
                lines = stream.readlines()
            for line in lines:
                key, value = line.rstrip('\n').split(': ')
                if key == 'Cohort':
                    value = int(value)
                elif key == 'Date':
                    value = date.fromisoformat(value)
                elif key == 'Session' and value == 'None':
                    value = None
                self._metadata[key] = value

        else:
            raise Exception('Could not locate metadata file')
        
        return
    
    def write(self, obj, key, initialize=True):
        """
        """

        # TODO: Make this fail safe such that a R/W error doesn't corrupt the output files

        #
        if self.outputFilePath.exists() == False:
            if initialize:
                with open(str(self.outputFilePath), 'wb') as stream:
                    pass
            else:
                raise Exception('Could not locate output file')

        #
        with open(str(self.outputFilePath), 'rb') as stream:
            try:
                container = pickle.load(stream)
            except EOFError:
                container = dict()

        # TODO: Wait to delete the output file until it passes a check
        self.outputFilePath.unlink() 

        #
        container.update({key: obj})
        with open(str(self.outputFilePath), 'wb') as stream:
            pickle.dump(container, stream)

        return
    
    def read(self, key):
        """
        """

        if self.outputFilePath.exists() == False:
            raise Exception('Could not locate output file')

        with open(self.outputFilePath, 'rb') as stream:
            try:
                container = pickle.load(stream)
            except EOFError as error:
                raise Exception(f'Ouptut file is corrupted') from None

        if key not in container.keys():
            raise Exception(f'Invalid key: {key}')
        else:
            return container[key]
    
    def delete(self, key):
        """
        """

        return
    
    def keys(self):
        """
        Return a list of the keys in the output.pkl file
        """

        if self.outputFilePath.exists() == False:
            raise Exception('Could not locate output file')

        with open(self.outputFilePath, 'rb') as stream:
            try:
                container = pickle.load(stream)
            except EOFError as error:
                raise Exception(f'Ouptut file is corrupted') from None
            
        return list(container.keys())
    
    @property
    def leftCameraMovie(self): return
    
    @property
    def rightCameraMovie(self):  return
    
    @property
    def leftEyePose(self): return
    
    @property
    def leftEyePose(self): return

    @property
    def outputFilePath(self):
        return self.sessionFolderPath.joinpath('output.pkl')
    
    @property
    def videosFolderPath(self):
        return self.sessionFolderPath.joinpath('videos')
    
    @property
    def missingDataMask(self):
        return self.read('missingDataMask')
    
    @property
    def eyePositionUncorrected(self): return self.read('eyePositionUncorrected')

    @property
    def eyePositionCorrected(self): return self.read('eyePositionCorrected')

    @property
    def eyePositionDecomposed(self): return self.read('eyePositionDecomposed')

    @property
    def eyePositionReoriented(self): return self.read('eyePositionReoriented')

    @property
    def eyePositionFiltered(self): return self.read('eyePositionFiltered')

    @property
    def animal(self):
        if self._metadata is not None:
            return self._metadata['Animal'].lower()
        
    @property
    def date(self):
        if self._metadata is not None:
            return self._metadata['Date']
        
    @property
    def cohort(self):
        if self._metadata is not None:
            return self._metadata['Cohort']
        
    @property
    def experiment(self):
        if self._metadata is not None:
            return self._metadata['Experiment']
        
    @property
    def treatment(self):
        if self._metadata is not None:
            return self._metadata['Treatment'].lower()
    
    @property
    def hemisphere(self):
        if self._metadata is not None:
            return self._metadata['Hemisphere'].lower()
        
    @property
    def eye(self):
        return self._eye
    
    @property
    def saccadeWaveformsIpsi(self):
        """
        Get the ipsilateral saccade waveforms
        """

        saccadeClassificationResults = self.read('saccadeClassificationResults')
        if self.eye == 'left' and self.hemisphere == 'left':
            waves = saccadeClassificationResults[self.eye]['temporal']['waveforms']
        elif self.eye == 'left' and self.hemisphere == 'right':
            waves = saccadeClassificationResults[self.eye]['nasal']['waveforms']
        elif self.eye == 'right' and self.hemisphere == 'left':
            waves = saccadeClassificationResults[self.eye]['nasal']['waveforms']
        elif self.eye == 'right' and self.hemisphere == 'right':
            waves = saccadeClassificationResults[self.eye]['temporal']['waveforms']

        return waves
    
    @property
    def saccadeWaveformsContra(self):
        """
        Get the ipsilateral saccade waveforms
        """

        saccadeClassificationResults = self.read('saccadeClassificationResults')
        if self.eye == 'left' and self.hemisphere == 'left':
            waves = saccadeClassificationResults[self.eye]['nasal']['waveforms']
        elif self.eye == 'left' and self.hemisphere == 'right':
            waves = saccadeClassificationResults[self.eye]['temporal']['waveforms']
        elif self.eye == 'right' and self.hemisphere == 'left':
            waves = saccadeClassificationResults[self.eye]['temporal']['waveforms']
        elif self.eye == 'right' and self.hemisphere == 'right':
            waves = saccadeClassificationResults[self.eye]['nasal']['waveforms']

        return waves

