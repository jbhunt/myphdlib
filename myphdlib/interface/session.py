import re
import yaml
import h5py
import pickle
import numpy as np
import pathlib as pl
from datetime import date
from types import SimpleNamespace
from scipy.interpolate import interp1d as interp
from myphdlib.general.labjack import loadLabjackData
from myphdlib.interface.ephys import Population

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

def mapSaccadeDirection(eye, hemisphere):
    """
    Map eye movement direction to egocentric space (i.e., ipsi/contra)
    """

    if eye == 'left' and hemisphere == 'left':
        ipsi, contra = 'temporal', 'nasal'
    elif eye == 'left' and hemisphere == 'right':
        ipsi, contra = 'nasal', 'temporal'
    elif eye == 'right' and hemisphere == 'left':
        ipsi, contra = 'nasal', 'temporal'
    elif eye == 'right' and hemisphere == 'right':
        ipsi, contra = 'temporal', 'nasal'

    return ipsi, contra

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
            self.home = pl.Path(home)
        else:
            raise Exception('session folder must be of type str of pathlib.Path')

        #
        self._metadata = None
        self._loadBasicMetadata()

        #
        self._eye = eye
        self._folders = None
        self._saccadeOnsetTimestamps = None
        self._units = None
        self._labjackSamplingRate = None
        self._population = None

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

        if key not in self.keys():
            raise Exception(f'{key} is not a valid key')

        if self.outputFilePath.exists() == False:
            raise Exception('Could not locate output file')

        with open(self.outputFilePath, 'rb') as stream:
            try:
                container = pickle.load(stream)
            except EOFError as error:
                raise Exception(f'Ouptut file is corrupted') from None
            
        #
        container_ = dict()
        for key_, value_ in container.items():
            if key_ == key:
                continue
            container_[key_] = value_
        self.outputFilePath.unlink()
        with open(str(self.outputFilePath), 'wb') as stream:
            pickle.dump(container_, stream)

        return
    
    def load(self, path):
        """
        """

        if self.hdf.exists() == False:
            raise Exception(f'Output file does not exist')
        
        obj = None
        with h5py.File(str(self.hdf), 'r') as file:
            try:
                obj = file[path]
                if type(obj) == h5py.Dataset:
                    return np.array(obj)
                elif type(obj) == h5py.Group:
                    return obj
            except KeyError:
                pass

        return None
    
    def save(self, path, value, overwrite=True):
        """
        """

        if self.hdf.exists() == False:
            file = h5py.File(str(self.hdf), 'w')
        else:
            file = h5py.File(str(self.hdf), 'a')
        
        #
        if path in file.keys():
            if overwrite:
                del file[path]
            else:
                raise Exception(f'{path} dataset already exists')
        
        #
        dataset = file.create_dataset(path, value.shape, value.dtype, data=value)

        #
        file.close()        

        return
    
    def remove(self, path):
        """
        Remove a group from the hdf file
        """

        if self.hdf.exists() == False:
            raise Exception('Output file does not exists')
        
        with h5py.File(str(self.hdf), 'a') as file:
            if path in file.keys():
                del file[path]

        return
    
    def save2(self, parent, stem, dataset=None, attribute=None, overwrite=True):
        """
        """

        if self.hdf.exists() == False:
            file = h5py.File(str(self.hdf), 'w')
        else:
            file = h5py.File(str(self.hdf), 'a')

        #
        if all([dataset is None, attribute is None]):
            raise Exception('Data must be either a dataset or attribute not both')
        
        if parent.endswith('/'):
            parent = path.rstrip('/')

        #
        if dataset is not None:
            path = f'{parent}/{stem}'
            if path in file.keys():
                if overwrite:
                    del file[path]
                else:
                    raise Exception(f'{path} dataset already exists')
            ds = file.create_dataset(path, dataset.shape, dataset.dtype, data=dataset)

        #
        elif attribute is not None:
            path = parent
            ds = file[path]
            if stem in ds.attrs.keys():
                if overwrite:
                    del ds.attrs[stem]
                else:
                    raise Exception(f'{stem} attribute already exists for the {path} dataset')
            ds.attrs[stem] = attribute

        file.close()

        return
    
    @property
    def hdf(self):
        """
        """

        return self.home.joinpath('output.hdf')
    
    @property
    def leftCameraMovie(self): return
    
    @property
    def rightCameraMovie(self):  return
    
    @property
    def leftEyePose(self): return
    
    @property
    def rightEyePose(self): return

    @property
    def leftCameraTimestamps(self): return

    @property
    def rightCameraTimestamps(self): return

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
    
    @eye.setter
    def eye(self, value):
        if value in ('left', 'right'):
            self._eye = value
        return
    
    @property
    def eventSampleNumber(self):
        return None
    
    @property
    def referenceSampleNumber(self):
        return
    
    @property
    def labjackDataMatrix(self):
        """
        """

        if 'labjackDataMatrix' in self.keys():
            return self.read('labjackDataMatrix')
        
        else:
            return loadLabjackData(self.folders.labjack)
    
    @property
    def saccadeWaveformsIpsi(self):
        """
        Get the ipsilateral saccade waveforms
        """

        ipsi, contra = mapSaccadeDirection(self.eye, self.hemisphere)
        saccadeClassificationResults = self.read('saccadeClassificationResults')
        waves = saccadeClassificationResults[self.eye][ipsi]['waveforms']

        return waves
    
    @property
    def saccadeWaveformsContra(self):
        """
        Get the ipsilateral saccade waveforms
        """

        ipsi, contra = mapSaccadeDirection(self.eye, self.hemisphere)
        saccadeClassificationResults = self.read('saccadeClassificationResults')
        waves = saccadeClassificationResults[self.eye][contra]['waveforms']

        return waves
    
    @property
    def saccadeIndicesIpsi(self):
        """
        """

        ipsi, contra = mapSaccadeDirection(self.eye, self.hemisphere)
        saccadeClassificationResults = self.read('saccadeClassificationResults')
        indices = saccadeClassificationResults[self.eye][ipsi]['indices']

        return indices
    
    @property
    def saccadeIndicesContra(self):
        """
        """

        ipsi, contra = mapSaccadeDirection(self.eye, self.hemisphere)
        saccadeClassificationResults = self.read('saccadeClassificationResults')
        indices = saccadeClassificationResults[self.eye][contra]['indices']

        return indices
    
    @property
    def folders(self):
        """
        """

        if self._folders is None:
            folders_ = {
                'videos': None,
                'labjack': None,
                'ephys': None
            }
            for folder in ('videos', 'Videos'):
                path = self.home.joinpath(folder)
                if path.exists():
                    folders_['videos'] = path
                    break
            for folder in ('labjack', 'LabJack'):
                path = self.home.joinpath(folder)
                if path.exists():
                    folders_['labjack'] = path
                    break
            for folder in ('ephys', 'neuropixels'):
                path = self.home.joinpath(folder)
                if path.exists():
                    folders_['ephys'] = path
                    break
            for folder in ('stimuli', 'stim'):
                path = self.home.joinpath(folder)
                if path.exists():
                    folders_['stimuli'] = path
                    break
            self._folders = SimpleNamespace(**folders_)

        return self._folders
    
    @property
    def population(self):
        """
        """

        if self._population is None:
            self._population = Population(self)

        return self._population
    
    def computeTimestamps(self, eventIndices):
        """
        Convert labjack indices to timestamps in the ephys recording
        """
        
        #
        with h5py.File(self.hdf, 'r') as file:
            keys = list(file.keys())
        if 'tfp' not in keys:
            raise Exception('Timestamping funciton has not been estimated')
        
        #
        # params = self.read('timestampingFunctionParameters')
        params = dict()
        for key in ('m', 'xp', 'fp', 'b'):
            value = self.load(f'tfp/{key}')
            if value.size == 1:
                value = value.item()
            params[key] = value

        f = interp(params['xp'], params['fp'], fill_value='extrapolate')

        #
        eventIndices = np.atleast_1d(eventIndices)
        nSamples = len(eventIndices)
        mask = np.invert(np.isnan(eventIndices)) 
        output = np.full(nSamples, np.nan)

        #
        timestamps = np.around(
            f(np.array(eventIndices)[mask]) * params['m'] + params['b'],
            3
        )
        output[mask] = timestamps

        return output
    
    @property
    def inputFilePath(self):
        """
        """

        return self.home.joinpath('input.txt')
    
    @property
    def labjackSamplingRate(self):
        """
        """

        files = [
            file for file in pl.Path(self.folders.labjack).iterdir()
                if file.suffix == '.dat'
        ]
        fileNumbers = [int(str(file).rstrip('.dat').split('_')[-1]) for file in files]
        sortedIndex = np.argsort(fileNumbers)
        dat = files[sortedIndex[0]]
        with open(dat, 'rb') as stream:
            lines = stream.readlines()
            for lineIndex, lineData in enumerate(lines):
                line = lineData.decode()
                if bool(re.search('Time.*\r\n', line)):
                    t1 = float(lines[lineIndex + 1].decode().split('\t')[0])
                    t2 = float(lines[lineIndex + 2].decode().split('\t')[0])
                    dt = t2 - t1
                    fs = int(round(1 / dt, 0))
                    break

        return fs
            
    def hasGroup(self, path):
        """
        """

        if self.hdf.exists() == False:
            return False
        

        with h5py.File(self.hdf, 'r') as file:
            try:
                group = file[path]
                return True
            except KeyError:
                return False

    @property
    def fps(self):
        """
        """

        file = None
        result = list(self.folders.videos.glob('*metadata.yaml'))
        if len(result) == 1:
            file = result.pop()
        if file is None:
            raise Exception()
        
        #
        with open(file, 'r') as stream:
            metadata = yaml.full_load(stream)
        framerate = None
        for key in metadata.keys():
            if key.startswith('cam'):
                if metadata[key]['ismaster'] == True:
                    framerate = metadata[key]['framerate']
                    break

        return framerate
    
    @property
    def isAutosorted(self):
        """
        """

        filenames = (
            'spike_times.npy',
            'spike_clusters.npy'
        )
        results = list()
        for filename in filenames:
            if len(list(self.folders.ephys.rglob(f'*{filename}'))) == 1:
                results.append(True)
            else:
                results.append(False)

        return all(results)
    
    @property
    def hasPoseEstimates(self):
        """
        """

        if self.leftEyePose is not None or self.rightEyePose is not None:
            return True
        else:
            return False
        
    def keys(self, path):
        """
        Check the available keys for a group specified by the path
        """

        if self.hdf.exists() == False:
            raise Exception('No output file found')
        

        with h5py.File(self.hdf, 'r') as file:
            try:
                group = file[path]
                return tuple(group.keys())
            except KeyError:
                raise Exception(f'{path} is not a valid group path') from None
            
    @property
    def saccadeOnsetTimestamps(self):
        """
        """

        if self._saccadeOnsetTimestamps is None:
            self._saccadeOnsetTimestamps = dict()
            for eye in ('left', 'right'):
                self._saccadeOnsetTimestamps[eye] = dict()
                for direction in ('nasal', 'temporal'):
                    path = f'saccades/predicted/{eye}/{direction}/timestamps'
                    if self.hasGroup(path):
                        timestamps = self.load(path)
                    else:
                        timestamps = np.array([])
                    self.saccadeOnsetTimestamps[eye][direction] = timestamps

        return self._saccadeOnsetTimestamps