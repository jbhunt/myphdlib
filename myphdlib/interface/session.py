import re
import yaml
import h5py
import numpy as np
import pathlib as pl
from datetime import date
from types import SimpleNamespace
from scipy.interpolate import interp1d as interp
from myphdlib.interface.ephys import Population

class SessionBase():
    """
    """

    def __init__(self, home, eye='left', loadEphysData=False, loadPropertyValues=False):
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
        self._eye = eye
        self._folders = None
        self._labjackSamplingRate = None
        self._population = None
        self._probeTimestamps = None
        self._probeLatencies = None
        self._gratingMotionDuringProbes = None
        self._saccadeIndicesChronological = None
        self._saccadeTimestamps = None
        self._saccadeLabels = None
        self._saccadeWaveforms = None
        self._saccadeLatencies = None
        self._gratingMotionDuringSaccades = None
        self._sampleIndicesRange = None
        self._tRange = None
        self._barcodeValues = None
        self._barcodeTimestamps = None

        #
        self._loadBasicMetadata()
        if loadPropertyValues:
            self._loadPropertyValues()

        #
        if loadEphysData:
            try:
                assert self.population is not None
            except AssertionError as error:
                self.log(f'Failed to load ephys data', level='warning')

        return
    
    def _loadBasicMetadata(self):
        """
        Read the basic metadata into class attributes
        """

        file = self.home.joinpath('metadata.txt')
        if file.exists():
            self._metadata = dict()

            #
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
        
            #
            letter = 'a'
            matches = list(re.findall('\d\D', self.home.name))
            if len(matches) == 1:
                match = matches.pop()
                number, letter = match
                self._metadata['letter'] = letter

        else:
            raise Exception('Could not locate metadata file')
        
        return

    def _loadPropertyValues(self):
        """
        """

        labels = (
            'probe timestamps',
            'probe latencies',
            'grating motion during probes',
            'saccade indices',
            'saccade timestamps',
            'saccade directions',
            'saccade waveforms',
            'grating motion during saccades',
        )

        for i, p in enumerate((
            self.probeTimestamps,
            self.probeLatencies,
            self.gratingMotionDuringProbes,
            self.saccadeIndicesChronological,
            self.saccadeTimestamps,
            self.saccadeDirections,
            self.saccadeWaveforms,
            self.gratingMotionDuringSaccades,
            )):
            try:
                assert p is not None
            except AssertionError as error:
                self.log(f'Could not load {labels[i]} data', level='warning')

        return

    def updateBasicMetadata(
        self,
        key,
        value,
        intitialize=False
        ):
        """
        Update the metadata dictionary at the head of the session folder
        """

        #
        filename = self.home.joinpath('metadata.txt')
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
            else:
                raise Exception(f'Undetermined data tyep for {key_} key')
            metadata[key_] = value_

        #
        metadata[key] = value

        #
        filename.unlink()
        with open(filename, 'w') as stream:
            for key_, value_ in metadata.items():
                line = f'{key_}: {value_}\n'
                stream.write(line)
    
    def load(self, path, returnMetadata=False):
        """
        """

        if self.hdf.exists() == False:
            raise Exception(f'Output file does not exist')
        
        obj = None
        with h5py.File(str(self.hdf), 'r') as file:
            try:
                obj = file[path]
                if type(obj) == h5py.Dataset:
                    if returnMetadata:
                        return np.array(obj), dict(obj.attrs)
                    else:
                        return np.array(obj)
                elif type(obj) == h5py.Group:
                    return obj
            except KeyError:
                pass

        if returnMetadata:
            return None, {}
        else:
            return None
    
    def save(self, path, value, overwrite=True, metadata={}):
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
        if len(metadata) != 0 and type(metadata) == dict:
            for k in metadata.keys():
                dataset.attrs[k] = metadata[k]

        #
        file.close()        

        return
    
    def remove(self, path):
        """
        Remove a group or dataset from the hdf file
        """

        if self.hdf.exists() == False:
            raise Exception('Output file does not exists')
        
        with h5py.File(str(self.hdf), 'a') as file:
            if path in file.keys():
                del file[path]

        return
    
    def log(self, message, level='info', end=None):
        """
        """

        if end is None:
            print(f'{level.upper()}: ({self.animal}, {self.date}) {message}')
        else:
            print(f'{level.upper()}: ({self.animal}, {self.date}) {message}', end=end)

        return

    def listAllDatasets(self, returnPaths=False):
        """
        """

        pathsInFile = list()
        with h5py.File(self.hdf, 'r') as file:
            file.visit(lambda name: pathsInFile.append(name))

        datasetsInFile = list()
        with h5py.File(self.hdf, 'r') as file:
            for path in pathsInFile:
                if type(file[path]) == h5py.Dataset:
                    datasetsInFile.append(path)

        #
        for path in datasetsInFile:
            print(path)
            

        if returnPaths:
            return datasetsInFile

    def _makeOutputFile(
        self,
        overwrite=False
        ):
        """
        """

        outputFile = self.home.joinpath('output.hdf')
        if outputFile.exists() and overwrite == False:
            return
        with h5py.File(str(outputFile), 'w') as file:
            pass

        return

    @property
    def hdf(self): return self.home.joinpath('output.hdf')
    
    @property
    def leftCameraMovie(self): return
    
    @property
    def rightCameraMovie(self):  return
    
    @property
    def leftEyePose(self):
        """
        """

        file = None
        result = list(self.folders.videos.glob('*leftCam*DLC*.csv'))
        if len(result) == 1:
            file = result.pop()

        return file
    
    @property
    def rightEyePose(self):
        """
        """

        file = None
        result = list(self.folders.videos.glob('*rightCam*DLC*.csv'))
        if len(result) == 1:
            file = result.pop()

        return file

    @property
    def leftCameraTimestamps(self):
        """
        """

        file = None
        result = list(self.folders.videos.glob('*leftCam_timestamps.txt'))
        if len(result) == 1:
            file = result.pop()

        return file
    
    @property
    def rightCameraTimestamps(self):
        """
        """

        file = None
        result = list(self.folders.videos.glob('*rightCam_timestamps.txt'))
        if len(result) == 1:
            file = result.pop()

        return file

    @property
    def animal(self):
        if self._metadata is not None:
            return self._metadata['Animal'].lower()
        
    @property
    def date(self):
        if self._metadata is not None:
            return self._metadata['Date']

    @property
    def letter(self):
        if self._metadata is not None:
            return self._metadata['letter']
        
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
    def eventSampleNumbers(self):
        """
        """

        if self.experiment == 'Mlati':
            file = self.folders.ephys.joinpath('events', 'Neuropix-PXI-100.ProbeA-AP', 'TTL', 'sample_numbers.npy')
        elif self.experiment == 'Dreadds':
            file = self.folders.ephys.joinpath('events', 'Neuropix-PXI-100.0', 'TTL_1', 'timestamps.npy')
        if file.exists() == False: 
            raise Exception('Could not locate ephys event timestamps file')
        
        #
        eventSampleNumbers = np.load(file)

        return eventSampleNumbers

    @property
    def tRange(self):
        """
        """

        if self._tRange is None:
            if self.experiment == 'Mlati':
                file = self.folders.ephys.joinpath('continuous', 'Neuropix-PXI-100.ProbeA-AP', 'sample_numbers.npy')
            elif self.experiment == 'Dreadds':
                file = self.folders.ephys.joinpath('continuous', 'Neuropix-PXI-100.0', 'timestamps.npy')
            if file.exists() == False:
                self._tRange = 0, np.inf
            else:
                sampleNumbers = np.load(str(file), mmap_mode='r')
                sampleNumbersRange = np.around(
                    np.array([sampleNumbers[0], sampleNumbers[-1]]) - self.referenceSampleNumber,
                    3
                )
                self._tRange = sampleNumbersRange / 30000

        return self._tRange
    
    @property
    def referenceSampleNumber(self):
        """
        """

        file = self.folders.ephys.joinpath('sync_messages.txt')
        if file.exists() == False:
            raise Exception('Could not locate the ephys sync messages file')
        
        #
        with open(file, 'r') as stream:
            referenceSampleNumber = None
            for line in stream.readlines():
                if self.experiment == 'Mlati':
                    pattern = '@.*30000.*Hz:.*\d*'
                elif self.experiment == 'Dreadds':
                    pattern = 'start time:.*@'
                result = re.findall(pattern, line)
                if len(result) == 1:
                    if self.experiment == 'Mlati':
                        referenceSampleNumber = int(result.pop().rstrip('\n').split(': ')[-1])
                    elif self.experiment == 'Dreadds':
                        referenceSampleNumber = int(result.pop().rstrip('@').split('start time: ')[1])
                    break
        
        #
        if referenceSampleNumber is None:
            raise Exception('Failed to parse sync messages file for first sample number')

        return referenceSampleNumber
    
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

    @property
    def barcodeValues(self):
        if self._barcodeValues is None:
            self._barcodeValues = dict()
            self._barcodeValues['labjack'] = self.load('barcodes/labjack/values')
            self._barcodeValues['neuropixels'] = self.load('barcodes/neuropixels/values')
        return self._barcodeValues

    @property
    def barcodeTimestamps(self):
        if self._barcodeTimestamps is None:
            self._barcodeTimestamps = dict()
            self._barcodeTimestamps['labjack'] = self.load('barcodes/labjack/indices')
            self._barcodeTimestamps['neuropixels'] = self.load('barcodes/neuropixels/indices')
        return self._barcodeTimestamps

    def computeTimestamps(
        self,
        eventIndices,
        neuropixelsSamplingRate=30000,
        useInterpolation=True,
        returnSampleIndices=False
        ):
        """
        """
        
        #
        eventIndices = np.atleast_1d(eventIndices)
        eventMask = np.invert(np.isnan(eventIndices))
        eventTimestamps = np.full(eventIndices.size, np.nan)

        barcodeValuesCommon, barcodeIndicesLabjack, barcodeIndicesNeuropixels = np.intersect1d(
            self.barcodeValues['labjack'], self.barcodeValues['neuropixels'], return_indices=True
        )

        barcodeTimestampsLabjack = self.barcodeTimestamps['labjack'][barcodeIndicesLabjack]
        barcodeTimestampsNeuropixels = self.barcodeTimestamps['neuropixels'][barcodeIndicesNeuropixels]
        
        #
        if useInterpolation:
            f = interp(
                barcodeTimestampsLabjack,
                barcodeTimestampsNeuropixels - self.referenceSampleNumber,
                fill_value='extrapolate'
            )
            try:
                eventTimestamps[eventMask] = np.around(f(eventIndices[eventMask]), 0)
            except:
                import pdb; pdb.set_trace()

        #
        else:
            m = (barcodeTimestampsNeuropixels[-1] - barcodeTimestampsNeuropixels[0]) / \
                (barcodeTimestampsLabjack[-1] - barcodeTimestampsLabjack[0])
            b = barcodeTimestampsNeuropixels[0] - barcodeTimestampsLabjack[0] * m
            t = np.around(
                (eventIndices[eventMask] * m + b) - self.referenceSampleNumber,
                0
            )
            eventTimestamps[eventMask] = t

        #
        if returnSampleIndices:
            return eventTimestamps
        else:
            return np.around(eventTimestamps / neuropixelsSamplingRate, 3)
    
    # NOTE: This version of the method is deprecated
    def computeTimestamps_(self, eventIndices):
        """
        Convert labjack indices to timestamps in the ephys recording
        """
        
        #
        with h5py.File(self.hdf, 'r') as file:
            keys = list(file.keys())
        if 'tfp' not in keys:
            raise Exception('Timestamping funciton has not been estimated')
        
        #
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

    def hasDataset(self, path):
        """
        Looks for a group/dataset in the output file
        """

        if self.hdf.exists() == False:
            return False
        

        with h5py.File(self.hdf, 'r') as file:
            try:
                dataset = file[path]
                return True
            except KeyError:
                return False

    def hasTrainingDataForSaccadeClassification(
        self,
        dataForBothEyes=True
        ):
        """
        Looks for saccade classification training data in the outptut file
        """

        flags = {
            'left': False,
            'right': False,
        }
        for eye in ('left', 'right'):
            if self.hasGroup(f'saccades/training/{eye}/X'):
                flags[eye] = True

        #
        if dataForBothEyes:
            return all(list(flags.values()))
        else:
            return any(list(flags.values()))

    @property
    def hasLabjackData(self):
        return True if self.home.joinpath('labjack').exists() else False

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
    def probeTimestamps(self):
        if self._probeTimestamps is None:
            if self.hasDataset('stimuli/dg/probe/timestamps'):
                self._probeTimestamps = self.load('stimuli/dg/probe/timestamps')
        return self._probeTimestamps

    @property
    def probeLatencies(self):
        if self._probeLatencies is None:
            if self.hasDataset('stimuli/dg/probe/tts'):
                self._probeLatencies = self.load('stimuli/dg/probe/tts')
        return self._probeLatencies

    @property
    def gratingMotionDuringProbes(self):
        if self._gratingMotionDuringProbes is None:
            if self.hasDataset('stimuli/dg/probe/motion'):
                self._gratingMotionDuringProbes = self.load('stimuli/dg/probe/motion')
        return self._gratingMotionDuringProbes

    @property
    def saccadeTimestamps(self):
        if self._saccadeTimestamps is None:
            if self.hasDataset(f'saccades/predicted/{self.eye}/timestamps'):
                self._saccadeTimestamps = self.load(f'saccades/predicted/{self.eye}/timestamps')

        return self._saccadeTimestamps

    @property
    def saccadeLatencies(self):
        if self._saccadeLatencies is None:
            if self.hasDataset(f'saccades/predicted/{self.eye}/ttp'):
                self._saccadeLatencies = self.load(f'saccades/predicted/{self.eye}/ttp')
        return self._saccadeLatencies

    @property
    def saccadeLabels(self):
        if self._saccadeLabels is None:
            if self.hasDataset(f'saccades/predicted/{self.eye}/labels'):
                self._saccadeLabels = self.load(f'saccades/predicted/{self.eye}/labels')

        return self._saccadeLabels

    @property
    def saccadeWaveforms(self):
        if self._saccadeWaveforms is None:
            if self.hasDataset(f'saccades/predicted/{self.eye}/waveforms'):
                self._saccadeWaveforms = self.load(f'saccades/predicted/{self.eye}/waveforms')

        return self._saccadeWaveforms

    @property
    def gratingMotionDuringSaccades(self):
        if self._gratingMotionDuringSaccades is None:
            if self.hasDataset(f'saccades/predicted/{self.eye}/gmds'):
                self._gratingMotionDuringSaccades = self.load(f'saccades/predicted/{self.eye}/gmds')

        return self._gratingMotionDuringSaccades

    def parseEvents(
        self,
        eventName='probe',
        coincident=False,
        eventDirection=None,
        coincidenceWindow=(-0.05, 0.05),
        ):
        """
        Create a mask that idenitfies coincident or non-coincident events
        """

        #
        if eventName == 'probe':
            if eventDirection is None:
                probeDirections = (-1, 1)
            else:
                probeDirections = (eventDirection,)
            f1 = np.array([
                True if probeDirection in probeDirections else False
                    for probeDirection in self.gratingMotionDuringProbes
            ])
            eventLatencies = self.probeLatencies

        #
        elif eventName == 'saccade':
            if eventDirection is None:
                saccadeDirections = (-1, +1)
            else:
                saccadeDirections = (eventDirection,)
            f1 = np.array([
                True if saccadeDirection in saccadeDirections else False
                    for saccadeDirection in self.saccadeLabels
            ])
            eventLatencies = self.saccadeLatencies

        #
        if coincident:
            try:
                f2 = np.logical_and(
                    eventLatencies >= coincidenceWindow[0],
                    eventLatencies <= coincidenceWindow[1]
                )
            except:
                import pdb; pdb.set_trace()

        #
        else:
            f2 = np.logical_or(
                eventLatencies < coincidenceWindow[0],
                eventLatencies > coincidenceWindow[1]
            )

        #
        eventMask = np.logical_and(
            f1,
            f2
        )

        return eventMask

    def filterProbes(
        self,
        trialType='ps',
        perisaccadicWindow=(-0.05, 0.1),
        probeDirections=(-1, 1),
        windowBufferForExtrasaccadicTrials=0,
        ):
        """
        Filter visual probes based on latency to nearest saccade
        """

        #
        if type(probeDirections) == int:
            probeDirections = (probeDirections,)
        probesByDirection = np.array([
            True if motion in probeDirections else False
                for motion in self.gratingMotionDuringProbes
        ])

        #
        if trialType == 'ps':
            trialMask = np.array([
                self.probeLatencies <= perisaccadicWindow[1],
                self.probeLatencies >= perisaccadicWindow[0],
                probesByDirection  
            ]).all(axis=0)

        #
        elif trialType == 'es':
            trialMask = np.logical_and(
                np.logical_or(
                    self.probeLatencies < perisaccadicWindow[0] - windowBufferForExtrasaccadicTrials,
                    self.probeLatencies > perisaccadicWindow[1] + windowBufferForExtrasaccadicTrials,
                ),
                probesByDirection
            )

        #
        elif trialType is None:
            trialMask = probesByDirection

        else:
            raise Exception(f'{trialType} is not a valid trial type')

        return trialMask

    def filterSaccades(
        self,
        trialType='es',
        saccadeDirections=('n', 't'),
        peristimulusWindow=(-0.1, 0.05),
        peristimulusWindowBuffer=0,
        ):
        """
        Create a mask which delimits extra-/peri-stimulus saccades
        """

        if type(saccadeDirections) in (int, str):
            saccadeDirections = (saccadeDirections,)
        saccadesByDirection = np.array([
            True if direction in saccadeDirections else False
                for direction in self.saccadeDirections
        ])

        # Extra-stimulus trial mask
        trialMask = np.array([
            np.logical_or(
                self.saccadeLatencies < (peristimulusWindow[0] - peristimulusWindowBuffer),
                self.saccadeLatencies > (peristimulusWindow[1] + peristimulusWindowBuffer),
            ),
            saccadesByDirection
        ]).all(axis=0)

        # Peri-stimulus trial mask
        if trialType == 'ps':
            trialMask = np.invert(trialMask)

        return trialMask

    @property
    def primaryCamera(self):
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
        primaryCamera = None
        for key in metadata.keys():
            if key.startswith('cam'):
                if metadata[key]['ismaster'] == True:
                    nickname = metadata[key]['nickname']
                    if 'left' in nickname:
                        primaryCamera = 'left'
                    elif 'right' in nickname:
                        primaryCamera = 'right'
                    break

        return primaryCamera