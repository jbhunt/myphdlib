import re
import yaml
import h5py
import pickle
import numpy as np
import pathlib as pl
from datetime import date
from types import SimpleNamespace
from scipy.stats import pearsonr
from scipy.interpolate import interp1d as interp
from myphdlib.general.labjack import loadLabjackData, filterPulsesFromPhotologicDevice
from myphdlib.interface.ephys import Population
from myphdlib.general.toolkit import psth2, smooth

# TODO:
# [ ]: Create properties for population mapped data

class StimulusProcessingMixinBase():
    """
    """

    def _processMovingBarsProtocol(
        self,
        invertOrientations=True
        ):
        """
        """

        print(f'INFO[{self.animal}, {self.date}]: Processing the moving bars stimulus data')

        #
        M = self.load('labjack/matrix')
        start, stop = self.load('epochs/mb')
        signal = M[start: stop, self.labjackChannelMapping['stimulus']]

        # Check for data loss
        if np.isnan(signal).sum() > 0:
            print(f'WARNING[{self.animal}, {self.date}]: Data loss detected during the moving bars stimulus')
            return
        
        #
        filtered = filterPulsesFromPhotologicDevice(signal, minimumPulseWidthInSeconds=0.03)

        #
        risingEdgeIndices = np.where(np.diff(filtered) > 0.5)[0]
        barOnsetIndices = risingEdgeIndices[0::2]
        barOffsetIndices = risingEdgeIndices[1::2]
        barCenteredIndices = barOffsetIndices - barOnsetIndices
        barOnsetTimestamps = self.computeTimestamps(
            barOnsetIndices + start
        )
        barOffsetTimestamps = self.computeTimestamps(
            barOffsetIndices + start
        )
        self.save('stimuli/mb/onset/timestamps', barOnsetTimestamps)
        self.save('stimuli/mb/offset/timestamps', barOffsetTimestamps)

        #
        result = list(self.folders.stimuli.rglob('*movingBarsMetadata*'))
        if len(result) != 1:
            raise Exception('Could not locate moving bars stimulus metadata')
        file = result.pop()
        with open(file, 'r') as stream:
            lines = stream.readlines()[5:]
        orientation = list()
        for line in lines:
            event, orientation_, timestamp = line.rstrip('\n').split(', ')
            if int(event) == 1:
                if invertOrientations:
                    orientation_ = round(np.mod(float(orientation_) + 180, 360), 2)
                orientation.append(float(orientation_))
        self.save('stimuli/mb/orientation', np.array(orientation))

        return

    def _correctForCableDisconnectionDuringDriftingGrating(
        self,
        filtered,
        maximumPulseWidthInSeconds=0.6,
        ):
        """
        """

        corrected = np.copy(filtered)
        risingEdgeIndices = np.where(np.diff(filtered) > 0.5)[0]
        fallingEdgeIndices = np.where(np.diff(filtered) * -1 > 0.5)[0] + 1
        pulseEpochIndices = np.hstack([
            risingEdgeIndices.reshape(-1, 1),
            fallingEdgeIndices.reshape(-1, 1)
        ])
        pulseWidthsInSeconds = np.diff(pulseEpochIndices, axis=1) / self.labjackSamplingRate
        for flag, epoch in zip(pulseWidthsInSeconds > maximumPulseWidthInSeconds, pulseEpochIndices):
            if flag:
                start, stop = epoch
                corrected[start: stop] = 0

        return corrected

    def processVisualEvents(self):
        return

class SessionBase():
    """
    """

    def __init__(self, home, eye='left', loadEphysData=True, loadPropertyValues=False):
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
        self._saccadeOnsetTimestamps = None
        self._labjackSamplingRate = None
        self._population = None
        self._probeTimestamps = None
        self._probeLatencies = None
        self._gratingMotionDuringProbes = None
        self._saccadeIndicesChronological = None
        self._saccadeTimestamps = None
        self._saccadeDirections = None
        self._saccadeWaveforms = None
        self._gratingMotionDuringSaccades = None

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
    
    def log(self, message, level='info'):
        """
        """

        print(f'{level.upper()}[{self.date}, {self.animal}]: {message}')

        return

    def listAllPaths(self):
        """
        """

        pathsInFile = list()
        with h5py.File(self.hdf, 'r') as file:
            file.visit(lambda name: pathsInFile.append(name))

        for path in pathsInFile:
            print(path)
            

        return pathsInFile

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

    def hasDeeplabcutPoseEstimates(self):
        """
        Looks for processed eye position data in the output file
        """

        if self.leftEyePose is not None or self.rightEyePose is not None:
            return True
        else:
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
    def saccadeIndicesChronological(self):
        if self._saccadeIndicesChronological is None:
            result = all([
                self.hasDataset(f'saccades/predicted/{self.eye}/nasal/timestamps'),
                self.hasDataset(f'saccades/predicted/{self.eye}/temporal/timestamps')
            ])
            if result:
                nasalSaccadeTimestamps = self.load(f'saccades/predicted/{self.eye}/nasal/timestamps')
                temporalSaccadeTimestamps = self.load(f'saccades/predicted/{self.eye}/temporal/timestamps')
                allSaccadeTimestamps = np.concatenate([
                    nasalSaccadeTimestamps,
                    temporalSaccadeTimestamps
                ])
                self._saccadeIndicesChronological = np.argsort(allSaccadeTimestamps)

        return self._saccadeIndicesChronological

    @property
    def saccadeTimestamps(self):
        if self._saccadeTimestamps is None:
            result = all([
                self.hasDataset(f'saccades/predicted/{self.eye}/nasal/timestamps'),
                self.hasDataset(f'saccades/predicted/{self.eye}/temporal/timestamps')
            ])
            if result:
                nasalSaccadeTimestamps = self.load(f'saccades/predicted/{self.eye}/nasal/timestamps')
                temporalSaccadeTimestamps = self.load(f'saccades/predicted/{self.eye}/temporal/timestamps')
                allSaccadeTimestamps = np.concatenate([
                    nasalSaccadeTimestamps,
                    temporalSaccadeTimestamps
                ])
                self._saccadeIndicesChronological = np.argsort(allSaccadeTimestamps)
                self._saccadeTimestamps = allSaccadeTimestamps[self.saccadeIndicesChronological]

        return self._saccadeTimestamps

    @property
    def saccadeLatencies(self):
        if self._saccadeLatencies is None:
            if self.hasDataset(f'saccades/predicted/{self.eye}/unsigned/ttp'):
                self._saccadeLatencies = self.load(f'saccades/predicted/{self.eye}/unsigned/ttp')
        return self._saccadeLatencies

    @property
    def saccadeDirections(self):
        if self._saccadeDirections is None:
            result = all([
                self.hasDataset(f'saccades/predicted/{self.eye}/nasal/timestamps'),
                self.hasDataset(f'saccades/predicted/{self.eye}/temporal/timestamps'),
            ])
            if result:
                nasalSaccadeTimestamps = self.load(f'saccades/predicted/{self.eye}/nasal/timestamps')
                temporalSaccadeTimestamps = self.load(f'saccades/predicted/{self.eye}/temporal/timestamps')
                saccadeDirectionsUnordered = np.concatenate([
                    np.full(nasalSaccadeTimestamps.size, 'n', dtype=str),
                    np.full(temporalSaccadeTimestamps.size, 't', dtype=str)
                ])
                self._saccadeDirections = saccadeDirectionsUnordered[self.saccadeIndicesChronological]

        return self._saccadeDirections

    @property
    def saccadeWaveforms(self):
        if self._saccadeWaveforms is None:
            result = all([
                self.hasDataset(f'saccades/predicted/{self.eye}/nasal/waveforms'),
                self.hasDataset(f'saccades/predicted/{self.eye}/temporal/waveforms'),
            ])
            if result:
                nasalSaccadeWaveforms = self.load(f'saccades/predicted/{self.eye}/nasal/waveforms')
                temporalSaccadeWaveforms = self.load(f'saccades/predicted/{self.eye}/temporal/waveforms')
                saccadeWaveformsUnordered = np.concatenate([
                    nasalSaccadeWaveforms,
                    temporalSaccadeWaveforms,
                ], axis=0)
                self._saccadeWaveforms = saccadeWaveformsUnordered[self.saccadeIndicesChronological]

        return self._saccadeWaveforms

    @property
    def gratingMotionDuringSaccades(self):
        if self._gratingMotionDuringSaccades is None:
            result = all([
                self.hasDataset(f'saccades/predicted/{self.eye}/nasal/motion'),
                self.hasDataset(f'saccades/predicted/{self.eye}/temporal/motion'),
            ])
            if result:
                gratingMotionDuringNasalSaccades = self.load(f'saccades/predicted/{self.eye}/nasal/motion')
                gratingMotionDuringTemporalSaccades = self.load(f'saccades/predicted/{self.eye}/temporal/motion')
                gratingMotionDuringSaccadesUnordered = np.concatenate([
                    gratingMotionDuringNasalSaccades,
                    gratingMotionDuringTemporalSaccades,
                ], axis=0)
                self._gratingMotionDuringSaccades = gratingMotionDuringSaccadesUnordered[self.saccadeIndicesChronological]

        return self._gratingMotionDuringSaccades

    def filterProbes(
        self,
        trialType='ps',
        perisaccadicWindow=(-0.05, 0.1),
        probeDirections=(-1, 1),
        windowBufferForExtrasaccadicTrials=0.5,
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
        else:
            raise Exception(f'{trialType} is not a valid trial type')

        return trialMask

    def filterSaccades(
        self,
        saccadeDirections=('n', 't'),
        peristimulusWindow=(-0.1, 0.05),
        peristimulusWindowBuffer=0,
        ):
        """
        Exlude saccades coincident with a probe as defined by the peristimulus window
        """

        if type(saccadeDirections) == int:
            saccadeDirections = (saccadeDirections,)
        saccadesByDirection = np.array([
            True if direction in saccadeDirections else False
                for direction in self.saccadeDirections
        ])

        #
        if peristimulusWindow is None:
            trialMask = saccadesByDirection
        else:
            trialMask = np.array([
                self.saccadeLatencies < peristimulusWindow[0] - peristimulusWindowBuffer[0],
                self.saccadeLatencies > peristimulusWindow[1] + peristimulusWindowBuffer[1],
                saccadesByDirection
            ]).all(axis=0)

        return trialMask

    def filterUnits(
        self,
        utypes=('vr', 'sr', 'nr', 'ud'),
        quality=('lq', 'hq'),
        minimumResponseAmplitude=0.3,
        ):
        """
        Filter single units based on unit type and spike-sorting quality
        """

        # Cast values to tuple if needed
        if type(utypes) == str:
            utypes = (utypes,)
        if type(quality) == str:
            quality = (quality,)

        # Filter
        populationMask = list()
        for unit in self.population:
            if unit.utype in utypes and unit.quality in quality and unit.gvr >= minimumResponseAmplitude:
                populationMask.append(True)
            else:
                populationMask.append(False)

        return np.array(populationMask)