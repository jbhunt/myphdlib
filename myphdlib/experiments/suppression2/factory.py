import os
import re
import yaml
import string
import pickle
import numpy as np
import pathlib as pl
from scipy.signal import find_peaks as findPeaks
from myphdlib.general.session import saveSessionData, locateFactorySource, SessionBase
from myphdlib.general.ephys import SpikeSortingResults

class EgocentricBilateralDirection():

    def __init__(self, value=None):
        self.value = None

    @property
    def opposite(self):
        if self.value is None:
            return None
        elif self.value == 'ipsi':
            return 'contra'
        else:
            return 'ipsi'
    
    @property
    def value(self):
        return self.value

class Session(SessionBase):
    def __init__(self, sessionFolder):
        """
        """

        #
        super().__init__(sessionFolder)

        # Folders
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
        self._saccadeOnsetTimestamps = None
        self._probeOnsetTimestamps = None
        self._probeMotionDirections = None
        self._gratingOnsetTimestamps = None
        self._motionOnsetTimestamps = None
        self._itiOnsetTimestamps = None
        self._spotOnsetTimestamps = None
        self._spotOffsetTimestamps = None
        self._barOnsetTimestamps = None
        self._barOffsetTimestamps = None

        return

    def movingBarTimestamps(self, orientation=0, position='center'):
        """
        Returns the timestamps for the moving bar stimulus
        """

        with open(self.videosFolderPath.joinpath('movingBarsMetadata.txt'), 'r') as stream:
            lines = stream.readlines()
        orientations = list()
        for line in lines:
            if re.match('.*,.*,.*', line) and line.startswith('Columns') == False:
                e, o, t = line.rstrip('\n').split(', ')
                if int(e) == 1:
                    orientations.append(int(round(float(o))))
        orientations = np.array(orientations)

        #
        timestamps = None
        if position == 'center':
            difference = self.barOffsetTimestamps - self.barOnsetTimestamps
            centered = self.barOnsetTimestamps + difference / 2
            timestamps = centered[:24][orientations[:24] == orientation]
        elif position == 'onset':
            timestamps = self.barOnsetTimestamps[:24][orientations[:24] == orientation]

        return timestamps

    def indexProbeTimestamps(self, index=0, minimumITI=5):
        """
        Return the timestamps for the ith probe in the sequence of probes for each presentation 
        of the drifting grating stimulus
        """

        dt = np.diff(self.probeOnsetTimestamps)
        peakIndices, peakProperties = findPeaks(dt, height=minimumITI)
        timestamps = self.probeOnsetTimestamps[peakIndices + 1 + index]
        timestamps = np.concatenate([
            np.array([self.probeOnsetTimestamps[index]]),
            timestamps
        ])

        return timestamps

    def saccadeOnsetTimestamps(self, eye='left'):
        """
        Categorize saccades by saccade direction and time relative to visual probes
        TODO: Categorize saccades as peri-stimulus or extra-stimulus
        """

        data = self.load('saccadeOnsetTimestamps')
        saccadeOnsetTimestamps = dict()

        if eye == 'left':
            saccadeOnsetTimestamps['ipsi'] = data[eye]['nasal']
            saccadeOnsetTimestamps['contra'] = data[eye]['temporal']
        else:
            saccadeOnsetTimestamps['ipsi'] = data[eye]['temporal']
            saccadeOnsetTimestamps['contra'] = data[eye]['nasal']

        return saccadeOnsetTimestamps

    def parseVisualProbes(self, perisaccadicTimeWindow=(-0.05, 0.1)):
        """
        Categorize probes by direction of the drifting grating and time relative to saccades
        """

        result = {
            'extrasaccadic': {
                'ipsi': {
                    'timestamps': list(),
                    'latencies': list(),
                    'directions': list() # TODO: Include the direction of the grating in addition to the saccade direction
                },
                'contra': {
                    'timestamps': list(),
                    'latencies': list(),
                    'directions': list() # TODO: Include the direction of the grating in addition to the saccade direction
                },
            },
            'perisaccadic': {
                'ipsi': {
                    'timestamps': list(),
                    'latencies': list(),
                    'directions': list() # TODO: Include the direction of the grating in addition to the saccade direction
                },
                'contra': {
                    'timestamps': list(),
                    'latencies': list(),
                    'directions': list() # TODO: Include the direction of the grating in addition to the saccade direction
                },
            }
        }

        #
        saccadeOnsetTimestamps = self.saccadeOnsetTimestamps()
        for probeIndex, (probeOnsetTimestamp, probeMotionDirection) in enumerate(zip(self.probeOnsetTimestamps, self.probeMotionDirections)):

            #
            saccadeDirection = 'ipsi' if probeMotionDirection == -1 else 'contra'
            
            #
            index = np.argmin(np.abs(saccadeOnsetTimestamps[saccadeDirection] - probeOnsetTimestamp))
            latency = saccadeOnsetTimestamps[saccadeDirection][index] - probeOnsetTimestamp

            # for direction in ('ipsi', 'contra'):
            #     closestSaccadeIndex = np.argmin(np.abs(saccadeOnsetTimestamps[direction] - probeOnsetTimestamp))
            #     latency = saccadeOnsetTimestamps[direction][closestSaccadeIndex] - probeOnsetTimestamp
            #     if abs(latency) < closestSaccadeAttributes['latency']:
            #         closestSaccadeAttributes['latency'] = latency
            #         closestSaccadeAttributes['direction'] = direction
            
            #
            if perisaccadicTimeWindow[0] <= latency <= perisaccadicTimeWindow[1]:
                category = 'perisaccadic'
            else:
                category = 'extrasaccadic'
            if probeMotionDirection == -1:
                motion = 'contra'
            else:
                motion = 'ipsi'
            result[category][motion]['timestamps'].append(probeOnsetTimestamp)
            result[category][motion]['latencies'].append(latency)
            result[category][motion]['directions'].append(saccadeDirection)

        #
        for category in result.keys():
            for motion in ('ipsi', 'contra'):
                for feature in ('timestamps', 'latencies', 'directions'):
                    result[category][motion][feature] = np.array(result[category][motion][feature]) 

        return result

    def createShareableSummary(self, outputFolder, overwrite=True):
        """
        """

        #
        outputFolderPath = pl.Path(outputFolder)
        spikesFilePath = outputFolderPath.joinpath('spikes.txt')
        eventsFilePath = outputFolderPath.joinpath('events.txt')
        if overwrite == False:
            if spikesFilePath.exists() and eventsFilePath.exists():
                return

        #
        nSpikes = 0
        for n in self.spikeSortingResults:
            nSpikes += n.timestamps.size
        spikeTimestamps = np.full([nSpikes, 2], np.nan)

        #
        events = (
            self.spotOnsetTimestamps,
            self.spotOffsetTimestamps,
            self.barOnsetTimestamps,
            self.barOffsetTimestamps,
            self.gratingOnsetTimestamps,
            self.motionOnsetTimestamps,
            self.probeOnsetTimestamps
        )
        nEvents = 0
        for event in events:
            mask = np.invert(np.isnan(event))
            nEvents += mask.sum()
        eventTimestamps = np.full([nEvents, 2], np.nan)

        #
        rowIndex = 0
        for n in self.spikeSortingResults:
            spikeTimestamps[rowIndex: rowIndex + n.timestamps.size, 0] = n.timestamps
            spikeTimestamps[rowIndex: rowIndex + n.timestamps.size, 1] = np.full(n.timestamps.size, n.clusterNumber)
            rowIndex += n.timestamps.size
        spikeTimestampsIndex = np.argsort(spikeTimestamps[:, 0])

        #
        rowIndex = 0
        for event, alias in zip(events, np.arange(len(events)) + 1):
            mask = np.invert(np.isnan(event))
            if mask.sum() == 0:
                continue
            eventTimestamps[rowIndex: rowIndex + event[mask].size, 0] = event[mask]
            eventTimestamps[rowIndex: rowIndex + event[mask].size, 1] = np.full(mask.sum(), alias)
            rowIndex += event[mask].size
        eventTimestampsIndex = np.argsort(eventTimestamps[:, 0])

        #
        np.savetxt(
            str(spikesFilePath),
            spikeTimestamps[spikeTimestampsIndex, :],
            fmt=['%.6f', '%d'],
            header=f'Timestamp, Unit',
            comments='',
            delimiter=', '
        )

        #
        np.savetxt(
            str(eventsFilePath),
            eventTimestamps[eventTimestampsIndex, :],
            fmt=['%.3f', '%d'],
            header=f'Timestamps, Event (1=Spot onset, 2=Spot offset, 3=Bar onset, 4=Bar offset, 5=Grating onset, 6=Motion Onset, 7=Probe onset)',
            comments='',
            delimiter=', '
        )

        return

    def getFrameTimestamps(self, eye='left'):
        """
        """

        # Compute timestamps for the trigger signal
        peakIndices, peakProps = findPeaks(
            np.abs(np.diff(self.load('exposureOnsetSignal'))),
            height=0.5,
        )
        params = self.load('timestampGeneratorParameters')
        timestamps = np.around(
            np.interp(peakIndices, params['xp'], params['fp']) * params['m'] + params['b'],
            3
        )

        #
        eyePositionReoriented = self.load('eyePositionReoriented')
        nFrames = eyePositionReoriented.shape[0]

        return timestamps[:nFrames]

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
    def probeMotionDirections(self):
        """
        """

        if self._probeMotionDirections is None:
            data = self.load('visualStimuliData')['dg']
            iterable = zip(
                data['i'],
                data['d'],
                data['e'],
                data['t']
            )
            self._probeMotionDirections = list()
            for i, d, e, t, in iterable:
                if e == 3:
                    self._probeMotionDirections.append(d)
            self._probeMotionDirections = np.array(self._probeMotionDirections)

        return self._probeMotionDirections

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
    def barOffsetTimestamps(self):
        """
        """

        if self._barOffsetTimestamps is None:
            self._barOffsetTimestamps = self.load('visualStimuliData')['mb']['t2']

        return self._barOffsetTimestamps

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

    def getMotionOnsetTimestamps(self, motion=-1):
        """
        """

        data = self.load('visualStimuliData')['dg']
        iterable = zip(
            data['i'],
            data['d'],
            data['e'],
            data['t']
        )
        motionOnsetTimestamps = list()
        for i, d, e, t, in iterable:
            if e == 2 and d == motion:
                motionOnsetTimestamps.append(t)

        return np.array(motionOnsetTimestamps)

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
        for date in self.rootFolderPath.iterdir():
            for animal in date.iterdir():
                self.sessionFolders.append(str(animal))
        
        self._loaded = False
        self._sessions = list()

        return

    def load(self, loadNeuralData=False):
        """
        """

        if self._loaded:
            return

        for sessionFolder in self.sessionFolders:
            session = Session(sessionFolder)
            if loadNeuralData:
                rez = session.rez
            self._sessions.append(session)

        self._loaded = True

        return self

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
        self._listIndex = 0
        return self

    def __next__(self):
        """
        """

        if self._listIndex < len(self.sessionFolders):
            if self._loaded:
                session = self._sessions[self._listIndex]
            else:
                session = Session(self.sessionFolders[self._listIndex])
            self._listIndex += 1
            return session
        else:
            raise StopIteration

    def __delitem__(self, key):
        raise Exception('Session objects are not able to be deleted')

    def __setitem__(self, key):
        raise Exception('Session objects are immutable')

    def __getitem__(self, key):
        if type(key) == slice:
            if key.stop > len(self.sessionFolders):
                raise Exception('Session index out of range')
            if self._loaded:
                return [session for session in self._sessions[key]]
            else:
                return [Session(sessionFolder) for sessionFolder in self.sessionFolders[key]]
        else:
            if key >= len(self.sessionFolders):
                raise Exception('Session index out of range')
            for sessionIndex in range(len(self.sessionFolders)):
                if sessionIndex == key:
                    if self._loaded:
                        return self._sessions[sessionIndex]
                    else:
                        return Session(self.sessionFolders[sessionIndex])