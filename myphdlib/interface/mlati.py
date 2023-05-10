import re
import yaml
import pickle
import numpy as np
from myphdlib.interface._session import SessionBase
from myphdlib.extensions.matplotlib import placeVerticalLines
from myphdlib.general.labjack import loadLabjackData, filterPulsesFromPhotologicDevice

class StimulusProcessingMixin():
    """
    """

    def identifyProtocolEpochs(self, xData=None, nBlocksTotal=8):
        """
        """

        #
        if xData is None:
            M = self.load('labjack/matrix')
            lightSensorSignal = M[:, self.labjackChannelMapping['stimulus']]
            xData = np.around(placeVerticalLines(lightSensorSignal), 0).astype(int)

        #
        if xData.size - 1 != nBlocksTotal:
            print('Warning: Number of line indices != number of blocks + 1')
            return xData
        
        #
        if self.cohort == 1:
            paths = (
                'epochs/sn/pre',
                'epochs/bn/hr/lf',
                'epochs/bn/hr/hf',
                'epochs/bn/lr/lf',
                'epochs/bn/lr/hf',
                'epochs/fs',
                'epochs/mb',
                'epochs/dg'
            )
        elif self.cohort == 2:
            paths = (
                'epochs/sn/pre',
                'epochs/bn/hr/lf',
                'epochs/bn/hr/hf',
                'epochs/bn/lr/lf',
                'epochs/bn/lr/hf',
                'epochs/sn/post'
                'epochs/fs',
                'epochs/mb',
                'epochs/dg'
            )

        #
        nEpochs = len(paths)
        if nEpochs != indices.shape[0]:
            raise Exception('User input does not match expected number of epochs')

        #
        indices = np.hstack([
            xData[0:-1].reshape(-1, 1),
            xData[1:  ].reshape(-1, 1)
        ])

        #
        for path, (start, stop) in zip(paths, indices):
            self.save(path, np.array([start, stop]))

        return

    def processVisualEvents(self):
        """
        """

        if self.hasGroup('epochs') == False:
            raise Exception('Protocol epochs have not been defined by the user')

        self._processSparseNoiseProtocol()
        self._processBinaryNoiseProtocol()
        self._processFictiveSaccadesProtocol()
        self._processMovingBarsProtocol()
        self._processDriftingGratingProtocol()

        return
    
    def _interpolateMissingSparseNoiseTrials(self, filtered, start=0, nTrialsExpected=1020):
        """
        """

        #
        missing = list()

        # Compute the inter-pulse intervals
        risingEdgeIndices = np.where(np.diff(filtered) > 0.5)[0]
        risingEdgeIndices += start # NOTE: Need to add the start index so that the timestamp function works properly
        interPulseIntervals = np.diff(risingEdgeIndices) / self.labjackSamplingRate

        #
        for ipi in interPulseIntervals:
            nTrialsDetected = int(round(ipi / 0.5))
            missing.append(False)
            nTrialsExtra = nTrialsDetected - 1
            if nTrialsExtra > 0:
                for iTrial in range(nTrialsExtra):
                    missing.append(True)
        missing.append(False)      

        #
        if len(missing) != nTrialsExpected:
            return False, np.array([]), np.array([])
        
        #
        x = np.arange(nTrialsExpected)[missing]
        xp = np.arange(nTrialsExpected)[np.invert(missing)]
        fp = risingEdgeIndices
        y = np.around(np.interp(x, xp, fp), 0).astype(int)
        interpolated = np.full(nTrialsExpected, 0).astype(int)
        interpolated[missing] = y
        interpolated[np.invert(missing)] = risingEdgeIndices.astype(int)
        
        return True, np.array(missing), interpolated
    
    def _processSparseNoiseProtocol(self):
        """
        """

        if self.cohort == 1:
            blocks = ('pre')
        elif self.cohort == 2:
            blocks = ('pre', 'post')

        #
        for block in blocks:
            data = {
                'signs': list(),
                'fields': list(),
                'coords': list(),
                'missing': list(),
                'timestamps': list()
            }

            # Extract raw signal
            M = self.load('labjack/matrix')
            start, stop = self.load(f'epochs/sn/{block}')
            signal = M[start: stop, self.labjackChannelMapping['stimulus']]

            # Check for data loss
            if np.isnan(signal).sum() > 0:
                print(f'WARNING[{self.animal}, {self.date}]: Data loss detected during the sparse noise stimulus')
                # return signal

            #
            filtered = filterPulsesFromPhotologicDevice(signal, minimumPulseWidthInSeconds=0.03)

            #
            result, missing, eventIndices = self._interpolateMissingSparseNoiseTrials(
                filtered,
                start,
                nTrialsExpected=1020
            )
            if result == False:
                print(f'WARNING[{self.animal}, {self.date}]: Failed to process sparse noise stimulus')
                return

            #
            data['missing'] = missing
            data['signs'] = np.full(missing.size, True)
            data['signs'][1::2] = False

            #
            data['timestamps'] = self.computeTimestamps(np.array(eventIndices))
            #data['signs'] = np.array(data['signs'])

            #
            result = list(self.folders.stimuli.rglob('sparseNoiseMetadata.pkl'))
            if len(result) != 1:
                raise Exception('Could not locate the sparse noise metadata file')
            file = result.pop()
            with open(file, 'rb') as stream:
                metadata = pickle.load(stream)

            #
            for key in ('fields', 'coords'):
                data[key] = metadata[key]

            #
            for key in data.keys():
                value = data[key]
                if type(value) != np.ndarray:
                    value = np.array(value)
                self.save(f'stimuli/sn/{block}/{key}', value)

        return
    
    def _processBinaryNoiseProtocol(self):
        """
        """

        data = {
            ('hr', 'lf'): {
                'length': None,
                'grids': list(),
                'fields': list(),
                'missing': list(),
                'timestamps': list(),
            },
            ('hr', 'hf'): {
                'length': None,
                'grids': list(),
                'fields': list(),
                'missing': list(),
                'timestamps': list(),
            },
            ('lr', 'lf'): {
                'length': None,
                'grids': list(),
                'fields': list(),
                'missing': list(),
                'timestamps': list(),
            },
            ('lr', 'hf'): {
                'length': None,
                'grids': list(),
                'fields': list(),
                'missing': list(),
                'timestamps': list(),
            }
        }

        # Load labjack data
        # M = self.read('labjackDataMatrix')
        M = self.load('labjack/matrix')

        #
        iterable = zip(
            ('epochs/bn/hr/lf', 'epochs/bn/hr/hf', 'epochs/bn/lr/lf','epochs/bn/lr/hf'),
            (('hr', 'lf'), ('hr', 'hf'), ('lr', 'lf'), ('lr', 'hf')),
            (10, 100, 10, 100),
            (True, False, True, False),
        )
        for blockIndex, (path, blockParams, expectedTrialCount, fieldOffsetSignaled) in enumerate(iterable):

            # Extract raw signal
            start, stop = self.load(path)
            signal = M[start: stop, self.labjackChannelMapping['stimulus']]

            # Check for data loss
            if np.isnan(signal).sum() > 0:
                resolution, frequency = blockParams
                print(f'WARNING[{self.animal}, {self.date}]: Data loss detected during the binary noise stimulus (resolution={resolution}, frequency={frequency})')
                continue

            #
            filtered = filterPulsesFromPhotologicDevice(signal, minimumPulseWidthInSeconds=0.03)

            # Segment the signal into sub-blocks using the full-field flashes as delimiters

            # Find full-field flashes
            peakIndices = np.where(np.abs(np.diff(filtered)) > 0.5)[0]
            pulseEdges = np.hstack([
                peakIndices[0::2].reshape(-1, 1),
                peakIndices[1::2].reshape(-1, 1)
            ])
            pulseWidths = np.diff(pulseEdges, axis=1) / self.labjackSamplingRate
            flashOnsetIndices = pulseEdges[np.where(pulseWidths > 0.08)[0][0::2], 0]
            flashOffsetIndices = pulseEdges[np.where(pulseWidths > 0.08)[0][1::2], 1] + 1
            sequenceIndices = np.hstack([
                flashOffsetIndices[:-1].reshape(-1, 1),
                flashOnsetIndices[1:].reshape(-1, 1)
            ])
            arrayList = (
                sequenceIndices,
                np.array([[flashOffsetIndices[-1], filtered.size]]),
            )
            sequenceIndices = np.concatenate(arrayList, axis=0)

            # Identify missing trials/compute timestamps
            for sequenceStartIndex, sequeneceStopIndex in sequenceIndices:
                sequence = filtered[sequenceStartIndex: sequeneceStopIndex]
                peakIndices_ = np.where(np.diff(sequence) > 0.5)[0]

                # NOTE: The low-frequency stimuli have an inter-field interval which is also signaled
                if fieldOffsetSignaled:
                    peakIndices_ = peakIndices_[::2]

                #
                nPulses = peakIndices_.size
                if peakIndices_.size != expectedTrialCount:
                    print(f'WARNING: Unexpected number of pulses detected for binary noise stimulus: {nPulses}')
                    for iTrial in range(expectedTrialCount):
                        data[blockParams]['missing'].append(True)
                        data[blockParams]['timestamps'].append(np.nan)
                else:
                    timestamps = self.computeTimestamps(
                        peakIndices_ + start + sequenceStartIndex
                    )
                    for timestamp in timestamps:
                        data[blockParams]['timestamps'].append(timestamp)
                    for iTrial in range(expectedTrialCount):
                        data[blockParams]['missing'].append(False)

            # Read metadata file
            blockNumber = blockIndex + 1
            file = self.folders.stimuli.joinpath('metadata', f'binaryNoiseMetadata{blockNumber}.pkl')
            if file.exists() == False:
                raise Exception(f'ERROR: Could not locate binary noise metadata: {file.name}')
            with open(str(file), 'rb') as stream:
                metadata = pickle.load(stream)

            #
            fields = metadata['values'].reshape(-1, *metadata['shape'])
            grid = metadata['coords'].reshape(*metadata['shape'], 2)
            jitter = metadata['length'] / 2 * np.array([1, -1])

            # Extract the metadata for each trial
            iField = 0
            nEvents = metadata['events'].shape[0]
            for iEvent in range(nEvents):
                event = metadata['events'][iEvent].item()
                if event == 'field onset':
                    field = fields[iField]
                    jittered = metadata['jittered'][iField].item()
                    data[blockParams]['fields'].append(field)
                    if jittered:
                        data[blockParams]['grids'].append(grid + jitter)
                    else:
                        data[blockParams]['grids'].append(grid)
                    iField += 1

            #
            for key in data[blockParams].keys():
                data[blockParams][key] = np.array(data[blockParams][key])

            #
            data[blockParams]['length'] = metadata['length']
            resolution, frequency = blockParams
            # self.save(f'stimuli/bn/{resolution}/{frequency}')

        #
        # struct = self.read('stimuli')
        # struct['bn'] = data
        # self.write(struct, 'stimuli')

        #
        for resolution, frequency in data.keys():
            for key in data[(resolution, frequency)]:
                value = data[(resolution, frequency)][key]
                if value is None:
                    continue
                if type(value) != np.ndarray:
                    value = np.array(value)
                self.save(f'stimuli/bn/{resolution}/{frequency}/{key}', value)

        return
    
    def _processFictiveSaccadesProtocol(self):
        """
        """

        #
        M = self.load('labjack/matrix')
        start, stop = self.load('epochs/fs')
        signal = M[start: stop, self.labjackChannelMapping['stimulus']]

        # Check for data loss
        if np.isnan(signal).sum() > 0:
            print(f'WARNING[{self.animal}, {self.date}]: Data loss detected during the fictive saccades stimulus')
            return

        #
        filtered = filterPulsesFromPhotologicDevice(signal, minimumPulseWidthInSeconds=0.03)

        #
        risingEdgeIndices = np.where(np.diff(filtered) > 0.5)[0]
        eventTimestamps = self.computeTimestamps(risingEdgeIndices + start)

        #
        probeEventMask = np.full(eventTimestamps.size, False).astype(bool)
        probeEventMask[1::3] = True
        probeTimestamps = eventTimestamps[probeEventMask]
        saccadeTimestamps = eventTimestamps[~probeEventMask]

        #
        trials = list()
        for saccadeTimestamp in saccadeTimestamps:
            probeTimestampsRelative = probeTimestamps - saccadeTimestamp
            closest = np.argsort(np.abs(probeTimestampsRelative))[0]
            dt = abs(probeTimestampsRelative[closest])
            if dt < 0.3:
                probeTimestamp = probeTimestamps[closest]
            else:
                probeTimestamp = np.nan
            entry = [saccadeTimestamp, probeTimestamp]
            trials.append(entry)


        #
        trials = np.array(trials)
        coincident = np.invert(np.isnan(trials[:, 1]))
        self.save('stimuli/fs/saccades/timestamps', trials[:, 0])
        self.save('stimuli/fs/probes/timestamps', trials[:, 1])
        self.save('stimuli/fs/coincident', coincident)

        return
    
    def _processMovingBarsProtocol(self, event='onset'):
        """
        """

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
        if event == 'onset':
            eventTimestamps = self.computeTimestamps(barOnsetIndices + start)
        elif event == 'offset':
            eventTimestamps = self.computeTimestamps(barOffsetIndices + start)
        elif event == 'centered':
            eventTimestamps = self.computeTimestamps(barCenteredIndices + start)
        else:
            raise Exception(f'{event} is not a valid event for the moving bar stimulus')
        self.save('stimuli/mb/timestamps', eventTimestamps)

        #
        result = list(self.folders.stimuli.rglob('*movingBarsMetadata.txt'))
        if len(result) != 1:
            raise Exception('Could not locate moving bars stimulus metadata')
        file = result.pop()
        with open(file, 'r') as stream:
            lines = stream.readlines()[5:]
        orientation = list()
        for line in lines:
            event, orientation_, timestamp = line.rstrip('\n').split(', ')
            if int(event) == 1:
                orientation.append(int(orientation_))
        self.save('stimuli/mb/orientation', np.array(orientation))

        return
    
    def _processDriftingGratingProtocol(self):
        """
        """

        #
        M = self.load('labjack/matrix')
        start, stop = self.load('epochs/dg')
        signal = M[start: stop, self.labjackChannelMapping['stimulus']]

        # Check for data loss
        # NOTE: This is the only case where the processing is robust to data loss
        if np.isnan(signal).sum() > 0:
            print(f'WARNING[{self.animal}, {self.date}]: Data loss detected during the drifting grating stimulus')

        #
        filtered = filterPulsesFromPhotologicDevice(signal, minimumPulseWidthInSeconds=0.03)

        #
        risingEdgeIndices = np.where(np.diff(filtered) > 0.5)[0]
        probeOnsetTimestamps = self.computeTimestamps(risingEdgeIndices + start)
        self.save('stimuli/dg/timestamps', probeOnsetTimestamps)

        #
        result = list(self.folders.stimuli.rglob('*driftingGratingMetadata.txt'))
        if len(result) != 1:
            raise Exception('Could not locate drifting grating stimulus metadata')
        file = result.pop()
        with open(file, 'r') as stream:
            lines = stream.readlines()[7:]
        motion = list()
        for line in lines:
            event, motion_, timestamp = line.rstrip('\n').split(', ')
            if int(event) == 3:
                motion.append(int(motion_))
        self.save('stimuli/dg/motion', np.array(motion))

        return

class MlatiSession(SessionBase, StimulusProcessingMixin):
    """
    """

    labjackChannelMapping = {
        'barcode': 5,
        'cameras': 6,
        'stimulus': 7,
    }

    def __init__(self, sessionFolder):
        """
        """

        super().__init__(sessionFolder)

        return
    
    @property
    def referenceSampleNumber(self):
        """
        """

        file = self.folders.ephys.joinpath('sync_messages.txt')
        if file.exists() == False:
            raise Exception('Could not locate the ephys sync messages file')
        
        #
        with open(file, 'r') as stream:
            ReferenceSampleNumber = None
            for line in stream.readlines():
                result = re.findall('@.*30000.*Hz:.*\d*', line)
                if len(result) == 1:
                    referenceSampleNumber = int(result.pop().rstrip('\n').split(': ')[-1])
                    break
        
        #
        if referenceSampleNumber is None:
            raise Exception('Failed to parse sync messages file for first sample number')

        return referenceSampleNumber
    
    @property
    def eventSampleNumbers(self):
        """
        """

        file = self.folders.ephys.joinpath('events', 'Neuropix-PXI-100.ProbeA-AP', 'TTL', 'sample_numbers.npy')
        if file.exists() == False: 
            raise Exception('Could not locate ephys event timestamps file')
        
        #
        eventSampleNumbers = np.load(file)

        return eventSampleNumbers
    
    @property
    def leftCameraMovie(self):
        """
        """

        movie = None
        result = list(self.folders.videos.glob('*leftCam*_reflected.mp4'))
        if len(result) == 1:
            movie = result.pop()

        return movie
    
    @property
    def rightCameraMovie(self):
        """
        """

        movie = None
        result = list(self.folders.videos.glob('*rightCam*.mp4'))
        if len(result) == 1:
            movie = result.pop()

        return movie
    
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
    