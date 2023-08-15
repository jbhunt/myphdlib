import re
import yaml
import pickle
import numpy as np
from myphdlib.interface.session import SessionBase
from myphdlib.extensions.matplotlib import placeVerticalLines
from myphdlib.general.labjack import loadLabjackData, filterPulsesFromPhotologicDevice

class StimulusProcessingMixin():
    """
    """

    def identifyProtocolEpochs(self, xData=None):
        """
        """

        #
        if xData is None:
            M = self.load('labjack/matrix')
            lightSensorSignal = M[:, self.labjackChannelMapping['stimulus']]
            xData = np.around(placeVerticalLines(lightSensorSignal), 0).astype(int)
        
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
        elif self.cohort in (2, 3):
            paths = (
                'epochs/sn/pre',
                'epochs/bn/hr/lf',
                'epochs/bn/hr/hf',
                'epochs/bn/lr/lf',
                'epochs/bn/lr/hf',
                'epochs/sn/post',
                'epochs/fs',
                'epochs/mb',
                'epochs/dg'
            )
        elif self.cohort == 4:
            paths = (
                'epochs/sn/pre',
                'epochs/fs',
                'epochs/mb',
                'epochs/dg'
            )
        elif self.cohort == 5:
            paths = (
                'epochs/sn/pre',
                'epochs/bn/hr/lf',
                'epochs/bn/hr/hf',
                'epochs/bn/lr/lf',
                'epochs/bn/lr/hf',
                'epochs/sn/post',
                'epochs/mb'
            )

        #
        indices = np.hstack([
            xData[0:-1].reshape(-1, 1),
            xData[1:  ].reshape(-1, 1)
        ])

        #
        nEpochs = len(paths)
        if nEpochs != indices.shape[0]:
            raise Exception('User input does not match expected number of epochs')

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
        if self.cohort in (1, 2, 3, 5):
            self._processBinaryNoiseProtocol()
        if self.cohort != 5:
            self._processFictiveSaccadesProtocol()
            self._processDriftingGratingProtocol()
        self._processMovingBarsProtocol()

        return
    
    def _interpolateMissingSparseNoiseTrials(
        self,
        filtered,
        start=0,
        nTrialsExpected=1020,
        ):
        """
        """

        #
        missing = list()

        # Compute the inter-pulse intervals
        risingEdgeIndices = np.where(np.diff(filtered) > 0.5)[0]
        risingEdgeIndices += start # NOTE: Need to add the start index so that the timestamp function works properly
        interPulseIntervals = np.diff(risingEdgeIndices) / self.labjackSamplingRate
        
        # In the case of exactly the number of expected trials
        if interPulseIntervals.size + 1 == nTrialsExpected:
            return True, np.full(nTrialsExpected, False), risingEdgeIndices

        # Figure out where the missing pulses happened
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

        print(f'INFO[{self.animal}, {self.date}]: Processing the sparse noise stimulus data')

        if self.cohort in (1, 2):
            blocks = ('pre',)
            nTrialsExpected=1020
        elif self.cohort == 2:
            blocks = ('pre', 'post')
            nTrialsExpected = 1020
        elif self.cohort == 3:
            blocks = ('pre', 'post')
            nTrialsExpected = 1700
        elif self.cohort == 4:
            blocks = ('pre',)
            nTrialsExpected = 1700
        elif self.cohort == 5:
            blocks = ('pre', 'post')
            nTrialsExpected = 1700

        #
        for iBlock, block in enumerate(blocks):
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
                nTrialsExpected=nTrialsExpected
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
            if self.cohort == 1:
                result = list(self.folders.stimuli.rglob(f'sparseNoiseMetadata.pkl'))
            elif self.cohort in (2, 3, 4, 5): 
                result = list(self.folders.stimuli.rglob(f'sparseNoiseMetadata-{iBlock + 1}.pkl'))
            if len(result) != 1:
                print(f'WARNING[{self.animal}, {self.date}]: Could not locate the sparse noise metadata file for block {iBlock + 1}')
                continue
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
    
    def _findBlockIndexSetsForBinaryNoiseStimulus(
        self,
        signalWholeStimulus,
        pulseEdges,
        nTrialsExpected,
        stepSize=0.001,
        pulseWidthMaximum=1,
        toleranceForPulseWidth=0.05,
        toleranceForMissingTrials=3,
        ):
        """
        """

        pulseWidths = np.diff(pulseEdges, axis=1).flatten() / self.labjackSamplingRate
        for threshold in np.arange(0, pulseWidthMaximum, stepSize):

            #
            pulsesInRange = np.logical_and(
                pulseWidths >= threshold,
                pulseWidths <= threshold + toleranceForPulseWidth
            )

            #
            try:
                fffPulseEdges = {
                    'white': {
                        'rising': pulseEdges[pulsesInRange, 0][0::2],
                        'falling': pulseEdges[pulsesInRange, 1][0::2]
                    },
                    'black': {
                        'rising': pulseEdges[pulsesInRange, 0][1::2],
                        'falling': pulseEdges[pulsesInRange, 1][1::2]
                    }
                }
                blockIndexSets = np.concatenate([
                    np.hstack([
                        fffPulseEdges['black']['falling'][:-1].reshape(-1, 1) - 1,
                        fffPulseEdges['white']['rising'][1:].reshape(-1, 1) + 1
                    ]),
                    np.array([[fffPulseEdges['black']['falling'][-1], signalWholeStimulus.size - 1]])
                ])
            except:
                result = False
                continue

            #
            observedTrialCounts = list()
            for startIndex, stopIndex in blockIndexSets:
                signalSingleBlock = signalWholeStimulus[startIndex: stopIndex]
                observedTrialCount = np.sum(np.diff(signalSingleBlock) > 0.5)
                observedTrialCounts.append(observedTrialCount)
            observedTrialCounts = np.array(observedTrialCounts)

            #
            checks = list()
            for iBlock, observedTrialCount in enumerate(observedTrialCounts):
                if nTrialsExpected - toleranceForMissingTrials <= observedTrialCount <= nTrialsExpected + toleranceForMissingTrials:
                    checks.append(True)
                else:
                    checks.append(False)
            result = np.all(np.array(checks))
            if result:
                break

        return result, threshold if result else None, blockIndexSets if result else None, fffPulseEdges if result else None
    
    def _interpolateMissingPulsesWithinBinaryNoiseBlock(
        self,
        peakIndices_,
        expectedTrialCount,
        data,
        blockParams,
        start,
        sequenceStartIndex
        ):
        """
        """

        # Try identifying and extrapolating the missing events
        expectedInterval = np.median(np.diff(peakIndices_))
        peakIndicesInterpolated = list()
        edgeIndex = 0
        for dt in np.diff(peakIndices_):
            nEdges = round(dt / expectedInterval)
            peakIndicesInterpolated.append(peakIndices_[edgeIndex])
            edgeIndex += 1
            for iEdge in range(nEdges - 1):
                peakIndicesInterpolated.append(np.nan)

        # Case where no pulses were detected in epoch
        if peakIndices_.size == 0:
            print(f'WARNING: Unexpected number of pulses detected for binary noise stimulus: 0')
            for iTrial in range(expectedTrialCount):
                data[blockParams]['missing'].append(True)
                data[blockParams]['timestamps'].append(np.nan)
                return data
        
        #
        peakIndicesInterpolated.append(peakIndices_[-1])
        peakIndicesInterpolated = np.array(peakIndicesInterpolated)
        xp = np.arange(peakIndicesInterpolated.size)
        xp = np.delete(xp, np.isnan(peakIndicesInterpolated))
        fp = np.delete(peakIndicesInterpolated, np.isnan(peakIndicesInterpolated))
        x = np.arange(peakIndicesInterpolated.size)
        x = np.delete(x, np.invert(np.isnan(peakIndicesInterpolated)))
        missingValues = np.around(np.interp(x, xp, fp), 0)
        peakIndicesInterpolated[np.isnan(peakIndicesInterpolated)] = missingValues

        #
        nPulses = peakIndicesInterpolated.size
        if nPulses != expectedTrialCount:
            print(f'WARNING: Unexpected number of pulses detected for binary noise stimulus: {nPulses}')
            for iTrial in range(expectedTrialCount):
                data[blockParams]['missing'].append(True)
                data[blockParams]['timestamps'].append(np.nan)
        else:
            timestamps = self.computeTimestamps(
                peakIndicesInterpolated + start + sequenceStartIndex
            )
            for timestamp in timestamps:
                data[blockParams]['timestamps'].append(timestamp)
            for iTrial in range(expectedTrialCount):
                data[blockParams]['missing'].append(False)

        return data
    
    def _processBinaryNoiseProtocol(
        self,
        pulseWidthMaximum=0.5
        ):
        """
        """

        print(f'INFO[{self.animal}, {self.date}]: Processing the binary noise stimulus data')

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
        M = self.load('labjack/matrix')

        #
        iterable = zip(
            ('epochs/bn/hr/lf', 'epochs/bn/hr/hf', 'epochs/bn/lr/lf','epochs/bn/lr/hf'),
            (('hr', 'lf'), ('hr', 'hf'), ('lr', 'lf'), ('lr', 'hf')),
            (20, 100, 20, 100),
            (True, False, True, False),
        )
        for blockIndex, (path, blockParams, expectedTrialCount, fieldOffsetSignaled) in enumerate(iterable):

            # Extract raw signal
            start, stop = self.load(path)
            signal = M[start: stop, self.labjackChannelMapping['stimulus']]

            # Check for data loss
            resolution, frequency = blockParams
            if np.isnan(signal).sum() > 0:
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

            #
            result, minimumPulseWidth, blockIndexSets, fffPulseEdges = self._findBlockIndexSetsForBinaryNoiseStimulus(
                filtered,
                pulseEdges,
                expectedTrialCount,
                stepSize=0.01,
                pulseWidthMaximum=pulseWidthMaximum,
            )

            #
            if result == False:
                print(f'WARNING[{self.animal}, {self.date}]: Failed to determine pulse width threshold for the binary noise stimulus (resolution={resolution}, frequency={frequency})')
                for iTrial in range(expectedTrialCount):
                    data[blockParams]['missing'].append(True)
                    data[blockParams]['timestamps'].append(np.nan)
                continue
            else:
                print(f'INFO[{self.animal}, {self.date}]: Pulse width threshold for binary noise stimulus determined: {minimumPulseWidth:.2f} seconds')

            # Identify missing trials/compute timestamps
            flashIndex = 0
            for blockStartIndex, blockStopIndex in blockIndexSets:

                # Save the timestamps for the full-field flash
                for color in ('white', 'black'):
                    timestamp = self.computeTimestamps(
                        np.array([fffPulseEdges[color]['rising'][flashIndex] + start])
                    ).item()
                    data[blockParams]['timestamps'].append(timestamp)
                    data[blockParams]['missing'].append(False)
                flashIndex += 1

                #
                sequence = filtered[blockStartIndex: blockStopIndex]
                peakIndices_ = np.where(np.diff(sequence) > 0.5)[0]

                #
                nPulses = peakIndices_.size
                if peakIndices_.size != expectedTrialCount:
                    data = self._interpolateMissingPulsesWithinBinaryNoiseBlock(
                        peakIndices_,
                        expectedTrialCount,
                        data,
                        blockParams,
                        start,
                        blockStartIndex
                    )
                else:
                    timestamps = self.computeTimestamps(
                        peakIndices_ + start + blockStartIndex
                    )
                    for timestamp in timestamps:
                        data[blockParams]['timestamps'].append(timestamp)
                    for iTrial in range(expectedTrialCount):
                        data[blockParams]['missing'].append(False)

            # Read metadata file
            blockNumber = blockIndex + 1
            if self.cohort == 1:
                file = self.folders.stimuli.joinpath('metadata', f'binaryNoiseMetadata{blockNumber}.pkl')
            elif self.cohort in (2, 3, 5):
                file = self.folders.stimuli.joinpath('metadata', f'binaryNoiseMetadata-{blockNumber}.pkl')
            if file.exists() == False:
                raise Exception(f'ERROR: Could not locate binary noise metadata: {file.name}')
            with open(str(file), 'rb') as stream:
                metadata = pickle.load(stream)

            #
            shape = metadata['shape']
            fields = metadata['values'].reshape(-1, *metadata['shape'])
            grid = metadata['coords'].reshape(*metadata['shape'], 2)
            jitter = metadata['length'] / 2 * np.array([1, -1])

            # Extract the metadata for each trial
            iField = 0
            nEvents = metadata['events'].shape[0]
            for iEvent in range(nEvents):
                event = metadata['events'][iEvent].item()

                #
                if event == 'flash onset':
                    field = np.full(shape, 1.0)
                    data[blockParams]['fields'].append(field)
                    data[blockParams]['grids'].append(grid)

                #
                elif event == 'flash offset' or event == 'field offset':
                    field = np.full(shape, -1.0)
                    data[blockParams]['fields'].append(field)
                    data[blockParams]['grids'].append(grid)

                #
                elif event == 'field onset':
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

        print(f'INFO[{self.animal}, {self.date}]: Processing the fictive saccades stimulus data')

        #
        M = self.load('labjack/matrix')
        start, stop = self.load('epochs/fs')
        signal = M[start: stop, self.labjackChannelMapping['stimulus']]

        # Check for data loss
        if np.isnan(signal).sum() > 0:
            print(f'WARNING[{self.animal}, {self.date}]: Data loss detected during the fictive saccades stimulus')
            return

        #
        filtered = filterPulsesFromPhotologicDevice(signal, minimumPulseWidthInSeconds=0.013)

        #
        risingEdgeIndices = np.where(np.diff(filtered) > 0.5)[0]
        eventTimestamps = self.computeTimestamps(risingEdgeIndices + start)

        #
        if self.cohort == 1:
            probeEventMask = np.full(eventTimestamps.size, False).astype(bool)
            probeEventMask[1::3] = True
            probeTimestamps = eventTimestamps[probeEventMask]
            saccadeTimestamps = eventTimestamps[~probeEventMask]

        #
        elif self.cohort in (2, 3, 4):
            result = list(self.folders.stimuli.joinpath('metadata').glob('*fictiveSaccadeMetadata*'))
            if len(result) != 1:
                print(f'WARNING[{self.animal}, {self.date}]: Could not locate the fictive saccades metadata')
                return
            with open(result.pop(), 'rb') as stream:
                metadata = pickle.load(stream)
            if risingEdgeIndices.size != metadata['events'].shape[0]:
                print(f'WARNING[{self.animal}, {self.date}]: Unexpected number of events detected during the fictive saccades stimulus')
                return
            probeEventMask = metadata['events'].flatten() == 'probe onset'
            saccadeEventMask = metadata['events'].flatten() == 'saccade onset'
            probeTimestamps = self.computeTimestamps(risingEdgeIndices[probeEventMask] + start)
            saccadeTimestamps = self.computeTimestamps(risingEdgeIndices[saccadeEventMask] + start)

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
        for probeTimestamp in probeTimestamps:
            saccadeTimestampsRelative = saccadeTimestamps - probeTimestamp
            closest = np.argsort(np.abs(saccadeTimestampsRelative))[0]
            dt = abs(saccadeTimestampsRelative[closest])
            if dt < 0.1:
                saccadeTimestamp = saccadeTimestamps[closest]
            else:
                saccadeTimestamp = np.nan
            entry = [saccadeTimestamp, probeTimestamp]
            trials.append(entry)

        #
        trials = np.unique(trials, axis=0)

        #
        coincident = np.invert(np.isnan(trials).any(axis=1))
        self.save('stimuli/fs/saccades/timestamps', trials[:, 0])
        self.save('stimuli/fs/probes/timestamps', trials[:, 1])
        self.save('stimuli/fs/coincident', coincident)

        return
    
    def _processMovingBarsProtocol(self, event='onset'):
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
                orientation.append(int(orientation_))
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
    
    def _interpolateMissingEventsFromDriftingGratingProtocol(
        self,
        trialParameters,
        risingEdgeIndices,
        ):
        """
        """

        #
        eventTimestampsExpected = np.array(trialParameters['timestamps']).astype(float)
        eventTimestampsExpected -= eventTimestampsExpected[0]
        eventTimestampsObserved = risingEdgeIndices / self.labjackSamplingRate
        eventTimestampsObserved -= eventTimestampsObserved[0]
        nEventsExpected = eventTimestampsExpected.size
        eventTimestampsCorrected = np.full(nEventsExpected, np.nan)
        eventTimestampsCorrected[:eventTimestampsObserved.size] = eventTimestampsObserved
        risingEdgeIndicesCorrected = np.copy(risingEdgeIndices).astype(float)

        #
        while True:

            # Computed the inter-pulse intervals
            notNanMask = np.invert(np.isnan(eventTimestampsCorrected))
            dtObs = np.diff(eventTimestampsCorrected[notNanMask]) # Observed inter-pulse intervals
            dtExp = np.diff(eventTimestampsExpected[notNanMask]) # Expected inter-pulse intervals

            # Look for interval mismatches
            difference = dtObs - dtExp
            indices = np.where(np.abs(difference) > 0.2)[0]
            if indices.size == 0:
                break

            #
            index = indices.min() + 1
            eventTimestampsCorrected = np.insert(eventTimestampsCorrected, index, np.nan)
            eventTimestampsCorrected = eventTimestampsCorrected[:nEventsExpected]
            risingEdgeIndicesCorrected = np.insert(risingEdgeIndicesCorrected, index, np.nan)

        #
        if risingEdgeIndicesCorrected.size == nEventsExpected:
            result = True
        else:
            result = False

        return result, risingEdgeIndicesCorrected
    
    def _processDriftingGratingProtocol(
        self,
        maximumPulseWidthForProbes=0.3,
        ):
        """
        """

        print(f'INFO[{self.animal}, {self.date}]: Processing the drifting grating stimulus data')

        #
        if self.hasGroup('stimuli/dg'):
            self.remove('stimuli/dg')

        # Read the metadata file
        result = list(self.folders.stimuli.rglob('*driftingGratingMetadata*'))
        if len(result) != 1:
            raise Exception('Could not locate drifting grating stimulus metadata')
        file = result.pop()
        if self.cohort != 4:
            startLineIndex = 6
        else:
            startLineIndex = 5
        with open(file, 'r') as stream:
            lines = stream.readlines()[startLineIndex:]
        trialParameters = {
            'events': list(),
            'motion': list(),
            'phase': list(),
            'contrast': list(),
            'timestamps': list()
        }
        for line in lines:
            if self.cohort in (1, 2, 3):
                contrast, phase = 1.0, np.nan
                event, motion, timestamp = line.rstrip('\n').split(', ')
            else:
                event, motion, contrast, phase, timestamp = line.rstrip('\n').split(', ')
            params = (event, motion, contrast, phase, timestamp)
            for key, value in zip(trialParameters.keys(), params):
                trialParameters[key].append(value)

        # Load the labjack data
        M = self.load('labjack/matrix')
        start, stop = self.load('epochs/dg')
        signal = M[start: stop, self.labjackChannelMapping['stimulus']]

        # Check for data loss
        dataLossDetected = False
        if np.isnan(signal).sum() > 0:
            print(f'WARNING[{self.animal}, {self.date}]: Data loss detected during the drifting grating stimulus')
            dataLossDetected = True

        # Parse protocol events
        filtered = filterPulsesFromPhotologicDevice(signal, minimumPulseWidthInSeconds=0.03)
        corrected = self._correctForCableDisconnectionDuringDriftingGrating(filtered)
        risingEdgeIndices = np.where(np.diff(corrected) > 0.5)[0]
        # return filtered, corrected, risingEdgeIndices

        # Cohorts 1, 2, and 3 ...
        if self.cohort in (1, 2, 3):
            nEventsExpected = np.array([1 for ev in trialParameters['events']
                if int(ev) == 3
            ]).sum()
            nEventsObserved = risingEdgeIndices.size
            if nEventsObserved != nEventsExpected: # TODO: Try to recover from missing events
                result, risingEdgeIndices = self._interpolateMissingEventsFromDriftingGratingProtocol(
                    trialParameters,
                    risingEdgeIndices
                )
                if result == False:
                    raise Exception()
            probeOnsetTimestamps = self.computeTimestamps(risingEdgeIndices + start)
            self.save(f'stimuli/dg/probe/timestamps', probeOnsetTimestamps)
            for eventName in ('grating', 'motion', 'iti'):
                self.save(f'stimuli/dg/{eventName}/timestamps', np.array([]).astype(float))

        # Cohorts 4 and greater ...
        elif self.cohort == 4:
            nEventsObserved = risingEdgeIndices.size
            nEventsExpected = len(trialParameters['timestamps'])
            if nEventsObserved != nEventsExpected: # TODO: Try to recover from missing events
                result, risingEdgeIndices = self._interpolateMissingEventsFromDriftingGratingProtocol(
                    trialParameters,
                    risingEdgeIndices
                )
                if result == False:
                    return corrected
                    raise Exception()
            for eventCode, eventName in zip(trialParameters['events'], ['grating', 'motion', 'probe', 'iti']):
                eventMask = np.array(trialParameters['events']).astype(int) == int(eventCode)
                eventTimestamps = self.computeTimestamps(
                    np.array(risingEdgeIndices[eventMask]) + start
                )
                self.save(f'stimuli/dg/{eventName}/timestamps', eventTimestamps)
            
        # Save the trial parameters
        dtypes = (
            int,
            float,
            float,
            float
        )
        probeMask = np.array(trialParameters['events']).astype(int) == 1
        for dtype, key in zip(dtypes, trialParameters.keys()):
            if key in ('events', 'timestamps'):
                continue
            else:
                value = np.array(trialParameters[key]).astype(dtype)[probeMask]
            self.save(f'stimuli/dg/probe/{key}', value)

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
    