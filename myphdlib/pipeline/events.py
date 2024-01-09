import os
import re
import numpy as np
import pathlib as pl

samplingRateNeuropixels = 30000

# TODO: Code this
def _getDataDimensions(dat):
    """
    Determine the number of samples and channels in a labjack data file
    """

    return 12000, 9

def _readDataFile(dat):
    """
    Read a single labjack dat file into a numpy array
    """

    #
    nSamples, nChannels = _getDataDimensions(dat)

    # read out the binary file
    with open(dat, 'rb') as stream:
        lines = stream.readlines()

    # Corrupted or empty dat files
    if len(lines) == 0:
        data = np.full((nSamples, nChannels), np.nan)
        return data

    #
    for iline, line in enumerate(lines):
        if bool(re.search('.*\t.*\t.*\t.*\t.*\t.*\t.*\r\n', line.decode())) and line.decode().startswith('Time') == False:
            break

    # split into header and content
    header  = lines[:iline ]
    content = lines[ iline:]

    # extract data and convert to float or int
    nrows = len(content)
    ncols = len(content[0].decode().rstrip('\r\n').split('\t'))
    shape = (nrows, ncols)
    data = np.zeros(shape)
    for iline, line in enumerate(content):
        elements = line.decode().rstrip('\r\n').split('\t')
        elements = [float(el) for el in elements]
        data[iline, :] = elements

    return np.array(data)

class EventsProcessingMixin(object):
    """
    """

    def _createLabjackDataMatrix(self, fileNumberRange=(None, None)):
        """
        Concatenate the dat files into a matrix of the shape N samples x N channels
        """

        self.log('Creating labjack data matrix')

        # determine the correct sequence of files
        files = [
            str(file)
                for file in self.folders.labjack.iterdir() if file.suffix == '.dat'
        ]
        file_numbers = [int(file.rstrip('.dat').split('_')[-1]) for file in files]
        sort_index = np.argsort(file_numbers)

        # create the matrix
        data = list()
        for ifile in sort_index:
            if fileNumberRange[0] is not None:
                if ifile < fileNumberRange[0]:
                    continue
            if fileNumberRange[1] is not None:
                if ifile > fileNumberRange[1]:
                    continue
            dat = self.folders.labjack.joinpath(files[ifile])
            mat = _readDataFile(dat)
            for irow in range(mat.shape[0]):
                data.append(mat[irow,:].tolist())

        #
        M = np.array(data)
        self.save('labjack/matrix', M)
        return

    def _extractLabjackTimespace(self):
        """
        """

        self.log('Extracting labjack timespace')

        labjackDataMatrix = self.load('labjack/matrix')
        labjackTimespace = labjackDataMatrix[:, 0]
        self.save('labjack/timespace', labjackTimespace)

        return

    def _extractBarcodeSignals(
        self,
        maximumWrapperPulseDuration=0.011,
        minimumBarcodeInterval=3,
        pad=100,
        ):
        """
        """

        self.log('Extracting barcode signals')
        
        #
        barcodes = {
            'labjack': {
                'pulses': None,
                'indices': None,
                'values': None,
            },
            'neuropixels': {
                'pulses': None,
                'indices': None,
                'values': None
            }
        }

        # Load labjack data matrix
        M = self.load('labjack/matrix')

        #
        for device in ('labjack', 'neuropixels'):

            # Identify pulse trains recorded by Neuropixels
            if device == 'neuropixels':
                stateTransitionIndices = self.eventSampleNumbers
                if stateTransitionIndices is None:
                    self.save(f'barcodes/neuropixels/trains', np.array([]).astype(float))
                    continue
                global samplingRateNeuropoixels
                samplingRate = samplingRateNeuropixels

            # Identify pulse trains recorded by labjack
            elif device == 'labjack':
                channelIndex = self.labjackChannelMapping['barcode']
                signal = M[:, channelIndex]
                stateTransitionIndices = np.where(
                    np.logical_or(
                        np.diff(signal) > +0.5,
                        np.diff(signal) < -0.5
                    )
                )[0]
                samplingRate = self.labjackSamplingRate

            # Parse individual barcode pulse trains
            longIntervalIndices = np.where(
                np.diff(stateTransitionIndices) >= minimumBarcodeInterval * samplingRate
            )[0]
            pulseTrains = np.split(stateTransitionIndices, longIntervalIndices + 1)

            # Filter out incomplete pulse trains
            pulseDurationThreshold = round(maximumWrapperPulseDuration * samplingRate)
            pulseTrainsFiltered = list()
            for pulseTrain in pulseTrains:

                # Need at least 1 pulse on each side for the wrapper
                if pulseTrain.size < 4:
                    continue

                # Wrapper pulses should be smaller than the encoding pulses
                firstPulseDuration = pulseTrain[1] - pulseTrain[0]
                finalPulseDuration = pulseTrain[-1] - pulseTrain[-2]
                if firstPulseDuration > pulseDurationThreshold:
                    continue
                elif finalPulseDuration > pulseDurationThreshold:
                    continue
                
                # Complete pulses
                pulseTrainsFiltered.append(pulseTrain)

            #
            padded = list()
            for pulseTrainFiltered in pulseTrainsFiltered:
                row = list()
                for index in pulseTrainFiltered:
                    row.append(index)
                nRight = pad - len(row)
                for value in np.full(nRight, np.nan):
                    row.append(value)
                padded.append(row)
            padded = np.array(padded)
            self.save(f'barcodes/{device}/trains', padded)

        return

    def _decodeBarcodeSignals(self, barcodeBitSize=0.03, wrapperBitSize=0.01):
        """
        """
        
        self.log('Decoding barcode signals')
        
        for device in ('labjack', 'neuropixels'):

            #
            if device == 'labjack':
                samplingRate = self.labjackSamplingRate
            elif device == 'neuropixels':
                global samplingRateNeuropixels
                samplingRate = samplingRateNeuropixels

            #
            pulseTrainsPadded = self.load(f'barcodes/{device}/trains') 
            if pulseTrainsPadded.size == 0:
                self.save(f'barcodes/{device}/indices', np.array([]).astype(float))
                self.save(f'barcodes/{device}/values', np.array([]).astype(float))
                continue
            pulseTrains = [
                row[np.invert(np.isnan(row))].astype(np.int32).tolist()
                    for row in pulseTrainsPadded
            ]
            barcodeValues, barcodeIndices = list(), list()

            #
            offset = 0
            for pulseTrain in pulseTrains:

                # 
                wrapperFallingEdge = pulseTrain[1]
                wrapperRisingEdge = pulseTrain[-2]
                barcodeLeftEdge = wrapperFallingEdge + round(wrapperBitSize * samplingRate)
                barcodeRightEdge = wrapperRisingEdge - round(wrapperBitSize * samplingRate)
                
                # Determine the state at the beginning and end of the data window
                firstStateTransition = pulseTrain[2]
                if (firstStateTransition - barcodeLeftEdge) / samplingRate < 0.001:
                    initialSignalState = currentSignalState = True
                else:
                    initialSignalState = currentSignalState = False
                finalStateTransition = pulseTrain[-3]
                if (barcodeRightEdge - finalStateTransition) / samplingRate < 0.001:
                    finalSignalState = True
                else:
                    finalSignalState = False

                # Determine what indices to use for computing time intervals between state transitions
                if initialSignalState == True and finalSignalState == True:
                    iterable = pulseTrain[2: -2]
                elif initialSignalState == True and finalSignalState == False:
                    iterable = np.concatenate([pulseTrain[2:-2], np.array([barcodeRightEdge])])
                elif initialSignalState == False and finalSignalState == False:
                    iterable = np.concatenate([np.array([barcodeLeftEdge]), pulseTrain[2:-2], np.array([barcodeRightEdge])])
                elif initialSignalState == False and finalSignalState == True:
                    iterable = np.concatenate([np.array([barcodeLeftEdge]), pulseTrain[2:-2]])
                
                # Determine how many bits are stored in each time interval and keep track of the signal state
                bitList = list()
                for nSamples in np.diff(iterable):
                    nBits = int(round(nSamples / (barcodeBitSize * samplingRate)))
                    for iBit in range(nBits):
                        bitList.append(1 if currentSignalState else 0)
                    currentSignalState = not currentSignalState

                # Decode the strings of bits
                bitString = ''.join(map(str, bitList[::-1]))
                if len(bitString) != 32:
                    raise Exception(f'More or less that 32 bits decoded')
                value = int(bitString, 2) + offset

                # 32-bit integer overflow
                if value == 2 ** 32 - 1:
                    offset = 2 ** 32

                #
                barcodeValues.append(value)
                barcodeIndices.append(pulseTrain[0])

            #
            self.save(f'barcodes/{device}/indices', np.array(barcodeIndices))
            self.save(f'barcodes/{device}/values', np.array(barcodeValues))

        return

    def _estimateTimestampingFunction(
        self
        ):
        """
        """

        self.log('Estimating timestamping function')

        # Load the barcode data
        barcodeValuesLabjack = self.load('barcodes/labjack/values')
        barcodeValuesNeuropixels = self.load('barcodes/neuropixels/values')
        barcodeIndicesLabjack = self.load('barcodes/labjack/indices')
        barcodeIndicesNeuropixels = self.load('barcodes/neuropixels/indices')

        # For sessions in which there is no associated ephys recording
        if barcodeValuesNeuropixels.size == 0 or barcodeValuesNeuropixels is None:

            timestampingFunctionParameters = {
                'm': np.nan,
                'b': np.nan,
                'xp': np.nan,
                'fp': np.nan
            }

        #
        else:

            # Find the shared barcode signals
            commonValues, barcodeFilterLabjack, barcodeFilterNeuropixels = np.intersect1d(
                barcodeValuesLabjack,
                barcodeValuesNeuropixels,
                return_indices=True
            )

            # Apply barcode filters and zero barcode values

            # Labjack
            barcodeValuesLabjack = barcodeValuesLabjack[barcodeFilterLabjack]
            barcodeValuesZeroedLabjack = barcodeValuesLabjack - barcodeValuesLabjack[0]
            barcodeIndicesLabjack = barcodeIndicesLabjack[barcodeFilterLabjack]

            # Neuropixels
            barcodeValuesNeuropixels = barcodeValuesNeuropixels[barcodeFilterNeuropixels]
            barcodeValuesZeroedNeuropixels = barcodeValuesNeuropixels - barcodeValuesNeuropixels[0]
            barcodeIndicesNeuropixels = barcodeIndicesNeuropixels[barcodeFilterNeuropixels]

            # NOTE: We have to subtract off the first sample of the recording
            barcodeIndicesNeuropixels -= self.referenceSampleNumber

            # Determine paramerts of linear equation for computing timestamps
            xlj = barcodeValuesZeroedLabjack
            xnp = barcodeValuesZeroedNeuropixels
            ylj = barcodeIndicesLabjack / self.labjackSamplingRate
            ynp = barcodeIndicesNeuropixels / samplingRateNeuropixels
            mlj = (ylj[-1] - ylj[0]) / (xlj[-1] - xlj[0])
            mnp = (ynp[-1] - ynp[0]) / (xnp[-1] - xnp[0])
            timestampingFunctionParameters = {
                'm': mlj + (mnp - mlj),
                'b': barcodeIndicesNeuropixels[0] / samplingRateNeuropixels,
                'xp': barcodeIndicesLabjack,
                'fp': barcodeValuesZeroedLabjack,
            }

        # Save the results
        for key, value in timestampingFunctionParameters.items():
            if type(value) != np.ndarray:
                value = np.array(value)
            self.save(f'tfp/{key}', value)

        return

    def _findDroppedFrames(self, pad=1000000):
        """
        """

        self.log('Finding dropped frames')

        #
        if self.primaryCamera == 'left':
            frameIntervals = np.loadtxt(self.leftCameraTimestamps, dtype=np.int64)
        elif self.primaryCamera == 'right':
            frameIntervals = np.loadtxt(self.rightCameraTimestamps, dtype=np.int64)
        else:
            raise Exception('Could not determine the primary camera')
        factor = np.median(frameIntervals)

        #
        for eye in ('left', 'right'):

            #
            file = self.leftCameraTimestamps if eye == 'left' else self.rightCameraTimestamps
            if file is None:
                self.log(f'Could not find the timestamps for the {eye} camera video', level='warning')
                continue
            observedFrameIntervals = np.loadtxt(file, dtype=np.int64) # in ms
            droppedFrames = np.full(observedFrameIntervals.size + 1 + pad, -1.0).astype(float)
            frameIntervals = np.full(observedFrameIntervals.size + 1 + pad, -1.0).astype(float)

            #
            frameIndex = 0
            nDropped = 0

            #
            for frameInterval in observedFrameIntervals:
                nFrames = int(round(frameInterval / factor, 0))
                if nFrames > 1:
                    nDropped += nFrames - 1
                    for iFrame in range(nFrames - 1):
                        droppedFrames[frameIndex] = 1
                        frameIntervals[frameIndex] = factor
                        frameIndex += 1
                droppedFrames[frameIndex] = 0
                frameIntervals[frameIndex] = frameInterval
                frameIndex += 1
            
            #
            excessFrameIndices = np.where(droppedFrames == -1)[0]
            droppedFrames = np.delete(droppedFrames, excessFrameIndices)
            droppedFrames = droppedFrames.astype(bool)
            self.save(f'frames/{eye}/dropped', droppedFrames)

            #
            frameIntervals = np.delete(frameIntervals, excessFrameIndices)
            self.save(f'frames/{eye}/intervals', frameIntervals)
            
        return

    def _timestampCameraTrigger(self, factor=1.3):
        """
        Compute the timestamps for all acquisition trigger events

        Notes
        -----
        This function will try to interpolate through periods in which
        the labjack device experienced data loss given the framerate of
        video acquisition
        """

        self.log('Timestamping camera trigger signal')

        # Load the raw signal
        M = self.load('labjack/matrix')
        signal = M[:, self.labjackChannelMapping['cameras']]

        # Find long intervals where data was dropped by the labjack device
        peaks = np.where(np.abs(np.diff(signal)) > 0.5)[0]
        intervals = np.diff(peaks) / self.labjackSamplingRate
        missing = list()

        #
        threshold = 1 / self.fps * factor
        for interval, flag in zip(intervals, intervals > threshold):
            missing.append(False)
            if flag:
                nFrames = int(round(interval / (1 / self.fps), 0))
                for iFrame in range(nFrames - 1):
                    missing.append(True)
        missing.append(False)
        missing = np.array(missing)

        #
        if missing.sum() > 0:
            nMissingEdges = missing.sum()
            self.log(f'{nMissingEdges} missing camera trigger events detected in labjack data', level='warning')

        # Interpolate across the missing pulses
        interpolated = np.full(missing.size, np.nan)
        x = np.arange(missing.size)[missing]
        xp = np.arange(missing.size)[np.invert(missing)]
        fp = peaks
        predicted = np.interp(x, xp, fp)
        interpolated[np.invert(missing)] = peaks
        interpolated[missing] = predicted

        # Compute the timestamps for all edges
        cameraTriggerTimestamps = self.computeTimestamps(interpolated)
        self.save('labjack/cameras/missing', missing)
        self.save('labjack/cameras/timestamps', cameraTriggerTimestamps)

        return

    def _timestampVideoFrames(
        self,
        ):
        """
        """

        self.log('Timestamping video frames')

        cameraTriggerTimestamps = self.load('labjack/cameras/timestamps')
        for camera in ('left', 'right'):
            dropped = self.load(f'frames/{camera}/dropped')
            nFrames = dropped.size
            videoFrameTimestamps = cameraTriggerTimestamps[:nFrames]
            self.save(f'frames/{camera}/timestamps', videoFrameTimestamps)

        return

    def _computeRelativeEventTiming(
        self,
        ):
        """
        Compute the timing of probes relative to saccades and vice-versa
        """

        self.log('Computing relative event timing')

        #
        if self.probeTimestamps is None:
            for k in ('dos', 'tts'):
                self.save(f'stimuli/dg/probe/{k}', np.array([]))
            for k in ('dop', 'ttp'):
                self.save(f'saccades/predicted/{self.eye}/{k}', np.array([]))
            self.log(f'No probe stimulus timestamps detected', level='warning')
            return

        #
        if np.isnan(self.saccadeTimestamps).all():
            for k in ('dos', 'tts'):
                self.save(f'stimuli/dg/probe/{k}', np.array([]))
            for k in ('dop', 'ttp'):
                self.save(f'saccades/predicted/{self.eye}/{k}', np.array([]))
            self.log(f'No saccade timestamps detected', level='warning')
            return

        #
        nProbes = self.probeTimestamps.size
        nSaccades = self.saccadeTimestamps.shape[0]
        data = {
            'tts': np.full(nProbes, np.nan).astype(np.float), # Time to saccade
            'dos': np.full(nProbes, np.nan).astype(np.float), # Direction of saccade
            'ttp': np.full(nSaccades, np.nan).astype(np.float), # Time to probe
            'dop': np.full(nSaccades, np.nan).astype(np.float), # direction of probe
        }

        #
        for trialIndex, probeTimestamp in enumerate(self.probeTimestamps):
            if np.isnan(probeTimestamp):
                continue
            saccadeTimestampsRelative = self.saccadeTimestamps[:, 0] - probeTimestamp
            saccadeIndex = np.nanargmin(np.abs(saccadeTimestampsRelative))
            saccadeLabel = self.saccadeLabels[saccadeIndex]
            probeLatency = round(probeTimestamp - self.saccadeTimestamps[:, 0][saccadeIndex], 3)
            data['dos'][trialIndex] = saccadeLabel
            data['tts'][trialIndex] = probeLatency

        #
        for trialIndex, saccadeTimestamp in enumerate(self.saccadeTimestamps[:, 0]):
            if np.isnan(saccadeTimestamp):
                saccadeLatency = np.nan
                probeDirection = np.nan
            else:
                probeTimestampsRelative = self.probeTimestamps - saccadeTimestamp
                probeIndex = np.nanargmin(np.abs(probeTimestampsRelative))
                probeDirection = self.gratingMotionDuringProbes[probeIndex]
                saccadeLatency = round(saccadeTimestamp - self.probeTimestamps[probeIndex], 3)
            data['ttp'][trialIndex] = saccadeLatency
            data['dop'][trialIndex] = probeDirection

        #
        for k in ('dos', 'tts'):
            self.save(f'stimuli/dg/probe/{k}', data[k])
        for k in ('dop', 'ttp'):
            self.save(f'saccades/predicted/{self.eye}/{k}', data[k])

        return

    def _timestampSaccades(
        self,
        saccadeEpochBoundaries=(-0.005, 0.005),
        ):
        """
        """

        self.log('Timestamping saccades')
        frameTimestamps = self.load('labjack/cameras/timestamps')

        for eye in ('left', 'right'):

            #
            saccadeEpochs = self.load(f'saccades/predicted/{eye}/epochs')
            saccadeIndices = self.load(f'saccades/predicted/{eye}/indices')
            nSaccades = saccadeEpochs.shape[0]
            droppedFrames = self.load(f'frames/{eye}/dropped')
            nFramesRecorded = droppedFrames.size

            #
            if nFramesRecorded > frameTimestamps.size:
                self.log(f'Saccade timestamping failed: {nFramesRecorded} frames in the {eye} camera movie but only {frameTimestamps.size} camera trigger pulses recorded', level='error')
                self.save(f'saccades/predicted/{eye}/timestamps', np.full([nSaccades, 2], np.nan))
                continue

            # Timestamp saccade peak velocity, onset, and offset
            peakVelocityTimestamps = np.interp(
                saccadeIndices,
                np.arange(nFramesRecorded),
                frameTimestamps[:nFramesRecorded]
            ).reshape(-1, 1)
            saccadeOnsetTimestamps = np.interp(
                saccadeEpochs[:, 0],
                np.arange(nFramesRecorded), 
                frameTimestamps[:nFramesRecorded]
            ).reshape(-1, 1)
            saccadeOffsetTimestamps = np.interp(
                saccadeEpochs[:, 1],
                np.arange(nFramesRecorded),
                frameTimestamps[:nFramesRecorded]
            ).reshape(-1, 1)
            saccadeEpochTimestamps = np.hstack([
                saccadeOnsetTimestamps,
                saccadeOffsetTimestamps
            ])

            # Mask saccade timestamps that violate the epoch boundaries
            iterable = zip(
                saccadeOnsetTimestamps,
                peakVelocityTimestamps,
                saccadeOffsetTimestamps,
            )
            for rowIndex, (t1, t2, t3) in enumerate(iterable):
                if np.around(t1 - t2, 3).item() > saccadeEpochBoundaries[0]:
                    saccadeEpochTimestamps[rowIndex, :] = np.array([np.nan, np.nan])
                if np.around(t3 - t2, 3).item() < saccadeEpochBoundaries[1]:
                    saccadeEpochTimestamps[rowIndex, :] = np.array([np.nan, np.nan])

            #            
            self.save(f'saccades/predicted/{eye}/timestamps', saccadeEpochTimestamps)

        return

    def _runEventsModule(self, redo=False):
        """
        """


        if self.hasDataset('labjack/matrix') == False or redo:
            self._createLabjackDataMatrix()
        if self.hasDataset('labjack/timespace') == False or redo:
            self._extractLabjackTimespace()
        if self.hasDataset('barcodes') == False or redo:
            self._extractBarcodeSignals()
            self._decodeBarcodeSignals()
        if self.hasDataset('tfp') == False or redo:
            self._estimateTimestampingFunction()
        self._findDroppedFrames()
        if self.hasDataset('labjack/cameras/timestamps') == False or redo:
            self._timestampCameraTrigger()
        self._timestampVideoFrames()
        self._timestampSaccades()
        self._computeRelativeEventTiming()

        return