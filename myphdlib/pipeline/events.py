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

def createLabjackDataMatrix(session, fileNumberRange=(None, None)):
    """
    Concatenate the dat files into a matrix of the shape N samples x N channels
    """

    # determine the correct sequence of files
    files = [
        str(file)
            for file in session.folders.labjack.iterdir() if file.suffix == '.dat'
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
        dat = session.folders.labjack.joinpath(files[ifile])
        mat = _readDataFile(dat)
        for irow in range(mat.shape[0]):
            data.append(mat[irow,:].tolist())

    #
    M = np.array(data)
    # session.write(M, 'labjackDataMatrix')
    session.save('labjack/matrix', M)
    return

def extractBarcodeSignals(
    session,
    maximumWrapperPulseDuration=0.011,
    minimumBarcodeInterval=3,
    pad=100,
    ):
    """
    """
    
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

    M = session.load('labjack/matrix')

    #
    for device in ('labjack', 'neuropixels'):

        # Identify pulse trains recorded by Neuropixels
        if device == 'neuropixels':
            stateTransitionIndices = session.eventSampleNumbers
            global samplingRateNeuropoixels
            samplingRate = samplingRateNeuropixels

        # Identify pulse trains recorded by labjack
        elif device == 'labjack':
            channelIndex = session.labjackChannelMapping['barcode']
            signal = M[:, channelIndex]
            stateTransitionIndices = np.where(
                np.logical_or(
                    np.diff(signal) > +0.5,
                    np.diff(signal) < -0.5
                )
            )[0]
            samplingRate = session.labjackSamplingRate

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
        # barcodes[device]['pulses'] = pulseTrainsFiltered

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
        session.save(f'barcodes/{device}/trains', padded)

    #
    # session.write(barcodes, 'barcodes')

    return

def decodeBarcodeSignals(session, barcodeBitSize=0.03, wrapperBitSize=0.01):
    """
    """

    #if 'barcodes' not in session.keys():
    #     raise Exception('Barcode pulse trains have not been extracted')
    # else:
    #     barcodes = session.read('barcodes')

    
    for device in ('labjack', 'neuropixels'):

        #
        if device == 'labjack':
            samplingRate = session.labjackSamplingRate
        elif device == 'neuropixels':
            global samplingRateNeuropixels
            samplingRate = samplingRateNeuropixels

        #
        # pulseTrains = barcodes[device]['pulses']
        pulseTrainsPadded = session.load(f'barcodes/{device}/trains') 
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
        # barcodes[device]['indices'] = np.array(barcodeIndices)
        # barcodes[device]['values'] = np.array(barcodeValues)
        session.save(f'barcodes/{device}/indices', np.array(barcodeIndices))
        session.save(f'barcodes/{device}/values', np.array(barcodeValues))

    #
    # session.write(barcodes, 'barcodes')
    return

def estimateTimestampingFunction(
    session
    ):
    """
    """

    # Load the barcode data
    barcodeValuesLabjack = session.load('barcodes/labjack/values')
    barcodeValuesNeuropixels = session.load('barcodes/neuropixels/values')
    barcodeIndicesLabjack = session.load('barcodes/labjack/indices')
    barcodeIndicesNeuropixels = session.load('barcodes/neuropixels/indices')

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
    barcodeIndicesNeuropixels -= session.referenceSampleNumber

    # Determine paramerts of linear equation for computing timestamps
    xlj = barcodeValuesZeroedLabjack
    xnp = barcodeValuesZeroedNeuropixels
    ylj = barcodeIndicesLabjack / session.labjackSamplingRate
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
        session.save(f'tfp/{key}', value)

    return

def findDroppedFrames(session, pad=1000000):
    """
    """

    #
    for eye in ('left', 'right'):

        #
        file = session.leftCameraTimestamps if eye == 'left' else session.rightCameraTimestamps
        if file is None:
            print(f'WARNING: Could not find the timestamps for the {eye} camera video')
            continue
        intervals = np.loadtxt(file, dtype=np.int64) / 1000000 # in ms
        expected = 1 / session.fps * 1000 # in ms
        dropped = np.full(intervals.size + 1 + pad, -1.0).astype(float)

        #
        index = 0
        nDropped = 0

        #
        for interval in intervals:
            nFrames = int(round(interval / expected, 0))
            if nFrames > 1:
                nDropped += nFrames - 1
                for iFrame in range(nFrames - 1):
                    dropped[index] = 1
                    index += 1
            dropped[index] = 0
            index += 1
        
        #
        dropped = np.delete(dropped, np.where(dropped == -1)[0])
        dropped = dropped.astype(bool)
        session.save(f'frames/{eye}/dropped', dropped)
        
    return

def timestampCameraTrigger(session, factor=1.3):
    """
    Compute the timestamps for all acquisition trigger events

    notes
    -----
    This function will try to interpolate through periods in which
    the labjack device experienced data loss given the framerate of
    video acquisition
    """

    # Load the raw signal
    M = session.load('labjack/matrix')
    signal = M[:, session.labjackChannelMapping['cameras']]

    # Find long intervals where data was dropped by the labjack device
    peaks = np.where(np.abs(np.diff(signal)) > 0.5)[0]
    intervals = np.diff(peaks) / session.labjackSamplingRate
    missing = list()

    #
    threshold = 1 / session.fps * factor
    for interval, flag in zip(intervals, intervals > threshold):
        missing.append(False)
        if flag:
            nFrames = int(round(interval / (1 / session.fps), 0))
            for iFrame in range(nFrames - 1):
                missing.append(True)
    missing.append(False)
    missing = np.array(missing)

    #
    if missing.sum() > 0:
        nMissingEdges = missing.sum()
        print(f'WARNING[{session.animal}, {session.date}]: {nMissingEdges} missing camera trigger events detected in labjack data')

    # Interpolate across the missing pulses
    interpolated = np.full(missing.size, np.nan)
    x = np.arange(missing.size)[missing]
    xp = np.arange(missing.size)[np.invert(missing)]
    fp = peaks
    predicted = np.interp(x, xp, fp)
    interpolated[np.invert(missing)] = peaks
    interpolated[missing] = predicted

    # Compute the timestamps for all edges
    timestamps = session.computeTimestamps(interpolated)
    session.save('labjack/cameras/missing', missing)
    session.save('labjack/cameras/timestamps', timestamps)

    return

def computeRelativeEventTiming(
    session,
    ):
    """
    Compute the timing of probes relative to saccades and vice-versa
    """

    #
    if session.probeTimestamps is None:
        for k in ('dos', 'tts'):
            session.save(f'stimuli/dg/probe/{k}', np.array([]))
        for k in ('dop', 'ttp'):
            session.save(f'saccades/predicted/{session.eye}/unsigned/{k}', np.array([]))
        session.log(f'No probe stimulus timestamps detected', level='warning')
        return

    #
    nProbes = session.probeTimestamps.size
    nSaccades = session.saccadeTimestamps.size
    data = {
        'tts': np.full(nProbes, np.nan).astype(np.float), # Time to saccade
        'dos': np.full(nProbes, np.nan).astype(np.int), # Direction of saccade
        'ttp': np.full(nSaccades, np.nan).astype(np.float), # Time to probe
        'dop': np.full(nSaccades, np.nan).astype(np.int), # direction of probe
    }

    #
    for trialIndex, probeTimestamp in enumerate(session.probeTimestamps):
        if np.isnan(probeTimestamp):
            continue
        saccadeTimestampsRelative = session.saccadeTimestamps - probeTimestamp
        closestSaccadeIndex = np.argmin(np.abs(saccadeTimestampsRelative))
        closestSaccadeDirection = session.saccadeDirections[closestSaccadeIndex]
        probeLatency = round(probeTimestamp - session.saccadeTimestamps[closestSaccadeIndex], 3)
        data['dos'][trialIndex] = -1 if closestSaccadeDirection == 't' else +1
        data['tts'][trialIndex] = probeLatency

    #
    for trialIndex, saccadeTimestamp in enumerate(session.saccadeTimestamps):
        probeTimestampsRelative = session.probeTimestamps - saccadeTimestamp
        closestProbeIndex = np.nanargmin(np.abs(probeTimestampsRelative))
        closestProbeDirection = session.gratingMotionDuringProbes[closestProbeIndex]
        saccadeLatency = round(saccadeTimestamp - session.probeTimestamps[closestProbeIndex], 3)
        data['ttp'][trialIndex] = saccadeLatency
        data['dop'][trialIndex] = closestProbeDirection

    #
    for k in ('dos', 'tts'):
        session.save(f'stimuli/dg/probe/{k}', data[k])
    for k in ('dop', 'ttp'):
        session.save(f'saccades/predicted/{session.eye}/unsigned/{k}', data[k])

    return