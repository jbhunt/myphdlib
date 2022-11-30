import numpy as np

samplingRateLabjack = 1000
samplingRateNeuropoixels = 30000

def extractPulseTrains(
    variableInput,
    device='lj',
    maximumWrapperPulseDuration=0.011,
    minimumBarcodeInterval=3,
    ):
    """
    """

    #
    global samplingRateLabjack
    global samplingRateNeuropoixels

    # Identify pulse trains recorded by Neuropixels
    if device in ('np', 'neuropixels', 'Neuropixels', 'NeuroPixels'):
        stateTransitionIndices = np.load(variableInput)
        samplingRate = samplingRateNeuropoixels

    # Identify pulse trains recorded by labjack
    elif device in ('lj', 'labjack', 'Labjack', 'LabJack'):
        stateTransitionIndices = np.where(
            np.logical_or(
                np.diff(variableInput) > +0.5,
                np.diff(variableInput) < -0.5
            )
        )[0]
        samplingRate = samplingRateLabjack

    #
    else:
        raise Exception(f'Invalid device: {device}')

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

    return pulseTrainsFiltered

def decodePulseTrains(pulseTrains, device='lj', barcodeBitSize=0.03, wrapperBitSize=0.01):
    """
    """

    global samplingRateLabjack
    global samplingRateNeuropoixels

    if device in ('lj', 'labjack', 'Labjack', 'LabJack'):
        samplingRate = samplingRateLabjack
    elif device in ('np', 'neuropixels', 'Neuropixels', 'NeuroPixels'):
        samplingRate = samplingRateNeuropoixels
    else:
        raise Exception(f'Invalid device: {device}')

    values, indices = list(), list()

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
        value = int(bitString, 2)
        values.append(value)
        indices.append(pulseTrain[0])

    return np.array(values), np.array(indices)