import numpy as np

samplingRateLabjack = 1000
samplingRateNeuropoixels = 30000

def getPulseTrains(timestampsFile, minimumBarcodeInterval=3, minimumBarcodeDuration=0.95):
    """
    """

    # Identify all pulse trains
    global samplingRateNeuropoixels
    stateTransitionIndices = np.load(timestampsFile)
    longIntervalIndices = np.where(
        np.diff(stateTransitionIndices) >= minimumBarcodeInterval * samplingRateNeuropoixels
    )[0]
    pulseTrains = np.split(stateTransitionIndices, longIntervalIndices + 1)

    # Filter out incomplete pulse trains
    pulseTrainsFiltered = list()
    for pulseTrain in pulseTrains:
        pulseTrainDuation = (pulseTrain[-1] - pulseTrain[0]) / samplingRateNeuropoixels
        if pulseTrainDuation >= minimumBarcodeDuration:
            pulseTrainsFiltered.append(pulseTrain)

    return pulseTrainsFiltered

def decodePulseTrains(pulseTrains, barcodeBitSize=0.03, wrapperBitSize=0.01):
    """
    """

    global samplingRateNeuropoixels

    values = list()

    for pulseTrain in pulseTrains:

        # 
        wrapperFallingEdge = pulseTrain[1]
        wrapperRisingEdge = pulseTrain[-2]
        barcodeLeftEdge = wrapperFallingEdge + round(wrapperBitSize * samplingRateNeuropoixels)
        barcodeRightEdge = wrapperRisingEdge - round(wrapperBitSize * samplingRateNeuropoixels)
        
        # Determine the state at the beginning and end of the data window
        firstStateTransition = pulseTrain[2]
        if (firstStateTransition - barcodeLeftEdge) / samplingRateNeuropoixels < 0.001:
            initialSignalState = currentSignalState = True
        else:
            initialSignalState = currentSignalState = False
        finalStateTransition = pulseTrain[-3]
        if (barcodeRightEdge - finalStateTransition) / samplingRateNeuropoixels < 0.001:
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
            nBits = round(nSamples / (barcodeBitSize * samplingRateNeuropoixels))
            for iBit in range(nBits):
                bitList.append(1 if currentSignalState else 0)
            currentSignalState = not currentSignalState

        # Decode the strings of bits
        bitString = ''.join(map(str, bitList[::-1]))
        if len(bitString) != 32:
            raise Exception(f'More or less that 32 bits decoded')
        value = int(bitString, 2)
        values.append(value)

    return np.array(values)