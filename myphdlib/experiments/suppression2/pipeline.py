from pickle import NONE
from re import L
from myphdlib.toolkit.custom import DotDict
from myphdlib.experiments.suppression2.constants import labjackChannelMapping
from myphdlib.experiments.suppression2.factory import loadSessionData, saveSessionData
from myphdlib.toolkit.labjack import loadLabjackData, extractLabjackEvent
from myphdlib.toolkit.sync import extractPulseTrains, decodePulseTrains

import re
import numpy as np

def extractLabjackData(sessionObject):
    """
    Extract the labjack data matrix
    TODO: Save the inidvi
    """

    labjackDataMatrix = loadLabjackData(sessionObject.labjackFolderPath)
    saveSessionData(sessionObject, 'labjackDataMatrix', labjackDataMatrix)

    return

def extractBarcodeSignals(sessionObject):
    """
    Extract barcode values and onset timestamps for the labjack and neuropixels devices
    """

    # Init data container
    rawBarcodeData = {
        'labjack': {
            'indices': None,
            'values': None
        },
        'neuropixels': {
            'indices': None,
            'values': None
        }
    }

    # Extract labjack barcodes
    labjackDataMatrix = loadSessionData(sessionObject, 'labjackDataMatrix')
    barcodeDigitalSignal = labjackDataMatrix[:, labjackChannelMapping['barcode']]
    barcodePulseTrains = extractPulseTrains(barcodeDigitalSignal, device='lj')
    barcodeValuesLabjack, barcodeIndicesLabjack = decodePulseTrains(barcodePulseTrains, device='lj')

    # Extract neuropixels barcodes
    barcodePulseTrains = extractPulseTrains(str(sessionObject.timestampsFilePath), device='np')
    barcodeValuesNeuropixels, barcodeIndicesNeuropixels = decodePulseTrains(barcodePulseTrains, device='np')

    # Save the raw barcode data
    rawBarcodeData['labjack']['indices'] = barcodeIndicesLabjack
    rawBarcodeData['labjack']['values'] = barcodeValuesLabjack
    rawBarcodeData['neuropixels']['indices'] = barcodeIndicesNeuropixels
    rawBarcodeData['neuropixels']['values'] = barcodeValuesNeuropixels
    saveSessionData(sessionObject, 'rawBarcodeData', rawBarcodeData)

    return

def determineTimestampsComputationParameters(sessionObject):
    """
    """

    # Load the barcode data
    rawBarcodeData = sessionObject.load('rawBarcodeData')
    barcodeValuesLabjack = rawBarcodeData['labjack']['values']
    barcodeValuesNeuropixels = rawBarcodeData['neuropixels']['values']
    barcodeIndicesLabjack = rawBarcodeData['labjack']['indices']
    barcodeIndicesNeuropixels = rawBarcodeData['neuropixels']['indices']
    filteredBarcodeData = {
        'labjack': {'indices': None, 'values': None},
        'neuropixels': {'indics': None, 'values': None}
    }

    # Find the shared barcode signals
    lowerBound = np.max([barcodeValuesLabjack.min(), barcodeValuesNeuropixels.min()])
    upperBound = np.min([barcodeValuesLabjack.max(), barcodeValuesNeuropixels.max()])
    barcodeFilterLabjack = np.logical_and(
        barcodeValuesLabjack >= lowerBound,
        barcodeValuesLabjack <= upperBound
        )
    barcodeFilterNeuropixels = np.logical_and(
        barcodeValuesNeuropixels >= lowerBound,
        barcodeValuesNeuropixels <= upperBound
    )

    # Apply barcode filters and zero barcode values
    barcodeValuesLabjack = barcodeValuesLabjack[barcodeFilterLabjack]
    barcodeValuesZeroedLabjack = barcodeValuesLabjack - barcodeValuesLabjack[0]
    barcodeIndicesLabjack = barcodeIndicesLabjack[barcodeFilterLabjack]
    filteredBarcodeData['labjack']['values'] = barcodeValuesLabjack
    filteredBarcodeData['labjack']['indices'] = barcodeIndicesLabjack

    # 
    barcodeValuesNeuropixels = barcodeValuesNeuropixels[barcodeFilterNeuropixels]
    barcodeValuesZeroedNeuropixels = barcodeValuesNeuropixels - barcodeValuesNeuropixels[0]
    barcodeIndicesNeuropixels = barcodeIndicesNeuropixels[barcodeFilterNeuropixels]
    filteredBarcodeData['neuropixels']['values'] = barcodeValuesNeuropixels
    filteredBarcodeData['neuropixels']['indices'] = barcodeIndicesNeuropixels

    # NOTE: We have to subtract off the first sample of the recording
    # so that these timestamps are in register with the spike timestamps
    barcodeIndicesNeuropixels -= sessionObject.ephysFirstSample

    # Determine paramerts of linear equation for computing timestamps
    xlj = barcodeValuesZeroedLabjack
    xnp = barcodeValuesZeroedNeuropixels
    ylj = barcodeIndicesLabjack / 1000
    ynp = (barcodeIndicesNeuropixels) / 30000 # - barcodeIndicesNeuropixels[0]) / 30000
    mlj = (ylj[-1] - ylj[0]) / (xlj[-1] - xlj[0])
    mnp = (ynp[-1] - ynp[0]) / (xnp[-1] - xnp[0])
    synchronizationFunctionParameters = {
        'm': mlj + (mnp - mlj),
        'b': barcodeIndicesNeuropixels[0] / 30000,
        'xp': barcodeIndicesLabjack,
        'fp': barcodeValuesZeroedLabjack,
    }
    saveSessionData(sessionObject, 'synchronizationFunctionParameters', synchronizationFunctionParameters)
    saveSessionData(sessionObject, 'filteredBarcodeData', filteredBarcodeData)

    return

def extractEventTimestamps(sessionObject):
    """
    """

    visualStimuliData = {
        'sparseNoise': {
            'i': list(),
            'x': list(),
            'y': list(),
            't': list()
        },
        'movingBars': None,
        'driftingGrating': None,
        'noisyGrating': None,
    }

    #
    labajackDataMatrix = sessionObject.load('labjackDataMatrix')
    visualStimuliSignal = labajackDataMatrix[:, labjackChannelMapping.stimulus]
    stateTransitionIndices = np.where(np.logical_or(
        np.diff(visualStimuliSignal) > +0.5,
        np.diff(visualStimuliSignal) < -0.5
    ))[0]
    import pdb; pdb.set_trace()

    #
    params = DotDict(sessionObject.load('synchronizationFunctionParameters'))
    stateTransitionTimestamps = np.interp(stateTransitionIndices, params.xp, params.fp) * params.m + params.b

    #
    trialData = list()
    trialIndex = 0

    # First bout of the sparse noise stimulus
    with open(sessionObject.sparseNoiseMetadataFilePath, 'r') as stream:
        lines = stream.readlines()
        for lineIndex, line in enumerate(line):
            if bool(re.search('-*\n', line)):
                lines = lines[lineIndex + 1:]
                break
        if len(lines) % 2 != 0:
            raise Exception('Trial count for sparse noise stimulus is not evenly divisible by 2')
        for line in line[:(len(lines) / 2)]:
            x, y, t1, t2 = line.split(', ')
            t = stateTransitionTimestamps[trialIndex]
            visualStimuliData['sparseNoise']['i'].append(trialIndex)
            visualStimuliData['sparseNoise']['x'].append(x)
            visualStimuliData['sparseNoise']['y'].append(y)
            visualStimuliData['sparseNoise']['t'].append(t)
            trialIndex += 1

    # First bout of the moving bars stimulus

    # Drifting grating stimulus (with probes)

    # Second bout of the sparse noise stimulus

    # Second bout of the moving bars stimulus

    # Noisy drifting grating stimulus

    saveSessionData(sessionObject, 'visualStimuliData', visualStimuliData)

    return