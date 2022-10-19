from pickle import NONE
from re import L
from myphdlib.toolkit.custom import DotDict
from myphdlib.experiments.suppression2.constants import labjackChannelMapping
from myphdlib.experiments.suppression2 import constants
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
    TODO: Move the barcode filtering into this function
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

def fitTimestampGenerator(sessionObject):
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
    timestampGeneratorParameters = {
        'm': mlj + (mnp - mlj),
        'b': barcodeIndicesNeuropixels[0] / 30000,
        'xp': barcodeIndicesLabjack,
        'fp': barcodeValuesZeroedLabjack,
    }
    saveSessionData(sessionObject, 'timestampGeneratorParameters', timestampGeneratorParameters)
    saveSessionData(sessionObject, 'filteredBarcodeData', filteredBarcodeData)

    return

def _extractSparseNoiseTimestamps(sessionObject, stateTransitionTimestamps, dataContainer):
    """
    Extract timestamps and metadata for the sparse noise stimulus
    """

    # TODO: Extract data for both presentations of the sparse noise stimulus

    # Determine which edges/timestamps to work with
    start = 0
    stop = start + constants.stateTransitionCounts[0]

    # Read metadata file
    with open(sessionObject.sparseNoiseMetadataFilePath, 'r') as stream:
        lines = [
            line for line in stream.readlines()
                if bool(re.search('.*, .*, .*, .*\n', line)) and line.startswith('Columns') == False
        ][:constants.trialCountSparseNoise]

    # Make sure the metadata file has the exact number of trials expected
    if len(lines) != constants.trialCountSparseNoise:
        import pdb; pdb.set_trace()
        raise Exception(f'Sparse noise stimulus metadata file contains too many or too few lines')

    #
    timestampIndices = np.arange(0, int(constants.trialCountSparseNoise * 2), 2)
    for trialIndex, (lineData, timestampIndex) in enumerate(zip(lines, timestampIndices)):
        x, y, t1, t2 = lineData.split(', ')
        t = stateTransitionTimestamps[timestampIndex]
        dataContainer['sparseNoise']['i'][trialIndex] = trialIndex
        dataContainer['sparseNoise']['x'][trialIndex] = x
        dataContainer['sparseNoise']['y'][trialIndex] = y
        dataContainer['sparseNoise']['t'][trialIndex] = t

    return dataContainer

def _extractMovingBarsTimestamps(sessionObject, stateTransitionTimestamps, dataContainer):
    """
    Extract timestamps and metadata for the moving bar stimulus
    """

    # TODO: Extract data for both presentations of the sparse noise stimulus

    # Determine which edges/timestamps to work with
    start = constants.stateTransitionCounts[0]
    stop = constants.stateTransitionCounts[0] + constants.stateTransitionCounts[1]

    #
    with open(sessionObject.movingBarsMetadataFilePath, 'r') as stream:
        lines = [
            line for line in stream.readlines()
                if bool(re.search('.*, .*, .*\n', line)) and line.startswith('Columns') == False
        ]
    for trialIndex, line in enumerate(lines[::2][:constants.trialCountMovingBars]):
        eventCode, orientation, approxTimestamp = line.rstrip('\n').split(', ')
        dataContainer['movingBars']['o'][trialIndex] = int(orientation)

    #
    for trialIndex, stateTransitionTimestamp in enumerate(stateTransitionTimestamps[start:stop][0::4]):
        dataContainer['movingBars']['i'][trialIndex] = trialIndex
        dataContainer['movingBars']['t1'][trialIndex] = stateTransitionTimestamp

    #
    for trialIndex, stateTransitionTimestamp in enumerate(stateTransitionTimestamps[start:stop][2::4]):
        dataContainer['movingBars']['i'][trialIndex] = trialIndex
        dataContainer['movingBars']['t2'][trialIndex] = stateTransitionTimestamp

    return dataContainer

def _extractDriftingGratingTimestamps(sessionObject, stateTransitionTimestamps, dataContainer):
    """
    """

    # TODO: Figure out how to handle Nan values associated with missing TTL pulses

    #
    start = constants.stateTransitionCounts[0] + constants.stateTransitionCounts[1]
    nEdgesTotal = np.sum([value for value in constants.stateTransitionCounts.values() if value is not None])
    nEdgesResidual = stateTransitionTimestamps.size - nEdgesTotal
    stop = start + nEdgesResidual
    nTrials = int(nEdgesResidual / 2 - constants.trialCountDriftingGrating)

    #
    dataContainer['driftingGrating'] = {
        'i': np.empty(nTrials),
        'd': np.empty(nTrials),
        't1': np.empty(nTrials),
        't2': np.empty(nTrials)
    }

    return

def _extractNoisyGratingTimestamps():
    """
    """

    return

def extractStimuliTimestamps(sessionObject):
    """
    """

    dataContainer = {
        'sparseNoise': {
            'i': np.empty(constants.trialCountSparseNoise),
            'x': np.empty(constants.trialCountSparseNoise),
            'y': np.empty(constants.trialCountSparseNoise),
            's': np.tile([1, 0], int(constants.trialCountSparseNoise / 2)),
            't': np.empty(constants.trialCountSparseNoise)
        },
        'movingBars': {
            'i': np.empty(constants.trialCountMovingBars),
            'o': np.empty(constants.trialCountMovingBars),
            't1': np.empty(constants.trialCountMovingBars),
            't2': np.empty(constants.trialCountMovingBars),
        },
        'driftingGrating': None, # This is dynamically populated
        'noisyGrating': None,
    }

    # Compute the timestamp for each state transition
    labajackDataMatrix = sessionObject.load('labjackDataMatrix')
    visualStimuliSignal = labajackDataMatrix[:, labjackChannelMapping.stimulus]
    stateTransitionIndices = np.where(np.logical_or(
        np.diff(visualStimuliSignal) > +0.5,
        np.diff(visualStimuliSignal) < -0.5
    ))[0]
    params = DotDict(sessionObject.load('timestampGeneratorParameters'))
    stateTransitionTimestamps = np.around(
        np.interp(stateTransitionIndices, params.xp, params.fp) * params.m + params.b,
        3
    )

    # Correct for missing TTL pulses
    missingStateTransitionIndices = np.loadtxt(
        sessionObject.missingFilePath,
        delimiter=', ',
        dtype=np.int
    )
    fillValues = np.full(missingStateTransitionIndices.size, np.nan)
    stateTransitionIndices = np.insert(stateTransitionIndices, missingStateTransitionIndices, fillValues)
    stateTransitionTimestamps = np.insert(stateTransitionTimestamps, missingStateTransitionIndices, fillValues)

    # First bout of the sparse noise stimulus
    dataContainer = _extractSparseNoiseTimestamps(sessionObject, stateTransitionTimestamps, dataContainer)

    # First bout of the moving bars stimulus
    dataContainer = _extractMovingBarsTimestamps(sessionObject, stateTransitionTimestamps, dataContainer)

    # Drifting grating stimulus (with probes)
    dataContainer = _extractDriftingGratingTimestamps(sessionObject, stateTransitionTimestamps, dataContainer)

    # Second bout of the sparse noise stimulus

    # Second bout of the moving bars stimulus

    # Noisy drifting grating stimulus


    saveSessionData(sessionObject, 'visualStimuliData', dataContainer)

    return