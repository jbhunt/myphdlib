from requests import session
from myphdlib.toolkit.custom import DotDict
from myphdlib.experiments.suppression2.constants import labjackChannelMapping
from myphdlib.experiments.suppression2 import constants
from myphdlib.experiments.suppression2.factory import loadSessionData, saveSessionData
from myphdlib.toolkit.labjack import loadLabjackData, extractLabjackEvent
from myphdlib.toolkit.sync import extractPulseTrains, decodePulseTrains

import re
import yaml
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

def extractStimulusDataSN(sessionObject, dataContainer):
    """
    """

    # Compute the timestamp for each state transition
    labjackDataMatrix = sessionObject.load('labjackDataMatrix')
    params = DotDict(sessionObject.load('timestampGeneratorParameters'))

    #
    with open(sessionObject.inputFilePath, 'r') as stream:
        curatedStimulusMetadata = yaml.full_load(stream)['curatedStimulusMetadata']

    #
    with open(sessionObject.sparseNoiseMetadataFilePath, 'r') as stream:
        lines = [
            line for line in stream.readlines()
                if bool(re.search('.*, .*, .*, .*\n', line)) and line.startswith('Columns') == False
        ]
    nEdgesPerBlock = int(curatedStimulusMetadata['sn']['b1']['nEdgesTotal'])
    nEdgesPerTrial = 4
    nTrialsPerBlock = int(nEdgesPerBlock / nEdgesPerTrial)
    nTrialsTotal = int(nTrialsPerBlock * 2)
    if len(lines) != nTrialsTotal:
        raise Exception('') # TODO: Describe this exception
    else:
        coords = list()
        for line in lines:
            x, y, t1, t2 = line.split(', ')
            coords.append([float(x), float(y)])
        coords = np.array(coords)
    for key in dataContainer['sn'].keys():
        if key == 'xy':
            dataContainer['sn'][key] = np.empty([nTrialsTotal, 2])
        elif key == 'i':
            dataContainer['sn'][key] = np.zeros(nTrialsTotal, dtype=int)
        else:
            dataContainer['sn'][key] = np.empty(nTrialsTotal)
    for block in ('b1', 'b2'):
        startIndex = curatedStimulusMetadata['sn'][block]['s1']
        stopIndex = curatedStimulusMetadata['sn'][block]['s2']
        if block == 'b1':
            trialIndexOffset = 0
        else:
            trialIndexOffset = nTrialsPerBlock
        trialIndices = np.arange(0, nTrialsPerBlock, 1) + trialIndexOffset
        dataContainer['sn']['i'][trialIndices] = trialIndices
        digitalSignal = labjackDataMatrix[startIndex: stopIndex, labjackChannelMapping.stimulus]
        edgeIndices = np.where(np.logical_or(
            np.diff(digitalSignal) > +0.5,
            np.diff(digitalSignal) < -0.5
        ))[0]
        nEdgesDetected = edgeIndices.size
        nEdgesExpected = curatedStimulusMetadata['sn'][block]['nEdgesTotal']
        if nEdgesDetected != nEdgesExpected:
            print(f'Warning: {nEdgesDetected} edges detected in block {block} but {nEdgesExpected} expected')
            continue
        else:
            timestamps = np.around(
                np.interp(edgeIndices, params.xp, params.fp) * params.m + params.b,
                3
            )
            dataContainer['sn']['xy'][trialIndices, :] = coords[trialIndices, :]
            dataContainer['sn']['t1'][trialIndices] = timestamps[0::4]
            dataContainer['sn']['t2'][trialIndices] = timestamps[2::4]

    return dataContainer

def extractStimuliTimestamps2(sessionObject):
    """
    """

    #
    dataContainer = {
        'sn': {
            'i': None,  # Trial index
            'xy': None, # x and y position of the dot (in degs)
            't1': None, # ON timestamp
            't2': None  # Off timestamp
        },
        'mb': {
            'i': None,  # Trial index
            'o': None,  # Orientation (in degs)
            't1': None, # Stimulus onset timestamp
            't2': None, # Stimulus offset timestamp
        },
        'dg': {
            'i': None,  # Trial index
            'e': None,  # Event code
            't': None,  # Timestamps for events
            'd': None,  # Direction of motion of the grating
        },
        'ng': {
            'i': None,  # Trial index
            't': None,  # Timestamps (N trials x N steps)
            's': None,  # Stimulus matrix indicating contrast levels (N trials x N steps) 
        },
    }
    
    # Extract metadata for the sparse noise stimulus
    dataContainer = extractStimulusDataSN(sessionObject, dataContainer)

    # Save results
    saveSessionData(sessionObject, 'visualStimuliData2', dataContainer)

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
        raise Exception(f'Sparse noise stimulus metadata file contains too many or too few lines')

    #
    timestampIndicesOnset = np.arange(0, int(constants.trialCountSparseNoise * 2), 1)[0::2]
    timestampIndicesOffset = np.arange(0, int(constants.trialCountSparseNoise * 2), 1)[1::2]
    for trialIndex, (lineData, timestampIndexOnset, timestampIndexOffset) in enumerate(zip(lines, timestampIndicesOnset, timestampIndicesOffset)):
        x, y, t1, t2 = lineData.split(', ')
        t1 = stateTransitionTimestamps[timestampIndexOnset]
        t2 = stateTransitionTimestamps[timestampIndexOffset]
        dataContainer['sparseNoise']['i'][trialIndex] = trialIndex
        dataContainer['sparseNoise']['x'][trialIndex] = x
        dataContainer['sparseNoise']['y'][trialIndex] = y
        dataContainer['sparseNoise']['t1'][trialIndex] = t1
        dataContainer['sparseNoise']['t2'][trialIndex] = t2

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

    #
    start = constants.stateTransitionCounts[0] + constants.stateTransitionCounts[1]
    nEdgesTotal = np.sum([value for value in constants.stateTransitionCounts.values() if value is not None])
    nEdgesResidual = stateTransitionTimestamps.size - nEdgesTotal
    stop = start + nEdgesResidual
    nTrials = int(nEdgesResidual / 2)

    #
    dataContainer['driftingGrating'] = {
        'i': np.empty(nTrials),
        'd': np.empty(nTrials),
        't1': np.empty(nTrials),
        't2': np.empty(nTrials)
    }

    #
    with open(sessionObject.driftingGratingMetadataFilePath, 'r') as stream:
        lines = [
            line for line in stream.readlines()
                if bool(re.search('.*, .*, .*\n', line)) and line.startswith('Columns') == False
        ]
    motionDirections = list()
    for line in lines:
        eventCode, motionDirection, approxTimestamp = line.rstrip('\n').split(', ')
        motionDirections.append(int(motionDirection))

    #
    risingEdgeTimestamps = stateTransitionTimestamps[start: stop: 2]
    fallingEdgeTimestamps = stateTransitionTimestamps[start + 1: stop: 2]
    iterable = zip(risingEdgeTimestamps, fallingEdgeTimestamps, motionDirections)
    for trialIndex, (risingEdgeTimestamp, fallingEdgeTimestamp, motionDirection) in enumerate(iterable):
        dataContainer['driftingGrating']['i'][trialIndex] = trialIndex
        dataContainer['driftingGrating']['d'][trialIndex] = motionDirection
        dataContainer['driftingGrating']['t1'][trialIndex] = risingEdgeTimestamp
        dataContainer['driftingGrating']['t2'][trialIndex] = fallingEdgeTimestamp

    return dataContainer

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
            't1': np.empty(constants.trialCountSparseNoise),
            't2': np.empty(constants.trialCountSparseNoise)
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

    # Correct for missing TTL pulses if they exist
    if sessionObject.missingFilePath.exists():
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

def runAll(sessionObject):
    """
    """

    extractLabjackData(sessionObject)
    extractBarcodeSignals(sessionObject)
    fitTimestampGenerator(sessionObject)
    extractStimuliTimestamps(sessionObject)

    return