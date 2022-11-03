from myphdlib.general.toolkit import DotDict
from myphdlib.experiments.suppression2.constants import labjackChannelMapping as LCM
from myphdlib.general.session import saveSessionData, loadSessionData
from myphdlib.general.labjack import loadLabjackData, extractLabjackEvent
from myphdlib.general.sync import extractPulseTrains, decodePulseTrains
from myphdlib.extensions.matplotlib import placeVerticalLines

import re
import yaml
import numpy as np
from scipy.signal import find_peaks as findPeaks

def extractLabjackData(sessionObject, saveDataMatrix=False):
    """
    Extract the digital signals recorded by the labjack device
    """

    labjackDataMatrix = loadLabjackData(sessionObject.labjackFolderPath)
    iterable = zip(
        ['lightSensorSignal', 'exposureOnsetSignal', 'digitalBarcodeSignal'],
        ['stimulus', 'cameras', 'barcode']
    )
    for channelName, channelKey in iterable:
        channelSignal = labjackDataMatrix[:, LCM[channelKey]]
        saveSessionData(sessionObject, channelName, channelSignal)

    if saveDataMatrix:
        saveSessionData(sessionObject, 'labjackDataMatrix', labjackDataMatrix)

    return

def createInputFile(sessionObject, xData=None, nBlocksTotal=25):
    """
    Create the input file which contains the sample indices which separate
    blocks of each stimulus presentation
    """

    #
    if xData is None:
        lightSensorSignal = sessionObject.load('lightSensorSignal')
        xData = np.around(placeVerticalLines(lightSensorSignal), 0).astype(int)

    #
    if xData.size - 1 != nBlocksTotal:
        print('Warning: Number of line indices != number of blocks + 1')
        return xData

    #
    data = dict()
    data['curatedStimulusMetadata'] = {
        'sn': {f'b{i + 1}': dict() for i in range(2)},
        'mb': {f'b{i + 1}': dict() for i in range(2)},
        'dg': {f'b{i + 1}': dict() for i in range(1)},
        'ng': {f'b{i + 1}': dict() for i in range(20)}
    }
    iterable = zip(
        ['sn', 'mb', 'dg', 'sn', 'mb', 'ng'],
        [2160, 96, 0, 2160, 96, 24040],
        [1, 1, 1, 2, 2, None]
    )
    lineIndex = 0
    # TODO: Clean up this loop
    for stimulusName, nEdgesTotal, blockNumber in iterable:
        if blockNumber is None:
            for blockIndex in range(20):
                block = data['curatedStimulusMetadata'][stimulusName][f'b{blockIndex + 1}']
                block['s1'] = int(xData[lineIndex])
                block['s2'] = int(xData[lineIndex + 1])
                block['nEdgesTotal'] = nEdgesTotal
                block['iEdgesMissing'] = list()
                lineIndex += 1
        else:
            block = data['curatedStimulusMetadata'][stimulusName][f'b{blockNumber}']
            block['s1'] = int(xData[lineIndex])
            block['s2'] = int(xData[lineIndex + 1])
            block['nEdgesTotal'] = nEdgesTotal
            block['iEdgesMissing'] = list()
            lineIndex += 1

    #
    with open(sessionObject.inputFilePath, 'w') as stream:
        yaml.dump(data, stream, default_flow_style=False, sort_keys=False)

    return xData

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
    barcodeDigitalSignal = labjackDataMatrix[:, LCM['barcode']]
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

    #
    labjackDataMatrix = sessionObject.load('labjackDataMatrix')
    params = DotDict(sessionObject.load('timestampGeneratorParameters'))
    with open(sessionObject.inputFilePath, 'r') as stream:
        curatedStimulusMetadata = yaml.full_load(stream)['curatedStimulusMetadata']
    with open(sessionObject.sparseNoiseMetadataFilePath, 'r') as stream:
        lines = [
            line for line in stream.readlines()
                if bool(re.search('.*, .*, .*, .*\n', line)) and line.startswith('Columns') == False
        ]

    #
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
        digitalSignal = labjackDataMatrix[startIndex: stopIndex, LCM.stimulus]
        # edgeIndices = np.where(np.logical_or(
        #     np.diff(digitalSignal) > +0.5,
        #     np.diff(digitalSignal) < -0.5
        # ))[0]
        edgeIndices, peakProperties = findPeaks(np.abs(np.diff(digitalSignal)), height=0.5)
        edgeIndices += startIndex
        nEdgesDetected = edgeIndices.size
        nEdgesExpected = curatedStimulusMetadata['sn'][block]['nEdgesTotal']
        if nEdgesDetected != nEdgesExpected:
            print(f'Warning: {nEdgesDetected} edges detected in block {block} but {nEdgesExpected} expected')
            continue
        else:
            timestamps = np.around(
                np.interp(edgeIndices, params.xp, params.fp) * params.m + params.b,
                6
            )
            dataContainer['sn']['xy'][trialIndices, :] = coords[trialIndices, :]
            dataContainer['sn']['t1'][trialIndices] = timestamps[0::4]
            dataContainer['sn']['t2'][trialIndices] = timestamps[2::4]

    return dataContainer

def extractStimulusDataMB(sessionObject, dataContainer):
    """
    """

    labjackDataMatrix = sessionObject.load('labjackDataMatrix')
    params = DotDict(sessionObject.load('timestampGeneratorParameters'))
    with open(sessionObject.inputFilePath, 'r') as stream:
        curatedStimulusMetadata = yaml.full_load(stream)['curatedStimulusMetadata']
    with open(sessionObject.movingBarsMetadataFilePath, 'r') as stream:
        lines = [
            line for line in stream.readlines()
                if bool(re.search('.*, .*, .*\n', line)) and line.startswith('Columns') == False
        ]
    orientations = list()
    for line in lines[::2]:
        eventCode, orientation, approxTimestamp = line.rstrip('\n').split(', ')
        orientations.append(float(orientation))
    orientations = np.array(orientations)

    #
    nBlocks = len(curatedStimulusMetadata['mb'].keys())
    nTrialsTotal = int(len(lines) / 2)
    nTrialsPerBlock = int(nTrialsTotal / nBlocks)
    dataContainer['mb']['i'] = np.arange(nTrialsTotal)
    dataContainer['mb']['o'] = np.empty(nTrialsTotal)
    dataContainer['mb']['t1'] = np.empty(nTrialsTotal)
    dataContainer['mb']['t2'] = np.empty(nTrialsTotal)

    #
    for block in ('b1', 'b2'):
        if block == 'b1':
            trialIndices = np.arange(nTrialsPerBlock)
        else:
            trialIndices = np.arange(nTrialsPerBlock) + nTrialsPerBlock
        startIndex = curatedStimulusMetadata['mb'][block]['s1']
        stopIndex = curatedStimulusMetadata['mb'][block]['s2']
        digitalSignal = labjackDataMatrix[startIndex: stopIndex, LCM.stimulus]
        edgeIndices = np.where(np.logical_or(
            np.diff(digitalSignal) > +0.5,
            np.diff(digitalSignal) < -0.5
        ))[0]
        nEdgesExpected = curatedStimulusMetadata['mb'][block]['nEdgesTotal']
        nEdgesDetected = edgeIndices.size
        if nEdgesDetected != nEdgesExpected:
            import pdb; pdb.set_trace()
            print('Warning: Ligma balls')
            continue
        else:
            timestamps = np.around(
                np.interp(edgeIndices, params.xp, params.fp) * params.m + params.b,
                3
            )
            try:
                dataContainer['mb']['t1'][trialIndices] = timestamps[0::4]
                dataContainer['mb']['t2'][trialIndices] = timestamps[2::4]
                dataContainer['mb']['o'][trialIndices] = orientations[trialIndices]
            except:
                import pdb; pdb.set_trace()
                continue

    return dataContainer

def extractStimulusDataDG(sessionObject, dataContainer):
    """
    """

    #
    labjackDataMatrix = sessionObject.load('labjackDataMatrix')
    params = DotDict(sessionObject.load('timestampGeneratorParameters'))
    with open(sessionObject.inputFilePath, 'r') as stream:
        curatedStimulusMetadata = yaml.full_load(stream)['curatedStimulusMetadata']

    #
    with open(sessionObject.driftingGratingMetadataFilePath, 'r') as stream:
        lines = [
            line for line in stream.readlines()
                if bool(re.search('.*, .*, .*\n', line)) and line.startswith('Columns') == False
        ]
    motionDirections, eventCodes = list(), list()
    for line in lines:
        eventCode, motionDirection, approxTimestamp = line.rstrip('\n').split(', ')
        motionDirections.append(int(motionDirection))
        eventCodes.append(int(eventCode))
    motionDirections = np.array(motionDirections)

    #
    startIndex = curatedStimulusMetadata['dg']['b1']['s1']
    stopIndex = curatedStimulusMetadata['dg']['b1']['s2']
    digitalSignal = labjackDataMatrix[startIndex: stopIndex, LCM.stimulus]
    edgeIndices = np.where(np.logical_or(
        np.diff(digitalSignal) > +0.5,
        np.diff(digitalSignal) < -0.5
    ))[0]

    #
    nEvents = len(lines)

    #
    dataContainer['dg']['i'] = np.arange(nEvents)
    dataContainer['dg']['d'] = np.empty(nEvents)
    dataContainer['dg']['e'] = np.empty(nEvents)
    dataContainer['dg']['t'] = np.empty(nEvents)

    nEdgesExpected = int(nEvents * 2)
    nEdgesDetected = edgeIndices.size
    if nEdgesDetected != nEdgesExpected:
        print('Warning: Gottem')
        return dataContainer
    else:
        timestamps = np.around(
            np.interp(edgeIndices, params.xp, params.fp) * params.m + params.b,
            3
        )
        trialIndices = np.arange(nEvents)
        dataContainer['dg']['d'][trialIndices] = motionDirections
        dataContainer['dg']['e'][trialIndices] = eventCodes
        dataContainer['dg']['t'][trialIndices] = timestamps[::2]
        return dataContainer

def extractStimulusDataNG(sessionObject, dataContainer, nBlocks=20):
    """
    """

    #
    labjackDataMatrix = sessionObject.load('labjackDataMatrix')
    params = DotDict(sessionObject.load('timestampGeneratorParameters'))
    with open(sessionObject.inputFilePath, 'r') as stream:
        curatedStimulusMetadata = yaml.full_load(stream)['curatedStimulusMetadata']

    #
    with open(sessionObject.stimuliMetadataFilePaths['ng'], 'r') as stream:
        lines = [
            line for line in stream.readlines()
                if bool(re.search('.*, .*, .*\n', line)) and line.startswith('Columns') == False
        ]
    contrastLevels, motionDirectionList = list(), list()
    for line in lines:
        contrastLevel, motionDirection, approxTimestamp = line.rstrip('\n').split(', ')
        contrastLevels.append(float(contrastLevel))
        motionDirectionList.append(int(motionDirection))
    contrastLevels = np.array(contrastLevels)
    motionDirectionList = np.array(motionDirectionList)
    nStepsPerBlock = int(len(lines) / nBlocks)
    nEdgesPerBlock = nStepsPerBlock + 1
    motionDirectionByBlock = motionDirectionList[::nStepsPerBlock]
    contrastMatrix = np.array(np.split(contrastLevels, nBlocks))

    #
    dataContainer['ng']['i'] = np.arange(nStepsPerBlock)
    dataContainer['ng']['d'] = np.empty(nBlocks)
    dataContainer['ng']['t'] = np.empty([nBlocks, nStepsPerBlock])
    dataContainer['ng']['s'] = np.empty([nBlocks, nStepsPerBlock])

    #
    for blockIndex, block in enumerate([f'b{i + 1}' for i in range(nBlocks)]):
        startIndex = curatedStimulusMetadata['ng'][block]['s1']
        stopIndex = curatedStimulusMetadata['ng'][block]['s2']
        digitalSignal = labjackDataMatrix[startIndex: stopIndex, LCM.stimulus]
        edgeIndices = np.where(np.logical_or(
            np.diff(digitalSignal) > +0.5,
            np.diff(digitalSignal) < -0.5
        ))[0]
        nEdgesDetected = edgeIndices.size
        if nEdgesDetected != nEdgesPerBlock:
            print('Warning: Never gonna give you up')
            continue
        else:
            timestamps = np.around(
                np.interp(edgeIndices, params.xp, params.fp) * params.m + params.b,
                3
            )
            dataContainer['ng']['t'][blockIndex, :] = timestamps[:-1]
            dataContainer['ng']['s'][blockIndex, :] = contrastMatrix[blockIndex, :]
            dataContainer['ng']['d'][blockIndex] = motionDirectionByBlock[blockIndex]

    return dataContainer

def extractVisualStimuliData(sessionObject, dataContainerName='visualStimuliData'):
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
            'd': None,  # Direction of grating motion
        },
        'ng': {
            'i': None,  # Trial indexm
            'd': None,  # Direction of grating motion
            't': None,  # Timestamps (N trials x N steps)
            's': None,  # Stimulus matrix indicating contrast levels (N trials x N steps) 
        },
    }
    
    # Extract data for the sparse noise stimulus
    dataContainer = extractStimulusDataSN(sessionObject, dataContainer)

    # Extract data for the moving bars stimulus
    dataContainer = extractStimulusDataMB(sessionObject, dataContainer)

    # Extract data for the drifting grating stimulus
    dataContainer = extractStimulusDataDG(sessionObject, dataContainer)

    # Extract data for the noisy grating stimulus
    dataContainer = extractStimulusDataNG(sessionObject, dataContainer, 20)

    # Save results
    saveSessionData(sessionObject, dataContainerName, dataContainer)

    return

def runAllModules(sessionObject):
    """
    """

    extractLabjackData(sessionObject)
    extractBarcodeSignals(sessionObject)
    fitTimestampGenerator(sessionObject)
    extractVisualStimuliData(sessionObject)

    return