from myphdlib.experiments.suppression2.constants import labjackChannelMapping
from myphdlib.experiments.suppression2.factory import loadSessionData, saveSessionData
from myphdlib.toolkit.labjack import loadLabjackData, extractLabjackEvent
from myphdlib.toolkit.sync import extractPulseTrains, decodePulseTrains
import numpy as np

def extractLabjackData(sessionObject):
    """
    Extract the labjack data matrix
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

def determineSyncFunction(sessionObject):
    """
    TODO: Figure out how to reference all computed time to the first ephys sample.
    Right now, everything is referenced to the first synchronous barcode
    """

    # Load the barcode data
    data = sessionObject.load('rawBarcodeData')
    barcodeValuesLabjack = data['labjack']['values']
    barcodeValuesNeuropixels = data['neuropixels']['values']
    barcodeIndicesLabjack = data['labjack']['indices']
    barcodeIndicesNeuropixels = data['neuropixels']['indices']

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
    barcodeValuesLabjack -= barcodeValuesLabjack[0]
    barcodeIndicesLabjack = barcodeIndicesLabjack[barcodeFilterLabjack]
    barcodeValuesNeuropixels = barcodeValuesNeuropixels[barcodeFilterNeuropixels]
    barcodeValuesNeuropixels -= barcodeValuesNeuropixels[0]
    barcodeIndicesNeuropixels = barcodeIndicesNeuropixels[barcodeFilterNeuropixels] - sessionObject.ephysFirstSample # Subtract first ephys sample here???

    # Compute the slope offset
    xlj = barcodeValuesLabjack
    xnp = barcodeValuesNeuropixels
    ylj = barcodeIndicesLabjack / 1000
    ynp = (barcodeIndicesNeuropixels) / 30000 # - barcodeIndicesNeuropixels[0]) / 30000
    mlj = (ylj[-1] - ylj[0]) / (xlj[-1] - xlj[0])
    mnp = (ynp[-1] - ynp[0]) / (xnp[-1] - xnp[0])

    # Define the sync function
    def f(si):
        """
        Convert a labjack sample index to time (sec) synchronized to the ephys recording
        """
        x = np.interp(si, barcodeIndicesLabjack, barcodeValuesLabjack) # Translae the sample index into units of barcodes
        y = x * (mlj + (mnp - mlj)) + (barcodeIndicesNeuropixels[0] / 30000)
        return y

    return f, barcodeIndicesLabjack, barcodeIndicesNeuropixels