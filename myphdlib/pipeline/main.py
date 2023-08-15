from myphdlib.pipeline.events import (
    createLabjackDataMatrix,
    extractBarcodeSignals,
    decodeBarcodeSignals,
    estimateTimestampingFunction,
    findDroppedFrames,
    timestampCameraTrigger,
)
from myphdlib.pipeline.saccades import (
    extractEyePosition,
    correctEyePosition,
    interpolateEyePosition,
    decomposeEyePosition,
    reorientEyePosition,
    filterEyePosition,
    detectPutativeSaccades,
    classifyPutativeSaccades,
    determineSaccadeOnset,
    sortProbeStimuli,
)

from myphdlib.pipeline.activity import (
    extractSingleUnitData,
    identifyUnitsWithEventRelatedActivity,
    measureSpikeSortingQuality,
)

from myphdlib.pipeline.cleanup import (
    cleanupOutputFile,
)

def process(
        session,
        redoManualInput=False
    ):
    """
    """

    # Extract the labjack data and the timestamps for the barcodes
    if session.hasGroup('labjack/matrix') == False:
        createLabjackDataMatrix(session)
    extractBarcodeSignals(session)
    decodeBarcodeSignals(session)
    estimateTimestampingFunction(session)

    # Find the boundaries between stimulus protocols
    if session.hasGroup('epochs') == False or redoManualInput:
        session.identifyProtocolEpochs()
    
    # Execute the session-specific processing of visual events
    if hasattr(session, 'processVisualEvents'):
        session.processVisualEvents()

    # Timestamp video acquisition
    findDroppedFrames(session)
    timestampCameraTrigger(session)

    # Extract and store single-unit data
    if session.isAutosorted:
        extractSingleUnitData(session)
        identifyVisualUnits(session)
        identifySaccadeRelatedUnits(session)
        measureSpikeSortingQuality(session)

    # Process eye position data and identify putative saccades
    if session.hasPoseEstimates:
        extractEyePosition(session)
        correctEyePosition(session)
        interpolateEyePosition(session)
        decomposeEyePosition(session)
        reorientEyePosition(session)
        filterEyePosition(session)
        detectPutativeSaccades(session)

    #
    cleanupOutputFile(session)

    return