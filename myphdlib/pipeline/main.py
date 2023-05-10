from myphdlib.pipeline.events import (
    createLabjackDataMatrix,
    extractBarcodeSignals,
    decodeBarcodeSignals,
    estimateTimestampingFunction,
    extractSingleUnitData,
    findDroppedFrames,
    timestampCameraTrigger,
)
from myphdlib.pipeline.saccades import (
    extractEyePosition,
    correctEyePosition,
    decomposeEyePosition,
    reorientEyePosition,
    detectPutativeSaccades,
    classifyPutativeSaccades,
    determineSaccadeOnset,
)

def process(
        session,
        redoEpochExtraction=False
    ):
    """
    """

    #
    createLabjackDataMatrix(session)
    extractBarcodeSignals(session)
    decodeBarcodeSignals(session)
    estimateTimestampingFunction(session)

    #


    return