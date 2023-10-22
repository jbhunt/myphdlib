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
    determineGratingMotionAssociatedWithEachSaccade
)

from myphdlib.pipeline.cleanup import (
    removeObsoleteDatasets,
)

from myphdlib.interface.factory import SessionFactory

class ModuleBase():
    
    def __init__(
        self,
        ):
        """
        """
        self._sessions = list()
        return

    def addSession(
        self,
        session,
        ):
        """
        """
        self._sessions.append(session)
        return

    def run(self):
        """
        """
        return

def processWholeDataset(
    sessions,
    experimentsForSaccadePrediction=('Mlati', 'Dreadds')
    ):
    """
    """

    # Extract eye position and detect saccades
    for session in sessions:
        if session.hasDeeplabcutPoseEstimates:
            extractEyePosition(session)
            correctEyePosition(session)
            interpolateEyePosition(session)
            decomposeEyePosition(session)
            reorientEyePosition(session)
            filterEyePosition(session)
            detectPutativeSaccades(session)

    # Classify putative saccades
    factory = SessionFactory()
    sessionsForSaccadePrediction = factory.produce(
        experiment=experimentsForSaccadePrediction
    )
    classifyPutativeSaccades(
        sessions,
        sessionsForSaccadePrediction
    )

    #
    for session in sessions:

        # Extract the labjack data and the timestamps for the barcodes
        if session.hasDataset('labjack/matrix') == False:
            createLabjackDataMatrix(session)
        extractBarcodeSignals(session)
        decodeBarcodeSignals(session)
        estimateTimestampingFunction(session)

        # Continue processing saccades
        if session.hasDeeplabcutPoseEstimates:
            determineSaccadeOnset(session)
            sortProbeStimuli(session)
            determineGratingMotionAssociatedWithEachSaccade(
                session
            )

        # Find the boundaries between stimulus protocols
        if session.hasGroup('epochs') == False:
            session.log('Protocol epochs need to be manually extracted', level='error')
            continue
        
        # Execute the session-specific processing of visual events
        if hasattr(session, 'processVisualEvents'):
            session.processVisualEvents()

        # Timestamp video acquisition
        findDroppedFrames(session)
        timestampCameraTrigger(session)

        # Process single-unit data
        if session.isAutosorted:
            extractSingleUnitData(session)
            measureSpikeSortingQuality(session)
            identifyUnitsWithEventRelatedActivity(session)
            estimateResponseLatency(session)
            predictUnitClassification(session)

        #
        # cleanupOutputFile(session)

    return