from myphdlib.pipeline.prediction import (
    predictSaccadeDirection,
    predictSaccadeEpochs
)
from myphdlib.interface.factory import SessionFactory

def process(
    sessions,
    redo=False,
    zeta=False,
    saccadePredictionExperiments=('Mlati', 'Dreadds', 'Muscimol'),
    ):
    """
    """

    # Process eye position data and detect putative saccades
    for session in sessions:
        if session.hasDataset('saccades/putative') == False or redo:
            session._runSaccadeModule()

    # Collect sessions missing predicted saccades
    sessionsToAnalyze = list()
    for session in sessions:
        if session.hasDataset('saccades/predicted')== False or redo:
            sessionsToAnalyze.append(session)

    # Predict saccade parameters
    if len(sessionsToAnalyze) != 0:

        # Collect sessions used to train saccade classifier
        factory = SessionFactory()
        sessionsForTraining = factory.produce(
            experiment=saccadePredictionExperiments
        )

        # Predict saccade direction
        predictSaccadeDirection(
            sessionsToAnalyze,
            sessionsForTraining
        )

        # Predict saccade onset and offset
        predictSaccadeEpochs(
            sessionsToAnalyze,
            sessionsForTraining,
        )

    # Main pipeline
    for session in sessions:
        session._runEventsModule(redo)
        session._runStimuliModule()
        session._runActivityModule(zeta, redo)
        session._runSpikesModule()

    return