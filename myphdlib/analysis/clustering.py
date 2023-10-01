import numpy as np
from myphdlib.general.toolkit import psth2

def bootsrtapBaselineResponseEstimation(
    unit,
    nRuns=1000,
    nTrials=50,
    baselineResponseWindow=(-11, -10),
    targetEventTimestamps=None,
    ):
    """
    """

    null = list()
    for iRun in range(nRuns):
        trialIndices = np.random.choice(np.arange(targetEventTimestamps.size), size=nTrials, replace=True)
        t, M = psth2(
            targetEventTimestamps[trialIndices],
            unit.timestamps,
            window=baselineResponseWindow,
            binsize=None
        )
        fr = np.mean(M.sum(1) / np.diff(baselineResponseWindow).item())
        null.append(fr)

    return np.array(null)

def createProbabilityCurve(
    unit,
    targetEventTimestamps=None,
    targetResponseWindow=(-5, 5),
    baselineResponseWindow=(-11, -10),
    binsize=0.02,
    ):
    """
    """

    t, M = psth2(
        targetEventTimestamps,
        unit.timestamps,
        window=targetResponseWindow,
        binsize=binsize
    )
    rTarget = M.mean(0) / binsize

    #
    null = bootsrtapBaselineResponseEstimation(
        unit,
        targetEventTimestamps=targetEventTimestamps,
        baselineResponseWindow=baselineResponseWindow
    )

    #
    curve = list()
    for fr in rTarget:

        #
        p = 1 - np.sum(
            null >= fr
        ) / null.size
        curve.append(p)

    return np.array(curve), null, rTarget

def extractFeaturesFromSingleUnit(
    unit,
    ):
    """
    """

    allTargetEventTimestamps = (
        unit.session.probeTimestamps[unit.session.gratingMotionDuringProbes == -1],
        unit.session.probeTimestamps[unit.session.gratingMotionDuringProbes == +1],
        unit.session.saccadeTimestamps[unit.session.saccadeDirections == 'n'],
        unit.session.saccadeTimestamps[unit.session.saccadeDirections == 't']
    )

    features = list()
    for targetEventTimestamps in allTargetEventTimestamps:
        curve, null, rTarget = createProbabilityCurve(
            unit,
            targetEventTimestamps=targetEventTimestamps
        )
        for feature in curve:
            features.append(feature)

    return np.array(features)