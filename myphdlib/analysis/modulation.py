import numpy as np
from scipy.signal import find_peaks
from joblib import Parallel, delayed
from myphdlib.general.toolkit import psth2, smooth
from decimal import Decimal

# TODO
# [X] If standardizing the response, use a common estimate of mu and sigma
# [X] Base the estimate of latency-shifted sacade-related activity on the direction of motion of the grating for each saccade
# [-] Write a funciton that determines the response window
# [X] Figure out how to deal with negative spike rates in the MI formula
#     - Wait until computing the expected response to subtract off the baseline activity
#     - Add an offset to the psths such that the actual and expected responses are non-zero
# [X] Make the functions that estimate/measure responses do one direction of motion at a time
# [-] Get rid of the "perisaccadicWindow" argument (it's redundant)
# [X] Write an alternative MI computation the measures modulation on a trial-by-trial basis
# [ ] Have the option to compute MI as the difference in expected and observed response amplitude
#     - Use the peak latency determined with the ZETA test to identify which bin to look in

def determineResponseWindow(
    unit,
    ):
    return

def measureVisualOnlyResponse(
    unit,
    probeMotion=-1,
    visualResponseWindow=(0, 0.3),
    baselineResponseWindow=(-5, -4),
    perisaccadicWindow=(-0.05, 0.1),
    perisaccadicTrialIndices=None,
    tBuffer=0.1,
    binsize=0.01,
    ):
    """
    Measure the visual-only response
    """

    # Filter out peri-saccadic trials
    trialIndices = np.where(unit.session.filterProbes(
        trialType='es',
        windowBufferForExtrasaccadicTrials=tBuffer,
        probeDirections=(probeMotion,),
    ))[0]

    #
    t1, M = psth2(
        unit.session.probeTimestamps[trialIndices],
        unit.timestamps,
        window=baselineResponseWindow,
        binsize=None
    )
    bl = np.mean(M.flatten() / np.diff(baselineResponseWindow).item())

    #
    if binsize is None:
        dt = np.diff(visualResponseWindow).item()
    else:
        dt = binsize
    t2, M = psth2(
        unit.session.probeTimestamps[trialIndices],
        unit.timestamps,
        window=visualResponseWindow,
        binsize=binsize
    )  
    fr = M.mean(0) / dt

    return t1, t2, bl, fr

def measurePerisaccadicVisualResponse(
    unit,
    probeMotion=-1,
    visualResponseWindow=(0, 0.3),
    baselineResponseWindow=(-5, -4),
    perisaccadicWindow=(-0.05, 0.1),
    perisaccadicTrialIndices=None,
    binsize=0.01,
    ):
    """
    Measure the observed peri-saccadic response
    """

    #
    perisaccadicProbesMask = unit.session.filterProbes(
        perisaccadicWindow=perisaccadicWindow
    )

    #
    if perisaccadicTrialIndices is None:
        perisaccadicTrialIndices_ = np.logical_and(
            perisaccadicProbesMask,
            unit.session.gratingMotionDuringProbes == probeMotion
        )
    else:
        perisaccadicTrialIndices_ = perisaccadicTrialIndices

    #
    t1, M = psth2(
        unit.session.probeTimestamps[perisaccadicTrialIndices_],
        unit.timestamps,
        window=baselineResponseWindow,
        binsize=None
    )
    # bl = np.mean(M.flatten() / np.diff(baselineResponseWindow).item())
    bl = M / np.diff(baselineResponseWindow).item()

    #
    if binsize is None:
        dt = np.diff(visualResponseWindow).item()
    else:
        dt = binsize
    t2, M = psth2(
        unit.session.probeTimestamps[perisaccadicTrialIndices_],
        unit.timestamps,
        window=visualResponseWindow,
        binsize=binsize
    )
    fr = M / dt

    return t1, t2, bl, fr

def estimateSaccadeRelatedActivity(
    unit,
    probeMotion=-1,
    visualResponseWindow=(0, 0.3),
    baselineResponseWindow=(-5, -4),
    perisaccadicWindow=(-0.05, 0.1),
    perisaccadicTrialIndices=None,
    binsize=0.01,
    ):
    """
    Estimate the activity attributable to saccades in the peri-saccadic window
    """

    #
    if perisaccadicTrialIndices is None:
        perisaccadicProbesMask = unit.session.filterProbes(
            trialType='ps',
            perisaccadicWindow=perisaccadicWindow
        )
        perisaccadictrialIndices_ = np.where(np.logical_and(
            unit.session.gratingMotionDuringProbes == probeMotion,
            perisaccadicProbesMask
        ))[0]
    else:
        perisaccadictrialIndices_ = perisaccadicTrialIndices

    # Estimate the baseline level of activity prior to saccades
    saccadeIndices = np.where(unit.session.gratingMotionDuringSaccades == probeMotion)[0]
    t1, M = psth2(
        unit.session.saccadeTimestamps[saccadeIndices],
        unit.timestamps,
        window=baselineResponseWindow,
        binsize=None
    )
    bl = np.mean(M.flatten() / np.diff(baselineResponseWindow).item())

    #
    fr = list()
    if binsize is None:
        dt = np.diff(visualResponseWindow).item()
    else:
        dt = binsize
    for probeLatency in unit.session.probeLatencies[perisaccadictrialIndices_]:

        # Shift the saccade psth by the latency from the saccade to the probe
        t2, M = psth2(
            unit.session.saccadeTimestamps[saccadeIndices] + probeLatency,
            unit.timestamps,
            window=visualResponseWindow,
            binsize=binsize
        )
        fri = M.mean(0) / dt
        fr.append(fri)

    #
    fr = np.array(fr)
    return t1, t2, bl, fr

def computeUnidirectionalModulationIndex(
    unit,
    probeMotion=-1,
    responseWindowSize=0.1,
    baselineResponseWindowRightEdge=-10,
    perisaccadicWindow=(-0.05, 0.1),
    perisaccadicTrialIndices=None,
    clipResidualActivity=False,
    removeBaselineActivity=False,
    binsize=None
    ):
    """
    """

    # Define the visual response window
    if probeMotion == -1:
        peakResponseLatency = unit.lvrl
    else:
        peakResponseLatency = unit.lvrr
    windowHalfWidth = round(responseWindowSize / 2, 2)
    visualResponseWindow = np.around(np.array([
        peakResponseLatency - windowHalfWidth,
        peakResponseLatency + windowHalfWidth
    ]), 2)

    # Define the baseline response window
    baselineResponseWindow = np.around(np.array([
        baselineResponseWindowRightEdge - (2 * windowHalfWidth),
        baselineResponseWindowRightEdge
    ]), 2)

    #
    kwargs = {
        'probeMotion': probeMotion,
        'visualResponseWindow': visualResponseWindow,
        'baselineResponseWindow': baselineResponseWindow,
        'perisaccadicWindow': perisaccadicWindow,
        'perisaccadicTrialIndices': perisaccadicTrialIndices,
        'binsize': binsize
    }

    #
    t1, t2, baselineResponseProbe, targetResponseProbe = measureVisualOnlyResponse(
        unit,
        **kwargs
    )
    t = np.copy(t2)

    #
    t1, t2, baselineResponseMixed, targetResponseMixed = measurePerisaccadicVisualResponse(
        unit,
        **kwargs
    )

    # Make the target resonse window the same as the baseline window
    t1, t2, baselineResponseControl, targetResponseControl = estimateSaccadeRelatedActivity(
        unit,
        probeMotion=kwargs['probeMotion'],
        visualResponseWindow=kwargs['baselineResponseWindow'],
        baselineResponseWindow=kwargs['baselineResponseWindow'],
        perisaccadicWindow=kwargs['perisaccadicWindow'],
        perisaccadicTrialIndices=kwargs['perisaccadicTrialIndices'],
        binsize=kwargs['binsize']
    )

    #
    t1, t2, baselineResponseSaccade, targetResponseSaccade = estimateSaccadeRelatedActivity(
        unit,
        **kwargs
    )

    # Shape of responses
    if binsize is None:
        nTrials = targetResponseMixed.size
        nBins = 1
    else:
        nTrials, nBins = targetResponseMixed.shape

    # Set the minimum for clipping residual activity
    if clipResidualActivity:
        minimumResidualActivity = 0
    else:
        minimumResidualActivity = -np.inf

    # Compute the response offsets
    if removeBaselineActivity:
        responseOffsetProbe = -1 * baselineResponseProbe
    else:
        responseOffsetProbe = 0

    #
    observedResponses = np.full([nTrials, nBins], np.nan)
    expectedResponses = np.full([nTrials, nBins], np.nan)
    modulationIndices = np.full(nTrials, np.nan)
    for iTrial in range(nTrials):

        #
        baselineResidualActivity = np.clip(
            targetResponseControl[iTrial] - baselineResponseControl,
            minimumResidualActivity,
            np.inf
        )
        saccadeRelatedResidualActivity = np.clip(
            targetResponseSaccade[iTrial] - baselineResponseSaccade,
            minimumResidualActivity,
            np.inf
        )

        #
        responseOffsetMixed = -1 * baselineResponseMixed[iTrial] if removeBaselineActivity else 0

        #
        ro = targetResponseMixed[iTrial] + responseOffsetMixed + baselineResidualActivity
        re = targetResponseProbe + responseOffsetProbe + saccadeRelatedResidualActivity
        mi = (np.clip(ro.sum(), 0, np.inf) - np.clip(re.sum(), 0, np.inf)) / (np.clip(ro.sum(), 0, np.inf) + np.clip(re.sum(), 0, np.inf))

        #
        observedResponses[iTrial] = ro
        expectedResponses[iTrial] = re
        modulationIndices[iTrial] = mi

    #
    # numerator = np.clip(
    #     observedResponses.mean(0).sum() - expectedResponses.mean(0).sum(),
    #     0,
    #     np.inf
    # )
    # denominator = np.clip(
    #     observedResponses.mean(0).sum() + expectedResponses.mean(0).sum(),
    #     0,
    #     np.inf
    # )
    # mi = numerator / denominator

    #
    return t, modulationIndices, observedResponses, expectedResponses

def computeUnidirectionalModulationIndexUsingAverageResponse(
    unit,
    visualResponseWindow=(0, 0.3),
    baselineResponseWindow=(-11, -10),
    perisaccadicWindow=(-0.05, 0.1),
    perisaccadicTrialIndices=None,
    baselineOffsetParams=(0, 100, 0.1),
    binsize=0.01,
    probeMotion=-1
    ):
    """
    Compute the MI for responses to probes in one direction of motion
    """
            
    # Visual-only response
    t1, t2, bProbe, rProbe = measureVisualOnlyResponse(
        unit,
        visualResponseWindow=visualResponseWindow,
        binsize=binsize,
        probeMotion=probeMotion,
        baselineResponseWindow=baselineResponseWindow
    )

    # Response attributable to saccades
    t1, t2, bSaccade, rSaccade = estimateSaccadeRelatedActivity(
        unit,
        visualResponseWindow=visualResponseWindow,
        binsize=binsize,
        perisaccadicTrialIndices=perisaccadicTrialIndices,
        probeMotion=probeMotion,
        baselineResponseWindow=baselineResponseWindow
    )

    # Observed peri-saccadic response to probes
    t1, t2, bMixed, rMixed = measurePerisaccadicVisualResponse(
        unit,
        visualResponseWindow=visualResponseWindow,
        perisaccadicWindow=perisaccadicWindow,
        binsize=binsize,
        perisaccadicTrialIndices=perisaccadicTrialIndices,
        probeMotion=probeMotion,
        baselineResponseWindow=baselineResponseWindow
    )

    # Align the baseline activity between the expected response and the observed response
    for rCorrect in np.arange(*baselineOffsetParams):
        bExpected = bProbe + bSaccade
        if abs((bExpected - rCorrect) - bMixed) < 0.5:
            break

    # Subtract off as much of the baseline response as possible
    # rOffset = np.min([
    #     np.min(rProbe + rSaccade - rCorrect),
    #     np.min(rMixed)
    # ])
    rOffset = 0

    # Compute the MI
    rExpected = rProbe + rSaccade - rCorrect - rOffset
    rObserved = rMixed - rOffset
    numerator = rObserved.sum() - rExpected.sum()
    denominator = rObserved.sum() + rExpected.sum()
    if denominator == 0:
        denominator += 0.001
    mi = round(numerator / denominator, 3)
    
    return mi, rObserved, rExpected

def computeUnidirectionalModulationIndexTrialByTrial(
    unit,
    visualResponseWindow=(0, 0.3),
    probeMotion=-1,
    perisaccadicWindow=(-0.05, 0.1),
    baselineResponseWindow=(-5, -4),
    perisaccadicTrialIndices=None,
    binsize=0.01,
    subtractBaselineResponse=False
    ):
    """
    Compute the MI on a trial-by-trial basis
    """

    # Visual-only response
    t1, t2, bProbe, rProbe = measureVisualOnlyResponse(
        unit,
        probeMotion=probeMotion,
        visualResponseWindow=visualResponseWindow,
        binsize=binsize,
        baselineResponseWindow=baselineResponseWindow
    )
    if subtractBaselineResponse:
        rProbe -= bProbe

    #
    if perisaccadicTrialIndices is None:
        perisaccadicTrialIndices_ = np.where(unit.session.filterProbes(
            trialType='ps',
            perisaccadicWindow=perisaccadicWindow
        ))[0]
    else:
        perisaccadicTrialIndices_ = perisaccadicTrialIndices

    #
    observedBaselines, expectedBaselines = list(), list()
    observedResponses, expectedResponses = list(), list() 

    #
    for trialIndex in perisaccadicTrialIndices_:

        #
        probeLatency = unit.session.probeLatencies[trialIndex]
        probeTimestamp = unit.session.probeTimestamps[trialIndex]
        try:
            saccadeTimestamps = unit.session.saccadeTimestamps[unit.session.gratingMotionDuringSaccades == probeMotion]
        except:
            import pdb; pdb.set_trace()

        #
        try:
            t, M = psth2(
                saccadeTimestamps,
                unit.timestamps,
                window=baselineResponseWindow,
                binsize=binsize
            )
            bSaccade = M.mean(0) / binsize
        except:
            import pdb; pdb.set_trace()

        #
        try:
            t, M = psth2(
                saccadeTimestamps + probeLatency,
                unit.timestamps,
                window=visualResponseWindow,
                binsize=binsize
            )
            rSaccade = (M.mean(0) / binsize) 
            if subtractBaselineResponse:
                rSaccade -= np.mean(bSaccade)
        except:
            import pdb; pdb.set_trace()

        #
        t, M = psth2(
            np.array([probeTimestamp]),
            unit.timestamps,
            window=baselineResponseWindow,
            binsize=binsize
        )
        observedBaseline = M.flatten() / binsize

        #
        t, M = psth2(
            np.array([probeTimestamp]),
            unit.timestamps,
            window=visualResponseWindow,
            binsize=binsize
        )
        observedResponse = (M.flatten() / binsize)
        if subtractBaselineResponse:
            observedResponse -= np.mean(observedBaseline)

        #
        expectedBaseline = bProbe + bSaccade
        expectedResponse = rProbe + rSaccade

        #
        expectedBaselines.append(expectedBaseline)
        observedBaselines.append(observedBaseline)
        expectedResponses.append(expectedResponse)
        observedResponses.append(observedResponse)

    #
    averageObservedBaseline = np.array(observedBaselines).mean(0)
    averageExpectedBaseline = np.array(expectedBaselines).mean(0)
    averageObservedResponse = np.array(observedResponses).mean(0)
    averageExpectedResponse = np.array(expectedResponses).mean(0)

    #
    if subtractBaselineResponse:
        rOffset = 0
    else:
        rOffset = averageObservedBaseline.mean() - averageExpectedBaseline.mean()

    #
    rObserved = averageObservedResponse
    rExpected = averageExpectedResponse + rOffset
    mi = (np.clip(rObserved.sum(), 0, np.inf) - rExpected.sum()) / (np.clip(rObserved.sum(), 0, np.inf) + rExpected.sum())
    
    return mi, rObserved, rExpected

def estimateModulationIndexNullDistribution(
    unit,
    nRuns=100,
    probeMotion=-1,
    visualResponseWindow=(0, 0.3),
    method='uar',
    subtractBaselineResponse=False
    ):
    """
    Bootstrap the MI computation to estimate a null distribution of the index
    """

    # load trial filters
    perisaccadicProbesMask = unit.session.filterProbes(trialType='ps')
    extrasaccdadicProbesMask = unit.session.filterProbes(
        trialType='es', 
        windowBufferForExtrasaccadicTrials=0.5
    )

    # Count the number of peri-saccadic trials
    perisaccadicTrialIndices = np.where(np.logical_and(
        perisaccadicProbesMask,
        unit.session.gratingMotionDuringProbes == probeMotion
    ))[0]
    nPerisaccadicTrials = perisaccadicTrialIndices.size

    # Get the indices for extrasaccadic trials
    extrasaccadicTrialIndices = np.where(unit.session.filterProbes(
        trialType='es',
        windowBufferForExtrasaccadicTrials=0.5,
        probeDirections=(probeMotion,)
    ))[0]

    # For each run select a set of random extra-saccadic trials and compute the MI
    sample = list()
    for iRun in range(nRuns):

        # Choose the new set of extra-saccadic trials
        perisaccadicTrialIndices_ = np.random.choice(
            extrasaccadicTrialIndices,
            size=nPerisaccadicTrials,
            replace=False
        )

        # Compute the MI using average response
        if method == 'uar':
            mi, rObserved, rExpected = computeUnidirectionalModulationIndexUsingAverageResponse(
                unit,
                perisaccadicTrialIndices=perisaccadicTrialIndices_,
                visualResponseWindow=visualResponseWindow
            )
        
        # Compute the MI on a trial-by-trial basis
        elif method == 'tbt':
            mi, rObserved, rExpected = computeUnidirectionalModulationIndexTrialByTrial(
                unit,
                perisaccadicTrialIndices=perisaccadicTrialIndices_,
                visualResponseWindow=visualResponseWindow,
                subtractBaselineResponse=subtractBaselineResponse
            )

        #
        sample.append(mi)

    return np.array(sample)

def measureSaccadicModulationForSingleUnit(
    unit,
    visualResponseWindow='dynamic',
    baselineResponseWindow=(-5, -4),
    perisaccadicWindow=(-0.05, 0.1),
    baselineOffsetParams=(0, 100, 0.1),
    binsize=0.01,
    nRunsForBootstrap=100,
    method='uar',
    ):
    """
    """

    #
    result = {
        'left': (np.nan, np.nan),
        'right': (np.nan, np.nan)
    }
    if unit is None:
        return result

    #
    if visualResponseWindow == 'dynamic':
        visualResponseWindow = determineResponseWindow(
            unit,
            binsize=binsize
        )

    #
    for km, probeMotion in zip(('left', 'right'), (-1, 1)):

        #
        unit.session.log(f'Measuring saccadic modulation of unit {unit.cluster} (motion={probeMotion})', level='info')

        # Copmute the MI using average response
        if method == 'uar':
            mi, rObserved, rExpected = computeUnidirectionalModulationIndexUsingAverageResponse(
                unit,
                visualResponseWindow=visualResponseWindow,
                baselineResponseWindow=baselineResponseWindow,
                perisaccadicWindow=perisaccadicWindow,
                baselineOffsetParams=baselineOffsetParams,
                binsize=binsize,
                probeMotion=probeMotion
            )

        # Compute the MI on a trial-by-trial basis
        elif method == 'tbt':
            mi, rObserved, rExpected = computeUnidirectionalModulationIndexTrialByTrial(
                unit,
                visualResponseWindow=visualResponseWindow,
                baselineResponseWindow=baselineResponseWindow,
                perisaccadicWindow=perisaccadicWindow,
                binsize=binsize,
                probeMotion=probeMotion
            )

         # Create the null distribution of MI
        if nRunsForBootstrap is None or nRunsForBootstrap == 0:
            p = np.nan

        else:
            null = estimateModulationIndexNullDistribution(
                unit,
                probeMotion=probeMotion,
                nRuns=nRunsForBootstrap,
                visualResponseWindow=visualResponseWindow,
                method=method
            )

            # Calculate the fraction of values in the null distribution equal to or more extreme that the actual MI
            p = round(
                np.sum(np.abs(null) >= abs(mi)) / null.size,
                3
            )

        #
        result[km] = (mi, p)

    return result

def measureSaccadicModulationForAllUnits(
    session,
    visualResponsewindow='dynamic',
    baselineResponseWindow=(-5, -4),
    perisaccadicWindow=(-0.05, 0.1),
    baselineOffsetParams=(0, 100, 0.1),
    binsize=0.01,
    nRunsForBootstrap=100,
    parallel=True,
    ):
    """
    """

    #
    nUnits = len(session.population)
    mi = {
        'left': np.full(nUnits, np.nan),
        'right': np.full(nUnits, np.nan)
    }
    p = {
        'left': np.full(nUnits, np.nan),
        'right': np.full(nUnits, np.nan)
    }

    #
    args = (
        visualResponsewindow,
        baselineResponseWindow,
        perisaccadicWindow,
        baselineOffsetParams,
        binsize,
        nRunsForBootstrap
    )

    # Filter units based on their type and the spike-sorting quality
    units = list()
    for unit in session.population:
        if unit.type == 'vr' and unit.quality == 'h':
            units.append(unit)
        else:
            units.append(None)

    # Run MI estimation in parallel
    if parallel:
        results = Parallel(n_jobs=-1)(delayed(measureSaccadicModulationForSingleUnit)(unit, *args)
            for unit in session.population
        )
        for iUnit, result in enumerate(results):
            for km in ('left', 'right'):
                mi[km][iUnit] = result[km][0]
                p[km][iUnit] = result[km][1]

    # Run MI estimation in serial
    else:
        for iUnit, unit in enumerate(session.population):
            result = measureSaccadicModulationForSingleUnit(unit, *args)
            for km in ('left', 'right'):
                mi[km][iUnit] = result[km][0]
                p[km][iUnit] = result[km][1]

    # Save the estimates
    for km in ('left', 'right'):
        session.save(f'population/modulation/{km}/mi', np.array(mi[km]))
        session.save(f'population/modulation/{km}/p', np.array(p[km]))
    
    return

# TODO: Figure out why the estimates of modulation are so variable
#       Does it have something to do with the trial selection for each time bin?
class SaccadicModulationAcrossTimeAnalysis():
    """
    """

    def __init__(
        self,
        ):
        """
        """

        return

    def run(
        self,
        sessions,
        window=(-0.35, 0.35),
        binsize=0.1,
        ):
        """
        """

        # Compute the bin edges
        binEdges = np.hstack([
            np.arange(window[0], float(Decimal(str(window[1])) +  Decimal(str(binsize))), binsize)[0:-1].reshape(-1, 1),
            np.arange(window[0], float(Decimal(str(window[1])) +  Decimal(str(binsize))), binsize)[1:  ].reshape(-1, 1)
        ])
        self.binCenters = np.around(binEdges[:, 0] + (binsize / 2), 2)
        nBins = self.binCenters.size

        #
        self.result = dict()

        #
        for session in sessions:

            # Create entry in the result dictionary
            nUnits = len(session.population)
            self.result[(str(session.date), session.animal)] = {
                'left': np.full([nUnits, nBins], np.nan),
                'right': np.full([nUnits, nBins], np.nan)
            }

            # Load datasets
            probeLatency = session.load('stimuli/dg/probe/latency')
            probeMotionDuringGrating = session.load('stimuli/dg/probe/motion')

            #
            for iBin, (leftEdge, rightEdge) in enumerate(binEdges):

                #
                for iUnit, unit in enumerate(session.population):

                    # Filter out non-visual and low-quality units
                    if unit.isVisual == False:
                        continue
                    if unit.isQuality == False:
                        continue

                    # Determine the response window
                    visualResponseWindow = determineResponseWindow(
                        unit
                    )

                    #
                    for probeMotion, km in zip((-1, 1), ('left', 'right')):

                        # Identify trial indices
                        trialIndices = np.where(np.array([
                            probeLatency >= leftEdge,
                            probeLatency <  rightEdge,
                            probeMotionDuringGrating == probeMotion
                        ]).all(0))[0]

                        # Compute the MI
                        mi, rObserved, rExpected = computeUnidirectionalModulationIndexUsingAverageResponse(
                            unit,
                            visualResponseWindow=visualResponseWindow,
                            probeMotion=probeMotion,
                            perisaccadicTrialIndices=trialIndices
                        )

                        # Store the result
                        self.result[(str(session.date), session.animal)][km][iUnit, iBin] = mi

        return self.result

    def save(
        self,
        dst
        ):
        """
        """

        return