from zetapy import getZeta
from joblib import Parallel, delayed
from myphdlib.general.toolkit import psth2
from myphdlib.pipeline.main import ModuleBase
import numpy as np

# TODO
# [ ] Recompute the visual response amplitudes and compute respone amplitudes to saccades
# [ ] Measure the latency to the very first peak response
# [ ] Measure baseline activity? Maybe don't do this yet

def _runZetaTestForBatch(
    units,
    eventTimestamps,
    responseWindow,
    latencyMetric='peak',
    ):
    """
    """

    #
    tOffset = 0 - responseWindow[0]
    responseWindowAdjusted = np.array(responseWindow) + tOffset

    #
    result = list()
    for unit in units:
        p, (tZenith, tInverse, tPeak), dZeta = getZeta(
            unit.timestamps,
            eventTimestamps - tOffset,
            intLatencyPeaks=3,
            dblUseMaxDur=np.max(responseWindowAdjusted),
            tplRestrictRange=responseWindowAdjusted,
            boolReturnZETA=True
        )
        if latencyMetric == 'zenith':
            tLatency = tZenith - tOffset
        elif latencyMetric == 'peak':
            tLatency = tPeak - tOffset
        else:
            tLatency = np.nan
        result.append([unit.index, p, tLatency])

    return np.array(result)

def runZetaTests(
    session,
    responseWindow=(-1, 1),
    parallelize=True,
    nUnitsPerBatch=3,
    latencyMetric='peak',
    overwrite=False
    ):
    """
    """

    eventTimestamps = (
        session.probeTimestamps[session.filterProbes(trialType='es', probeDirections=(-1,))],
        session.probeTimestamps[session.filterProbes(trialType='es', probeDirections=(+1,))],
        session.saccadeTimestamps[session.saccadeDirections == 'n'],
        session.saccadeTimestamps[session.saccadeDirections == 't']
    )
    eventNames = (
        'probe',
        'probe',
        'saccade',
        'saccade',
    )
    eventDirections = (
        'left',
        'right',
        'nasal',
        'temporal'
    )

    #
    for ev, n, d in zip(eventTimestamps, eventNames, eventDirections):

        # Check if dataset already exists
        if session.hasDataset(f'population/zeta/{n}/{d}/p') and overwrite == False:
            session.log(f'Skipping ZETA test for activity related to {n}s (direction={d}, window=[{responseWindow[0]}, {responseWindow[1]}] sec)', level='info')
            continue

        #
        session.log(f'Running ZETA test for activity related to {n}s (direction={d}, window=[{responseWindow[0]}, {responseWindow[1]}] sec)', level='info')

        # Create batches of units
        batches = [
            session.population[i:i + nUnitsPerBatch]
                for i in range(0, len(session.population), nUnitsPerBatch)
        ]

        # Parallel processsing
        if parallelize:
            results = Parallel(n_jobs=-1)(delayed(_runZetaTestForBatch)(
                batch,
                ev,
                responseWindow,
                latencyMetric
                )
                    for batch in batches
            )

        # Serial processing
        else:
            results = list()
            for batch in batches:
                result = _runZetaTestForBatch(
                    batch,
                    session.probeTimestamps[session.filterProbes(trialType='es', probeDirections=(-1, ))],
                    responseWindow=responseWindow,
                    latencyMetric=latencyMetric
                )
                results.append(result)

        # Stack and sort the results
        results = np.vstack([result for result in results])
        unitIndices = results[:, 0]
        results = results[np.argsort(unitIndices), :]

        # Save p-values and latencies
        session.save(f'population/zeta/{n}/{d}/p', results[:, 1])
        session.save(f'population/zeta/{n}/{d}/latency', results[:, 2])

    return

def _runBaselineEstimationForBatch(
    units,
    eventTimestamps,
    baselineWindowSize,
    baselineBoundaries,
    nRuns
    ):
    """
    """

    params = list()
    for unit in units:
        mu, sigma = unit.estimateBaselineParameters(
            eventTimestamps,
            baselineWindowSize,
            baselineBoundaries,
            nRuns,
        )
        params.append([mu, sigma, unit.index])

    return params

def measureBaselineActivity2(
    session,
    ):
    """
    """

    return

def measureBaselineActivity(
    session,
    baselineBoundaries=(-10, -5),
    baselineWindowSize=0.1,
    nRunsPerUnit=100,
    nUnitsPerBatch=3,
    parallelize=True,
    overwrite=False,
    nJobs=-3
    ):
    """
    """

    #
    nUnits = len(session.population)

    #
    datasetKeys = (
        ('probe', 'left'),
        ('probe', 'right'),
        ('saccade', 'nasal'),
        ('saccade', 'temporal')
    )

    # Skip sessions with no visual probes
    if session.probeTimestamps is None:
        for (eventName, eventDirection) in datasetKeys:
            for feature in ('mu', 'sigma'):
                session.save(f'population/baseline/{eventName}/{eventDirection}/{feature}', np.full(nUnits, np.nan))
        return


    #
    eventTimestamps = (
        session.probeTimestamps[session.filterProbes(trialType='es', probeDirections=(-1,))],
        session.probeTimestamps[session.filterProbes(trialType='es', probeDirections=(+1,))],
        session.saccadeTimestamps[session.filterSaccades(trialType='es', saccadeDirections=('n',))],
        session.saccadeTimestamps[session.filterSaccades(trialType='es', saccadeDirections=('t',))]
    )

    #
    for (eventName, eventDirection), evt in zip(datasetKeys, eventTimestamps):

        #
        if session.hasDataset(f'population/baseline/{eventName}/{eventDirection}/mu') and overwrite == False:
            session.log(f'Skipping baseline estimation (event={eventName}, direction={eventDirection})', level='info')
            continue

        #
        session.log(f'Estimating baseline parameters (event={eventName}, direction={eventDirection})', level='info')

        #
        if parallelize:
            batches = list()
            for i in range(0, nUnits, nUnitsPerBatch):
                batch = session.population[i:i + nUnitsPerBatch]
                batches.append(batch)
            results = Parallel(n_jobs=nJobs)(delayed(_runBaselineEstimationForBatch)(
                batch,
                evt,
                baselineWindowSize,
                baselineBoundaries,
                nRunsPerUnit
                )
                    for batch in batches
            )
            params = np.vstack([
                np.array(result) for result in results
            ])
            average, deviation, indices = params[:, 0], params[:, 1], params[:, 2]
            index = np.argsort(indices)
            average = average[index]
            deviation = deviation[index]

        #
        else:
            average, deviation = np.full(nUnits, np.nan), np.full(nUnits, np.nan)
            for iUnit, unit in enumerate(session.population):
                mu, sigma = unit.estimateBaselineParameters(
                    evt,
                    baselineWindowSize=baselineWindowSize,
                    baselineBoundaries=baselineBoundaries,
                    nRuns=nRunsPerUnit
                )
                if sigma == 0:
                    average[iUnit] = np.nan
                    deviation[iUnit] = np.nan
                else:
                    average[iUnit] = mu
                    deviation[iUnit] = sigma

        #
        session.save(f'population/baseline/{eventName}/{eventDirection}/mu', average)
        session.save(f'population/baseline/{eventName}/{eventDirection}/sigma', deviation)

    return

def measureVisualResponseAmplitude(
    session,
    responseWindowSize=0.1,
    baselineWindowEdge=-5,
    zscoreMethod=2,
    nRuns=100,
    overwrite=False
    ):
    """
    """

    nUnits = len(session.population)
    responseAmplitudes = np.full([nUnits, 2], np.nan)

    #
    if session.hasDataset('population/metrics/gvr') and overwrite == False:
        session.log('Skipping estimation of visual response amplitude', level='info')
        return

    # Skip this session
    if session.probeTimestamps is None:
        session.save('population/metrics/gvr', np.full(nUnits, np.nan))
        return

    #
    for probeDirection, columnIndex in zip(('left', 'right'), (0, 1)):

        #
        probeMotion = -1 if probeDirection == 'left' else +1

        #
        responseProbabilities = session.load(f'population/zeta/probe/{probeDirection}/p')
        responseLatencies = session.load(f'population/zeta/probe/{probeDirection}/latency')

        #
        nFailed = 0
        for unit in session.population:

            #
            p = responseProbabilities[unit.index]
            l = responseLatencies[unit.index]

            # Define the response and baseline windows
            windowHalfWidth = responseWindowSize / 2
            visualResponseWindow = np.around(np.array([
                l - windowHalfWidth,
                l + windowHalfWidth
            ]), 2)
            baselineResponseWindow = np.around(np.array([
                baselineWindowEdge - responseWindowSize,
                baselineWindowEdge,
            ]), 2)

            # Measure the mean FR in the response window
            t, M = psth2(
                session.probeTimestamps[session.gratingMotionDuringProbes == probeMotion],
                unit.timestamps,
                window=visualResponseWindow,
                binsize=None
            )
            fr = M.mean(0) / responseWindowSize

            # Estimate the mean and std of the baseline FR
            if zscoreMethod == 1:
                mu, sigma = unit.describe(
                    session.probeTimestamps[session.gratingMotionDuringProbes == probeMotion],
                    window=baselineResponseWindow,
                binsize=None
                )
            elif zscoreMethod == 2:
                mu, sigma = (unit.upl, unit.spl) if probeMotion == -1 else (unit.upr, unit.spr)
                if mu is None and sigma is None:
                    print('Uh-oh')
                    mu, sigma = unit.estimateBaselineParameters(
                        session.probeTimestamps[session.gratingMotionDuringProbes == probeMotion],
                        baselineWindowSize=responseWindowSize,
                        nRuns=nRuns,
                    )

            # Standardization is undefined
            if sigma == 0:
                nFailed += 1
                continue
            else:
                z = (fr - mu) / sigma

            # 
            responseAmplitudes[unit.index, columnIndex] = z

        #
        session.log(f'Response amplitude estimation failed for {nFailed} out of {nUnits} units (probe motion={probeMotion})', level='warning')

    # Select the largest amplitude response (across motion directions)
    greatestVisualResponse = np.max(responseAmplitudes, axis=1)
    session.save('population/metrics/gvr', greatestVisualResponse)

    return

def computeStandardizedResponseCurves(
    session,
    binsize=0.01,
    responseWindow=(-0.5, 0.5),
    baselineWindowEdge=-5,
    zscoreMethod=1,
    nRuns=100,
    overwrite=False
    ):
    """
    """

    #
    responseWindowSize = np.diff(responseWindow).item()
    baselineWindow = np.around(np.array([
        baselineWindowEdge - responseWindowSize,
        baselineWindowEdge
    ]))

    #
    events = (
        session.probeTimestamps,
        session.probeTimestamps,
        session.saccadeTimestamps,
        session.saccadeTimestamps,
    )
    masks = (
        session.filterProbes(trialType='es', probeDirections=(-1,)),
        session.filterProbes(trialType='es', probeDirections=(+1,)),
        session.filterSaccades(trialType='es', saccadeDirections=('n',)),
        session.filterSaccades(trialType='es', saccadeDirections=('t',))
    )
    keys = (
        ('probe', 'left'),
        ('probe', 'right'),
        ('saccade', 'nasal'),
        ('saccade', 'temporal'),
    )

    #
    nBins = int(round(np.diff(responseWindow).item() / binsize, 0))
    nUnits = len(session.population)

    #
    for eventTimestamps, eventMask, (eventType, eventDirection) in zip(events, masks, keys):

        #
        if session.hasDataset(f'population/psths/{eventType}/{eventDirection}') and overwrite == False:
            session.log(f'Skipping estimation of standardized psth (event={eventType}, direction={eventDirection})', level='info')
            continue

        # Initialize the dataset
        Z = np.full([nUnits, nBins], np.nan)

        # Skip this event
        if eventTimestamps is None:
            session.save(f'population/psths/{eventType}/{eventDirection}', Z)
            continue

        #
        for iUnit, unit in enumerate(session.population):

            session.log(f'Computing standardized psth for unit {unit.cluster} (event={eventType}, direction={eventDirection})', level='info')

            # Compute the event-related response
            t, R1 = psth2(
                eventTimestamps[eventMask],
                unit.timestamps,
                window=responseWindow,
                binsize=binsize
            )
            fr = R1.mean(0) / binsize

            # Estimate the mean and std of the baseline FR
            if zscoreMethod == 1:
                mu, sigma = unit.describe(
                    eventTimestamps,
                    window=baselineWindow,
                    binsize=binsize
                )
            elif zscoreMethod == 2:
                if eventType == 'probe':
                    mu, sigma = (unit.upl, unit.spl) if eventDirection == 'left' else (unit.upr, unit.spr)
                elif eventType == 'saccade':
                    mu, sigma = (unit.usn, unit.ssn) if eventDirection == 'nasal' else (unit.ust, unit.sst)
                if mu is None and sigma is None:
                    mu, sigma = unit.estimateBaselineParameters(
                        session.probeTimestamps[session.gratingMotionDuringProbes == probeMotion],
                        baselineWindowSize=responseWindowSize,
                        nRuns=nRuns,
                    )

            # Standardize (z-score)
            if sigma == 0:
                continue
            z = (fr - mu) / sigma
            Z[iUnit, :] = z
    
        #
        session.save(f'population/psths/{eventType}/{eventDirection}', Z)

    return

class ActivityModule():

    def __init__(
        self
        ):
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

    def run(
        self,
        overwrite=False,
        ):
        """
        """

        submodules = (
            runZetaTests,
            measureBaselineActivity,
            measureVisualResponseAmplitude,
            computeStandardizedResponseCurves,
        )

        for session in self._sessions:
            for submodule in submodules:
                submodule(session, overwrite=overwrite)

        return