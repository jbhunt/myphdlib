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

def measureVisualResponseAmplitude(
    session,
    responseWindowSize=0.05,
    baselineWindowEdge=-2,
    baselineWindowSize=3,
    overwrite=True
    ):
    """
    """

    nUnits = len(session.population)
    responseAmplitudes = np.full([nUnits, 2], np.nan)

    #
    if session.hasDataset('population/metrics/vra/left') and overwrite == False:
        session.log('Skipping estimation of visual response amplitude', level='info')
        return

    # Skip this session
    if session.probeTimestamps is None:
        for probeDirection in ('left', 'right'):
            session.save(f'population/metrics/vra/{probeDirection}', np.full(nUnits, np.nan))
            return

    #
    for probeDirection, columnIndex in zip(('left', 'right'), (0, 1)):

        #
        probeMotion = -1 if probeDirection == 'left' else +1
        probeTimestamps = session.probeTimestamps[
            session.filterProbes('es', probeDirections=(probeMotion,))
        ]

        #
        # responseProbabilities = session.load(f'population/zeta/probe/{probeDirection}/p')
        responseLatencies = session.load(f'population/zeta/probe/{probeDirection}/latency')

        #
        nFailed = 0
        for iUnit, unit in enumerate(session.population):

            session.log(
                f'Measuring visual response amlplitude for unit {unit.cluster} ({iUnit +z})',
                level='info'
            )

            #
            # p = responseProbabilities[unit.index]
            l = responseLatencies[unit.index]

            # Define the response and baseline windows
            windowHalfWidth = responseWindowSize / 2
            responseWindow = np.around(np.array([
                l - windowHalfWidth,
                l + windowHalfWidth
            ]), 2)
            baselineWindow = np.around(np.array([
                baselineWindowEdge - baselineWindowSize,
                baselineWindowEdge,
            ]), 2)

            # Measure the mean FR in the response window
            t, M = psth2(
                probeTimestamps,
                unit.timestamps,
                window=responseWindow,
                binsize=None
            )
            fr = M.mean(0) / responseWindowSize

            # Estimate the mean and std of the baseline FR
            mu, sigma = unit.describe2(
                probeTimestamps,
                baselineWindowBoundaries=baselineWindow,
                binsize=responseWindowSize
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
    for iColumn, probeDirection in zip([0, 1], ['left', 'right']):
        session.save(f'population/metrics/vra/{probeDirection}', responseAmplitudes[:, iColumn])

    return

def measureEventRelatedBaselines(
    session,
    baselineWindowBoundaries=(-5, -2),
    slidingWindowSize=0.5,
    binsize=0.02,
    nRuns=30,
    filterUnits=False,
    ):
    """
    """

    #
    if filterUnits:
        session.population.filter()
    nUnits = session.population.count(filtered=False)

    #
    datasets = {
        'population/metrics/bl/probe/left/mu': np.full(nUnits, np.nan),
        'population/metrics/bl/probe/left/sigma': np.full(nUnits, np.nan),
        'population/metrics/bl/probe/right/mu': np.full(nUnits, np.nan),
        'population/metrics/bl/probe/right/sigma': np.full(nUnits, np.nan),
        'population/metrics/bl/saccade/nasal/mu': np.full(nUnits, np.nan),
        'population/metrics/bl/saccade/nasal/sigma': np.full(nUnits, np.nan),
        'population/metrics/bl/saccade/temporal/mu': np.full(nUnits, np.nan),
        'population/metrics/bl/saccade/temporal/sigma': np.full(nUnits, np.nan),
    }

    #
    eventTimestamps = (
        session.filterProbes('es', probeDirections=(-1,)),
        session.filterProbes('es', probeDirections=(+1,)),
        session.filterSaccades('es', saccadeDirections=('n',)),
        session.filterSaccades('es', saccadeDirections=('t',)),
    )

    #
    eventKeys = (
        ('probe', 'left'),
        ('probe', 'right'),
        ('saccade', 'nasal'),
        ('saccade', 'temporal')
    )
    
    #
    for unit in session.population:
        for timestamps, (eventType, eventDirection) in zip(eventTimestamps, eventKeys):
            xb, sd = unit.describe2(
                timestamps,
                baselineWindowBoundaries=baselineWindowBoundaries,
                windowSize=slidingWindowSize,
                binsize=binsize
            )
            for feature, value in zip(('mu', 'sigma'), (xb, sd)):
                if np.isnan(value):
                    continue
                datasetPath = f'population/metrics/bl/{eventType}/{eventDirection}/{feature}'
                datasets[datasetPath][unit.index] = round(value, 2)

    #
    for datasetPath, datasetArray in datasets.items():
        session.save(datasetPath, datasetArray)

    return

def computeStandardizedResponseCurves(
    session,
    binsize=0.02,
    responseWindow=(-0.5, 0.5),
    baselineWindowEdge=-5,
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

            session.log(f'Computing standardized psth for unit {iUnit + 1} out of {nUnits} (event={eventType}, direction={eventDirection})', level='info')

            # Compute the event-related response
            t, R = psth2(
                eventTimestamps[eventMask],
                unit.timestamps,
                window=responseWindow,
                binsize=binsize
            )
            fr = R.mean(0) / binsize

            # Estimate the mean and std of the baseline FR
            mu, sigma = unit.describe2(
                eventTimestamps[eventMask],
                baselineWindowBoundaries=baselineWindow,
                binsize=binsize,
                nRuns=100,
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
            measureVisualResponseAmplitude,
            computeStandardizedResponseCurves,
        )

        for session in self._sessions:
            for submodule in submodules:
                submodule(session, overwrite=overwrite)

        return