from zetapy import getZeta
from joblib import Parallel, delayed
from myphdlib.general.toolkit import psth2
# from myphdlib.pipeline.main import ModuleBase
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
    minimumSpikeCount=100,
    iBatch=0,
    nBatches=0,
    ):
    """
    """

    units[0].session.log(f'Running batch ZETA test ({iBatch + 1} out of {nBatches})')

    #
    tOffset = 0 - responseWindow[0]
    responseWindowAdjusted = np.array(responseWindow) + tOffset

    #
    result = list()
    for unit in units:
        if unit.timestamps.size < minimumSpikeCount:
            p, tLatency = np.nan, np.nan
        else:
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

class ActivityProcessingMixin(object):
    """
    """

    def _runZetaTests(
        self,
        responseWindow=(-1, 1),
        parallelize=True,
        nUnitsPerBatch=10,
        latencyMetric='peak',
        minimumSpikeCount=100,
        overwrite=False
        ):
        """
        """

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
        if self.probeTimestamps is None:
            self.log('No probe stimuli detected: skipping ZETA test', level='warning')
            nUnits = len(self.population)
            for n, d in zip(eventNames, eventDirections):
                self.save(f'population/zeta/{n}/{d}/p', np.full(nUnits, np.nan))
                self.save(f'population/zeta/{n}/{d}/latency', np.full(nUnits, np.nan))
            return

        #
        eventTimestamps = (
            self.probeTimestamps[self.filterProbes(trialType='es', probeDirections=(-1,))],
            self.probeTimestamps[self.filterProbes(trialType='es', probeDirections=(+1,))],
            self.saccadeTimestamps[self.saccadeDirections == 'n'],
            self.saccadeTimestamps[self.saccadeDirections == 't']
        )

        #
        for ev, n, d in zip(eventTimestamps, eventNames, eventDirections):

            # Check if dataset already exists
            if self.hasDataset(f'population/zeta/{n}/{d}/p') and overwrite == False:
                self.log(f'Skipping ZETA test for activity related to {n}s (direction={d}, window=[{responseWindow[0]}, {responseWindow[1]}] sec)', level='info')
                continue

            #
            self.log(f'Running ZETA test for activity related to {n}s (direction={d}, window=[{responseWindow[0]}, {responseWindow[1]}] sec)', level='info')

            # Create batches of units
            batches = [
                self.population[i:i + nUnitsPerBatch]
                    for i in range(0, len(self.population), nUnitsPerBatch)
            ]
            nBatches = len(batches)

            # Parallel processsing
            if parallelize:
                results = Parallel(n_jobs=-1)(delayed(_runZetaTestForBatch)(
                    batch,
                    ev,
                    responseWindow,
                    latencyMetric,
                    minimumSpikeCount,
                    iBatch,
                    nBatches
                    )
                        for iBatch, batch in enumerate(batches)
                )

            # Serial processing
            else:
                results = list()
                for iBatch, batch in enumerate(batches):
                    result = _runZetaTestForBatch(
                        batch,
                        self.probeTimestamps[self.filterProbes(trialType='es', probeDirections=(-1, ))],
                        responseWindow=responseWindow,
                        latencyMetric=latencyMetric,
                        minimumSpikeCount=minimumSpikeCount,
                        iBatch=iBatch,
                        nBatches=nBatches
                    )
                    results.append(result)

            # Stack and sort the results
            results = np.vstack([result for result in results])
            unitIndices = results[:, 0]
            results = results[np.argsort(unitIndices), :]

            # Save p-values and latencies
            self.save(f'population/zeta/{n}/{d}/p', results[:, 1])
            self.save(f'population/zeta/{n}/{d}/latency', results[:, 2])

        return

    def _measureVisualResponseAmplitude(
        self,
        responseWindowSize=0.05,
        baselineWindowEdge=-2,
        baselineWindowSize=3,
        overwrite=True
        ):
        """
        """

        nUnits = len(self.population)
        responseAmplitudes = np.full([nUnits, 2], np.nan)

        #
        if self.hasDataset('population/metrics/vra/left') and overwrite == False:
            self.log('Skipping estimation of visual response amplitude', level='info')
            return

        # Skip this self
        if self.probeTimestamps is None:
            for probeDirection in ('left', 'right'):
                self.save(f'population/metrics/vra/{probeDirection}', np.full(nUnits, np.nan))
                return

        #
        for probeDirection, columnIndex in zip(('left', 'right'), (0, 1)):

            #
            probeMotion = -1 if probeDirection == 'left' else +1
            probeTimestamps = self.probeTimestamps[
                self.filterProbes('es', probeDirections=(probeMotion,))
            ]

            #
            # responseProbabilities = self.load(f'population/zeta/probe/{probeDirection}/p')
            responseLatencies = self.load(f'population/zeta/probe/{probeDirection}/latency')

            #
            nFailed = 0
            nUnits = len(self.population)
            for iUnit, unit in enumerate(self.population):

                self.log(
                    f'Measuring visual response amlplitude for unit {unit.cluster} ({iUnit + 1} / {nUnits})',
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
            self.log(f'Response amplitude estimation failed for {nFailed} out of {nUnits} units (probe motion={probeMotion})', level='warning')

        # Select the largest amplitude response (across motion directions)
        for iColumn, probeDirection in zip([0, 1], ['left', 'right']):
            self.save(f'population/metrics/vra/{probeDirection}', responseAmplitudes[:, iColumn])

        return

    def _measureEventRelatedBaselines(
        self,
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
            self.population.filter()
        nUnits = self.population.count(filtered=False)

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
            self.filterProbes('es', probeDirections=(-1,)),
            self.filterProbes('es', probeDirections=(+1,)),
            self.filterSaccades('es', saccadeDirections=('n',)),
            self.filterSaccades('es', saccadeDirections=('t',)),
        )

        #
        eventKeys = (
            ('probe', 'left'),
            ('probe', 'right'),
            ('saccade', 'nasal'),
            ('saccade', 'temporal')
        )
        
        #
        for unit in self.population:
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
            self.save(datasetPath, datasetArray)

        return

    def _computeStandardizedResponseCurves(
        self,
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
            self.probeTimestamps,
            self.probeTimestamps,
            self.saccadeTimestamps,
            self.saccadeTimestamps,
        )
        masks = (
            self.filterProbes(trialType='es', probeDirections=(-1,)),
            self.filterProbes(trialType='es', probeDirections=(+1,)),
            self.filterSaccades(trialType='es', saccadeDirections=('n',)),
            self.filterSaccades(trialType='es', saccadeDirections=('t',))
        )
        keys = (
            ('probe', 'left'),
            ('probe', 'right'),
            ('saccade', 'nasal'),
            ('saccade', 'temporal'),
        )

        #
        nBins = int(round(np.diff(responseWindow).item() / binsize, 0))
        nUnits = len(self.population)

        #
        for eventTimestamps, eventMask, (eventType, eventDirection) in zip(events, masks, keys):

            #
            if self.hasDataset(f'population/psths/{eventType}/{eventDirection}') and overwrite == False:
                self.log(f'Skipping estimation of standardized psth (event={eventType}, direction={eventDirection})', level='info')
                continue

            # Initialize the dataset
            Z = np.full([nUnits, nBins], np.nan)

            # Skip this event
            if eventTimestamps is None:
                self.save(f'population/psths/{eventType}/{eventDirection}', Z)
                continue

            #
            for iUnit, unit in enumerate(self.population):

                self.log(f'Computing standardized psth for unit {iUnit + 1} out of {nUnits} (event={eventType}, direction={eventDirection})', level='info')

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
            self.save(f'population/psths/{eventType}/{eventDirection}', Z)

        return

    def _runActivityModule(
        self,
        zeta=False,
        redo=False
        ):
        """
        """

        if zeta:
            if self.hasDataset('population/zeta/probe/left/p') == False or redo:
                self._runZetaTests()

        return