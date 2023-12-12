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

    def _estimateBaselineParameters(
        self,
        baselineWindow=(-20, -10),
        binsize=0.02,
        ):
        """
        """

        eventTimestamps = (
            self.probeTimestamps[self.filterProbes(trialType=None, probeDirections=(+1,))],
            self.probeTimestamps[self.filterProbes(trialType=None, probeDirections=(-1,))]
        )

        #
        for evt, probeDirection in zip(eventTimestamps, ('right', 'left')):
            mu, sigma = list(), list()
            for unit in self.population:
                mu_, sigma_ = unit.describe4(
                    evt,
                    baselineWindow=baselineWindow,
                    binsize=binsize,
                )
                mu.append(mu_)
                sigma.append(sigma_)
            self.save(f'population/baselines/probe/{probeDirection}/mu', np.array(mu))
            self.save(f'population/baselines/probe/{probeDirection}/sigma', np.array(sigma))

        return

    def _measureVisualResponseAmplitude(
        self,
        responseWindowSize=0.05,
        ):
        """
        """

        nUnits = len(self.population)
        responseAmplitudes = np.full([nUnits, 2], np.nan)

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
                self.filterProbes(trialType=None, probeDirections=(probeMotion,))
            ]

            #
            mu = self.load(f'population/baselines/probe/{probeDirection}/mu')
            sigma = self.load(f'population/baselines/probe/{probeDirection}/sigma')
            responseLatencies = self.load(f'population/zeta/probe/{probeDirection}/latency')

            #
            nFailed = 0
            nUnits = len(self.population)
            for iUnit, unit in enumerate(self.population):

                self.log(
                    f'Measuring visual response amplitude for unit {unit.cluster} ({iUnit + 1} / {nUnits})',
                    level='info'
                )

                #
                l = responseLatencies[unit.index]
                mu_ = mu[unit.index]
                sigma_ = sigma[unit.index]

                # Define the response and baseline windows
                windowHalfWidth = responseWindowSize / 2
                responseWindow = np.around(np.array([
                    l - windowHalfWidth,
                    l + windowHalfWidth
                ]), 2)

                # Measure the mean FR in the response window
                t, M = psth2(
                    probeTimestamps,
                    unit.timestamps,
                    window=responseWindow,
                    binsize=None
                )
                fr = M.mean(0) / responseWindowSize

                # Standardization is undefined
                if sigma_ == 0:
                    nFailed += 1
                    continue
                else:
                    z = (fr - mu_) / sigma_

                # 
                responseAmplitudes[unit.index, columnIndex] = z

            #
            self.log(f'Response amplitude estimation failed for {nFailed} out of {nUnits} units (probe motion={probeMotion})', level='warning')

        # Select the largest amplitude response (across motion directions)
        for iColumn, probeDirection in zip([0, 1], ['left', 'right']):
            self.save(f'population/metrics/vra/{probeDirection}', responseAmplitudes[:, iColumn])

        return

    def _measureVisualResponseShape(
        self,
        responseWindow=(0, 0.3),
        binsize=0.02,
        ):
        """
        """

        #
        for probeDirection, columnIndex in zip(('left', 'right'), (0, 1)):

            #
            responseCurves = list()

            #
            probeMotion = -1 if probeDirection == 'left' else +1
            probeTimestamps = self.probeTimestamps[
                self.filterProbes(trialType=None, probeDirections=(probeMotion,))
            ]

            #
            mu = self.load(f'population/baselines/probe/{probeDirection}/mu')
            sigma = self.load(f'population/baselines/probe/{probeDirection}/sigma')

            #
            nFailed = 0
            nUnits = len(self.population)
            for iUnit, unit in enumerate(self.population):

                self.log(
                    f'Measuring visual response shape for unit {unit.cluster} ({iUnit + 1} / {nUnits})',
                    level='info'
                )

                #
                mu_ = mu[unit.index]
                sigma_ = sigma[unit.index]

                # Measure the mean FR in the response window
                t, M = psth2(
                    probeTimestamps,
                    unit.timestamps,
                    window=responseWindow,
                    binsize=binsize
                )
                fr = M.mean(0) / binsize

                # Standardization is undefined
                if sigma_ == 0:
                    nFailed += 1
                    responseCurves.append(np.full(t.size, np.nan))
                    continue
                else:
                    z = (fr - mu_) / sigma_

                responseCurves.append(z)

            #
            self.save(f'population/peths/probe/{probeDirection}', np.array(responseCurves))

        return

    def _measureVisualResponsePolarity(
        self,
        ):
        """
        """

        return

    def _measureDirectionSelectivity(
        self,
        ):
        """
        """

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