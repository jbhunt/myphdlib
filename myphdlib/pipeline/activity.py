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

    def _extractResponseWaveforms(
        self,
        responseWindow=(-1, 1),
        baselineWindow=(-11, -10),
        normalizingWindow=(-0.25, 0.25),
        minimumBaselineActivity=2,
        binsize=0.02,
        overwrite=False
        ):
        """
        """

        self.log(f'Extracting event-related response waveforms')

        #
        nBins = 101
        nUnits = self.population.count()

        #
        datasetKeys = (
            ('probe', 'left'),
            ('probe', 'right'),
            ('saccade', 'nasal'),
            ('saccade', 'temporal')
        )

        # Check if data is alread processed 
        flags = np.full(4, False)
        for i, (eventType, eventDirection) in enumerate(datasetKeys):
            datasetPath = f'population/peths/{eventType}/{eventDirection}'
            if self.hasDataset(datasetPath):
                flags[i] = True
        if flags.all() and overwrite == False:
           return

        # Skip sessions without the drifting grating stimulus
        if self.probeTimestamps is None:
            for eventType, eventDirection in enumerate(datasetKeys):
                self.save(datasetPath, np.full([nUnits, nBins], np.nan))
            return

        # Event type, event direction, event timestamps
        iterable = (
            ('probe', 'left', self.probeTimestamps[self.gratingMotionDuringProbes == -1]),
            ('probe', 'right', self.probeTimestamps[self.gratingMotionDuringProbes == +1]),
            ('saccade', 'nasal', self.saccadeTimestamps[self.saccadeLabels == +1, 0]),
            ('saccade', 'temporal', self.saccadeTimestamps[self.saccadeLabels == -1, 1])
        )

        #
        for eventType, eventDirection, eventTimestamps in iterable:

            self.log(f'Computing response curves for {eventDirection} {eventType}s')

            #
            curves = np.full([nUnits, nBins], np.nan)

            #
            for iUnit, unit in enumerate(self.population):

                # Estimate baseline activity
                bl, sigma = unit.describeAcrossTrials(
                    eventTimestamps,
                    responseWindow=baselineWindow,
                )
                if bl < minimumBaselineActivity:
                    continue

                # Measure the evoked response
                t, fr1 = unit.peth(
                    eventTimestamps,
                    responseWindow=responseWindow,
                    binsize=binsize,
                    kde=True,
                    sd=0.02,
                )

                # Get the maximum of the evoked response in a smaller response window
                t, fr2 = unit.peth(
                    eventTimestamps,
                    responseWindow=normalizingWindow,
                    binsize=binsize,
                    kde=True,
                    sd=0.02,
                )
                factor = np.max(fr2)
                if factor == 0:
                    continue

                #
                r = (fr1 - bl) / factor
                curves[iUnit, :] = r

            #
            self.save(f'population/peths/{eventType}/{eventDirection}', np.array(curves))

        return

    def _measureLuminancePolarity(
        self,
        responseWindow=(0, 0.3),
        baselineWindow=(-0.2, 0),
        ):
        """
        """

        #
        spotTimestamps = list()
        spotPolarities = list()
        for phase in ('pre', 'post'):
            spotTimestamps_ = self.load(f'stimuli/sn/{phase}/timestamps')
            spotPolarities_ = self.load(f'stimuli/sn/{phase}/signs')
            if spotTimestamps_ is None:
                continue
            for t in spotTimestamps_:
                spotTimestamps.append(t)
            for p in spotPolarities_:
                spotPolarities.append(p)
        spotTimestamps = np.array(spotTimestamps)
        spotPolarities = np.array(spotPolarities)

        #
        nUnits = self.population.count()
        polarityIndices = np.full(nUnits, np.nan)
        for iUnit, unit in enumerate(self.population):

            #
            trialIndices = np.where(spotPolarities == True)
            t, M = psth2(
                spotTimestamps[trialIndices],
                unit.timestamps,
                window=responseWindow,
                binsize=None
            )
            rOn = M.mean(0).item() / np.diff(responseWindow).item()

            #
            trialIndices = np.where(spotPolarities == False)
            t, M = psth2(
                spotTimestamps[trialIndices],
                unit.timestamps,
                window=responseWindow,
                binsize=None
            )
            rOff = M.mean(0).item() / np.diff(responseWindow).item()

            #
            if rOn + rOff == 0:
                continue
            else:
                polarityIndex = (rOn - rOff) / (rOn + rOff)
                polarityIndices[iUnit] = round(polarityIndex, 2)

        #
        self.save(f'population/metrics/lpi', polarityIndices)

        return

    def _measureDirectionSelectivity(
        self,
        ):
        """
        """

        # Load stimulus metadata
        movingBarOrientations = self.load('stimuli/mb/orientation')
        barOnsetTimestamps = self.load('stimuli/mb/onset/timestamps')
        barOffsetTimestamps = self.load('stimuli/mb/offset/timestamps')
        movingBarTimestamps = np.hstack([
            barOnsetTimestamps.reshape(-1, 1),
            barOffsetTimestamps.reshape(-1, 1)
        ])

        #
        uniqueOrientations = np.unique(movingBarOrientations)
        uniqueOrientations.sort()

        #
        nUnits = self.population.count()
        directionSelectivityIndices = np.full(nUnits, np.nan).astype(float)

        #
        for unitIndex, unit in enumerate(self.population):

            #
            vectors = np.full([uniqueOrientations.size, 2], np.nan)
            for rowIndex, orientation in enumerate(uniqueOrientations):

                #
                trialIndices = np.where(movingBarOrientations == orientation)[0]
                amplitudes = list()
                for trialIndex in trialIndices:
                    t1, t2 = movingBarTimestamps[trialIndex, :]
                    dt = t2 - t1
                    t, M = psth2(
                        np.array([t1]),
                        unit.timestamps,
                        window=(0, dt),
                        binsize=None
                    )
                    fr = M.item() / dt
                    amplitudes.append(fr)

                #
                vectors[rowIndex, 0] = np.mean(amplitudes)
                vectors[rowIndex, 1] = np.deg2rad(orientation)

            # Compute the coordinates of the polar plot vertices
            vertices = np.vstack([
                vectors[:, 0] * np.cos(vectors[:, 1]),
                vectors[:, 0] * np.sin(vectors[:, 1])
            ]).T

            # Compute direction selectivity index
            a, b = vertices.sum(0) / vectors[:, 0].sum()
            dsi = np.sqrt(np.power(a, 2) + np.power(b, 2))
            directionSelectivityIndices[unitIndex] = dsi

        #
        self.save('population/metrics/dsi', directionSelectivityIndices)

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