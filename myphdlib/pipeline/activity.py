from zetapy import getZeta, zetatest
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
    result = np.full([len(units), 3], np.nan)
    for i, unit in enumerate(units):
        if unit.timestamps.size < minimumSpikeCount:
            p, tLatency = np.nan, np.nan
        else:
            p, dZeta, dRate = zetatest(
                unit.timestamps,
                eventTimestamps - tOffset,
                dblUseMaxDur=np.max(responseWindowAdjusted),
                tplRestrictRange=responseWindowAdjusted,
            )

            if latencyMetric == 'zenith':
                tLatency = dZeta['vecLatencies'][0].item() - tOffset
            elif latencyMetric == 'peak':
                tLatency = dZeta['vecLatencies'][2].item() - tOffset
            else:
                tLatency = np.nan
        result[i, :] = np.array([unit.index, p, tLatency])

    return result

class ActivityProcessingMixin(object):
    """
    """

    def _runZetaTests(
        self,
        responseWindow=(-0.2, 0.5),
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
                self.save(f'zeta/{n}/{d}/p', np.full(nUnits, np.nan))
                self.save(f'zeta/{n}/{d}/latency', np.full(nUnits, np.nan))
            return

        #
        eventTimestamps = (
            self.probeTimestamps[self.filterProbes(trialType='es', probeDirections=(-1,))],
            self.probeTimestamps[self.filterProbes(trialType='es', probeDirections=(+1,))],
            self.saccadeTimestamps[self.saccadeLabels ==  1, 0],
            self.saccadeTimestamps[self.saccadeLabels == -1, 0]
        )

        #
        for ev, n, d in zip(eventTimestamps, eventNames, eventDirections):

            # Check if dataset already exists
            if self.hasDataset(f'zeta/{n}/{d}/p') and overwrite == False:
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
                        ev,
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
            self.save(f'zeta/{n}/{d}/p', results[:, 1])
            self.save(f'zeta/{n}/{d}/latency', results[:, 2])

        return

    def _runActivityModule(
        self,
        zeta=False,
        redo=False,
        parallelize=True,
        ):
        """
        """

        if zeta:
            if self.hasDataset('zeta/probe/left/p') == False or redo:
                self._runZetaTests(overwrite=True, parallelize=parallelize)

        return