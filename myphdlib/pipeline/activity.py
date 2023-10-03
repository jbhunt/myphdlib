import os
import numpy as np
from zetapy import getZeta
from joblib import Parallel, delayed
from myphdlib.general.toolkit import psth2, smooth, computeAngleFromStandardPosition
from simple_spykes.util.ecephys import run_quality_metrics
from scipy.stats import ttest_rel, ttest_ind
from scipy.signal import find_peaks as findPeaks

samplingRateNeuropixels = 30000.0

def extractSpikeSortingData(session):
    """
    """

    session.log(f'Extracting spike clusters and timestamps', level='info')

    spikeTimestamps = np.array([])
    result = list(session.folders.ephys.joinpath('sorting').glob('spike_times.npy'))
    if len(result) != 1:
        raise Exception('Could not locate the spike times data')
    else:
        spikeTimestamps = np.around(
            np.load(str(result.pop())).flatten() / samplingRateNeuropixels,
            3
        )
    
    #
    spikeClusters = np.array([])
    result = list(session.folders.ephys.joinpath('sorting').glob('spike_clusters.npy'))
    if len(result) != 1:
        raise Exception('Could not locate the cluster ID data')
    else:
        spikeClusters = np.around(
            np.load(str(result.pop())).flatten(),
            3
        )

    #
    session.save('spikes/timestamps', spikeTimestamps)
    session.save('spikes/clusters', spikeClusters)

    #
    clusterNumbers = np.unique(spikeClusters)
    session.save('population/clusters', clusterNumbers)
    nUnits = clusterNumbers.size
    session.log(f'{nUnits} units detected')

    # Extract the label assigned to each unit by Kilosort
    clusterLabels = list()
    clusterNumbers = list()
    result = list(session.folders.ephys.joinpath('sorting').glob('cluster_KSLabel.tsv'))
    if len(result) != 1:
        session.log(f'Could not locate Kilosort labels', level='warning')
        clusterLabels = np.full(nUnits, np.nan)
    else:
        tsv = result.pop()
        with open(tsv, 'r') as stream:
            lines = stream.readlines()[1:]
        for line in lines:
            cluster, label = line.rstrip('\n').split('\t')
            clusterNumbers.append(int(cluster))
            clusterLabels.append(0 if label == 'mua' else 1)
        clusterLabels = np.array(clusterLabels)[np.argsort(clusterNumbers)]
    
    #
    session.save('population/metrics/ksl', clusterLabels)

    return

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

        #
        session.log(f'Running ZETA test for activity related to {n}s, (direction={d}, window=[{responseWindow[0]}, {responseWindow[1]}] sec)', level='info')

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

def measureSpikeSortingQuality(
    session,
    **kwargs
    ):
    """
    Notes
    -----
    Default threshold values are based on the quality metrics tutorial from the Allen Institute:
    https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_quality_metrics.html
    """

    print(f'INFO[{session.animal}, {session.date}]: Measuring spike-sorting quality')

    # ecephys spike sorting backend
    params_ = {
        "isi_threshold": 0.0015,
        "min_isi": 0.000166,
        "num_channels_to_compare": 7,
        "max_spikes_for_unit": 500,
        "max_spikes_for_nn": 10000,
        "n_neighbors": 4,
        'n_silhouette': 10000,
        "drift_metrics_interval_s": 51,
        "drift_metrics_min_spikes_per_interval": 10,
        "include_pc_metrics": False
    }
    params_.update(kwargs)

    #
    sortingResultsFolder = session.folders.ephys.joinpath('sorting')
    metrics = run_quality_metrics(
        str(sortingResultsFolder),
        30000.0,
        params_,
        save_to_file=str(sortingResultsFolder.joinpath('quality_metrics.json'))
    )
    presenceRatios = np.array(list(metrics['presence_ratio'].values())).astype(float)
    isiViolationRates = np.array(list(metrics['isi_viol'].values())).astype(float)
    amplitudeCutoffs = np.array(list(metrics['amplitude_cutoff'].values())).astype(float)

    #
    session.save('population/metrics/pr', presenceRatios)
    session.save('population/metrics/rpvr', isiViolationRates)
    session.save('population/metrics/ac', amplitudeCutoffs)

    return

def measureVisualResponseAmplitude(
    session,
    visualResponseWindow=(-0.1, 0.3),
    baselineResponseWindow=(-11, -10),
    binsize=0.01
    ):
    """
    """

    nUnits = len(session.population)
    responseAmplitudes = np.full([nUnits, 2], np.nan)
    for probeDirection, columnIndex in zip(('left', 'right'), (0, 1)):

        #
        probeMotion = -1 if probeDirection == 'left' else +1

        #
        responseProbabilities = session.load(f'population/zeta/probe/{probeDirection}/p')
        responseLatencies = session.load(f'population/zeta/probe/{probeDirection}/latency')

        #
        for unit in session.population:

            #
            p = responseProbabilities[unit.index]
            l = responseLatencies[unit.index]

            # Peak response is outside of the target response window
            if l < visualResponseWindow[0] and l > visualResponseWindow[1]:
                continue

            #
            t, M = psth2(
                session.probeTimestamps[session.gratingMotionDuringProbes == probeMotion],
                unit.timestamps,
                window=visualResponseWindow,
                binsize=binsize
            )
            fr = M.mean(0) / binsize
            mu, sigma = unit.describe(
                session.probeTimestamps[session.gratingMotionDuringProbes == probeMotion],
                window=baselineResponseWindow,
                binsize=binsize
            )

            # Standardization is undefined
            if sigma == 0:
                continue
            else:
                z = (fr - mu) / sigma

            # Figure out which bin contains the peak response
            binIndex = np.argmin(np.abs(t - l))
            responseAmplitudes[unit.index, columnIndex] = z[binIndex]

    # Select the largest amplitude response (across motion directions)
    greatestVisualResponse = np.max(responseAmplitudes, axis=1)
    session.save('population/metrics/gvr', greatestVisualResponse)

    return

def createPopulationMasks(
    session,
    minimumResponseProbability=0.05,
    maximumResponseLatencyForSaccadeRelatedUnits=0.04,
    minimumPresenceRatio=0.9,
    maximumRefractoryPeriodViolationRate=0.5,
    maximumAmplitudeCutoff=0.1,
    ):
    """
    """

    #
    nUnits = len(session.population)
    maskVisuallyResponsive = np.full(nUnits, False)
    maskSaccadeRelated = np.full(nUnits, False)

    for probeDirection in ('left', 'right'):
        responseProbabilities = session.load(f'population/zeta/probe/{probeDirection}/p')
        for unit in session.population:
            p = responseProbabilities[unit.index]
            if p < minimumResponseProbability:
                maskVisuallyResponsive[unit.index] = True

    for saccadeDirection in ('nasal', 'temporal'):
        responseProbabilities = session.load(f'population/zeta/saccade/{saccadeDirection}/p')
        responseLatencies = session.load(f'population/zeta/saccade/{saccadeDirection}/latency')
        for unit in session.population:
            p = responseProbabilities[unit.index]
            l = responseLatencies[unit.index]
            if p < minimumResponseProbability and l < maximumResponseLatencyForSaccadeRelatedUnits:
                maskSaccadeRelated[unit.index] = True

    #
    session.save('population/masks/sr', maskSaccadeRelated)
    session.save('population/masks/vr', maskVisuallyResponsive)

    #
    spikeSortingQualityFilters = np.vstack([
        session.load('population/metrics/pr') >= minimumPresenceRatio,
        session.load('population/metrics/rpvr') <= maximumRefractoryPeriodViolationRate,
        session.load('population/metrics/ac') <= maximumAmplitudeCutoff
    ])
    maskHighQuality = spikeSortingQualityFilters.all(0)
    session.save('population/masks/hq', maskHighQuality)

    return

def processEphysData(session):
    """
    """

    modules = (
        extractSpikeSortingData,
        runZetaTests,
        measureSpikeSortingQuality,
        measureVisualResponseAmplitude,
        createPopulationMasks,
    )

    for module in modules:
        module(session)

    return