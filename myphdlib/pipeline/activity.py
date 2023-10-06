import os
import time
import mat73
import numpy as np
from zetapy import getZeta
from joblib import Parallel, delayed
from myphdlib.general.toolkit import psth2, smooth, computeAngleFromStandardPosition
from myphdlib.extensions.matlab import runMatlabScript, locatMatlabAddonsFolder
from simple_spykes.util.ecephys import run_quality_metrics
from scipy.stats import ttest_rel, ttest_ind
from scipy.signal import find_peaks as findPeaks

#
samplingRateNeuropixels = 30000.0

#
matlabScriptTemplate = """
addpath('{0}/npy-matlab-master/npy-matlab')
addpath('{0}/spikes-master/analysis')
spikeTimesFile = '{1}'
spikeClustersFile = '{2}'
gwf.dataDir = '{3}'
gwf.fileName = 'continuous.dat'
gwf.dataType = 'int16'
gwf.nCh = 384
gwf.wfWin = [-31 30]
gwf.nWf = {4}
gwf.spikeTimes = readNPY(spikeTimesFile)
gwf.spikeClusters = readNPY(spikeClustersFile)
result = getWaveForms(gwf)
waveforms = result.waveFormsMean;
fname = '{5}'
writeNPY(waveforms, fname)
exit
"""


def extractSpikeDatasets(session):
    """
    """

    if session.hasDataset('population/spikes/clusters') and session.hasDataset('population/spikes/timestamps'):
        return
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

    return

def extractKilosortLabels(
    session,
    overwrite=True,
    ):
    """
    """

    #
    if session.hasDataset('population/metrics/ksl') and overwrite == False:
        return

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
    clusterNumbers1 = np.unique(spikeClusters)
    nUnits = clusterNumbers1.size

    # Extract the label assigned to each unit by Kilosort
    clusterLabels = list()
    clusterNumbers2 = list()
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
            clusterNumbers2.append(int(cluster))
            clusterLabels.append(0 if label == 'mua' else 1)
        clusterLabels = np.array(clusterLabels)[np.argsort(clusterNumbers2)]

    # Need to delete labels where the cluster number is missing
    missingClusterNumbers = np.setdiff1d(clusterNumbers2, clusterNumbers1)
    missingClusterIndices = np.array([
        np.where(clusterNumbers2 == missingClusterNumber)[0].item()
            for missingClusterNumber in missingClusterNumbers
    ])
    if missingClusterIndices.size != 0:
        clusterLabels = np.delete(clusterLabels, missingClusterIndices)
    
    #
    session.save('population/metrics/ksl', clusterLabels)

    return

def extractSpikeWaveforms(
    session,
    nWaveforms=50,
    nBestChannels=5,
    nogui=True,
    windowsProcessTimeout=60*10, # Ten minutes
    ):

    #
    # if session.hasDataset('population/metrics/bsw'):
    #     return

    #
    partsFromEphysFolder = (
        'sorting',
    )
    spikeWaveformsFile = session.folders.ephys.joinpath(*partsFromEphysFolder, 'spike_waveforms.npy')

    #
    if spikeWaveformsFile.exists() == False:
        matlabAddonsFolder = locatMatlabAddonsFolder()
        matlabScriptLines = matlabScriptTemplate.format(
            matlabAddonsFolder,
            session.folders.ephys.joinpath(*partsFromEphysFolder, 'spike_times.npy'),
            session.folders.ephys.joinpath(*partsFromEphysFolder, 'spike_clusters.npy'),
            session.folders.ephys.joinpath(*partsFromEphysFolder),
            nWaveforms,
            session.folders.ephys.joinpath(*partsFromEphysFolder, 'spike_waveforms.npy'),
        ).strip('\n')
        scriptFilePath = session.folders.ephys.joinpath('sorting', 'extractSpikeWaveforms.m')
        with open(scriptFilePath, 'w') as stream:
            for line in matlabScriptLines:
                stream.write(line)

        #
        runMatlabScript(
            scriptFilePath,
            nogui=nogui
        )

        #
        if os.name == 'nt':
            t0 = time.time()
            while True:
                if time.time() - t0 > windowsProcessTimeout:
                    raise Exception(f'Failed to extract spike waveforms')
                spikeWaveformsFile = session.folders.ephys.joinpath(*partsFromEphysFolder, 'spike_waveforms.npy')
                if spikeWaveformsFile.exists():
                    break

        #
        elif os.name == 'posix':
            if spikeWaveformsFile.exists() == False:
                raise Exception(f'Failed to extract spike waveforms')
            
        #
        scriptFilePath.unlink() # Delete script

    #
    spikeWaveformsArray = np.load(spikeWaveformsFile)

    #
    nUnits, nChannels, nSamples = spikeWaveformsArray.shape
    bestSpikeWaveforms = np.full([nUnits, nSamples], np.nan)
    for iUnit in range(nUnits):
        lfp = spikeWaveformsArray[iUnit, :, :].mean(0)
        adv = np.array([
            np.abs(wf - lfp) for wf in spikeWaveformsArray[iUnit]
        ])
        channelIndices = np.argsort(adv.sum(axis=1))[::-1][:nBestChannels]
        bestSpikeWaveform = spikeWaveformsArray[iUnit, channelIndices, :].mean(0)
        bestSpikeWaveforms[iUnit :] = bestSpikeWaveform

    #
    session.save(f'population/metrics/bsw', bestSpikeWaveforms)

    return

def extractUnitPositions(session):
    """
    """

    #
    if session.hasDataset('population/metrics/msp'):
        return

    #
    kilosortResultsFile = session.folders.ephys.joinpath('sorting', 'rez.mat')
    kilosortResults = mat73.loadmat(kilosortResultsFile)['rez']
    spikeCoordinates = kilosortResults['xy']
    spikeClustersFile = session.folders.ephys.joinpath('sorting', 'spike_clusters.npy')
    spikeClusters = np.load(spikeClustersFile)
    
    #
    uniqueSpikeClusters = np.unique(spikeClusters)
    nUnits = uniqueSpikeClusters.size
    meanSpikePositions = np.full([nUnits, 2], np.nan)
    for iUnit, uniqueSpikeCluster in enumerate(uniqueSpikeClusters):
        mask = spikeClusters.flatten() == uniqueSpikeCluster
        meanSpikePosition = np.around(spikeCoordinates[mask, :].mean(0), 2)
        meanSpikePositions[iUnit] = meanSpikePosition

    # Need to swap x and y coordinates
    meanSpikePositions = np.fliplr(meanSpikePositions)
    session.save('population/metrics/msp', meanSpikePositions)


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
    responseWindowSize=0.1,
    baselineWindowEdge=-10,
    ):
    """
    """

    nUnits = len(session.population)
    responseAmplitudes = np.full([nUnits, 2], np.nan)

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
            mu, sigma = unit.describe(
                session.probeTimestamps[session.gratingMotionDuringProbes == probeMotion],
                window=baselineResponseWindow,
                binsize=None
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
    baselineWindowEdge=-10,
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
        session.filterSaccades(saccadeDirections=('n',)),
        session.filterSaccades(saccadeDirections=('t',))
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

        # Initialize the dataset
        Z = np.full([nUnits, nBins], np.nan)

        # Skip this event
        if eventTimestamps is None:
            session.save(f'population/psths/{eventType}/{eventDirection}', Z)
            continue

        #
        for iUnit, unit in enumerate(session.population):

            # Compute the event-related response
            t, R1 = psth2(
                eventTimestamps,
                unit.timestamps,
                window=responseWindow,
                binsize=binsize
            )
            fr = R1.mean(0) / binsize

            # Estimate the mean and std of the baseline FR
            mu, sigma = unit.describe(
                eventTimestamps,
                unit.timestamps,
                window=baselineWindow,
                binsize=binsize
            )

            # Standardize (z-score)
            if sigma == 0:
                continue
            z = (fr - mu) / sigma
            Z[iUnit, :] = z
    
        #
        session.save(f'population/psths/{eventType}/{eventDirection}', Z)

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