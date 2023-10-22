import os
import time
import mat73
import numpy as np
from myphdlib.general.toolkit import psth2
from myphdlib.extensions.matlab import runMatlabScript, locatMatlabAddonsFolder
from simple_spykes.util.ecephys import run_quality_metrics

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

class SpikesModule():
    """
    """

    def __init__(self):
        self._sessions = list()
        return

    def addSession(
        self,
        session
        ):
        """
        """

        self._sessions.append(session)

        return

    def run(
        self,
        ):
        """

        """

        submodules = (
            extractSpikeDatasets,
            extractKilosortLabels,
            extractSpikeWaveforms,
            extractUnitPositions,
            measureSpikeSortingQuality,
            createPopulationMasks,
        )

        for session in self._sessions:
            for submodule in submodules:
                submodule(session)

        return