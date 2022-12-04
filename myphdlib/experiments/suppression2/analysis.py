import numpy as np
from scipy.ndimage import gaussian_filter as gaussianFilter
from myphdlib.general.toolkit import psth, smooth
from myphdlib.general.ephys import SpikeSortingResults

def computeSRF(
    neuronObject, 
    sessionObject,
    binSize=0.01,
    gridShape=(10, 18),
    blur=True,
    kernelSpread=0.8,
    standardize=False,
    ):
    """
    Estimate the spatial receptive field from to the sparse noise stimulus
    """

    #
    data = sessionObject.load('visualStimuliData')
    stimulusTimestamps = np.hstack([
        data['sparseNoise']['t1'].reshape(-1, 1),
        data['sparseNoise']['t2'].reshape(-1, 1)
    ])
    spatialCoordinates = np.hstack([
        data['sparseNoise']['x'].reshape(-1, 1),
        data['sparseNoise']['y'].reshape(-1, 1)
    ])
    uniqueGridPoints = np.unique(spatialCoordinates, axis=0)

    # Left-to-right, top-to-bottom
    sortedIndex = np.lexsort([
        uniqueGridPoints[:, 0],
        uniqueGridPoints[:, 1][::-1],
    ])

    #
    resultMatrix = np.empty([gridShape[0], gridShape[1], 2])
    resultIndices = np.array([
        [i, j] for (i, j), v in np.ndenumerate(np.empty(gridShape))
    ])

    #
    if standardize:
        mu, sigma = neuronObject.baselineFiringRate

    #
    for (x, y), (i, j) in zip(uniqueGridPoints[sortedIndex, :], resultIndices):
        rowIndices = np.where([np.array_equal(row, np.array([x, y])) for row in spatialCoordinates])[0]
        stimulusOnsetTimestamps = stimulusTimestamps[rowIndices, 0]
        stimulusOffsetTimestamps = stimulusTimestamps[rowIndices, 1]
        edges, M = psth(stimulusOnsetTimestamps, neuronObject.timestamps)
        averageSpikeCount = M[:, 50:70].sum(1).mean()
        averageSpikeRate = averageSpikeCount / 0.2
        if standardize:
            averageSpikeRate = (averageSpikeRate - mu) / sigma
        resultMatrix[i, j, 0] = averageSpikeRate
        edges, M = psth(stimulusOffsetTimestamps, neuronObject.timestamps)
        averageSpikeCount = M[:, 50:100].sum(1).mean()
        averageSpikeRate = averageSpikeCount / 0.2
        if standardize:
            averageSpikeRate = (averageSpikeRate - mu) / sigma
        resultMatrix[i, j, 1] = averageSpikeRate

    if blur:
        resultMatrix = gaussianFilter(resultMatrix, kernelSpread)

    return resultMatrix

def computeSTA(sessionObject, neuronObject, timeWindow=(-1, 0.5), samplingRate=1000):
    """
    Compute a spike-triggered average of the noisy grating stimulus
    """

    data = sessionObject.load('visualStimuliData')['ng']
    trials = list()

    for timestamps, contrast in zip(data['t'], data['s']):

        #
        nSamples = round((timestamps.max() - timestamps.min()) * 1000)
        resampled = np.linspace(
            timestamps.min() * samplingRate,
            timestamps.max() * samplingRate,
            nSamples
        ) / samplingRate

        #
        stimulus = np.empty(resampled.size)
        s1 = 0
        for c, dt in zip(contrast[1:], np.diff(timestamps)):
            ds = round(dt * 1000)
            stimulus[s1: s1 + ds] = c
            s1 = s1 + ds

        #
        spikingMask = np.logical_and(
            neuronObject.timestamps > timestamps.min(),
            neuronObject.timestamps < timestamps.max()
        )

        #
        for t1 in neuronObject.timestamps[spikingMask]:
            relative = resampled - t1
            closest = np.argmin(np.abs(relative))
            start = closest + round(timeWindow[0] * samplingRate)
            stop = closest + round(timeWindow[1] * samplingRate)
            if start < 0:
                right = stimulus[:stop]
                left = np.full(abs(start), np.nan)
                trial = np.concatenate([left, right])
            elif stop > stimulus.size:
                left = stimulus[start:]
                right = np.full(stop - stimulus.size, np.nan)
                trial = np.concatenate([left, right])
            else:
                trial = stimulus[start: stop]

            trials.append(trial)

    return np.array(trials)

def computeSEA(sessionObject, minimumSpikeCount=10000, binsize=0.02):
    """
    Stimulus-evoked activity (normalized to baseline activity)
    """

    #
    stimulusData = sessionObject.load('visualStimuliData')['dg']
    probeOnsetTimestamps = list()
    for e, t in zip(stimulusData['e'], stimulusData['t']):
        if e == 3:
            if np.isnan(t) == False:
                probeOnsetTimestamps.append(t)
    
    #
    resultsFolderPath = sessionObject.ephysFolderPath.joinpath('continuous', 'Neuropix-PXI-100.0')
    rez = SpikeSortingResults(resultsFolderPath)

    # TODO: Normalize the FR first, then compute the response energy
    responses = list()
    uids = list()
    for unit in rez._neuronList:
        if unit.timestamps.size < minimumSpikeCount:
            continue
        edges, M = psth(
            probeOnsetTimestamps,
            unit.timestamps,
            window=(-0.3, 0.3),
            binsize=binsize
        )
        nBins = M.shape[1]
        halfway = int(M.shape[1] / 2)
        ref = np.mean(M[:, :halfway].mean(0) / binsize) # average spikes per second per bin
        fac = np.mean(M[:, :halfway].std(0) / binsize) # std of spikes per second per bin
        if fac == 0:
            continue
        raw = np.mean(M[:, halfway:] / binsize, axis=0) # average spikes per second across bins
        response = (raw - ref) / fac
        energy = np.sqrt(np.power(response, 2).sum())
        responses.append(energy)
        uids.append(unit.clusterNumber)

    return np.array(responses), np.array(uids)

def computeVisualResponses(factory):
    """
    """

    sessions = [s for s in factory]
    M = list()

    for s in sessions:
        try:
            rez = s.spikeSortingResults
        except:
            continue
        events = (
            s.spotOnsetTimestamps,
            s.spotOffsetTimestamps,
            s.gratingOnsetTimestamps,
            s.motionOnsetTimestamps,
            s.probeOnsetTimestamps,
        )

    return
