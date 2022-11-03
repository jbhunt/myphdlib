import numpy as np
import pathlib as pl
from myphdlib.general.toolkit import psth, smooth

class Neuron():
    """
    """

    def __init__(self, clusterNumber, singleUnitData, samplingRate=30000):
        self.clusterNumber = clusterNumber
        clusterMask = singleUnitData[:, 0] == clusterNumber
        self._timestamps = singleUnitData[clusterMask, 1] / samplingRate
        return

    @property
    def baselineFiringRate(self):
        """
        Compute the average and standard deviation FR (spikes/second)
        """
        binEdges = np.arange(0, self.timestamps.max(), 0.02)
        spikeCounts, binEdges = np.histogram(self.timestamps, binEdges)
        mu = np.mean(spikeCounts / 0.02)
        sigma = np.std(spikeCounts / 0.02)
        return mu, sigma

    @property
    def timestamps(self):
        return self._timestamps

class SpikeSortingResults():
    """
    """

    def __init__(self, resultsFolder):
        """
        """

        resultsFolderPath = pl.Path(resultsFolder)
        clusterNumbers, clusterLabels = list(), list()
        with open(resultsFolderPath.joinpath('cluster_group.tsv'), 'r') as stream:
            for line in stream.readlines()[1:]:
                clusterNumber, clusterLabel = line.rstrip('\n').split('\t')
                clusterNumbers.append(int(clusterNumber))
                clusterLabels.append(clusterLabel)

        #
        singleUnitData = np.hstack([
            np.load(resultsFolderPath.joinpath('spike_clusters.npy')),
            np.load(resultsFolderPath.joinpath('spike_times.npy'))
        ])

        #
        self._neuronList = list()
        for clusterNumber in clusterNumbers:
            self._neuronList.append(Neuron(clusterNumber, singleUnitData))

        return

    def __iter__(self):
        return

    def __next__(self):
        return

    def __getitem__(self, listIndex):
        """
        """

        try:
            neuron = self._neuronList[listIndex]
        except:
            raise Exception() from None

        return neuron

    