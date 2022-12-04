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

    def describe(self, event, window=(0, 0.5), binsize=0.02):
        """
        Estimate the mean and standard deviation of the neuron's FR within a single time bin
        """

        edges, M = psth(
            event,
            self.timestamps,
            window=window,
            binsize=binsize
        )
        mu = M.flatten().mean() / binsize
        sigma = M.flatten().std() / binsize

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

        #
        self._listIndex = 0

        return

    def search(self, clusterNumber):
        """
        """

        for neuron in self._neuronList:
            if neuron.clusterNumber == clusterNumber:
                return neuron

        return None

    def __iter__(self):
        self._listIndex =0
        return self

    def __next__(self):
        if self._listIndex < len(self._neuronList):
            neuron = self._neuronList[self._listIndex]
            self._listIndex += 1
            return neuron
        else:
            raise StopIteration()

    def __len__(self):
        return len(self._neuronList)

    def __getitem__(self, listIndex):
        """
        """

        try:
            neuron = self._neuronList[listIndex]
        except:
            raise Exception() from None

        return neuron

    