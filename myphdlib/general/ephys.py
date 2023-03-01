import numpy as np
import pathlib as pl
from myphdlib.general.toolkit import psth, smooth

class Neuron():
    """
    """

    def __init__(self, clusterNumber, singleUnitData, samplingRate=30000):
        self.cluster = clusterNumber
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
        mu = np.mean(M.mean(1) / binsize)
        sigma = np.std(M.mean(1) / binsize)

        return mu, sigma

    @property
    def timestamps(self):
        return self._timestamps

class SpikeSortingResults():
    """
    """

    def __init__(self, resultsFolder, autoload=True):
        """
        """

        self.resultsFolderPath = pl.Path(resultsFolder)
        self._neurons = list()
        self._index = 0
        if autoload:
            self.load()

        return

    def load(self):
        """
        """

        if self.isComplete() == False:
            return

        clusterNumbers, clusterLabels = list(), list()
        with open(self.resultsFolderPath.joinpath('cluster_group.tsv'), 'r') as stream:
            for line in stream.readlines()[1:]:
                clusterNumber, clusterLabel = line.rstrip('\n').split('\t')
                clusterNumbers.append(int(clusterNumber))
                clusterLabels.append(clusterLabel)

        #
        singleUnitData = np.hstack([
            np.load(self.resultsFolderPath.joinpath('spike_clusters.npy')),
            np.load(self.resultsFolderPath.joinpath('spike_times.npy'))
        ])

        #
        self._neurons = list()
        for clusterNumber in clusterNumbers:
            self._neurons.append(Neuron(clusterNumber, singleUnitData))
        self._neurons = np.array(self._neurons)

        #
        self._index = 0

        return

    def isComplete(self):
        """
        """

        filenames = (
            'cluster_group.tsv',
            'spike_clusters.npy',
            'spike_times.npy'
        )
        for filename in filenames:
            if self.resultsFolderPath.joinpath(filename).exists():
                continue
            else:
                return False

        return True

    def search(self, clusterNumber):
        """
        """

        for neuron in self._neurons:
            if neuron.clusterNumber == clusterNumber:
                return neuron

        return None

    def __iter__(self):
        self._index =0
        return self

    def __next__(self):
        if self._index < len(self._neurons):
            neuron = self._neurons[self._index]
            self._index += 1
            return neuron
        else:
            raise StopIteration()

    def __len__(self):
        return len(self._neurons)

    def __getitem__(self, key):
        """
        """

        try:
            neuron = self._neurons[key]
        except:
            raise Exception() from None

        return neuron

    