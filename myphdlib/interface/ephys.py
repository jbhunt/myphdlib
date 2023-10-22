import h5py
import numpy as np
from myphdlib.general.toolkit import psth2

# TODO
# [ ] Load all of the unit property values on instatiation

class SingleUnit():
    """
    """

    def __init__(
        self,
        session,
        cluster=None
        ):
        """
        """

        self._session = session
        self._cluster = cluster
        self._timestamps = None
        self._utype = None
        self._quality = None
        self._index = None
        self._ksl = None
        self._lpl = None
        self._lpr = None
        self._gvr = None
        self._upl = None
        self._upr = None
        self._usn = None
        self._ust = None
        self._spl = None
        self._spr = None
        self._ssn = None
        self._sst = None
        self._ppl = None
        self._ppr = None
        self._psn = None
        self._pst = None

        return
    
    def describe(
        self,
        event=None,
        window=(-1, 0),
        binsize=1,
        ):

        #
        if event is None:
            t0 = 0
            t1 = np.ceil(self.timestamps.max())
            nBins = int((t1 - t0) // binsize) 
            nSpikes, binEdges = np.histogram(
                self.timestamps,
                range=(t0, t1),
                bins=nBins
            )
            dt = binEdges[1] - binEdges[0]
            mu = round(np.mean(nSpikes) / dt, 2)
            sigma = round(nSpikes.std() / dt, 2)
    
        #
        else:
            t, M = psth2(
                event,
                self.timestamps,
                window=window,
                binsize=binsize
            )
            if binsize is None:
                dt = np.diff(window).item()
                fr = M.flatten() / dt
                mu = np.mean(fr)
                sigma = np.std(fr)
            else:
                fr = M.flatten() / binsize
                mu = fr.mean()
                sigma = fr.std()

        return mu, sigma

    def estimateTrueBaseline(
        self,
        ):
        """
        """

        epochs = list()
        gratingOnsetTimestamps = self.session.load('stimuli/dg/grating/timestamps')
        gratingOffsetTimestamps = self.session.load('stimuli/dg/iti/timestamps')
        nBlocks = gratingOnsetTimestamps.size
        for iBlock in range(nBlocks):
            epoch = gratingOnsetTimestamps[iBlock], gratingOffsetTimestamps[iBlock]
            epochs.append(epoch)
        epochs = np.array(epochs)
        sample = list()
        for epoch in epochs:
            nSpikes = np.sum(np.logical_and(
                self.timestamps >= epoch[0],
                self.timestamps <= epoch[1]
            ))
            dt = np.diff(epoch)
            fr = nSpikes / dt
            sample.append(fr)

        return np.array(sample), np.mean(sample), np.std(sample)

    def bootstrapBaselineDescription(
        self,
        eventTimestamps,
        baselineWindowSize=0.1,
        baselineBoundaries=(-10, -5),
        nRuns=30,
        ):
        """
        Estimate mean and std of baseline FR using boostrap procedure
        """

        mu, sigma = np.full(nRuns, np.nan), np.full(nRuns, np.nan)
        windowHalfWidth = round(baselineWindowSize / 2, 2)
        for iRun in range(nRuns):
            baselineWindowCenter = np.around(np.random.uniform(
                low=baselineBoundaries[0] + windowHalfWidth,
                high=baselineBoundaries[1] - windowHalfWidth,
                size=1,
            ), 2).item()
            baselineWindowEdges = np.array([
                baselineWindowCenter - windowHalfWidth,
                baselineWindowCenter + windowHalfWidth
            ])
            t, R = psth2(
                eventTimestamps,
                self.timestamps,
                window=baselineWindowEdges,
                binsize=None
            )
            mu[iRun] = R.mean(0) / baselineWindowSize
            sigma[iRun] = R.std(0) / baselineWindowSize

        return round(mu.mean(), 3), round(sigma.mean(), 3)

    def peth(
        self,
        eventTimestamps,
        responseWindow=(-0.3, 0.5),
        baselineWindow=(-8, -5),
        binsize=0.01,
        standardize=True
        ):
        """
        """

        #
        t, m = psth2(
            eventTimestamps,
            self.timestamps,
            window=responseWindow,
            binsize=binsize
        )
        fr = m.mean(0) / binsize

        #
        if standardize == False:
            return t, fr

        #
        mu, sigma = self.describe(
            eventTimestamps,
            window=baselineWindow,
            binsize=None
        )
        if sigma == 0:
            z = np.full(t.size, np.nan)
        else:
            z = (fr - mu) / sigma

        return t, z

    @property
    def index(self):
        """
        """

        if self._index is None:
            # with h5py.File(str(self.session.hdf), 'r') as stream:
            #     spikeClusters = stream['spikes/clusters']
            #     uniqueSpikeClusters = np.unique(spikeClusters)
            #     self._index = np.where(uniqueSpikeClusters == self.cluster)[0].item()
            self._index = np.where(self.session.population.uniqueSpikeClusters == self.cluster)[0].item()

        return self._index

    @property
    def session(self):
        """
        """

        return self._session

    @property
    def cluster(self):
        """
        """

        return self._cluster
    
    @property
    def timestamps(self):
        """
        """

        if self._timestamps is None:
            # with h5py.File(str(self.session.hdf), 'r') as stream:
            #     spikeTimestamps = stream['spikes/timestamps']
            #     spikeClusters = stream['spikes/clusters']
            #     spikeIndices = np.where(spikeClusters == self.cluster)
            #     self._timestamps = np.array(spikeTimestamps[spikeIndices])
            spikeIndices = np.where(self.session.population.allSpikeClusters == self.cluster)[0]
            self._timestamps = self.session.population.allSpikeTimestamps[spikeIndices]

        return self._timestamps
    
    @property
    def utype(self):
        """
        """

        if self._utype is None:

            self._utype = list()

            #
            if self.session.population.datasets[('masks', 'vr')] is not None:
                vr = self.session.population.datasets[('masks', 'vr')][self.index]
                if vr:
                    self._utype.append('vr')

            #
            if self.session.population.datasets[('masks', 'sr')] is not None:
                sr = self.session.population.datasets[('masks', 'sr')][self.index]
                if sr:
                    self._utype.append('sr')

            #
            if len(self._utype) == 0:
                self._utype.append('nr')
            self._utype = tuple(self._utype)

        return self._utype

    # Spike-sorting quality (high/low)
    @property
    def quality(self):
        if self._quality is None:
            if self.session.population.datasets[('masks', 'hq')] is not None:
                hq = self.session.population.datasets[('masks', 'hq')][self.index]
                if hq:
                    self._quality = 'hq'
                else:
                    self._quality = 'lq'

        return self._quality

    # Kilosort label
    @property
    def ksl(self):
        if self._ksl is None:
            if self.session.population.datasets[('metrics', 'ksl')] is not None:
                label = self.session.population.datasets[('metrics', 'ksl')][self.index]
                self._ksl = 'm' if label == 0 else 'g'

        return self._ksl

    # Probability visually responsive (left)
    @property
    def ppl(self):
        if self._ppl is None:
            if self.session.population.datasets[('zeta', 'probe', 'left', 'p')] is not None:
                self._ppl = round(self.session.population.datasets[('zeta', 'probe', 'left', 'p')][self.index], 2)
        return self._ppl
    
    # Probability visually responsive (right)
    @property
    def ppr(self):
        if self._ppr is None:
            if self.session.population.datasets[('zeta', 'probe', 'right', 'p')] is not None:
                self._ppr = round(self.session.population.datasets[('zeta', 'probe', 'right', 'p')][self.index], 2)
        return self._ppr

    # Probability saccade related (nasal)
    @property
    def psn(self):
        if self._psn is None:
            if self.session.population.datasets[('zeta', 'probe', 'left', 'latency')] is not None:
                self._psn = round(self.session.population.datasets[('zeta', 'saccade', 'nasal', 'p')][self.index], 2)
        return self._psn

    # Probability saccade related (temporal)
    @property
    def pst(self):
        if self._pst is None:
            if self.session.population.datasets[('zeta', 'probe', 'left', 'latency')] is not None:
                self._pst = round(self.session.population.datasets[('zeta', 'saccade', 'nasal', 'p')][self.index], 2)
        return self._pst

    #
    @property
    def lpl(self):
        if self._lpl is None:
            if self.session.population.datasets[('zeta', 'probe', 'left', 'latency')] is not None:
                self._lpl = round(self.session.population.datasets[('zeta', 'probe', 'left', 'latency')][self.index], 2)
        return self._lpl

    #
    @property
    def lpr(self):
        if self._lpr is None:
            if self.session.population.datasets[('zeta', 'probe', 'right', 'latency')] is not None:
                self._lpr = round(self.session.population.datasets[('zeta', 'probe', 'right', 'latency')][self.index], 2)
        return self._lpr

    # Greatest visual response (amplitude, z-scored spikes/second)
    @property
    def gvr(self):
        if self._gvr is None:
            if self.session.population.datasets[('metrics', 'gvr')] is not None:
                self._gvr = round(self.session.population.datasets[('metrics', 'gvr')][self.index], 3)
        return self._gvr

    # Mean baseline FR preceding probe stimulus during leftward motion
    @property
    def upl(self):
        if self._upl is None:
            if self.session.population.datasets[('baseline', 'probe', 'left', 'mu')] is not None:
                self._upl = round(self.session.population.datasets[('baseline', 'probe', 'left', 'mu')][self.index], 3)
        return self._upl
    
    # Mean baseline FR preceding probe stimulus during rightward motion
    @property
    def upr(self):
        if self._upr is None:
            if self.session.population.datasets[('baseline', 'probe', 'right', 'mu')] is not None:
                self._upr = round(self.session.population.datasets[('baseline', 'probe', 'right', 'mu')][self.index], 3)
        return self._upr
    
    # Mean baseline FR preceding nasal saccades
    @property
    def usn(self):
        if self._upr is None:
            if self.session.population.datasets[('baseline', 'probe', 'right', 'mu')] is not None:
                self._upr = round(self.session.population.datasets[('baseline', 'probe', 'right', 'mu')][self.index], 3)
        return self._upr

    # Mean baseline FR preceding temporal saccades
    @property
    def ust(self):
        if self._upr is None:
            if self.session.population.datasets[('baseline', 'probe', 'right', 'mu')] is not None:
                self._upr = round(self.session.population.datasets[('baseline', 'probe', 'right', 'mu')][self.index], 3)
        return self._upr

    @property
    def spl(self):
        if self._spl is None:
            if self.session.population.datasets[('baseline', 'probe', 'left', 'sigma')] is not None:
                self._spl = round(self.session.population.datasets[('baseline', 'probe', 'left', 'sigma')][self.index], 3)
        return self._spl
    
    @property
    def spr(self):
        if self._spr is None:
            if self.session.population.datasets[('baseline', 'probe', 'right', 'sigma')] is not None:
                self._spr = round(self.session.population.datasets[('baseline', 'probe', 'right', 'sigma')][self.index], 3)
        return self._spr
    
    @property
    def ssn(self):
        if self._ssn is None:
            if self.session.population.datasets[('baseline', 'saccade', 'nasal', 'sigma')] is not None:
                self._snn = round(self.session.population.datasets[('baseline', 'saccade', 'nasal', 'sigma')][self.index], 3)
        return self._snn

    @property
    def sst(self):
        if self._sst is None:
            if self.session.population.datasets[('baseline', 'saccade', 'temporal', 'sigma')] is not None:
                self._sst = round(self.session.population.datasets[('baseline', 'saccade', 'temporal', 'sigma')][self.index], 3)
        return self._sst

class Population():
    """
    """

    def __init__(self, session, autoload=True):
        """
        """

        self._session = session
        self._units = None
        self._index = 0
        self._datasets = {
            ('masks', 'vr'): None,
            ('masks', 'sr'): None,
            ('masks', 'hq'): None,
            ('metrics', 'pr'): None,
            ('metrics', 'rpvr'): None,
            ('metrics', 'ac'): None,
            ('metrics', 'gvr'): None,
            ('metrics', 'ksl'): None,
            ('zeta', 'probe', 'left', 'p'): None,
            ('zeta', 'probe', 'left', 'latency'): None,
            ('zeta', 'probe', 'right', 'p'): None,
            ('zeta', 'probe', 'right', 'latency'): None,
            ('zeta', 'saccade', 'nasal', 'p'): None,
            ('zeta', 'saccade', 'nasal', 'latency'): None,
            ('zeta', 'saccade', 'temporal', 'p'): None,
            ('zeta', 'saccade', 'temporal', 'latency'): None,
        }

        if autoload:
            self._loadSingleUnitData()
            self._loadPopulationDatasets()

        return
    
    def _loadSingleUnitData(self):
        """
        """

        #
        if self._units is not None:
            del self._units
        self._units = list()

        #
        self._allSpikeClusters = self._session.load('spikes/clusters')
        self._allSpikeTimestamps = self._session.load('spikes/timestamps')
        self._uniqueSpikeClusters = np.unique(self.allSpikeClusters)
        # with h5py.File(str(self._session.hdf), 'r') as stream:
        #     spikeClusters = stream['spikes/clusters']
        #    uniqueSpikeClusters = np.unique(spikeClusters)

        #
        # for cluster in np.unique(self._allSpikeClusters):
        for cluster in self.uniqueSpikeClusters:
            unit = SingleUnit(self._session, cluster)
            self._units.append(unit)

        return

    # TODO: Code this
    def _loadPopulationDatasets(
        self
        ):
        """
        """

        for k in self._datasets.keys():
            if self._datasets[k] is None:
                parts = list(k)
                parts.insert(0, 'population')
                datasetPath = '/'.join(parts)
                if self._session.hasDataset(datasetPath):
                    self._datasets[k] = self._session.load(datasetPath)

        return

    def indexByCluster(self, cluster):
        """
        """

        for unit in self._units:
            if unit.cluster == cluster:
                return unit

        return

    @property
    def allSpikeClusters(self):
        return self._allSpikeClusters
    
    @property
    def allSpikeTimestamps(self):
        return self._allSpikeTimestamps

    @property
    def uniqueSpikeClusters(self):
        return self._uniqueSpikeClusters

    @property
    def datasets(self):
        return self._datasets
    
    def __iter__(self):
        self._index = 0
        return self
    
    def __next__(self):
        if self._index < len(self._units):
            unit = self._units[self._index]
            self._index += 1
            return unit
        else:
            self._index = 0 # reset the index
            raise StopIteration
        
    def __getitem__(self, index):
        if type(index) == int:
            return self._units[index]
        elif type(index) in (list, np.ndarray):
            return np.array(self._units)[index].tolist()
        elif type(index) == slice:
            return self._units[index.start: index.stop: index.step]

    def __len__(self):
        return len(self._units)