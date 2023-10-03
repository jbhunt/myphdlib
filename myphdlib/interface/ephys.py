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
                dt = np.diff(window)
                fr = M.flatten() / dt
                mu = np.mean(fr)
                sigma = np.std(fr)
            else:
                fr = M.flatten() / binsize
                mu = fr.mean()
                sigma = fr.std()

        return mu, sigma

    @property
    def index(self):
        """
        """

        if self._index is None:
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
            spikeIndices = np.where(self.session.population.allSpikeClusters == self.cluster)[0]
            self._timestamps = self.session.population.allSpikeTimestamps[spikeIndices]

        return self._timestamps
    
    @property
    def utype(self):
        """
        """

        if self._utype is None:

            #
            if self.session.population.datasets[('masks', 'vr')] is not None:
                vr = self.session.population.datasets[('masks', 'vr')][self.index]
            else:
                vr = None

            #
            if self.session.population.datasets[('masks', 'sr')] is not None:
                sr = self.session.population.datasets[('masks', 'sr')][self.index]
            else:
                sr = None

            #
            if vr == True and sr == True:
                self._utype = 'vm'
            elif vr == True and sr == False:
                self._utype = 'vr'
            elif vr == False and sr == True:
                self._utype = 'sr'
            elif vr == False and sr == False:
                self._utype = 'nr'
            else:
                self._utype = 'ud' # undefined

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
                self._ksl = self.session.population.datasets[('metrics', 'ksl')][self.index]

        return self._ksl

    # TODO: Code these properties

    # Probability visually responsive (left)
    @property
    def pvrl(self):
        return
    
    # Probability visually responsive (right)
    @property
    def pvrr(self):
        return

    # Probability saccade related (nasal)
    @property
    def psrn(self):
        return

    # Probability saccade related (temporal)
    @property
    def psrt(self):
        return

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

        #
        for cluster in np.unique(self._allSpikeClusters):
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