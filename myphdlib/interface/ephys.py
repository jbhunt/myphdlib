import numpy as np
from myphdlib.general.toolkit import psth2

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
        self._dsi = None
        self._type = None
        self._index = None
        self._stability = None
        self._contamination = None
        self._isHighQuality = None
        self._isVisualUnit = None
        self._isMotorUnit = None

        return
    
    def describe(
        self,
        event,
        window=(-0.1, 0)
        ):

        t, M = psth2(
            event,
            self.timestamps,
            window=window,
            binsize=None
        )
        dt = np.diff(window)
        fr = M.flatten() / dt
        mu = np.mean(fr)
        sigma = np.std(fr)

        return mu, sigma
    
    @property
    def timestamps(self):
        return self._timestamps
    
    @property
    def cluster(self):
        return self._cluster
    
    @property
    def timestamps(self):
        """
        """

        if self._timestamps is None:
            if self._session.hasGroup('spikes') == False:
                return None
            spikes = self._session.load('spikes/timestamps')
            clusters = self._session.load('spikes/clusters')
            self._timestamps = spikes[np.where(clusters == self.cluster)[0]]

        return self._timestamps
    
    @property
    def dsi(self):
        """
        """

        if self._dsi is None:
            if self._session.hasGroup('analysis/population/dsi') == False:
                return None
            sample = self._session.load('analysis/population/dsi')
            self._dsi = sample[self.index]

        return self._dsi
    
    @property
    def type(self):
        """
        """

        if self._type is None:
            if self._session.hasGroup('analysis/typing/visual') == False:
                return None
            mask = self._session.load('analysis/typing/visual')
            if mask[self.index]:
                self._type = 'visual'
            else:
                self._type = 'unclassified'

        return self._type
    
    @property
    def index(self):
        """
        """

        if self._index is None:
            uids = self._session.load('population/uids')
            if self.cluster in uids:
                self._index = np.where(uids == self.cluster)[0].item()
            else:
                return None

        return self._index
    
    @property
    def isHighQuality(self):
        """
        """

        if self._isHighQuality is None:
            sample = self._session.load('population/filters/quality')
            self._isHighQuality = sample[self.index]

        return self._isHighQuality
    
    @property
    def stability(self):
        """
        """

        if self._stability is None:
            sample = self._session.load('population/metrics/presence_ratio')
            self._stability = sample[self.index]

        return self._stability
    
    @property
    def contamination(self):
        """
        """

        if self._contamination is None:
            sample = self._session.load('population/metrics/isi_violation_rate')
            self._contamination = sample[self.index]
    
        return self._contamination 
    
    @property
    def isVisualUnit(self):
        """
        """

        if self._isVisualUnit is None:
            sample = self._session.load('population/filters/visual')
            self._isVisualUnit = sample[self.index]
        
        return self._isVisualUnit
    
    @property
    def isMotorUnit(self):
        """
        """

        if self._isMotorUnit is None:
            sample = self._session.load('population/filters/motor')
            self._isMotorUnit = sample[self.index]
        
        return self._isMotorUnit

class Population():
    """
    """

    def __init__(self, session, autoload=True):
        """
        """

        self._session = session
        self._units = None
        if autoload:
            self._loadSingleUnitData()
        self._index = 0

        return
    
    def index(self, cluster):
        """
        """

        for unit in self._units:
            if unit.cluster == cluster:
                return unit

        return
    
    def _loadSingleUnitData(self):
        """
        """

        #
        if self._units is not None:
            del self._units
        self._units = list()

        #
        clusters = self._session.load('spikes/clusters')

        #
        for cluster in np.unique(clusters):
            unit = SingleUnit(self._session, cluster)
            self._units.append(unit)

        return
    
    def __iter__(self):
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