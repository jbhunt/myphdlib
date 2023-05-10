import numpy as np
from myphdlib.general.toolkit import psth2

class SingleUnit():
    """
    """

    def __init__(
        self,
        timestamps=None,
        cluster=None
        ):
        """
        """
        self._timestamps = timestamps
        self._cluster = cluster
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
        spikes = self._session.load('spikes/timestamps')
        clusters = self._session.load('spikes/clusters')

        #
        for cluster in np.unique(clusters):
            timestamps = spikes[clusters == cluster]
            unit = SingleUnit(timestamps, cluster)
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