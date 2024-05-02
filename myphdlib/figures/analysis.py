import numpy as np
from myphdlib.interface.factory import SessionFactory
from scipy.optimize import curve_fit as fitCurve
import h5py

def g(x, a, mu, sigma, d):
    """
    """

    return a * np.exp(-((x - mu) / 4 / sigma) ** 2) + d

class GaussianMixturesModel():
    """
    """

    def __init__(self, k=1, maxfev=1000000):
        """
        """
        self._popt = None
        self._k = k
        self._maxfev = maxfev
        return

    def fit(self, x, y, p0=None, bounds=None):
        """
        """

        nParams = self.k * 3 + 1
        if p0 is not None and len(p0) != nParams:
            raise Exception('Invalid number of initial parameter values')
        if p0 is None:
            p0 = np.ones(nParams)
        if bounds is None:
            bounds = np.vstack([
                np.full(nParams, -np.inf),
                np.full(nParams, +np.inf)
            ])
        self._popt, pcov = fitCurve(
            self.f,
            x,
            y,
            p0=p0,
            bounds=bounds,
            maxfev=self._maxfev
        )

        return

    def predict(self, x):
        """
        """

        if self._popt is None:
            raise Exception('Optimal parameters undefined')

        #
        y = self.f(x, *self._popt)

        return y

    @property
    def f(self):
        def inner(x, d, *params):
            if self.k == 1:
                A, B, C = [params[0]], [params[1]], [params[2]]
            else:
                A, B, C = np.split(np.array(params), 3)
            y = np.zeros(x.size)
            for i in range(self.k):
                a, b, c = A[i], B[i], C[i]
                y += a * np.exp(-((x - b) / 4 / c) ** 2)
            y += d
            return y
        return inner

    @property
    def k(self):
        return self._k
    @k.setter
    def k(self, value):
        self._k = value

class AnalysisBase():
    """
    """

    def __init__(
        self, 
        ukey=None,
        hdf=None,
        tag='JH-DATA-',
        mount=False,
        experiments=('Mlati',),
        ):
        """
        """

        self._ukeys = None
        self._ukey = None
        if mount:
            self._factory = SessionFactory(mount=tag)
        else:
            self._factory = SessionFactory(tag=tag)
        self._session = None
        self._unit = None
        if ukey is not None:
            self.ukey = ukey
        if hdf is not None:
            self._hdf = hdf

        #
        self._loadSessions(experiments)
        self._loadUnitKeys()

        return

    @property
    def session(self):
        return self._session

    @property
    def sessions(self):
        return self._sessions
    
    @property
    def unit(self):
        return self._unit

    @property
    def ukey(self):
        return self._ukey

    @ukey.setter
    def ukey(self, value):
        date, animal, cluster = value
        if self.session is None or str(self.session.date) != date:
            self._session = None
            for session in self.sessions:
                if str(session.date) == date and session.animal == animal:
                    self._session = session
                    break
        self._unit = self.session.population.indexByCluster(cluster)
        self._ukey = value
        return

    @property
    def ukeys(self):
        return self._ukeys
    
    @property
    def hdf(self):
        return self._hdf

    @property
    def iUnit(self):
        """
        """

        iUnit = None
        for i, (date, animal, cluster) in enumerate(self.ukeys):
            if date == self.ukey[0] and animal == self.ukey[1] and cluster == self.ukey[2]:
                iUnit = i
                break

        if iUnit is None:
            raise Exception('Could not determine unit index')

        return iUnit
    
    def _loadSessions(
        self,
        experiments
        ):
        """
        """

        self._sessions = self._factory.produce(experiment=experiments)

        return
    
    def _loadUnitKeys(
        self,
        **kwargs_,
        ):
        """
        """

        if self.hdf is not None:
            with h5py.File(self.hdf, 'r') as stream:
                dates = np.array(stream['ukeys/date'])
                animals = np.array(stream['ukeys/animal'])
                clusters = np.array(stream['ukeys/cluster'])
            self._ukeys = [
                (date.item().decode(), animal.item().decode(), int(cluster.item().decode())) for (date, animal, cluster)
                    in zip(dates, animals, clusters)
            ]
            return

        kwargs = {
            'maximumAmplitudeCutoff': 0.1,
            'minimumPresenceRatio': 0.9,
            'maximumIsiViolations': 0.5,
            'maximumProbabilityValue': 0.01,
            'minimumFiringRate': 0.2,
        }
        kwargs.update(kwargs_)

        #
        self._ukeys = list()
        nSessions = len(self._sessions)

        #
        for i, session in enumerate(self._sessions):

            end = '\r' if i + 1 != nSessions else None
            print(f'Filtering units from session {i + 1} out of {nSessions}', end=end)

            #
            if session.probeTimestamps is None:
                continue

            #
            amplitudeCutoff = session.load('metrics/ac')
            presenceRatio = session.load('metrics/pr')
            isiViolations = session.load('metrics/rpvr')
            firingRate = session.load('metrics/fr')
            probabilityValues = np.nanmin(np.vstack([
                session.load('zeta/probe/left/p'),
                session.load('zeta/probe/right/p')
            ]), axis=0)

            #
            nUnits = len(session.population)
            clusterNumbers = np.unique(session.load('spikes/clusters'))
            qualityLabels = session.load('metrics/ql')
            for iUnit in range(nUnits):

                # First check if the unit has a low p-value from the ZETA test
                if kwargs['maximumProbabilityValue'] is not None and probabilityValues[iUnit] > kwargs['maximumProbabilityValue']:
                    continue

                #
                if kwargs['minimumFiringRate'] is not None and firingRate[iUnit] <  kwargs['minimumFiringRate']:
                    continue

                # Check if unit was labeled as "good"
                if qualityLabels[iUnit] == 0:

                    #
                    if kwargs['minimumPresenceRatio'] is not None and  presenceRatio[iUnit] < kwargs['minimumPresenceRatio']:
                        continue

                    #
                    if kwargs['maximumAmplitudeCutoff'] is not None and amplitudeCutoff[iUnit] > kwargs['maximumAmplitudeCutoff']:
                        continue

                    if kwargs['maximumIsiViolations'] is not None and isiViolations[iUnit] > kwargs['maximumIsiViolations']:
                        continue
                
                #
                ukey = (
                    str(session.date),
                    session.animal,
                    clusterNumbers[iUnit]
                )
                self._ukeys.append(ukey)

        return

    def _indexUnitKey(self, ukey):
        """
        """

        date, animal, cluster = ukey
        unitIndex = None
        for iUnit, (d, a, c) in enumerate(self.ukeys):
            if d == date and a == animal and c == cluster:
                unitIndex = iUnit
                break

        return unitIndex

    def _intersectUnitKeys(
        self,
        ukeys,
        ):
        """
        """

        with h5py.File(self.hdf, 'r') as stream:
            dates = np.array(stream['ukeys/date'])
            animals = np.array(stream['ukeys/animal'])
            clusters = np.array(stream['ukeys/cluster'])
        referenceUnitKeys = [
            (date.item().decode(), animal.item().decode(), int(cluster.item().decode())) for (date, animal, cluster)
                in zip(dates, animals, clusters)
        ]


        #
        mask = np.full(len(referenceUnitKeys), False)
        for i, ukey in enumerate(referenceUnitKeys):
            if ukey in ukeys:
                mask[i] = True

        return mask

    def _saveLargeDataset(
        self,
        hdf,
        path,
        dataset,
        nUnitsPerChunk=100
        ):
        """
        """

        mask = self._intersectUnitKeys(self.ukeys)
        shape = dataset.shape[1:]

        with h5py.File(hdf, 'a') as stream:

            # Re-sample PETHs
            if path in stream:
                del stream[path]
            ds = stream.create_dataset(
                path,
                shape=[mask.size, *shape],
                dtype=np.float64,
            )
            for start in np.arange(0, mask.size, nUnitsPerChunk):
                stop = start + nUnitsPerChunk
                if stop >= mask.size:
                    stop = mask.size
                    data = np.full([mask.size - start, *shape], np.nan)
                else:
                    data = np.full([nUnitsPerChunk, *shape], np.nan)
                # iUnit1 - Unit index for filtered units
                # iUnit2 - Unit index for unfiltered units
                # iUnit3 - Unit index for target unit within data chunk
                for iUnit1, iUnit2 in enumerate(np.where(mask)[0]):
                    indices = np.where(np.arange(start, stop) == iUnit2)[0]
                    if len(indices) == 1:
                        iUnit3 = indices.item()
                    else:
                        continue
                    data[iUnit3] = dataset[iUnit1]
                ds[start: stop, :, :] = data

        return
    
    def _initializeStore(
        self,
        filename
        ):
        """
        """

        #
        with h5py.File(filename, 'w') as stream:
            date = np.array([
                ukey[0] for ukey in self.ukeys
            ], dtype='S').reshape(-1, 1)
            animal = np.array([
                ukey[1] for ukey in self.ukeys
            ], dtype='S').reshape(-1, 1)
            cluster = np.array([
                ukey[2] for ukey in self.ukeys
            ], dtype='S').reshape(-1, 1)
            datasets = {
                'ukeys/date': date,
                'ukeys/animal': animal,
                'ukeys/cluster': cluster
            }
            for path, data in datasets.items():
                stream.create_dataset(
                    path,
                    shape=data.shape,
                    dtype=data.dtype,
                    data=data
                )

        #
        self._hdf = filename

        return

    def createFilter(
        self,
        minimumResponseLatency=0.03,
        minimumResponseAmplitude=5,
        ):
        """
        """

        # Make sure the model data is loaded
        try:
            assert hasattr(self, 'model')
            assert ('params' in self.model.keys() or 'params1' in self.model.keys())
        except AssertionError:
            raise Exception('Model data is not available') from None

        # Create the filter
        nUnits = len(self.ukeys)
        self.filter = np.full(nUnits, False)
        for iUnit in range(nUnits):
            params = self.model['params'][iUnit]
            mask = np.invert(np.isnan(params))
            if np.all(np.isnan(params)):
                continue
            abcd = params[mask]
            abc, d = abcd[:-1], abcd[-1]
            A, B, C = np.split(abc, 3)
            if np.max(np.abs(A)) >= minimumResponseAmplitude and B.min() >= minimumResponseLatency:
                self.filter[iUnit] = True

        return