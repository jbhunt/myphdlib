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
        tag='JH-DATA-',
        mount=True
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
        return
    
    @property
    def session(self):
        return self._session
    
    @property
    def unit(self):
        return self._unit

    @property
    def ukey(self):
        return self._ukey

    @ukey.setter
    def ukey(self, value):
        date, animal, cluster = value
        if self._session is None or str(self._session.date) != date:
            self._session = self._factory.produce(
                dates=(date,),
                animals=(animal,),
            ).pop()
        self._unit = self._session.population.indexByCluster(cluster)
        self._ukey = value
        return

    @property
    def ukeys(self):
        return self._ukeys

    def loadUnitKeys(
        self,
        experiments=('Mlati',),
        minimumBaselineLevel=0.5,
        minimumResponseAmplitude=5,
        minimumPeakLatency=0.05,
        responseWindow=(0, 0.5),
        baselineWindow=(-0.2, 0),
        ):
        """
        """

        self._ukeys = list()
        sessions = self._factory.produce(experiment=experiments)
        n = len(sessions)
        for i, session in enumerate(sessions):

            end = '\r' if i + 1 == n else None
            print(f'Filtering units from session {i + 1} out of {n}', end=end)

            #
            if session.probeTimestamps is None:
                continue

            #
            peths = {
                ('rProbe', 'left'): session.load('peths/rProbe/dg/left/fr'),
                ('rProbe', 'right'): session.load('peths/rProbe/dg/right/fr'),
            }
            peth, metadata = session.load('peths/rProbe/dg/left/fr', returnMetadata=True)
            clusterNumbers = session.load('population/clusters')
            t = metadata['t']


            # Define the baseline and response windows
            binIndicesForBaselineWindow = np.where(np.logical_and(
                t >= baselineWindow[0],
                t <= baselineWindow[1]
            ))[0]
            binIndicesForResponseWindow = np.where(np.logical_and(
                t >= responseWindow[0],
                t <= responseWindow[1]
            ))[0]

            #
            nUnits = peths[('rProbe', 'left')].shape[0]

            # Iterate over units
            for iUnit in range(nUnits):

                #
                lowestBaselineLevel = np.max([
                    peths[('rProbe', 'left')][iUnit, binIndicesForBaselineWindow].mean(),
                    peths[('rProbe', 'right')][iUnit, binIndicesForBaselineWindow].mean(),
                ])
                if lowestBaselineLevel < minimumBaselineLevel:
                    continue

                #
                greatestPeakAmplitude = np.max([
                    np.max(np.abs(peths[('rProbe', 'left')][iUnit, binIndicesForResponseWindow] - \
                        peths[('rProbe', 'left')][iUnit, binIndicesForBaselineWindow].mean())),
                    np.max(np.abs(peths[('rProbe', 'right')][iUnit, binIndicesForResponseWindow] - \
                        peths[('rProbe', 'right')][iUnit, binIndicesForBaselineWindow].mean())),
                ])
                if greatestPeakAmplitude < minimumResponseAmplitude:
                    continue

                shortestPeakLatency = np.min([
                    t[np.argmax(np.abs(peths[('rProbe', 'left')][iUnit]))],
                    t[np.argmax(np.abs(peths[('rProbe', 'right')][iUnit]))],
                ])
                if shortestPeakLatency < minimumPeakLatency:
                    continue
                
                #
                ukey = (
                    str(session.date),
                    session.animal,
                    clusterNumbers[iUnit]
                )
                self._ukeys.append(ukey)

        return

    def lookupUnitKey(self, ukey):
        """
        """

        date, animal, cluster = ukey
        unitIndex = None
        for iUnit, (d, a, c) in enumerate(self.ukeys):
            if d == date and a == animal and c == cluster:
                unitIndex = iUnit
                break

        return unitIndex

def findOverlappingUnits(ukeys1, hdf):
    """
    """

    with h5py.File(hdf, 'r') as stream:
        dates = np.array(stream['date'])
        animals = np.array(stream['animal'])
        clusters = np.array(stream['unitNumber'])
    ukeys2 = [
        (date.item().decode(), animal.item().decode(), cluster.item()) for (date, animal, cluster)
            in zip(dates, animals, clusters)
    ]

    mask = np.full(len(ukeys2), False)
    for i, ukey in enumerate(ukeys2):
        if ukey in ukeys1:
            mask[i] = True

    return mask