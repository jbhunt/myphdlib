import numpy as np
import pathlib as pl
from myphdlib.interface.factory import SessionFactory
from scipy.optimize import curve_fit as fitCurve
import h5py

def convertGratingMotionToSaccadeDirection(
    gratingMotion=-1,
    referenceEye='left'
    ):
    """
    """

    saccadeDirection = None
    if referenceEye == 'left':
        if gratingMotion == -1:
            saccadeDirection = 'nasal'
        else:
            saccadeDirection = 'temporal'
    elif referenceEye == 'right':
        if gratingMotion == -1:
            saccadeDirection = 'temporal'
        else:
            saccadeDirection = 'nasal'

    return saccadeDirection

def convertSaccadeDirectionToGratingMotion(
    saccadeDirection,
    referenceEye='left'
    ):
    """
    """

    gratingMotion = None
    if referenceEye == 'left':
        if saccadeDirection == 'nasal':
            gratingMotion = -1
        else:
            gratingMotion = +1
    else:
        if saccadeDirection == 'nasal':
            gratingMotion = +1
        else:
            gratingMotion = -1

    return gratingMotion

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

class Namespace():
    """
    """

    def __init__(self):
        """
        """

        self._data = {

            # Peri-probe time histograms
            'ppths/pref/real/extra': None,
            'ppths/pref/real/peri': None,
            'ppths/pref/fictive/extra': None,
            'ppths/pref/fictive/peri': None,
            'ppths/null/real/extra': None,
            'ppths/null/real/peri': None,
            'ppths/null/fictive/extra': None,
            'ppths/null/fictive/peri': None,
            'ppths/pref/real/resampled': None,
            'ppths/pref/fictive/resampled': None,
            'ppths/null/real/resampled': None,
            'ppths/null/fictive/resampled': None,

            # Model (GMM) parameters
            'params/pref/real/extra': None,
            'params/pref/real/peri': None,
            'params/pref/fictive/extra': None,
            'params/pref/fictive/peri': None,
            'params/null/real/extra': None,
            'params/null/real/peri': None,
            'params/null/fictive/extra': None,
            'params/null/fictive/peri': None,

            # Mean and standard deviation of baseline FR
            'stats/pref/real/extra': None,
            'stats/pref/real/peri': None,
            'stats/pref/fictive/extra': None,
            'stats/pref/fictive/peri': None,
            'stats/null/real/extra': None,
            'stats/null/real/peri': None,
            'stats/null/fictive/extra': None,
            'stats/null/fictive/peri': None,

            # Peri-saccade time histograms
            'psths/nasal/real': None,
            'psths/temporal/real': None,
            'psths/nasal/fictive': None,
            'psths/temporal/fictive': None,
            'psths/pref/real': None,
            'psths/null/real': None,
            'psths/pref/fictive': None,
            'psths/null/fictive': None,

            # Response terms
            'terms/pref/real/extra': None,
            'terms/pref/real/mixed': None,
            'terms/pref/real/saccade': None,
            'terms/pref/real/peri': None,
            'terms/null/real/extra': None,
            'terms/null/real/mixed': None,
            'terms/null/real/saccade': None,
            'terms/null/real/peri': None,
            'terms/pref/fictive/extra': None,
            'terms/pref/fictive/mixed': None,
            'terms/pref/fictive/saccade': None,
            'terms/pref/fictive/peri': None,
            'terms/null/fictive/extra': None,
            'terms/null/fictive/mixed': None,
            'terms/null/fictive/saccade': None,
            'terms/null/fictive/peri': None,

            # Modulation index
            'mi/pref/real': None,
            'mi/pref/fictive': None,
            'mi/null/real': None,
            'mi/null/fictive': None,

            # Boostrapped null samples
            'samples/pref/real': None,
            'samples/pref/fictive': None,
            'samples/null/real': None,
            'samples/null/fictive': None,

            # p-values from boostrap
            'p/pref/real': None,
            'p/pref/fictive': None,
            'p/null/real': None,
            'p/null/fictive': None,

            # Direction selectivity index
            'dsi/probe/extra': None,
            'dsi/probe/peri': None,
            'dsi/saccade/real': None,
            'dsi/saccade/fictive': None,
            'dsi/bar': None,
            'dsi/dg': None,

            # Global variables
            'globals/factor': None, # Scaling factor (standard deviation of baseline firing rate for preferred direction)
            'globals/preference': None, # Preferred direction of motion
            'globals/ssi': None, # Saccade selectivity index
            'globals/labels': None, # Response complexity categories
            'globals/windows': None, # Peri-saccadic time windows

        }

        return

    @property
    def data(self):
        return self._data

    @property
    def paths(self):
        return list(self.data)

    def __getitem__(self, name):
        if name in self.data.keys():
            return self.data[name]
        else:
            raise Exception(f'{name} is not available')

    def __setitem__(self, name, value):
        self.data[name] = value


class AnalysisBase():
    """
    """

    def __init__(
        self, 
        ukey=None,
        hdf=None,
        event='probe',
        **kwargs_
        ):
        """
        """


        self._ukeys = None
        self._ukey = None
        self._session = None
        self._unit = None
        if ukey is not None:
            self.ukey = ukey
        self._hdf = hdf
        self.ns = Namespace()

        #
        kwargs = {
            'tag': 'JH-DATA-',
            'mount': False,
            'experiment': ('Mlati',),
            'dates': (None, None),
            'cohort': None,
            'animals': None,
        }
        kwargs.update(kwargs_)
        if kwargs['mount']:
            self._factory = SessionFactory(mount=kwargs['tag'])
        else:
            self._factory = SessionFactory(tag=kwargs['tag'])
        subset = {
            'experiment': kwargs['experiment'],
            'dates': kwargs['dates'],
            'cohort': kwargs['cohort'],
            'animals': kwargs['animals']
        }
        self._loadSessions(**subset)
        self._loadUnitKeys(event)

        # Globals
        self.tProbe = None
        self.tSaccade = None
        self.windows = None
        self.factor = None
        self.filter = None
        self.labels = None
        self.preference = None

        return

    def loadNamespace(
        self,
        ):
        """
        """

        if pl.Path(self.hdf).exists() == False:
            raise Exception('h5 file does not exist')

        mask = self._intersectUnitKeys(self.ukeys)
        with h5py.File(self.hdf, 'r') as stream:
            for path in self.ns.paths:
                if path in stream.keys():

                    #
                    ds = stream[path]
                    data = np.array(ds)

                    #
                    if 'ppths' in pl.Path(path).parts and self.tProbe is None:
                        if 't' in ds.attrs.keys():
                            self.tProbe = ds.attrs['t']
                    if 'psths' in pl.Path(path).parts and self.tSaccade is None:
                        if 't' in ds.attrs.keys():
                            self.tSaccade = ds.attrs['t']
            
                    # Assign global variables to attributes
                    if 'globals' in pl.Path(path).parts:
                        name = path.split('/')[-1]
                        self.__setattr__(name, data)
                        self.ns[path] = data

                    #
                    else:
                        if data.size == data.shape[0]:
                            data = data.flatten()
                        try:
                            self.ns[path] = data[mask]
                        except:
                            print(f'Warning: failed to load {path}')
                            pass

        return

    def saveNamespace(
        self,
        ):
        """
        """

        if pl.Path(self.hdf).exists() == False:
            raise Exception('h5 file does not exist')

        mask = self._intersectUnitKeys(self.ukeys)
        with h5py.File(self.hdf, 'a') as stream:
            for path in self.ns.paths:

                #
                data = self.ns[path]
                if data is None:
                    continue

                #
                if path in stream.keys():
                    del stream[path]

                #
                if 'globals' in pl.Path(path).parts:
                    
                    #
                    ds = stream.create_dataset(
                        path,
                        data.shape,
                        data.dtype,
                        data=data,
                    )

                else:

                    # Re-shape into single column
                    if len(data.shape) == 1:
                        data = data.reshape(-1, 1)

                    #
                    filled = np.full([mask.size, *data.shape[1:]], np.nan)
                    filled[mask] = data

                    #
                    try:
                        ds = stream.create_dataset(
                            path,
                            filled.shape,
                            filled.dtype,
                            data=filled,
                        )
                    except:
                        import pdb; pdb.set_trace()

                    #
                    metadata = None
                    if 'ppths' in pl.Path(path).parts and self.tProbe is not None:
                        metadata = {'t': self.tProbe}
                    if 'psths' in pl.Path(path).parts and self.tSaccade is not None:
                        metadata = {'t': self.tSaccade}
                    if metadata is not None:
                        for k, v in metadata.items():
                            ds.attrs[k] = v

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
            if self.session is None:
                raise Exception(f'Could not determine session from unit key: {value}')
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
        **kwargs
        ):
        """
        """

        self._sessions = self._factory.produce(**kwargs)

        return
    
    # TODO: Implement a filter for excluding units that response too fast
    def _loadUnitKeys(
        self,
        event='probe',
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
            'minimumResponseLatency': 0.025,
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
            if session.hasDataset(f'zeta/{event}/left/p') == False:
                probabilityValues = np.full(len(firingRate), np.nan)
                responseLatency = np.full(len(firingRate), np.nan)
            else:
                probabilityValues = np.nanmin(np.vstack([
                    session.load(f'zeta/{event}/left/p'),
                    session.load(f'zeta/{event}/right/p')
                ]), axis=0)
                responseLatency = np.nanmin(np.vstack([
                    session.load(f'zeta/{event}/left/latency'),
                    session.load(f'zeta/{event}/right/latency'),
                ]), axis=0)

            #
            nUnits = len(session.population)
            clusterNumbers = np.unique(session.load('spikes/clusters'))
            qualityLabels = session.load('metrics/ql')
            for iUnit in range(nUnits):

                # First check if the unit has a low p-value from the ZETA test
                if kwargs['maximumProbabilityValue'] is not None and probabilityValues[iUnit] > kwargs['maximumProbabilityValue']:
                    continue

                # Exclude noise units (based on firing rate)
                if kwargs['minimumFiringRate'] is not None and firingRate[iUnit] <  kwargs['minimumFiringRate']:
                    continue

                # Exclude incomplete units (based on presence ratio)
                if kwargs['minimumPresenceRatio'] is not None and  presenceRatio[iUnit] < kwargs['minimumPresenceRatio']:
                    continue

                # Exclude incomplete units (based on amplitude cutoff)
                if kwargs['maximumAmplitudeCutoff'] is not None and amplitudeCutoff[iUnit] > kwargs['maximumAmplitudeCutoff']:
                    continue

                # Exclude contaminated units
                if kwargs['maximumIsiViolations'] is not None and isiViolations[iUnit] > kwargs['maximumIsiViolations']:
                    continue

                # Exclude units that response too fast
                # NOTE: These are likely the artificial units produced by the photologic device
                if kwargs['minimumResponseLatency'] is not None and responseLatency[iUnit] < kwargs['minimumResponseLatency']:
                    continue

                # Exclude units Anna identified as noise or multi-unit
                # NOTE: 0 codes noise units and 1 codes multi-units
                if qualityLabels is not None and qualityLabels[iUnit] in (0, 1):
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

    def _getSessionFromUnitKey(
        self,
        ukey
        ):
        """
        """

        date, animal, cluster = ukey
        session = None
        for session_ in self.sessions:
            if str(session_.date) == date and session_.animal == animal:
                session = session_
                break

        if session is None:
            raise Exception('Could not get session')

        return session

    def _matchUnitKey(self, test, unitkeys):
        """
        """

        result = False
        for ukey in unitkeys:
            matched = np.all([
                str(ukey[0]) == str(test[0]),
                ukey[1] == test[1],
                ukey[2] == test[2]
            ])
            if matched:
                result = True
                break

        return result

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