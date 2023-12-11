import h5py
import numpy as np
from dotmap import DotMap
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
        self._kilosortLabel = None
        self._visualResponseLatency = None
        self._visualResponseAmplitude = None
        self._visualResponseProbability = None
        self._presenceRatio = None
        self._refractoryPeriodViolationRate = None
        self._amplitudeCutoff = None
        self._deltaResponseValue = None
        self._deltaResponseProbability = None
        self._visualResponseCurve = None

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

    def describe2(
        self,
        eventTimestamps,
        baselineWindowBoundaries=(-5, -2),
        windowSize=0.5,
        nRuns=100,
        binsize=None,
        ):
        """
        """

        if binsize is None:
            dt = windowSize
        else:
            dt = binsize

        #
        samples = list()
        for i in range(nRuns):

            #
            baselineWindowEdge = np.random.uniform(
                low=baselineWindowBoundaries[0] + windowSize,
                high=baselineWindowBoundaries[1],
                size=1
            ).item()
            baselineWindow = np.array([
                baselineWindowEdge,
                baselineWindowEdge + windowSize
            ])

            #
            t, M = psth2(
                eventTimestamps,
                self.timestamps,
                window=baselineWindow,
                binsize=binsize,
            )
            fr = M.mean(0) / dt
            samples.append([fr.mean(), fr.std()])

        #
        samples = np.around(np.array(samples), 2)

        return round(samples[:, 0].mean(), 2), round(samples[:, 1].mean(), 2)

    def describe3(
        self,
        eventTimestamps,
        baselineWindow=(-5, -2),
        binsize=None,
        ):
        """
        Get the mean and SD of the mean FR in a baseline window
        """

        if binsize is None:
            dt = np.diff(baselineWindow).item()
        else:
            dt = binsize

        #
        t, M = psth2(
            eventTimestamps,
            self.timestamps,
            baselineWindow,
            binsize=binsize
        )
        fr = M.mean(0) / dt

        return round(fr.mean(), 2), round(fr.std(), 2)

    def peth(
        self,
        eventTimestamps,
        responseWindow=(-0.3, 0.5),
        baselineWindow=(-5, -2),
        binsize=0.02,
        standardize=True,
        nRuns=100,
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
        mu, sigma = self.describe2(
            eventTimestamps,
            baselineWindowBoundaries=baselineWindow,
            windowSize=0.5,
            binsize=binsize,
            nRuns=nRuns,
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

    # Kilosort label
    @property
    def kilosortLabel(self):
        if self._kilosortLabel is None:
            if self.session.population.datasets[('metrics', 'ksl')] is not None:
                label = self.session.population.datasets[('metrics', 'ksl')][self.index]
                self._kilosortLabel = 'm' if label == 0 else 'g'

        return self._kilosortLabel

    # Response latency (onset-to-peak)
    @property
    def visualResponseLatency(self):
        """
        """

        if self._visualResponseLatency is None:
            vrl = {
                'left': None,
                'right': None
            }
            keys = (
                ('zeta', 'probe', 'left', 'latency'),
                ('zeta', 'probe', 'right', 'latency')
            )
            for key in keys:
                if self.session.population.datasets[key] is None:
                    continue
                probeDirection = key[2]
                vrl[probeDirection] = round(self.session.population.datasets[key][self.index], 3)
            self._visualResponseLatency = DotMap(vrl)

        return self._visualResponseLatency

    # Response amplitude (z-scored)
    @property
    def visualResponseAmplitude(self):
        """
        """

        if self._visualResponseAmplitude is None:
            vra = {
                'left': None,
                'right': None
            }
            keys = (
                ('metrics', 'vra', 'left'),
                ('metrics', 'vra', 'right')
            )
            for key in keys:
                if self.session.population.datasets[key] is None:
                    continue
                probeDirection = key[-1]
                vra[probeDirection] = round(self.session.population.datasets[key][self.index], 3)
            self._visualResponseAmplitude = DotMap(vra)

        return self._visualResponseAmplitude

    # Probability of being a visually-responsive unit
    @property
    def visualResponseProbability(self):
        """
        """

        if self._visualResponseProbability is None:
            vrp = {
                'left': None,
                'right': None
            }
            keys = (
                ('zeta', 'probe', 'left', 'p'),
                ('zeta', 'probe', 'right', 'p')
            )
            for key in keys:
                if self.session.population.datasets[key] is None:
                    continue
                probeDirection = key[2]
                vrp[probeDirection] = round(1 - self.session.population.datasets[key][self.index], 3)
            self._visualResponseProbability = DotMap(vrp)

        return self._visualResponseProbability

    #
    @property
    def visualResponseCurve(self):
        """
        """

        if self._visualResponseCurve is None:
            vrc = {
                'left': None,
                'right': None
            }
            keys = (
                ('psths', 'probe', 'left'),
                ('psths', 'probe', 'right')
            )
            for key in keys:
                if self.session.population.datasets[key] is None:
                    continue
                probeDirection = key[2]
                vrc[probeDirection] = self.session.population.datasets[key][self.index]
            self._visualResponseCurve = DotMap(vrc)

        return self._visualResponseCurve
    
    #
    @property
    def deltaResponseValue(self):
        """
        """

        if self._deltaResponseValue is None:
            dr = {
                'left': None,
                'right': None,
            }
            keys = (
                ('metrics', 'dr', 'left', 'x'),
                ('metrics', 'dr', 'right', 'x')
            )
            for key in keys:
                if self.session.population.datasets[key] is None:
                    continue
                probeDirection = key[2]
                dr[probeDirection] = round(self.session.population.datasets[key][self.index], 3)
            self._deltaResponseValue = DotMap(dr)

        return self._deltaResponseValue

    #
    @property
    def deltaResponseProbability(self):
        """
        """

        if self._deltaResponseProbability is None:
            ps = {
                'left': None,
                'right': None,
            }
            keys = (
                ('metrics', 'dr', 'left', 'p'),
                ('metrics', 'dr', 'right', 'p')
            )
            for key in keys:
                if self.session.population.datasets[key] is None:
                    continue
                probeDirection = key[2]
                ps[probeDirection] = round(1 - self.session.population.datasets[key][self.index], 3)
            self._deltaResponseProbability = DotMap(ps)

        return self._deltaResponseProbability

    # Presence ratio metric
    @property
    def presenceRatio(self):
        """
        """

        if self._presenceRatio is None:
            keys = (
                ('metrics', 'pr'),
            )
            for key in keys:
                if self.session.population.datasets[key] is None:
                    continue
                self._presenceRatio = self.session.population.datasets[key][self.index]

        return self._presenceRatio

    #
    @property
    def refractoryPeriodViolationRate(self):
        """
        """

        if self._refractoryPeriodViolationRate is None:
            keys = (
                ('metrics', 'rpvr'),
            )
            for key in keys:
                if self.session.population.datasets[key] is None:
                    continue
                self._refractoryPeriodViolationRate = self.session.population.datasets[key][self.index]

        return self._refractoryPeriodViolationRate

    #
    @property
    def amplitudeCutoff(self):
        """
        """

        if self._amplitudeCutoff is None:
            keys = (
                ('metrics', 'ac'),
            )
            for key in keys:
                if self.session.population.datasets[key] is None:
                    continue
                self._amplitudeCutoff = self.session.population.datasets[key][self.index]

        return self._amplitudeCutoff

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
            ('metrics', 'pr'): None,
            ('metrics', 'rpvr'): None,
            ('metrics', 'ac'): None,
            ('metrics', 'ksl'): None,
            ('metrics', 'vra', 'left'): None,
            ('metrics', 'vra', 'right'): None,
            ('zeta', 'probe', 'left', 'p'): None,
            ('zeta', 'probe', 'left', 'latency'): None,
            ('zeta', 'probe', 'right', 'p'): None,
            ('zeta', 'probe', 'right', 'latency'): None,
            ('zeta', 'saccade', 'nasal', 'p'): None,
            ('zeta', 'saccade', 'nasal', 'latency'): None,
            ('zeta', 'saccade', 'temporal', 'p'): None,
            ('zeta', 'saccade', 'temporal', 'latency'): None,
            ('metrics', 'dr', 'left', 'x'): None,
            ('metrics', 'dr', 'left', 'p'): None,
            ('metrics', 'dr', 'right', 'x'): None,
            ('metrics', 'dr', 'right', 'p'): None,
            ('psths', 'probe', 'left'): None,
            ('psths', 'probe', 'right'): None,
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
        if self._allSpikeClusters is None:
            self._units = list()
            self._uniqueSpikeClusters = None
            self._allSpikeTimestamps = None
            return

        self._allSpikeTimestamps = self._session.load('spikes/timestamps')
        self._uniqueSpikeClusters = np.unique(self.allSpikeClusters)

        #
        for cluster in self.uniqueSpikeClusters:
            unit = SingleUnit(self._session, cluster)
            self._units.append(unit)

        return

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

    def filter(
        self,
        probeMotion=None,
        presenceRatio=0.9,
        refractoryPeriodViolationRate=0.5,
        amplitudeCutoff=0.1,
        visualResponseProbability=0.99,
        visualResponseAmplitude=None,
        visualResponseLatencyRange=(0.04, 0.25),
        reload=True,
        returnMask=False
        ):
        """
        """

        # Reset the list of units
        if reload:
            self.unfilter()
        filtered = np.full(len(self._units), False)

        #
        if self.count() == 0:
            return
        if self._units[0].session.probeTimestamps is None:
            return

        #
        if probeMotion is not None:
            probeDirections = ('left',) if probeMotion == -1 else ('right',)
        else:
            probeDirections = ('left', 'right')

        #
        units = list()
        for unitIndex, unit in enumerate(self._units):

            # Filter out units with poor clustering quality metric scores
            if presenceRatio is not None and unit.presenceRatio < presenceRatio:
                continue
            if refractoryPeriodViolationRate is not None and unit.refractoryPeriodViolationRate > refractoryPeriodViolationRate:
                continue
            if amplitudeCutoff is not None and unit.amplitudeCutoff > amplitudeCutoff:
                continue


            # Filter out units with no or weak visual responses
            filtersPassed = False
            for probeDirection in probeDirections:
                if visualResponseProbability is not None and unit.visualResponseProbability[probeDirection] < visualResponseProbability:
                    continue
                if visualResponseAmplitude is not None and unit.visualResponseAmplitude[probeDirection] < visualResponseAmplitude:
                    continue
                filtersPassed = True
            if filtersPassed == False:
                continue

            # Filter out units with peak responses that are too fast or too delayed
            filterPassed = True
            for probeDirection in probeDirections:
                if visualResponseLatencyRange is not None:
                    if unit.visualResponseLatency[probeDirection] < visualResponseLatencyRange[0]:
                        filterPassed = False
                        break
                    if unit.visualResponseLatency[probeDirection] > visualResponseLatencyRange[1]:
                        filterPassed = False
                        break
            if filterPassed == False:
                continue

            # All filters passed
            units.append(unit)
            filtered[unitIndex] = True

        #
        self._units = units

        #
        if returnMask:
            return filtered

    def unfilter(
        self,
        ):
        """
        """

        self._loadSingleUnitData()

        return

    def count(self):
        """
        Return a count of the number of units in the population
        """

        return len(self._units)

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