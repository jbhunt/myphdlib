import h5py
import numpy as np
from dotmap import DotMap
from myphdlib.general.toolkit import psth2
from sklearn.neighbors import KernelDensity
from scipy.stats import gaussian_kde

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
        self._label = None
        self._directionSelectivityIndex = None
        self._luminancePolarityIndex = None
        self._visualResponseSign = None
        self._motionPreferenceIndex = None
        self._preferredProbeDirection = None

        return

    def describeAcrossTrials(
        self,
        eventTimestamps,
        responseWindow=(-0.2, 0),
        ):
        """
        Estimate baseline activity mean and std across trials
        """

        #
        t, M = psth2(
            eventTimestamps,
            self.timestamps,
            window=responseWindow,
            binsize=None
        )
        fr = M.flatten() / np.diff(responseWindow).item()

        return round(fr.mean(), 2), round(fr.std(), 2)
    
    def describeAcrossBins(
        self,
        eventTimestamps,
        responseWindow=(-30, -10),
        binsize=0.02,
        ):
        """
        Estimate baseline activity mean and std across time bins
        """

        t, M = psth2(
            eventTimestamps,
            self.timestamps,
            window=responseWindow,
            binsize=binsize
        )
        fr = M.mean(0) / binsize
        mu, sigma = round(fr.mean(), 2), round(fr.std(), 2)

        return mu, sigma

    def describeWithBootstrap(
        self,
        eventTimestamps,
        baselineWindowBoundaries=(-0.2, 0),
        windowSize=0.02,
        nRuns=30,
        ):
        """
        Estimate baseline activity mean and std with bootstrap
        """

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
                binsize=None,
            )
            fr = M.flatten() / windowSize
            samples.append([fr.mean(), fr.std()])

        #
        mu, sigma = np.around(np.array(samples).mean(0), 2)

        return mu, sigma

    def describeGlobalActivity(
        self,
        binsize=None,
        returnFiringRate=False
        ):
        """
        """

        tStart, tStop = self.session.tRange
        if binsize is None:
            fr = None
            mu = round(self.timestamps.size / tStop, 3)
            sigma = np.nan
        else:
            nBins = int(tStop // binsize)
            nSpikes, binEdges = np.histogram(self.timestamps, bins=nBins, range=(tStart, tStop))
            fr = nSpikes / binsize
            mu, sigma = round(fr.mean(), 3), round(fr.std(), 3)

        if returnFiringRate:
            return mu, sigma, fr
        else:
            return mu, sigma

    def peth(
        self,
        eventTimestamps,
        responseWindow=(-1, 1),
        binsize=0.02,
        kde=False,
        sd=0.005,
        buffer=0.5
        ):
        """
        """

        #
        if kde:
            responseWindowBuffered = (
                responseWindow[0] - buffer,
                responseWindow[1] + buffer
            )
            t_, M, sample = psth2(
                eventTimestamps,
                self.timestamps,
                window=responseWindowBuffered,
                binsize=None,
                returnTimestamps=True
            )
            # t = np.linspace(responseWindow[0], responseWindow[1], nt)
            leftEdges = np.arange(responseWindow[0], responseWindow[1], binsize)
            t = np.around(leftEdges + (binsize / 2), 3)
            if sample.size < 3:
                fr = np.full(t.size, np.nan)
            else:
                f = gaussian_kde(sample)
                f.set_bandwidth(sd / sample.std())
                y = f(t)
                fr = y * (sample.size * binsize) / M.shape[0] / binsize

        else:
            t, m = psth2(
                eventTimestamps,
                self.timestamps,
                window=responseWindow,
                binsize=binsize
            )
            fr = m.mean(0) / binsize

        return t, fr

    def kde(
        self,
        eventTimestamps,
        responseWindow=(-1, 1),
        binsize=0.02,
        sigma=0.005,
        buffer=0.5,
        t=None,
        sample=None,
        nTrials=None
        ):
        """
        Estimate FR using kernel density estimation
        """

        responseWindowBuffered = (
            responseWindow[0] - buffer,
            responseWindow[1] + buffer
        )
        if sample is None:
            t_, M, sample_ = psth2(
                eventTimestamps,
                self.timestamps,
                window=responseWindowBuffered,
                binsize=None,
                returnTimestamps=True
            )
            sample = list()
            for tr in sample_:
                for ts in tr:
                    sample.append(ts)
            sample = np.array(sample)
            nTrials = M.shape[0]

        #
        if sample.size < 3:
            raise Exception('Not enough events to perform KDE')
        else:
            f = gaussian_kde(sample)
            f.set_bandwidth(sigma / sample.std())

        #
        if t is None:
            leftEdges = np.arange(responseWindow[0], responseWindow[1], binsize)
            t = np.around(leftEdges + (binsize / 2), 3)

        y = f(t)
        fr = y * (sample.size * binsize) / nTrials / binsize

        return t, fr

    def determineResponseWindow(
        self,
        probeMotion=-1,
        windowSize=0.1,
        ):
        """
        """

        probeDirection = 'left' if probeMotion == -1 else 'right'
        responsePolarityIndices = np.array([
            self.session.population.datasets[('population', 'metrics', 'rsi', 'left')][self.index],
            self.session.population.datasets[('population', 'metrics', 'rsi', 'right')][self.index]
        ])
        responseLatencyIndex = 0 if responsePolarityIndices.mean() >= 0 else 1
        windowCenter = self.visualResponseLatency[probeDirection][responseLatencyIndex]
        windowEdges = np.around(np.array([
            windowCenter - (windowSize / 2),
            windowCenter + (windowSize / 2)
        ]), 3)

        return windowEdges

    @property
    def visualResponseSign(
        self,
        ):
        """
        """

        if self._visualResponseSign is None:
            responsePolarityIndices = np.array([
                self.session.population.datasets[('population', 'metrics', 'rsi', 'left')][self.index],
                self.session.population.datasets[('population', 'metrics', 'rsi', 'right')][self.index]
            ])
            self._visualResponseSign = -1 if responsePolarityIndices.mean() < 0 else +1

        return self._visualResponseSign

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
                ('population', 'metrics', 'rl', 'probe', 'left'),
                ('population', 'metrics', 'rl', 'probe', 'right')
            )
            for key in keys:
                if self.session.population.datasets[key] is None:
                    continue
                probeDirection = key[-1]
                vrl[probeDirection] = np.around(self.session.population.datasets[key][self.index], 3)
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
                ('population', 'metrics', 'vra', 'left'),
                ('population', 'metrics', 'vra', 'right')
            )
            for key in keys:
                if self.session.population.datasets[key] is None:
                    continue
                probeDirection = key[-1]
                ri = 1 if self.visualResponseSign == -1 else 0 # Choose the amplitude of the positive or negative peak
                vra[probeDirection] = np.around(self.session.population.datasets[key][self.index][ri], 3)
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
                ('population', 'zeta', 'probe', 'left', 'p'),
                ('population', 'zeta', 'probe', 'right', 'p')
            )
            for key in keys:
                if self.session.population.datasets[key] is None:
                    continue
                probeDirection = key[3]
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
                ('population', 'psths', 'probe', 'left'),
                ('population', 'psths', 'probe', 'right')
            )
            for key in keys:
                if self.session.population.datasets[key] is None:
                    continue
                probeDirection = key[3]
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
                ('population', 'metrics', 'dr', 'left', 'x'),
                ('population', 'metrics', 'dr', 'right', 'x')
            )
            for key in keys:
                if self.session.population.datasets[key] is None:
                    continue
                probeDirection = key[3]
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
                ('population', 'metrics', 'dr', 'left', 'p'),
                ('population', 'metrics', 'dr', 'right', 'p')
            )
            for key in keys:
                if self.session.population.datasets[key] is None:
                    continue
                probeDirection = key[3]
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
                ('population', 'metrics', 'pr'),
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
                ('population', 'metrics', 'rpvr'),
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
                ('population', 'metrics', 'ac'),
            )
            for key in keys:
                if self.session.population.datasets[key] is None:
                    continue
                self._amplitudeCutoff = self.session.population.datasets[key][self.index]

        return self._amplitudeCutoff

    @property
    def label(self):
        """
        """

        if self._label is None:
            keys = (
                ('clustering', 'probe', 'labels'),
            )
            for key in keys:
                if self.session.population.datasets[key] is None:
                    continue
                label = self.session.population.datasets[key][self.index]         
                if np.isnan(label) == True:
                    self._label = np.nan
                else:
                    self._label = int(label)

        return self._label

    @property
    def directionSelectivityIndex(self):
        if self._directionSelectivityIndex is None:
            keys = (
                ('population', 'metrics', 'dsi'),
            )
            for key in keys:
                if self.session.population.datasets[key] is None:
                    continue
                self._directionSelectivityIndex = self.session.population.datasets[key][self.index]

        return self._directionSelectivityIndex

    @property
    def luminancePolarityIndex(self):
        if self._luminancePolarityIndex is None:
            keys = (
                ('population', 'metrics', 'lpi'),
            )
            for key in keys:
                if self.session.population.datasets[key] is None:
                    continue
                self._luminancePolarityIndex = self.session.population.datasets[key][self.index]

        return self._luminancePolarityIndex

    @property
    def motionPreferenceIndex(self):
        if self._motionPreferenceIndex is None:
            keys = (
                ('population', 'metrics', 'rpi'),
            )
            for key in keys:
                if self.session.population.datasets[key] is None:
                    continue
                self._motionPreferenceIndex = self.session.population.datasets[key][self.index]

        return self._motionPreferenceIndex

    @property
    def preferredProbeDirection(self):
        """
        """

        if self._preferredProbeDirection is None:
            self._preferredProbeDirection = 'left' if self.motionPreferenceIndex < 0 else 'right'

        return self._preferredProbeDirection


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
            ('population', 'metrics', 'pr'): None,
            ('population', 'metrics', 'rpvr'): None,
            ('population', 'metrics', 'ac'): None,
            ('population', 'metrics', 'ksl'): None,
            ('population', 'metrics', 'vra', 'left'): None,
            ('population', 'metrics', 'vra', 'right'): None,
            ('population', 'zeta', 'probe', 'left', 'p'): None,
            ('population', 'zeta', 'probe', 'left', 'latency'): None,
            ('population', 'zeta', 'probe', 'right', 'p'): None,
            ('population', 'zeta', 'probe', 'right', 'latency'): None,
            ('population', 'zeta', 'saccade', 'nasal', 'p'): None,
            ('population', 'zeta', 'saccade', 'nasal', 'latency'): None,
            ('population', 'zeta', 'saccade', 'temporal', 'p'): None,
            ('population', 'zeta', 'saccade', 'temporal', 'latency'): None,
            ('population', 'metrics', 'dr', 'left', 'x'): None,
            ('population', 'metrics', 'dr', 'left', 'p'): None,
            ('population', 'metrics', 'dr', 'right', 'x'): None,
            ('population', 'metrics', 'dr', 'right', 'p'): None,
            ('clustering', 'probe', 'labels'): None,
            ('population', 'metrics', 'rl', 'probe', 'left'): None,
            ('population', 'metrics', 'rl', 'probe', 'right'): None,
            ('population', 'metrics', 'dsi'): None,
            ('population', 'metrics', 'lpi'): None,
            ('population', 'metrics', 'rsi', 'left'): None,
            ('population', 'metrics', 'rsi', 'right'): None,
            ('population', 'metrics', 'rpi'): None,
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

    def filter2(
        self,
        minimumFiringRate=None,
        minimumResponseAmplitude=None,
        probeDirection='left',
        applyFilter=True,
        ):
        """
        """

        # Reset the list of units
        self.unfilter()

        #
        unitFilter = np.full(len(self._units), True)
        for unitIndex, unit in enumerate(self._units):
            if minimumFiringRate is not None and unit.timestamps.size / unit.session.tRange[-1] < minimumFiringRate:
                unitFilter[unitIndex] = False
            if minimumResponseAmplitude is not None and unit.visualResponseAmplitude[probeDirection] < minimumResponseAmplitude:
                unitFilter[unitIndex] = False

        #
        if applyFilter:
            self._units = self[unitFilter]

        return unitFilter

    def filter(
        self,
        probeMotion=None,
        presenceRatio=0.9,
        refractoryPeriodViolationRate=0.7,
        amplitudeCutoff=0.1,
        visualResponseProbability=0.99,
        visualResponseAmplitude=None,
        visualResponseLatencyRange=(0.05, 0.5),
        spikeCountMinimum=3000,
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

            #
            if unit.timestamps.size < spikeCountMinimum:
                continue

            # All filters passed
            units.append(unit)
            filtered[unitIndex] = True

        #
        if returnMask:
            return filtered
        else:
            self._units = units

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