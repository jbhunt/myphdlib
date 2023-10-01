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
        self._pd = None
        self._nd = None
        self._dsi = None
        self._utype = None
        self._index = None
        self._stability = None
        self._contamination = None
        self._isQuality = None
        self._isVisual = None
        self._isMotor = None
        self._quality = None

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
            spikeIndices = np.where(self.session.population.allSpikeClusters == self.cluster)[0]
            self._timestamps = self.session.population.allSpikeTimestamps[spikeIndices]

        return self._timestamps
    
    @property
    def utype(self):
        """
        """

        if self._utype is None:
            visuallyResponsive = self.session.population.filters['vr'][self.index]
            if self.session.population.filters['sr'] is None:
                saccadeRelated = False
            else:
                saccadeRelated = self.session.population.filters['sr'][self.index]
            if visuallyResponsive == True and saccadeRelated == True:
                self._utype = 'vm'
            elif visuallyResponsive == True and saccadeRelated == False:
                self._utype = 'vr'
            elif visuallyResponsive == False and saccadeRelated == True:
                self._utype = 'sr'
            else:
                self._utype = 'nr'

        return self._utype

    @property
    def quality(self):
        if self._quality is None:
            if self.session.population.filters['hq'][self.index]:
                self._quality = 'h'
            else:
                self._quality = 'l'
        return self._quality
    
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

class Population():
    """
    """

    def __init__(self, session, autoload=True):
        """
        """

        self._session = session
        self._units = None
        self._index = 0
        self._filters = {
            'vr': None,
            'sr': None,
            'hq': None
        }
        self._spikeSortingQualityMetrics = {
            'presence': None,
            'contamination': None,
            'completeness': None,
        }
        self._zetaTestPs = None
        self._visualResponseAmplitudes = None

        if autoload:
            self._loadSingleUnitData()
            self._loadPopulationFilters()
            self._loadZetaTestPs()
            self._loadSpikeSortinQualityMetrics()
            self._loadResponseAmplitudes()

        return
    
    def indexByCluster(self, cluster):
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
        self._allSpikeClusters = self._session.load('spikes/clusters')
        self._allSpikeTimestamps = self._session.load('spikes/timestamps')
        self._uniqueSpikeClusters = np.unique(self.allSpikeClusters)

        #
        for cluster in np.unique(self._allSpikeClusters):
            unit = SingleUnit(self._session, cluster)
            self._units.append(unit)

        return

    def _loadPopulationFilters(
        self
        ):
        """
        """

        datasetNames = (
            'visual',
            'motor',
            'quality'
        )
        for filterKey, datasetName in zip(self._filters.keys(), datasetNames):
            if self._filters[filterKey] is None:
                self._filters[filterKey] = self._session.load(f'population/filters/{datasetName}')

        return

    def _loadZetaTestPs(self):
        """
        """

        datasetPaths = (
            'population/metrics/zeta/visual/pvalues',
            'population/metrics/zeta/motor/left/nasal/pvalues',
            'population/metrics/zeta/motor/left/temporal/pvalues',
        )
        pvalues = list()
        for datasetPath in datasetPaths:
            pvalues.append(self._session.load(datasetPath))
        self._zetaTestPs = np.array(pvalues).T

        return

    def _loadSpikeSortinQualityMetrics(self):
        """
        """

        for metricKey in self._spikeSortingQualityMetrics.keys():
            datasetPath = f'population/metrics/{metricKey}'
            metricValues = self._session.load(datasetPath)
            self._spikeSortingQualityMetrics[metricKey] = metricValues

        return

    def _loadResponseAmplitudes(self):
        """
        """

        if self._visualResponseAmplitudes is None:
            if self._session.hasDataset('population/metrics/visual_response_amplitude'):
                self._visualResponseAmplitudes = self._session.load('population/metrics/visual_response_amplitude')
            # else:
            #     self._visualResponseAmplitudes = np.full(len(self, np.nan))

        return

    def filterUnits(
        self,
        pResponsiveThreshold=0.001,
        presenceRatioThreshold=0.9,
        isiViolationRateThreshold=0.5,
        amplitudeCutoffThreshold=0.1,
        responseAmplitudeThreshold=None,
        targetUnitQuality='high',
        ):
        """

        """

        units = list()
        for unit in self:

            # Is the unit responsive?
            if np.min(self._zetaTestPs[unit.index]) >= pResponsiveThreshold:
                continue

            # Does the unit pass quality metric thresholds?
            if targetUnitQuality is not None:
                qualityMetricFlags = np.array([
                    self._spikeSortingQualityMetrics['presence'][unit.index] > presenceRatioThreshold,
                    self._spikeSortingQualityMetrics['contamination'][unit.index] < isiViolationRateThreshold,
                    self._spikeSortingQualityMetrics['completeness'][unit.index] < amplitudeCutoffThreshold
                ])
                passedQualityMetricsFilter = qualityMetricFlags.all()
                if targetUnitQuality in ('l', 'low', 0) and passedQualityMetricsFilter:
                    continue
                if targetUnitQuality in ('h', 'high', 1) and passedQualityMetricsFilter == False:
                    continue
    
            # Does the unit have a large enough response?
            if responseAmplitudeThreshold is not None:
                if self._visualResponseAmplitudes is None:
                    raise Exception('Response amplitudes not available')
                responseAmlitude = self._visualResponseAmplitudes[unit.index]
                if np.isnan(responseAmlitude):
                    continue
                if responseAmlitude <= responseAmplitudeThreshold:
                    continue

            # All checks passed
            units.append(unit)

        return units

    @property
    def filters(self):
        return self._filters

    @property
    def allSpikeClusters(self):
        return self._allSpikeClusters
    
    @property
    def allSpikeTimestamps(self):
        return self._allSpikeTimestamps

    @property
    def uniqueSpikeClusters(self):
        return self._uniqueSpikeClusters
    
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