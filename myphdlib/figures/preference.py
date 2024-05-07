import h5py
import numpy as np
from scipy.signal import find_peaks as findPeaks
from myphdlib.general.toolkit import psth2
from myphdlib.figures.analysis import AnalysisBase, GaussianMixturesModel

class DirectionSectivityAnalysis(
    AnalysisBase,
    ):
    """
    """

    def __init__(self, **kwargs):
        """
        """

        super().__init__(**kwargs)

        self.dsi = {
            'bar': None,
            'probe': None,
            'saccade': None,
        }
        self.peths = {
            'left': None,
            'right': None,
            'preferred': None,
            'null': None,
        }
        self.tProbe = None
        self.tSaccade = None

        return

    def loadNamespace(
        self,
        ):
        """
        """

        datasets = {
        }
        with h5py.File(self.hdf, 'r') as stream:
            for path, (attr, key) in datasets.items():
                parts = path.split('/')
                if path in stream:
                    ds = stream[path]
                    if path == 'fictive/peths/extra':
                        self.tProbe = ds.attrs['t']
                    if path == 'fictive/templates/nasal':
                        self.tSaccade = ds.attrs['t']
                    value = np.array(ds)
                    if 'filter' in parts:
                        value = value.astype(bool)
                    if len(value.shape) == 2 and value.shape[-1] == 1:
                        value = value.flatten()
                    if key is None:
                        setattr(self, attr, value)
                    else:
                        attr[key] = value

        return

    def saveNamespace(
        self,
        ):
        """
        """

        return

    def _extractDirectionSelectivityForMovingBars(
        self,
        ):
        """
        Extract DSI for the moving bars stimulus
        """

        self.dsi['bar'] = np.full(len(self.ukeys), np.nan)
        date = None
        for ukey in self.ukeys():
            self.ukey = ukey
            if date is None or ukey[0] != date:
                dsi = self.session.load('metrics/dsi')
                date = self.session.date
            iUnit = self._indexUnitKey(ukey)
            self.dsi['bar'][iUnit] = dsi[self.unit.index]

        return

    def _computePeths(
        self,
        responseWindow=(-0.2, 0.5),
        baselineWindow=(-0.2, 0),
        standardizationWindow=(-20, -10),
        perisaccadicWindow=(-0.5, 0.5),
        binsize=0.01,
        gaussianKernelWidth=0.01,
        ):
        """
        """

        #
        t, nTrials, nBins = psth2(
            np.array([0]),
            np.array([0]),
            window=responseWindow,
            returnShape=True
        )
        if self.tProbe is None:
            self.tProbe = t

        #
        for key in ('left', 'right'):
            self.peths[key] = np.full(len(self.ukeys), np.nan)

        #
        for ukey in self.ukeys:

            #
            self.ukey = ukey

            #
            for gratingMotion, key in zip([-1, 1], ['left', 'right']):

                #
                trialIndices = np.where(np.vstack([
                    np.logical_or(
                        self.session.probeLatencies < perisaccadicWindow[0],
                        self.session.probeLatencies > perisaccadicWindow[1],
                    ),
                    self.session.gratingMotionDuringProbes == gratingMotion
                ]).all(0))

                #
                t, fr = self.unit.kde(
                    self.session.probeTimestamps[trialIndices],
                    responseWindow=responseWindow,
                    binsize=binsize,
                    sigma=gaussianKernelWidth,
                )

                # Estimate baseline firing rate
                t, bl1 = self.unit.kde(
                    self.session.probeTimestamps[trialIndices],
                    responseWindow=baselineWindow,
                    binsize=binsize,
                    sigma=gaussianKernelWidth
                )

                # Estimate standard deviation of firing rate
                t, bl2 = self.unit.kde(
                    self.session.probeTimestamps[trialIndices],
                    responseWindow=standardizationWindow,
                    binsize=binsize,
                    sigma=gaussianKernelWidth
                )

                #
                peth = (fr - bl1.mean()) / bl2.std()
                self.peths[key][self.iUnit] = peth

        return

    def _determinePreferredDirection(
        self,
        kmax=5,
        **kwargs_
        ):
        """
        """

        kwargs = {
            'minimumPeakHeight': 0.15,
            'maximumPeakHeight': 1,
            'minimumPeakProminence': 0.05,
            'minimumPeakWidth': 0.001,
            'maximumPeakWidth': 0.02,
            'minimumPeakLatency': 0,
            'initialPeakWidth': 0.001,
            'maximumLatencyShift': 0.003,
            'maximumBaselineShift': 0.001,
            'maximumAmplitudeShift': 0.01 
        }
        kwargs.update(kwargs_)

        #
        amplitude = {
            'left': np.full(len(self.ukeys), np.nan),
            'right': np.full(len(self.ukeys), np.nan),
        }

        #
        for ukey in self.ukeys:
            self.ukey = ukey
            for key in ('left', 'right'):

                #
                yStandard = self.peths[key][self.iUnit]
                yNormal /= np.max(np.abs(yStandard)) # Normalize

                #
                peakIndices = list()
                peakProminences = list()
                for coef in (-1, 1):
                    peakIndices_, peakProperties = findPeaks(
                        coef * yNormal,
                        height=kwargs['minimumPeakHeight'],
                        prominence=kwargs['minimumPeakProminence']
                    )
                    if peakIndices_.size == 0:
                        continue
                    for iPeak in range(peakIndices_.size):

                        # Exclude peaks detected before the stimulus  onset
                        if self.tProbe[peakIndices_[iPeak]] <= 0:
                            continue

                        #
                        peakIndices.append(peakIndices_[iPeak])
                        peakProminences.append(peakProperties['prominences'][iPeak])

                # 
                peakIndices = np.array(peakIndices)
                if peakIndices.size == 0:
                    continue
                peakProminences = np.array(peakProminences)
                peakAmplitudes = yStandard[peakIndices]
                peakLatencies = self.tProbe[peakIndices]

                # Use only the k largest peaks
                if peakIndices.size > kmax:
                    index = np.argsort(np.abs(peakAmplitudes))[::-1]
                    peakIndices = peakIndices[index][:kmax]
                    peakProminences = peakProminences[index][:kmax]
                    peakAmplitudes = peakAmplitudes[index][:kmax]
                    peakLatencies = peakLatencies[index][:kmax]

                #
                k = peakIndices.size

                # Initialize the parameter space
                p0 = np.concatenate([
                    np.array([0]),
                    peakAmplitudes,
                    peakLatencies,
                    np.full(k, kwargs['initialPeakWidth'])
                ])
                bounds = np.vstack([
                    np.array([[
                        -1 * kwargs['maximumBaselineShift'],
                        kwargs['maximumBaselineShift']
                    ]]),
                    np.vstack([
                        peakAmplitudes - kwargs['maximumAmplitudeShift'],
                        peakAmplitudes + kwargs['maximumAmplitudeShift']
                    ]).T,
                    np.vstack([
                        peakLatencies - kwargs['maximumLatencyShift'],
                        peakLatencies + kwargs['maximumLatencyShift']
                    ]).T,
                    np.repeat([[
                        kwargs['minimumPeakWidth'],
                        kwargs['maximumPeakWidth']
                    ]], k, axis=0)
                ]).T

                # Fit the GMM and compute the residual sum of squares (rss)
                gmm = GaussianMixturesModel(k)
                gmm.fit(
                    self.tProbe,
                    yStandard,
                    p0=p0,
                    bounds=bounds
                )

                # Extract the parameters of the fit GMM
                d, abc = gmm._popt[0], gmm._popt[1:]
                A, B, C = np.split(abc, 3)
                amplitude[key][self.iUnit] = np.abs(A).max()

        #
        for key in ('preferred', 'null'):
            self.peths[key] = np.full([len(self.ukeys), self.tProbe.size], np.nan)
        for iUnit, ukey in enumerate(self.ukeys):
            aL, aR = amplitude['left'][iUnit], amplitude['right'][iUnit]
            if aL > aR:
                pd = 'left'
                nd = 'right'
            else:
                pd = 'right'
                nd = 'left'
            self.peths['preferred'][iUnit] = self.peths[pd][iUnit]
            self.peths['null'][iUnit] = self.peths[nd][iUnit]  

        return

    def measureDirectionSelectivityForSaccades(
        self,
        ):
        """
        """

        self.dsi['saccade'] = np.full(len(self.ukeys), np.nan)

        return

    def run(
        self,
        ):
        """
        """

        self._computePeths()
        self._determinePreferredDirection()
        self._extractDirectionSelectivityForMovingBars()

        return