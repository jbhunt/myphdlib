import h5py
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks as findPeaks
from myphdlib.figures.modulation import SimpleSaccadicModulationAnalysis
from myphdlib.figures.analysis import findOverlappingUnits, GaussianMixturesModel

class BoostrappedSaccadicModulationAnalysis(SimpleSaccadicModulationAnalysis):
    """
    """

    def __init__(self):
        """
        """

        super().__init__()

        self.pethsResampled = None
        self.probabilityValues = None
        self.modulationPolarity = None

        return

    def loadNamespace(self, hdf):
        """
        """

        super().loadNamespace(hdf)
        self.templates = {
            'nasal': None,
            'temporal': None
        }
        unitIndices = np.where(findOverlappingUnits(self.ukeys, hdf))[0]
        with h5py.File(hdf, 'r') as stream:
            paths = (
                'rSaccade/dg/preferred/nasal/fr',
                'rSaccade/dg/preferred/temporal/fr',
            )
            for path in paths:
                direction = 'nasal' if 'nasal' in path else 'temporal'
                if path in stream:
                    self.templates[direction] = np.array(stream[path])[unitIndices]
            path = 'rProbe/dg/preferred/perisaccadic/fr'
            if path in stream:
                self.peths['perisaccadic'] = np.array(stream[path][unitIndices])
            path = 'rProbe/dg/preferred/resampled/fr'
            if path in stream:
                self.pethsResampled = np.array(stream[path])[unitIndices]
            path = 'gmm/modulation'
            if path in stream:
                self.modulation = np.array(stream[path])[unitIndices]
            path = 'gmm/k'
            if path in stream:
                self.k = np.array(stream[path])[unitIndices]
            path = 'gmm/labels'
            if path in stream:
                self.labels = np.array(stream[path])[unitIndices]
        
        return

    def saveNamespace(
        self,
        hdf
        ):
        """
        """

        m = findOverlappingUnits(self.ukeys, hdf)
        unitIndices = np.where(m)[0]
        nUnits, nBins, nRuns = self.pethsResampled.shape
        with h5py.File(hdf, 'a') as stream:
            path = 'rProbe/dg/preferred/resampled/fr'
            if path in stream:
                del stream[path]
            data = np.full([m.size, nBins, nRuns], np.nan)
            data[unitIndices] = self.pethsResampled
            ds = stream.create_dataset(
                path,
                dtype=data.dtype,
                shape=data.shape,
                data=data,
            )

        return

    def resamplePeths(
        self,
        nRuns=30,
        responseWindow=(-0.2, 0.5),
        perisaccadicWindow=(-0.05, 0.1),
        binsize=0.01,
        smoothingKernelWidth=0.01
        ):
        """
        """

        nUnits, nBins = self.peths['extrasaccadic'].shape
        self.pethsResampled = np.full([nUnits, nBins, nRuns], np.nan)
        for iUnit in range(nUnits):
            end = '\r' if iUnit + 1 != nUnits else None
            print(f'Re-sampling PETHs for unit {iUnit + 1} out of {nUnits}', end=end)
            self.ukey = self.ukeys[iUnit]
            trialIndices = np.where(self.session.parseEvents(
                eventName='probe',
                coincident=False,
                eventDirection=self.ambc[iUnit, 1],
                coincidenceWindow=perisaccadicWindow,
            ))[0]
            nTrials = trialIndices.size
            mu, sigma = self.ambc[iUnit, 2], self.ambc[iUnit, 3]
            for iRun in range(nRuns):
                t, fr = self.unit.kde(
                    self.session.probeTimestamps[np.random.choice(trialIndices, size=nTrials)],
                    responseWindow=responseWindow,
                    binsize=binsize,
                    sigma=smoothingKernelWidth
                )
                self.pethsResampled[iUnit, :, iRun] = (fr - mu) / sigma

        return

    def refitExtrasaccadicPeth(
        self,
        ukey=None,
        iRun=0,
        **kwargs_
        ):
        """
        Re-fit the resampled extra-saccadic PETHs
        """

        #
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
            'maximumAmplitudeShift': 0.001 
        }
        kwargs.update(kwargs_)

        #
        if ukey is None:
            self.ukey = ukey
        for iUnit in range(len(self.ukeys)):
            date, animal, cluster = self.ukeys[iUnit]
            if date == self.ukey[0] and animal == self.ukey[1] and cluster == self.ukey[2]:
                break

        #
        y = self.pethsResampled[iUnit ,:, iRun]

        # Load parameters estimated for extra-saccadic PSTH
        params = self.params[iUnit]
        if np.isnan(params).all():
            return
        abcd = params[np.invert(np.isnan(params))]
        abc, d = abcd[:-1], abcd[-1]
        A1, B1, C1 = np.split(abc, 3)
        k = A1.size
        order = np.argsort(np.abs(A1))[::-1]
        A1 = A1[order]
        B1 = B1[order]
        C1 = C1[order]

        #
        peakIndices = list()
        for coef in (-1, 1):
            peakIndices_, peakProperties = findPeaks(
                coef * y,
                height=kwargs['minimumPeakHeight'],
                prominence=kwargs['minimumPeakProminence']
            )
            if peakIndices_.size == 0:
                continue
            for iPeak in range(peakIndices_.size):

                # Exclude peaks detected before the stimulus  onset
                if self.t[peakIndices_[iPeak]] <= 0:
                    continue

                #
                peakIndices.append(peakIndices_[iPeak])
        peakIndices = np.array(peakIndices)

        # Find the peak most similar to the original largest peak
        peakIndex = peakIndices[np.argmin(np.abs(self.t[peakIndices] - B1[0]))]

        # Compute peak properties
        peakAmplitude = y[peakIndex]
        peakLatency = self.t[peakIndex]
        peakAmplitudes = A1
        peakAmplitudes[0] = peakAmplitude
        peakLatencies = B1
        peakLatencies[0] = peakLatency
        peakWidths = C1
        peakWidths[0] = kwargs['initialPeakWidth']

        # Define the parameter boundaries
        amplitudeBoundaries = np.vstack([
            peakAmplitudes - 0.001,
            peakAmplitudes + 0.001
        ]).T
        amplitudeBoundaries[0, :] = np.array([
            peakAmplitude - kwargs['maximumAmplitudeShift'],
            peakAmplitude + kwargs['maximumAmplitudeShift']
        ])
        latencyBoundaries = np.vstack([
            peakLatencies - 0.001,
            peakLatencies + 0.001
        ]).T
        latencyBoundaries[0, :] = np.array([
            peakLatency - kwargs['maximumLatencyShift'],
            peakLatency + kwargs['maximumLatencyShift']
        ])
        widthBoundaries = np.vstack([
            peakWidths - 0.001,
            peakWidths + 0.001
        ]).T
        widthBoundaries[0, :] = np.array([
            kwargs['minimumPeakWidth'],
            kwargs['maximumPeakWidth']
        ])

        # Initialize the parameter space and bounds
        p0 = np.concatenate([
            np.array([d]),
            peakAmplitudes,
            peakLatencies,
            peakWidths
        ])
        bounds = np.vstack([
            np.array([[
                d - 0.001,
                d + 0.001
            ]]),
            amplitudeBoundaries,
            latencyBoundaries,
            widthBoundaries
        ]).T

        # Fit the GMM and compute the residual sum of squares (rss)
        gmm = GaussianMixturesModel(k)
        gmm.fit(
            self.t,
            y,
            p0=p0,
            bounds=bounds
        )
        yFit = gmm.predict(self.t)

        # Extract the parameters of the fit GMM
        d, abc = gmm._popt[0], gmm._popt[1:]
        A2, B2, C2 = np.split(abc, 3)
        order = np.argsort(np.abs(A1))[::-1] # Sort by amplitude
        params = np.concatenate([
            A2[order],
            B2[order],
            C2[order],
        ])
        paramsPadded = np.full(self.params.shape[1], np.nan)
        paramsPadded[:params.size] = params
        paramsPadded[-1] = d

        return yFit, paramsPadded

    def computeProbabilities(
        self,
        ):
        """
        """

        nUnits, nBins, nRuns = self.pethsResampled.shape
        self.probabilityValues = np.full(nUnits, np.nan)
        self.modulationPolarity = np.full(nUnits, np.nan)
        for iUnit in range(nUnits):
            end = '\r' if iUnit + 1 != nUnits else None
            print(f'Computing p-values for unit {iUnit + 1} out of {nUnits}', end=end)
            self.ukey = self.ukeys[iUnit]
            sample = np.full(nRuns, np.nan)
            for iRun in range(nRuns):
                yFit, params = self.refitExtrasaccadicPeth(
                    self.ukey,
                    iRun=iRun
                )
                self.params[iUnit, :] = params # set the new parameters
                # peth = self.pethsResampled[iUnit, :, iRun]
                dr, latencies, popt = super().refitPerisaccadicPeth(
                    self.ukey,
                )
                sample[iRun] = dr[0]
            sign = -1 if sample.mean() < 0 else +1
            if sign == -1:
                p = (1 - (np.sum(sample < 0) / sample.size)) / 0.5
            else:
                p = (1 - (np.sum(sample > 0) / sample.size)) / 0.5
            self.probabilityValues[iUnit] = p
            self.modulationPolarity[iUnit] = sign

        return

    def computeProbabilities2(
        self,
        alpha=0.05
        ):
        """
        """

        nUnits, nBins, nRuns = self.pethsResampled.shape
        self.probabilityValues = np.full(nUnits, np.nan)
        self.modulationPolarity = np.full(nUnits, np.nan)
        for iUnit in range(nUnits):
            end = '\r' if iUnit + 1 != nUnits else None
            print(f'Computing p-values for unit {iUnit + 1} out of {nUnits}', end=end)
            self.ukey = self.ukeys[iUnit]
            sample = np.full(nRuns, np.nan)
            for iRun in range(nRuns):
                peth = self.pethsResampled[iUnit, :, iRun]
                dr, latencies, popt = super().measureModulation(
                    self.ukey,
                    peth=peth
                )
                sample[iRun] = dr[0]
            tv = self.modulation[iUnit, 0]
            sign = -1 if tv < 0 else +1
            p = np.sum(np.abs(sample) > np.abs(tv)) / sample.size
            self.probabilityValues[iUnit] = p
            self.modulationPolarity[iUnit] = sign
        
        #
        # self.modulationPolarity[self.probabilityValues < alpha] = np.nan

        return

    def run(
        self,
        hdf=None
        ):
        """
        """

        if hdf is None == False:
            self.loadNamespace(hdf)
        if self.pethsResampled is None:
            self.resamplePeths()
        if self.probabilityValues is None:
            self.computeProbabilities()
        if hdf is None == False:
            self.saveNamespace(hdf)

        return

    def plotModulationDistributionsWithScatterplot(
        self,
        threshold=0.05,
        figsize=(3, 4),
        scale=0.07,
        **kwargs_
        ):
        """
        """

        kwargs = {
            'marker': '.',
            's': 12,
            'alpha': 0.3
        }
        kwargs.update(kwargs_)

        fig, ax = plt.subplots()
        for i, l in enumerate(np.unique(self.k)):
            m = np.vstack([
                np.ravel(self.k) == l,
                self.probabilityValues < threshold,
                self.modulationPolarity == -1
            ]).all(0)
            jitter = np.random.normal(loc=0, scale=scale, size=m.sum())
            ax.scatter(
                self.modulation[m, 0],
                np.full(m.sum(), l) + jitter,
                color='b',
                **kwargs
            )
            m = np.vstack([
                np.ravel(self.k) == l,
                self.probabilityValues < threshold,
                self.modulationPolarity == +1
            ]).all(0)
            jitter = np.random.normal(loc=0, scale=scale, size=m.sum())
            ax.scatter(
                self.modulation[m, 0],
                np.full(m.sum(), l) + jitter,
                color='r',
                **kwargs
            )
            m = np.logical_and(
                np.ravel(self.k) == l,
                self.probabilityValues >= threshold
            )
            jitter = np.random.normal(loc=0, scale=scale, size=m.sum())
            ax.scatter(
                self.modulation[m, 0],
                np.full(m.sum(), l) + jitter,
                color='k',
                **kwargs
            )
        ylim = ax.get_ylim()
        ax.vlines(0, *ylim, color='k')
        ax.set_ylim(ylim)
        xmax = np.max(np.abs(ax.get_xlim()))
        ax.set_xlim([-xmax, xmax])
        ax.set_ylabel('# of components (k)')
        ax.set_xlabel(r'$\Delta R$')
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, ax