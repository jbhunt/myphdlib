import h5py
import pathlib as pl
import numpy as np
from matplotlib import pyplot as plt
from myphdlib.general.toolkit import smooth, stretch, psth2
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from scipy.optimize import curve_fit as fitCurve
from scipy.signal import find_peaks as findPeaks
from scipy.signal import peak_prominences
from scipy.ndimage import gaussian_filter1d as smooth2
from scipy.ndimage import gaussian_filter as gaussianFilter
from myphdlib.figures.analysis import AnalysisBase, GaussianMixturesModel, g

class GaussianMixturesFittingAnalysis(AnalysisBase):
    """
    """

    def __init__(
        self,
        **kwargs
        ):
        """
        Analysis for fitting visual responses with a Gaussian mixtures model

        Method sequence
        ---------------
        1. computeExtrasaccadicPeths
        2. fitExtrasaccadicPeths
        3. predictLabels
        4. createFilter
        """

        super().__init__(**kwargs)

        #
        self.examples = (
            ('2023-07-11', 'mlati10', 291),
            ('2023-07-19', 'mlati10', 327),
            ('2023-07-11', 'mlati10', 295),
            ('2023-07-14', 'mlati9', 165),
        )

        return

    def _loadEventDataForSaccades(
        self,
        ):
        """
        """

        return (
            self.session.saccadeTimestamps[:, 0],
            self.session.saccadeLatencies,
            self.session.saccadeLabels,
            self.session.gratingMotionDuringSaccades
        )
    
    def _loadEventDataForProbes(
        self,
        ):
        """
        """

        saccadeLabels = self.session.load('stimuli/dg/probe/dos')

        return self.session.probeTimestamps, self.session.probeLatencies, saccadeLabels, self.session.gratingMotionDuringProbes

    def computeExtrasaccadicPeths(
        self,
        responseWindow=(-0.2, 0.5),
        baselineWindow=(-0.2, 0),
        standardizationWindow=(-20, -10),
        binsize=0.01,
        smoothingKernelWidth=0.01,
        perisaccadicWindow=(-0.1, 0.1),
        saccadeType='real',
        ):
        """
        """

        self.tProbe, nTrials, nBins = psth2(
            np.zeros(1),
            np.zeros(1),
            window=responseWindow,
            binsize=binsize,
            returnShape=True
        )

        #
        nUnits = len(self.ukeys)

        #
        for motionDirection in ('pref', 'null'):
            self.ns[f'ppths/{motionDirection}/{saccadeType}/extra'] = np.full([nUnits, nBins], np.nan)
            self.ns[f'stats/{motionDirection}/{saccadeType}/extra'] = np.full([nUnits, 2], np.nan)
        if saccadeType == 'real':
            self.ns[f'globals/factor'] = np.full(nUnits, np.nan)
            self.ns[f'globals/preference'] = np.full(nUnits, np.nan)

        for iUnit, ukey in enumerate(self.ukeys):

            #
            end = None if iUnit + 1 == nUnits else '\r'
            print(f'Working on {iUnit + 1} out of {nUnits} units', end=end)
            self.ukey = ukey # NOTE: Very important, this invokes the ukey.setter situation

            #
            features = {
                'ppths': np.full([2, nBins], np.nan),
                'stats': np.full([2, 2], np.nan),
                'amplitude': np.full(2, np.nan),
            }

            #
            probeTimestamps, probeLatencies, saccadeLabels, gratingMotionDuringProbes = self._loadEventDataForProbes()

            #
            for i, gratingMotion in enumerate([-1, 1]):

                # Select just the extra-saccadic trials
                trialIndices = np.where(np.vstack([
                    gratingMotionDuringProbes == gratingMotion,
                    np.logical_or(
                        probeLatencies > perisaccadicWindow[1],
                        probeLatencies < perisaccadicWindow[0]
                    )
                ]).all(0))[0]

                #
                try:

                    # Compute firing rate
                    t, fr = self.unit.kde(
                        probeTimestamps[trialIndices],
                        responseWindow=responseWindow,
                        binsize=binsize,
                        sigma=smoothingKernelWidth,
                    )

                    # Estimate baseline firing rate
                    t, bl1 = self.unit.kde(
                        probeTimestamps[trialIndices],
                        responseWindow=baselineWindow,
                        binsize=binsize,
                        sigma=smoothingKernelWidth
                    )

                    # Estimate standard deviation of firing rate
                    t, bl2 = self.unit.kde(
                        probeTimestamps[trialIndices],
                        responseWindow=standardizationWindow,
                        binsize=binsize,
                        sigma=smoothingKernelWidth
                    )

                #
                except:
                    continue

                #
                features['ppths'][i] = np.around(fr, 3)
                features['stats'][i] = np.around(np.array([bl1.mean(), bl2.std()]), 3)
                features['amplitude'][i] = np.abs(fr[self.tProbe > 0] - bl1.mean()).max()

            #
            iPref = np.argmax(features['amplitude'])
            iNull = 0 if iPref == 1 else 1
            if saccadeType == 'real':
                self.ns[f'globals/preference'][iUnit] = np.array([-1, 1])[iPref]
                self.ns[f'globals/factor'][iUnit] = features['stats'][iPref, 1]

            # Z-score the firing rate
            # TODO: Correct the z-scoring method; right now the estimate of SD is wrong
            for motionDirection, iRow in zip(['pref', 'null'], [iPref, iNull]):
                z = (features['ppths'][iRow] - features['stats'][iRow, 0]) / self.ns[f'globals/factor'][iUnit]
                self.ns[f'ppths/{motionDirection}/{saccadeType}/extra'][iUnit] = z
                self.ns[f'stats/{motionDirection}/{saccadeType}/extra'][iUnit] = features['stats'][iRow]

        return

    def fitExtrasaccadicPeths(
        self,
        kmax=5,
        saccadeType='real',
        probeCondition='extra',
        **kwargs_
        ):
        """
        Algorithm
        ---------
        1. Find peaks using the normalized PSTH
        2. Discard all but the k largest peaks
        3. Use peak positions and amplitudes from the standardized PSTHs to initialize the GMM
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
        nUnits = len(self.ukeys)

        #
        for motionDirection in ('pref', 'null'):
            self.ns[f'params/{motionDirection}/{saccadeType}/{probeCondition}'] = np.full([nUnits, int(3 * kmax + 1)], np.nan)

        #
        for iUnit in range(nUnits):

            #
            end = None if iUnit + 1 == nUnits else '\r'
            print(f'Fitting GMM for unit {iUnit + 1} out of {nUnits} units', end=end)

            #
            for motionDirection in ('pref', 'null'):

                #
                yStandard = self.ns[f'ppths/{motionDirection}/{saccadeType}/{probeCondition}'][iUnit]
                yNormal = yStandard / np.max(np.abs(yStandard)) #amplitude normalized so peak of largest component = 1

                #
                peakIndices = list()
                peakProminences = list()
                for coef in (-1, 1): #this allows us to flip signal so we can get max and min peaks since this only finds max
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
                peakProminences = np.array(peakProminences) #amplitude from shoulders of peak
                peakAmplitudes = yStandard[peakIndices] #amplitude from baseline
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
                p0 = np.concatenate([ #these are the parameters we need to fit the gaussian function
                    np.array([0]), #constant offset for some reason
                    peakAmplitudes,
                    peakLatencies, 
                    np.full(k, kwargs['initialPeakWidth']) #set width manually bc find peaks does not estimate this well
                ])
                bounds = np.vstack([ #set min and max values for each parameter intelligently :)
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
                order = np.argsort(np.abs(A))[::-1] # Sort by amplitude
                params = np.concatenate([
                    A[order],
                    B[order],
                    C[order],
                    np.array([d,])
                ])
                self.ns[f'params/{motionDirection}/{saccadeType}/{probeCondition}'][iUnit, :params.size] = params

        return

    def predictLabels(
        self,
        ):
        """
        """

        nUnits = len(self.ukeys)
        self.ns['globals/labels'] = np.full(nUnits, np.nan)
        for i in range(nUnits):

            #
            y = self.ns[f'ppths/pref/real/extra'][i]

            # Determine the number of components in the fit
            params = self.ns[f'params/pref/real/extra'][i]
            abcd = np.delete(params, np.isnan(params))
            A, B, C = np.split(abcd[:-1], 3)
            k = A.size

            # No peaks detected
            if k == 0:
                self.ns['globals/labels'][i] = np.nan
                continue

            # Negative
            if y[np.argmax(np.abs(y))] < 0:
                self.ns['globals/labels'][i] = -1

            # Positive (multiphasic)
            elif k >= 3:
                self.ns['globals/labels'][i] = 3

            # Positive (Mono- or Biphasic)
            else:
                self.ns['globals/labels'][i] = k

        return

    def _plotReceptiveFields(
        self,
        axs,
        threshold=2,
        smoothingKernelWidth=0.5,
        colors={'on': '0.5', 'off': 'k'},
        linestyles={'on': '-', 'off': '-'}
        ):
        """
        """

        for i, ukey in enumerate(self.examples):
            if (ukey in self.ukeys) == False:
                continue
            self.ukey = ukey
            for phase in ('on', 'off'):
                heatmaps = self.session.load(f'rf/{phase}')
                if heatmaps is None:
                    continue
                hm = heatmaps[self.unit.index]
                if smoothingKernelWidth is not None:
                    hm = gaussianFilter(hm, smoothingKernelWidth)
                # X, Y = np.meshgrid(np.arange(hm.shape[1]), np.arange(hm.shape[0]))
                # mesh = axs[i].pcolor(X, Y, hm, vmin=vrange[0], vmax=vrange[1], cmap=cmap)
                if hm.max() < threshold:
                    continue
                lines = axs[i].contour(hm, np.array([threshold]), colors=colors[phase], linestyles=linestyles[phase])
                centroid = lines.allsegs[0][0].mean(0)
                axs[i].scatter(*centroid, color=colors[phase], marker='.', s=5)
            x = hm.shape[1] / 2
            y = hm.shape[0] / 2
            axs[i].vlines(x, y - 1, y + 1, color='k')
            axs[i].hlines(y, x - 1, x, color='k')

        return

    def _plotExampleRasterplots(
        self,
        axs,
        nTrials=1000,
        responseWindow=(-0.2, 0.5),
        perisaccadicWindow=(-0.05, 0.1),
        **kwargs_,
        ):
        """
        """

        kwargs = {
            'marker': '.',
            'color': 'k',
            's': 7,
            'alpha': 0.3
        }
        kwargs.update(kwargs_)
        
        for i in range(len(self.examples)):

            print(self.examples[i])
            #
            if (self.examples[i] in self.ukeys) == False:
                continue

            #
            self.ukey = self.examples[i]
            gratingMotion = self.preference[self.iUnit]
            trialIndices = np.where(self.session.parseEvents(
                eventName='probe',
                coincident=False,
                eventDirection=gratingMotion,
                coincidenceWindow=perisaccadicWindow
            ))[0]
            trialIndices = np.random.choice(trialIndices, size=nTrials, replace=False)
            t, M, spikeTimestamps = psth2(
                self.session.probeTimestamps[trialIndices],
                self.unit.timestamps,
                window=responseWindow,
                binsize=None,
                returnTimestamps=True
            )
            x, y = list(), list()
            for iTrial, ts in enumerate(spikeTimestamps):
                for iSpike in range(len(ts)):
                    x.append(ts[iSpike])
                    y.append(iTrial)
            axs[i].scatter(
                x,
                y,
                rasterized=True,
                ec='none',
                **kwargs
            )
            axs[i].set_ylim([-1, nTrials + 1])

        return

    def _plotFittingDemo(
        self,
        axs,
        ):
        """
        """

        k = (self.ns[f'params/pref/real/extra'].shape[1] - 1) // 3
        cmap = plt.get_cmap('rainbow', k)
        for i, ukey in enumerate(self.examples):

            #
            if (ukey in self.ukeys) == False:
                continue

            #
            self.ukey = ukey

            #
            yRaw = self.ns[f'ppths/pref/real/extra'][self.iUnit]
            params = self.ns[f'params/pref/real/extra'][self.iUnit]
            abcd = np.delete(params, np.isnan(params))
            A, B, C = np.split(abcd[:-1], 3)
            d = abcd[-1]
            paramsOrdered = np.concatenate([
                np.array([d,]),
                abcd[:-1],
            ])
            gmm = GaussianMixturesModel(k=A.size)
            gmm._popt = paramsOrdered
            yFit = gmm.predict(self.tProbe)

            #
            axs[i].plot(self.tProbe, yRaw, color='k', alpha=0.3)

            #
            for ii in range(gmm.k):
                t = np.linspace(-15 * C[ii], 15 * C[ii], 100) + B[ii]
                yComponent = g(t, A[ii], B[ii], C[ii], d)
                axs[i].plot(t, yComponent, color=cmap(ii), alpha=1)

            ylim = axs[i].get_ylim()
            ymax = np.max(np.abs(ylim))
            axs[i].set_ylim([-ymax, ymax])

        return

    def plotExamples(
        self,
        responseWindow=(-0.2, 0.5),
        figsize=(1.5, 4),
        ):
        """
        """

        fig, axs = plt.subplots(
            ncols=3,
            nrows=len(self.examples),
            gridspec_kw={'height_ratios': (1, 1, 1, 1)},
        )
        figsize = (
            figsize[0] * len(self.examples),
            figsize[1]
        )

        #
        self._plotReceptiveFields(axs[:, 0])
        self._plotExampleRasterplots(axs[:, 1])
        self._plotFittingDemo(axs[:, 2])

        #
        for ax in axs[:, 1:].flatten():
            ax.set_xlim(responseWindow)
        for ax in axs.flatten():
            for sp in ('top', 'right', 'bottom', 'left'):
                ax.spines[sp].set_visible(False)
        for ax in axs[:, 0].flatten():
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect('equal')
        for ax in axs[:, 1]:
            ax.set_yticks([0, 100])
        for ax in axs[:-1, 1:].flatten():
            ax.set_xticklabels([])

        #
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, axs

    def plotHeatmapsByCluster(
        self,
        figsize=(4, 5),
        vrange=(-1, 1),
        cmap='coolwarm',
        ):
        """
        """

        #
        if type(cmap) == str:
            cmap = plt.get_cmap(cmap)

        #
        labels = self.ns['globals/labels']
        include = np.invert(np.isnan(labels))
        labelCounts = np.array([
            np.sum(labels[include] == l)
                for l in (1, 2, 3, -1)
        ])
        fig, axs = plt.subplots(
            ncols=2,
            nrows=4,
            gridspec_kw={'height_ratios': labelCounts},
        )

        #
        peths = self.ns[f'ppths/pref/real/extra'][include]
        amplitudes = np.array([
            np.max(np.abs(peth))
                for peth in peths
        ]).reshape(-1, 1)
        pethsNormalized = peths / amplitudes
        latency = np.array([np.argmax(np.abs(y)) for y in pethsNormalized])
        pethsReverseLatencySorted = pethsNormalized[np.argsort(latency)[::-1]]

        #
        start = 0
        for i, label in enumerate([1, 2, 3, -1]):

            #
            maskByLabel = labels[include] == label

            # Plot unsorted PETHs
            stop = start + maskByLabel.sum()
            y = np.arange(0, maskByLabel.sum(), 1)[::-1]
            axs[i, 0].pcolor(
                self.tProbe,
                y,
                pethsReverseLatencySorted[start: stop, :],
                vmin=vrange[0],
                vmax=vrange[1],
                cmap=cmap,
                rasterized=True
            )
            axs[i, 0].vlines(0, y.max() + 0.5, y.min() - 0.5, color='k')

            # Plot sorted PETHs
            n = np.arange(maskByLabel.sum())
            index = np.argsort(latency[maskByLabel])
            axs[i, 1].pcolor(
                self.tProbe,
                n,
                pethsNormalized[maskByLabel][index],
                vmin=vrange[0],
                vmax=vrange[1],
                cmap=cmap,
                rasterized=True,
            )
            axs[i, 1].vlines(0, -0.5, n.max() + 0.5, color='k')

            # Indicate examples
            xlim = axs[i, 1].get_xlim()
            x = xlim[0] - 0.05
            ukeysByLabel = [ukey
                for ukey, flag in zip(self.ukeys, labels == label)
                    if flag
            ]
            for y, ukey in zip(index, ukeysByLabel):
                if ukey in self.examples:
                    axs[i, 1].plot(x, y, marker='>', color='k', clip_on=False)
                    break
            axs[i, 1].set_xlim(xlim)

            #
            start += maskByLabel.sum()

        #
        for ax in axs[:-1, :].flatten():
            ax.set_xticks([])
        for ax in axs[:, 1].flatten():
            ax.set_yticks([])
        for ax in axs[:-1, 0].flatten():
            ax.spines['bottom'].set_visible(False)
        for ax in axs[1:, 0].flatten():
            ax.spines['top'].set_visible(False)
        for ax in axs[:, 0]:
            ax.set_yticks([])
        axs[-1, 0].set_yticks([0, 100])
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()
        fig.subplots_adjust(hspace=0)

        return fig, axs

    def scatterAmplitudeByComplexity(
        self,
        ):
        """
        """

        #
        amplitude = list()
        pethsNormalized = list()
        for iUnit in range(len(self.ukeys)):
            peth = self.ns[f'ppths/pref/real/extra'][iUnit]
            if np.isnan(peth).all():
                continue
            a = np.abs(peth).max()
            amplitude.append(a)
            pethNormalized = peth / a
            pethsNormalized.append(pethNormalized)
        pethsNormalized = np.array(pethsNormalized)
        complexity = np.abs(pethsNormalized.sum(1)) / (pethsNormalized.shape[1])
        amplitude = np.array(amplitude)

        #
        fig, ax = plt.subplots()
        ax.scatter(
            complexity,
            amplitude,
            marker='.',
            color='k',
            s=10,
            alpha=0.5
        )

        return fig, ax

    def scatterPolarityByLatency(
        self,
        ):
        """
        """

        #
        pethsNormalized = list()
        latency = list()
        for iUnit in range(len(self.ukeys)):
            peth = self.ns[f'ppths/pref/real/extra'][iUnit]
            if np.isnan(peth).all():
                continue
            pethNormalized = peth / np.abs(peth).max()
            pethsNormalized.append(pethNormalized)
            params = self.ns[f'params/pref/real/extra'][iUnit]
            abcd = np.delete(params, np.isnan(params))
            A, B, C = np.split(abcd[:-1], 3)
            latency.append(B[0])
        pethsNormalized = np.array(pethsNormalized)
        polarity = pethsNormalized.sum(1) / np.abs(pethsNormalized).sum(1)
        latency = np.array(latency)

        #
        fig, ax = plt.subplots()
        ax.scatter(
            polarity,
            latency,
            marker='.',
            color='k',
            s=10,
            alpha=0.5
        )

        return