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
from myphdlib.figures.analysis import AnalysisBase, GaussianMixturesModel, g, findOverlappingUnits

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
        self.peths = {
            'raw': None,
            'normal': None,
            'standard': None
        }
        self.model = {
            'params': None,
            'labels': None,
            'rss': None,
            'k': None
        }
        self.features = {
            'a': None,
            'd': None,
            'm': None,
            's': None,
        }
        self.t = None
        self.filter = None

        #
        self.examples = (
            ('2023-07-11', 'mlati10', 291),
            ('2023-07-19', 'mlati10', 327),
            ('2023-07-11', 'mlati10', 295),
            ('2023-07-19', 'mlati10', 268),
        )

        return

    def saveNamespace(
        self,
        ):
        """
        """

        if pl.Path(self.hdf).exists() == False:
            raise Exception('Data store does not exist')

        #
        d = {
            'clustering/peths/raw': self.peths['raw'],
            'clustering/peths/normal': self.peths['normal'],
            'clustering/peths/standard': self.peths['standard'],
            'clustering/model/params': self.model['params'],
            'clustering/model/labels': self.model['labels'],
            'clustering/model/rss': self.model['rss'],
            'clustering/model/k': self.model['k'],
            'clustering/description/a': self.features['a'],
            'clustering/description/d': self.features['d'],
            'clustering/description/m': self.features['m'],
            'clustering/description/s': self.features['s'],
            'clustering/filter': self.filter,
        }

        #
        mask = self._intersectUnitKeys(self.ukeys)

        #
        with h5py.File(self.hdf, 'a') as stream:
            for k, v in d.items():
                if v is None:
                    continue
                nCols = 1 if len(v.shape) == 1 else v.shape[1]
                data = np.full([mask.size, nCols], np.nan)
                data[mask, :] = v.reshape(-1, nCols)
                if k in stream:
                    del stream[k]
                ds = stream.create_dataset(
                    k,
                    data.shape,
                    data.dtype,
                    data=data
                )

                # Save the bin centers for all PETH datasets
                if 'peths' in pl.Path(k).parts:
                    ds.attrs['t'] = self.t

        return

    def loadNamespace(
        self,
        ):
        """
        """

        d = {
            'clustering/peths/raw': (self.peths, 'raw'),
            'clustering/peths/normal': (self.peths, 'normal'),
            'clustering/peths/standard': (self.peths, 'standard'),
            'clustering/model/params': (self.model, 'params'),
            'clustering/model/labels': (self.model, 'labels'),
            'clustering/model/rss': (self.model, 'rss'),
            'clustering/model/k': (self.model, 'k'),
            'clustering/description/a': (self.features, 'a'),
            'clustering/description/d': (self.features, 'd'),
            'clustering/description/m': (self.features, 'm'),
            'clustering/description/s': (self.features, 's'),
            'clustering/filter': ('filter', None),
        }

        with h5py.File(self.hdf, 'r') as stream:
            for path, (attr, key) in d.items():
                if path in stream:
                    ds = stream[path]
                    if 't' in ds.attrs.keys() and self.t is None:
                        self.t = ds.attrs['t']
                    value = np.array(ds)
                    if len(value.shape) == 2 and value.shape[-1] == 1:
                        value = value.flatten()
                    if key is None:
                        setattr(self, attr, value)
                    else:
                        attr[key] = value

        return

    def computeExtrasaccadicPeths(
        self,
        responseWindow=(-0.2, 0.5),
        baselineWindow=(-0.3, 0),
        binsize=0.01,
        smoothingKernelWidth=0.01,
        perisaccadicWindow=(-0.1, 0.1),
        ):
        """
        """

        self.t, nTrials, nBins = psth2(
            np.zeros(1),
            np.zeros(1),
            window=responseWindow,
            binsize=binsize,
            returnShape=True
        )

        #
        nUnits = len(self.ukeys)
        for k in self.features.keys():
            self.features[k] = np.full(nUnits, np.nan)
        for k in self.peths.keys():
            self.peths[k] = np.full([nUnits, nBins], np.nan)
        for iUnit, ukey in enumerate(self.ukeys):

            #
            end = None if iUnit + 1 == nUnits else '\r'
            print(f'Working on {iUnit + 1} out of {nUnits} units', end=end)
            self.ukey = ukey # NOTE: Very important

            # Initialize feature set
            y = np.full(nBins, np.nan)
            a = None # Amplitude
            d = None # Probe direction
            m = None # Mean FR
            s = None # Standard deviation

            #
            for gratingMotion in (-1, 1):

                # Select just the extra-saccadic trials
                trialIndices = np.where(np.vstack([
                    self.session.gratingMotionDuringProbes == gratingMotion,
                    np.logical_or(
                        self.session.probeLatencies > perisaccadicWindow[1],
                        self.session.probeLatencies < perisaccadicWindow[0]
                    )
                ]).all(0))[0]

                # Compute firing rate
                t, y_ = self.unit.kde(
                    self.session.probeTimestamps[trialIndices],
                    responseWindow=responseWindow,
                    binsize=binsize,
                    sigma=smoothingKernelWidth,
                )

                # Estimate baseline mean and standard deviation
                t, bl = self.unit.kde(
                    self.session.probeTimestamps[trialIndices],
                    responseWindow=baselineWindow,
                    binsize=binsize,
                    sigma=smoothingKernelWidth
                )
                m_ = bl.mean()
                s_ = bl.std()

                # Compute new features
                a_ = np.abs(y_[self.t > 0] - m_).max()
                d_ = gratingMotion

                # Override current feature set if amplitude is greater
                if a is None or a_ > a:
                    y = y_
                    a = a_
                    d = d_
                    m = m_
                    s = s_

            #
            self.features['a'][iUnit] = a
            self.features['d'][iUnit] = d
            self.features['m'][iUnit] = m
            self.features['s'][iUnit] = s if s != 0 else np.nan

            # Store the raw PSTH
            self.peths['raw'][iUnit] = y

            # Normalize
            self.peths['normal'][iUnit] = (y - m) / a

            # Standardize
            if np.isnan(s):
                self.peths['standard'][iUnit] = np.full(y.size, np.nan)
            else:
                self.peths['standard'][iUnit] = (y - m) / s

        return

    def fitExtrasaccadicPeths(
        self,
        kmax=5,
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
        self.model['k'] = np.full(nUnits, np.nan)
        self.model['rss'] = np.full(nUnits, np.nan)
        self.model['params'] = np.full([nUnits, int(3 * kmax + 1)], np.nan)

        #
        for iUnit in range(nUnits):

            end = None if iUnit + 1 == nUnits else '\r'
            print(f'Fitting GMM for unit {iUnit + 1} out of {nUnits} units', end=end)

            #
            yNormal = self.peths['normal'][iUnit]
            yStandard = self.peths['standard'][iUnit]

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
                    if self.t[peakIndices_[iPeak]] <= 0:
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
            peakLatencies = self.t[peakIndices]

            # Use only the k largest peaks
            if peakIndices.size > kmax:
                index = np.argsort(np.abs(peakAmplitudes))[::-1]
                peakIndices = peakIndices[index][:kmax]
                peakProminences = peakProminences[index][:kmax]
                peakAmplitudes = peakAmplitudes[index][:kmax]
                peakLatencies = peakLatencies[index][:kmax]
            
            #
            k = peakIndices.size
            self.model['k'][iUnit] = k

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
                self.t,
                yStandard,
                p0=p0,
                bounds=bounds
            )
            yFit = gmm.predict(self.t)
            self.model['rss'][iUnit] = np.sum(np.power(yFit - yStandard, 2)) / np.sum(np.power(yStandard, 2))

            # Extract the parameters of the fit GMM
            d, abc = gmm._popt[0], gmm._popt[1:]
            A, B, C = np.split(abc, 3)
            order = np.argsort(np.abs(A))[::-1] # Sort by amplitude
            params = np.concatenate([
                A[order],
                B[order],
                C[order],
            ])
            self.model['params'][iUnit, :params.size] = params
            self.model['params'][iUnit, -1] = d

        return

    def predictLabels(
        self,
        ):
        """
        """

        nUnits = len(self.ukeys)
        self.model['labels'] = np.full(nUnits, np.nan)
        for i, y in enumerate(self.peths['normal']):

            # No peaks detected
            if np.isnan(self.model['k'][i]):
                self.model['labels'][i] = np.nan
                continue

            # Negative
            if y[np.argmax(np.abs(y))] < 0:
                self.model['labels'][i] = -1

            # Positive (multiphasic)
            elif self.model['k'][i] >= 3:
                self.model['labels'][i] = 3

            # Positive (Mono- or Biphasic)
            else:
                self.model['labels'][i] = self.model['k'][i]

        return

    def createFilter(
        self,
        minimumResponseLatency=0.03,
        minimumResponseAmplitude=5,
        ):
        """
        """

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
        nTrials=300,
        responseWindow=(-0.2, 0.5),
        perisaccadicWindow=(-0.05, 0.1),
        **kwargs_,
        ):
        """
        """

        kwargs = {
            'marker': '.',
            'color': 'k',
            's': 3,
            'alpha': 0.25
        }
        kwargs.update(kwargs_)
        
        for i in range(len(self.examples)):

            #
            self.ukey = self.examples[i]
            gratingMotion = self.features['d'][self.iUnit]
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

        cmap = plt.get_cmap('rainbow', np.nanmax(self.model['k']))
        for i, ukey in enumerate(self.examples):

            #
            self.ukey = ukey

            #
            yRaw = self.peths['standard'][self.iUnit]
            gmm = GaussianMixturesModel(k=int(self.model['k'][self.iUnit]))
            params = self.model['params'][self.iUnit][np.invert(np.isnan(self.model['params'][self.iUnit]))]
            paramsOrdered = np.concatenate([
                np.array([params[-1],]),
                params[:-1]
            ])
            gmm._popt = paramsOrdered
            yFit = gmm.predict(self.t)

            #
            axs[i].plot(self.t, yRaw, color='k', alpha=0.3)

            #
            A, B, C = np.split(params[:-1], 3)
            d = params[-1]
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
        form='normal',
        figsize=(4, 5),
        vrange=(-1, 1),
        cmap='coolwarm',
        ):
        """
        """

        #
        mask = np.logical_and(
            np.invert(np.isnan(self.model['labels'])),
            self.filter
        )
        labelCounts = np.array([
            np.sum(self.model['labels'][mask] == l)
                for l in (1, 2, 3, -1)
        ])
        fig, axs = plt.subplots(
            ncols=2,
            nrows=4,
            gridspec_kw={'height_ratios': labelCounts},
        )

        #
        peths = self.peths[form][mask]
        latency = np.array([np.argmax(np.abs(y)) for y in peths])
        pethsReverseLatencySorted = peths[np.argsort(latency)[::-1]]

        #
        start = 0
        for i, label in enumerate([1, 2, 3, -1]):

            #
            maskByLabel = self.model['labels'][mask] == label

            # Plot unsorted PETHs
            stop = start + maskByLabel.sum()
            y = np.arange(0, maskByLabel.sum(), 1)[::-1]
            axs[i, 0].pcolor(
                self.t,
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
                self.t,
                n,
                peths[maskByLabel][index],
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
                for ukey, flag in zip(self.ukeys, np.vstack([self.filter, self.model['labels'] == label]).all(0))
                    if flag
            ]
            for y, ukey in enumerate(ukeysByLabel):
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

    def plotComplexityByAmplitude(
        self,
        ):
        """
        """

        fig, ax = plt.subplots()
        x = self.k.flatten()
        y = self.params[:, 0]
        ax.scatter(x, y)

        return fig, ax