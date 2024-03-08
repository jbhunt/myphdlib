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

    def __init__(self):
        """
        """

        super().__init__()
        self.peths = {
            'raw': None,
            'normalized': None,
            'standardized': None
        }
        self.ambc = None
        self.params = None
        self.t = None
        self.rss = None
        self.labels = None
        self.k = None

        #
        self.examples = (
            ('2023-05-15', 'mlati7', 226),
            ('2023-06-30', 'mlati9', 14),
            ('2023-07-05', 'mlati9', 107),
            ('2023-07-14', 'mlati9', 40),
            ('2023-07-25', 'mlati10', 237)
        )

        return

    def saveNamespace(
        self,
        hdf,
        ):
        """
        """

        if type(hdf) != pl.Path:
            hdf = pl.Path(hdf)
        if hdf.exists() == False:
            raise Exception('Base table does not exist')

        d = {
            'rProbe/dg/preferred/raw/fr': self.peths['raw'],
            'rProbe/dg/preferred/normalized/fr': self.peths['normalized'],
            'rProbe/dg/preferred/standardized/fr': self.peths['standardized'],
            'rProbe/dg/preferred/ambc': self.ambc,
            'gmm/params': self.params,
            'gmm/rss': self.rss,
            'gmm/labels': self.labels,
            'gmm/k': self.k
        }
        m = findOverlappingUnits(self.ukeys, hdf)
        nRows = m.size
        with h5py.File(hdf, 'a') as stream:
            for k, v in d.items():
                if v is None:
                    continue
                nCols = 1 if len(v.shape) == 1 else v.shape[1]
                data = np.full([m.size, nCols], np.nan)
                data[m, :] = v.reshape(-1, nCols)
                if k in stream:
                    del stream[k]
                ds = stream.create_dataset(
                    k,
                    data.shape,
                    data.dtype,
                    data=data
                )
                if k.endswith('fr'):
                    ds.attrs['t'] = self.t

        return

    def loadNamespace(
        self,
        ):
        """
        """

        return

    def computePeths(
        self,
        responseWindow=(-0.2, 0.5),
        baselineWindow=(-0.25, -0.05),
        binsize=0.01,
        smoothingKernelWidth=0.01,
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
        self.ambc = np.full([nUnits, 4], np.nan)
        for k in self.peths.keys():
            self.peths[k] = np.full([nUnits, nBins], np.nan)
        for iUnit, ukey in enumerate(self.ukeys):

            #
            end = None if iUnit + 1 == nUnits else '\r'
            print(f'Working on {iUnit + 1} out of {nUnits} units', end=end)
            self.ukey = ukey

            # Initialize parameters
            x = np.full(nBins, np.nan)
            a = 0 # Amplitude of preferred direction
            m = None # Probe direction
            b = None # Baseline
            c = None # Scaling factor

            # TODO: Exclude peri-saccadic trials
            for probeMotion, probeDirection in zip([-1, 1], ['left', 'right']):
                t, fr = self.unit.kde(
                    self.session.probeTimestamps[self.session.gratingMotionDuringProbes == probeMotion],
                    responseWindow=responseWindow,
                    binsize=binsize,
                    sigma=smoothingKernelWidth,
                )
                t, M = psth2(
                    self.session.probeTimestamps[self.session.gratingMotionDuringProbes == probeMotion],
                    self.unit.timestamps,
                    window=baselineWindow,
                    binsize=None
                )
                bl = M.flatten() / np.diff(baselineWindow).item()
                if np.abs(fr - bl.mean()).max() > a:
                    x = fr
                    a = np.abs(fr - bl.mean()).max()
                    m = probeMotion
                    b = bl.mean()
                    c = bl.std()

            #
            if c == 0:
                c = np.nan
            self.ambc[iUnit] = np.array([a, m, b, c])

            # Store the raw PSTH
            self.peths['raw'][iUnit] = x

            # Normalize
            self.peths['normalized'][iUnit] = (x - b) / a

            # Standardize
            if np.isnan(c):
                self.peths['standardized'][iUnit] = np.full(x.size, np.nan)
            else:
                self.peths['standardized'][iUnit] = (x - b) / c

        return

    def fitPeths(
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
            'maximumAmplitudeShift': 0.001 
        }
        kwargs.update(kwargs_)

        #
        nUnits = len(self.ukeys)
        self.k = np.full(nUnits, np.nan)
        self.rss = np.full(nUnits, np.nan)
        self.params = np.full([nUnits, int(3 * kmax + 1)], np.nan)

        #
        for iUnit in range(nUnits):

            end = None if iUnit + 1 == nUnits else '\r'
            print(f'Fitting GMM for unit {iUnit + 1} out of {nUnits} units', end=end)

            #
            # yRaw = self.peths['raw'][iUnit]
            yNormal = self.peths['normalized'][iUnit]
            yStandard = self.peths['standardized'][iUnit]

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
            self.k[iUnit] = k

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
            self.rss[iUnit] = np.sum(np.power(yFit - yStandard, 2)) / np.sum(np.power(yStandard, 2))

            # Extract the parameters of the fit GMM
            d, abc = gmm._popt[0], gmm._popt[1:]
            A, B, C = np.split(abc, 3)
            order = np.argsort(B)[::-1] # Sort by amplitude
            params = np.concatenate([
                A[order],
                B[order],
                C[order],
            ])
            self.params[iUnit, :params.size] = params
            self.params[iUnit, -1] = d

        return

    def predictLabels(
        self,
        ):
        """
        """

        nUnits = len(self.ukeys)
        self.labels = np.full(nUnits, np.nan)
        for i, y in enumerate(self.peths['normalized']):

            # No peaks detected
            if np.isnan(self.k[i]):
                self.labels[i] = np.nan
                continue

            # Negative
            if y[np.argmax(np.abs(y))] < 0:
                self.labels[i] = -1

            # Positive
            else:
                self.labels[i] = self.k[i]

        return

    def plotReceptiveFields(
        self,
        threshold=2,
        vrange=(-5, 5),
        cmap='binary_r',
        figsize=(7, 2),
        phase='on',
        smoothingKernelWidth=0.5,
        ):
        """
        """

        fig, axs = plt.subplots(ncols=len(self.examples))
        if len(self.examples) == 1:
            axs = [axs,]
        for i, ukey in enumerate(self.examples):
            self.ukey = ukey
            heatmaps = self.session.load(f'population/rf/{phase}')
            if heatmaps is None:
                continue
            hm = heatmaps[self.unit.index]
            if smoothingKernelWidth is not None:
                hm = gaussianFilter(hm, smoothingKernelWidth)
            X, Y = np.meshgrid(np.arange(hm.shape[1]), np.arange(hm.shape[0]))
            mesh = axs[i].pcolor(X, Y, hm, vmin=vrange[0], vmax=vrange[1], cmap=cmap)
            if hm.max() < threshold:
                continue
            lines = axs[i].contour(hm, np.array([threshold]), colors=['k'])

        #
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, axs

    def plotFittingDemo(
        self,
        figsize=(3.5, 9),
        ):
        """
        """

        if len(self.examples) == 1:
            fig, ax = plt.subplots()
            axs = [ax,]
        else:
            fig, axs = plt.subplots(nrows=len(self.examples), sharey=False, sharex=True)
        cmap = plt.get_cmap('rainbow', np.nanmax(self.k))
        for i, ukey in enumerate(self.examples):

            #
            self.ukey = ukey

            #
            for j, (date, animal, cluster) in enumerate(self.ukeys):
                if date == ukey[0] and animal == ukey[1] and cluster == ukey[2]:
                    break

            yRaw = self.peths['standardized'][j]
            gmm = GaussianMixturesModel(k=int(self.k[j]))
            params = self.params[j][np.invert(np.isnan(self.params[j]))]
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
            axs[i].set_title(f'k={gmm.k}', fontsize=10)

        #
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, axs

    def plotPeths(
        self,
        form='normalized',
        order=None,
        figsize=(4, 5),
        vrange=(-1, 1),
        cmap='coolwarm',
        ):
        """
        """

        if self.labels is None:
            self.predictLabels()
        uniqueLabels, labelCounts = np.unique(self.labels, return_counts=True)
        labelCounts = np.delete(labelCounts, np.isnan(uniqueLabels))
        uniqueLabels = np.delete(uniqueLabels, np.isnan(uniqueLabels))
        if order is not None:
            uniqueLabels = uniqueLabels[order]
            labelCounts = labelCounts[order]
        fig, axs = plt.subplots(
            ncols=2,
            nrows=uniqueLabels.size,
            gridspec_kw={'height_ratios': labelCounts},
            # sharey=True
        )
        latency = np.array([np.argmax(np.abs(y)) for y in self.peths['normalized']])
        pethsReverseLatencySorted = self.peths[form][np.argsort(latency)[::-1]]
        start = 0
        for i, label in enumerate(uniqueLabels):

            #
            m = self.labels == label

            #
            stop = start + m.sum()
            y = np.arange(0, m.sum(), 1)[::-1]
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
            start += m.sum()

            #
            n = np.arange(m.sum())
            index = np.argsort(latency[m])
            axs[i, 1].pcolor(
                self.t,
                n,
                self.peths[form][m][index],
                vmin=vrange[0],
                vmax=vrange[1],
                cmap=cmap,
                rasterized=True,
            )
            axs[i, 1].vlines(0, -0.5, n.max() + 0.5, color='k')

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