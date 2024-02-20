import h5py
import numpy as np
from matplotlib import pyplot as plt
from myphdlib.general.toolkit import smooth, stretch
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from scipy.optimize import curve_fit as fitCurve
from scipy.signal import find_peaks as findPeaks
from scipy.ndimage import gaussian_filter1d as smooth2

#
exampleUnitIndices = (
    [38, 1], # Mono-phasic, positive/negative
    [3,  296], # Bi-phasic, positive/negative
    [27, 80] # Multi-phasic, positive/negative
)

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

class ClusteringAnalysis():
    """
    """

    def __init__(self):
        """
        """

        self.peths = None
        self.labels = None
        self.include = None
        self.X = None
        self.t = None
        self.te = None
        self.fits = None

        return

    def loadPeths(
        self,
        hdf,
        normalize=True,
        baselineWindow=(-0.2, 0),
        responseWindow=(0, 0.3),
        minimumBaselineLevel=0.5,
        minimumResponseAmplitude=5,
        smoothingKernelWidth=1.5,
        ):
        """
        """

        # Load the PETHs
        with h5py.File(hdf, 'r') as stream:
            rProbe = {
                'left': np.array(stream['rProbe/dg/left/fr']),
                'right': np.array(stream['rProbe/dg/right/fr'])
            }
            np.array(stream['rProbe/dg/right'])
            xProbe = np.array(stream['xProbe'])
            xSaccade = np.array(stream['xSaccade'])
            self.t = np.array(stream['rProbe/dg/left/fr'].attrs['t'])
            dsi = np.array(stream['directionSelectivityIndex'])

        # Define the baseline and response windows
        binIndicesForBaselineWindow = np.where(np.logical_and(
            self.t >= baselineWindow[0],
            self.t <= baselineWindow[1]
        ))[0]
        binIndicesForResponseWindow = np.where(np.logical_and(
            self.t >= responseWindow[0],
            self.t <= responseWindow[1]
        ))[0]

        # Initialize variables
        nUnits = rProbe['left'].shape[0]
        nBins = rProbe['left'].shape[1]
        exclude = np.full(nUnits, False)
        peths = {
            'probe': np.full([nUnits, nBins], np.nan),
            'saccade': np.full([nUnits, nBins], np.nan)
        }

        # Iterate over units
        for iUnit in range(nUnits):

            #
            lowestBaselineLevel = np.max([
                rProbe['left'][iUnit, binIndicesForBaselineWindow].mean(),
                rProbe['right'][iUnit, binIndicesForBaselineWindow].mean(),
            ])
            if lowestBaselineLevel < minimumBaselineLevel:
                exclude[iUnit] = True

            #
            greatestPeakAmplitude = np.max([
                np.max(np.abs(rProbe['left'][iUnit, binIndicesForResponseWindow] - rProbe['left'][iUnit, binIndicesForBaselineWindow].mean())),
                np.max(np.abs(rProbe['right'][iUnit, binIndicesForResponseWindow] - rProbe['right'][iUnit, binIndicesForBaselineWindow].mean())),
            ])
            if greatestPeakAmplitude < minimumResponseAmplitude:
                exclude[iUnit] = True

            #
            if normalize:
                peths['probe'][iUnit, :] = xProbe[iUnit]
                peths['saccade'][iUnit, :] = xSaccade[iUnit]
            else:
                peths['probe'][iUnit, :] = rProbe[iUnit]

        # Filter and smooth PETHs
        self.include = np.invert(exclude)
        self.peths = {
            'probe': peths['probe'][self.include, :],
            'saccade': peths['saccade'][self.include, :]
        }
        if smoothingKernelWidth is not None:
            for k in self.peths.keys():
                for i in range(self.peths[k].shape[0]):
                    self.peths[k][i, :] = smooth2(self.peths[k][i, :], smoothingKernelWidth)

        return dsi[self.include]

    def fitPeths(
        self,
        event='probe',
        sortby='amplitude',
        minimumPeakHeight=0.15,
        maximumPeakWidth=0.1,
        responseWindow=(-0.1, 0.5),
        nPoints=None,
        fillValue=np.nan,
        normalize=True,
        returnFitCurves=False,
        ):
        """
        """

        peths = self.peths[event]
        nUnits = peths.shape[0]
        binIndices = np.where(np.logical_and(
            self.t >= responseWindow[0],
            self.t <= responseWindow[1]
        ))[0]
        if nPoints is None:
            self.te = self.t
        else:
            self.te = np.linspace(self.t.min(), self.t.max(), nPoints)
        fitCurves = list()
        fitParams = list()

        #
        for iUnit in range(nUnits):

            #
            yTrue = peths[iUnit]
            peakIndices = list()
            for coef in (-1, 1):
                ySigned = yTrue[binIndices] * coef
                peakIndices_, peakProps = findPeaks(
                    ySigned,
                    height=minimumPeakHeight,
                )
                for binIndex in peakIndices_:
                    peakIndices.append(binIndex)

            # Detect the peaks in the PSTHs
            peakIndices = np.array(peakIndices)
            k = peakIndices.size
            peakIndices += binIndices.min()
            peakAmplitudes = yTrue[peakIndices]
            peakLatencies = self.t[peakIndices]
            order = np.argsort(peakLatencies)
            peakIndices = peakIndices[order]
            peakAmplitudes = peakAmplitudes[order]

            # Initialize the parameter space
            p0 = np.concatenate([
                np.array([0]),
                peakAmplitudes,
                peakLatencies,
                np.full(k, 0.03)
            ])
            bounds = np.vstack([
                np.array([[-0.02, 0.02]]),
                np.repeat([[
                    -1.05 * np.abs(peakAmplitudes).max(),
                    +1.05 * np.abs(peakAmplitudes).max()
                ]], k, axis=0),
                np.repeat([[responseWindow[0], responseWindow[1]]], k, axis=0),
                np.repeat([[0.005, maximumPeakWidth]], k, axis=0)
            ]).T

            # Fit psths
            gmm = GaussianMixturesModel(k)
            gmm.fit(
                self.t,
                yTrue,
                p0=p0,
                bounds=bounds
            )
            yFit = gmm.predict(self.te)
            fitCurves.append(yFit)

            # Extract parameters
            A, B, C = np.split(gmm._popt[1:], 3)

            #
            if k == 1:
                sample = np.array([
                    A[0], np.nan, np.nan,
                    B[0], np.nan, np.nan,
                    C[0], np.nan, np.nan,
                ])
            elif k == 2:
                sample = np.array([
                    A[0], A[1], np.nan,
                    B[0], B[1], np.nan,
                    C[0], C[1], np.nan,
                ])
            else:
                sample = np.array([*A[:3], *B[:3], *C[:3]])

            # Sort parameter sets
            if sortby == 'latency':
                sortingFeatures = sample[3:6]
                reverseOrder = False
            elif sortby == 'amplitude':
                sortingFeatures = np.abs(sample[0:3])
                reverseOrder = True
            mask = np.invert(np.isnan(sortingFeatures))
            order = np.argsort(sortingFeatures[mask])
            order = np.concatenate([
                np.where(np.isnan(sortingFeatures))[0],
                order,
            ])
            if reverseOrder:
                order = order[::-1]

            #
            sample[0:3] = sample[0:3][order] # Amplitude (Signed)
            sample[3:6] = sample[3:6][order] # Latency
            sample[6:9] = sample[6:9][order] # Width

            #
            fitParams.append(sample)

        #
        self.X = np.array(fitParams)

        # Normalize
        if normalize:
            for start in (0, 3, 6):
                stop = start + 3
                sample = self.X[:, start: stop]
                fmin, fmax = np.nanmin(sample), np.nanmax(sample)
                for j in range(3):
                    self.X[:, start + j] = stretch(
                        self.X[:, start + j],
                        b=(fmin, fmax),
                        c=(0, 1)
                    )

        # Impute
        for j in range(self.X.shape[1]):
            column = self.X[:, j]
            if fillValue == 'mean':
                fillValue_ = np.nanmean(column)
            elif fillValue == 'median':
                fillValue_ = np.nanmedian(column)
            else:
                fillValue_ = fillValue
            self.X[np.isnan(column), j] = np.full(np.isnan(column).sum(), fillValue_)

        #
        self.fits = np.array(fitCurves)
        if returnFitCurves:
            return self.fits

    def predictLabels(
        self,
        clustersByType=(2, 2, 2),
        ):
        """
        """

        # Mono-phasic PSTHs
        iMonophasic = np.where(np.isnan(self.X[:, :3]).sum(1) == 2)[0] # 2 NaNs indicates only one component
        xMonophasic = self.X[iMonophasic, 0:9:3]

        # Bi-phasic PSTHs
        iBiphasic = np.where(np.isnan(self.X[:, :3]).sum(1) == 1)[0]
        xBiphasic = np.concatenate([
            self.X[iBiphasic, 0:9:3],
            self.X[iBiphasic, 1:9:3],
        ], axis=1)

        # Multi-phasic PSTHs
        iMultiphasic = np.where(np.isnan(self.X[:, :3]).sum(1) == 0)[0]
        xMultiphasic = np.concatenate([
            self.X[iMultiphasic, 0:9:3],
            self.X[iMultiphasic, 1:9:3],
            self.X[iMultiphasic, 2:9:3],
        ], axis=1)

        #
        xSet = (
            xMonophasic,
            xBiphasic,
            xMultiphasic,
        )
        iSet = (
            iMonophasic,
            iBiphasic,
            iMultiphasic,
        )

        #
        nUnits = self.peths['probe'].shape[0]
        labels = np.full(nUnits, np.nan)
        c = 0
        for X, i, k in zip(xSet, iSet, clustersByType):
            labelsWithinType = AgglomerativeClustering(n_clusters=k).fit_predict(X)
            labels[i] = labelsWithinType + c
            c += labelsWithinType.max() + 1

        self.labels = labels
        return labels

    def saveLabels(
        self,
        hdf,
        name,
        ):
        """
        """

        with h5py.File(hdf, 'a') as stream:
            nUnits = stream['rProbe/dg/left/fr'].shape[0]
            labels = np.full(nUnits, np.nan)
            labels[self.include] = self.labels
            labels = labels.reshape(-1, 1)
            if f'clusterLabels/{name}' in stream:
                del stream[f'clusteringLabels/{name}']
            ds = stream.create_dataset(
                f'clusteringLabels/{name}',
                shape=labels.shape,
                dtype=labels.dtype,
                data=labels
            )

        return

    def measureLatency(
        self,
        alignment='peak',
        minimumResponseAmplitude=0.3,
        responseSigns=(-1, 1),
        ):
        """
        """

        latency = list()
        for y in self.fits:
            l = np.inf
            a = 0
            for coef in responseSigns:

                #
                if alignment == 'onset':
                    peakIndices, peakProps = findPeaks(coef * y, height=minimumResponseAmplitude)
                    for peakIndex in peakIndices:
                        if self.te[peakIndex] < l:
                            l = self.te[peakIndex]
                
                #
                elif alignment == 'peak':
                    if np.max(coef * y) > a:
                        a = np.max(coef * y)
                        l = self.te[np.argmax(coef * y)]
                    
            #
            latency.append(l)

        #
        latency =  np.array(latency)

        return latency

    def plotPeths(
        self,
        figsize=(2, 8.5),
        vrange=(-0.7, 0.7),
        cmap='coolwarm',
        ):
        """
        """

        fig, ax = plt.subplots()
        n = np.arange(self.fits.shape[0])
        latency = self.measureLatency()
        index = np.argsort(latency)
        ax.pcolor(
            self.te,
            n,
            self.fits[index],
            vmin=vrange[0],
            vmax=vrange[1],
            cmap=cmap,
            rasterized=True,
        )
        ax.vlines(0, 0, n.max(), color='k')
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, ax

    def plotPethsWithClustering(
        self,
        order=None,
        figsize=(2, 8.5),
        vrange=(-0.7, 0.7),
        cmap='Blues_r',
        ):
        """
        """

        labels = self.predictLabels([2, 2, 2])
        uniqueLabels, labelCounts = np.unique(labels, return_counts=True)
        if order is not None:
            uniqueLabels = uniqueLabels[order]
            labelCounts = labelCounts[order]
        fig, axs = plt.subplots(
            nrows=uniqueLabels.size,
            gridspec_kw={'height_ratios': labelCounts},
        )
        latency = self.measureLatency(
            responseSigns=(-1, +1)
        )
        for i, label in enumerate(uniqueLabels):
            m = labels == label
            n = np.arange(m.sum())
            fits = self.fits[m]
            index = np.argsort(latency[m])
            axs[i].pcolor(
                self.te,
                n,
                fits[index],
                vmin=vrange[0],
                vmax=vrange[1],
                cmap=cmap,
                rasterized=True,
            )
            axs[i].vlines(0, 0, n.max(), color='k')
        for ax in axs[:-1]:
            ax.set_xticks([])
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()
        fig.subplots_adjust(hspace=0)

        return fig, axs

    def plotExampleCurves(
        self,
        figsize=(4, 5),
        colormap='coolwarm',
        alpha=0.7,
        lineWeight=0.15,
        ):
        """
        """

        fig, axs = plt.subplots(nrows=3, ncols=2, sharey=True)
        cmap = plt.get_cmap(colormap, 2)
        for i, (u1, u2) in enumerate(exampleUnitIndices):
            axs[i, 0].plot(
                self.t,
                self.peths['probe'][u1],
                color='gray',
                lw=lineWeight
            )
            axs[i, 0].plot(
                self.te,
                self.fits[u1],
                color=cmap(1),
                lw=lineWeight
            )
            axs[i, 1].plot(
                self.t,
                self.peths['probe'][u2],
                color='gray',
                lw=lineWeight
            )
            axs[i, 1].plot(
                self.te,
                self.fits[u2],
                color=cmap(0),
                alpha=alpha,
                lw=lineWeight   
            )
        for ax in axs.flatten():
            ax.set_ylim([-1, 1])
        for ax in axs[:-1, :].flatten():
            ax.set_xticks([])
        for ax in axs[-1, :].flatten():
            ax.set_xticks([-0.2, 0, 0.2, 0.4])
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.15)

        return fig, axs

