import h5py
import numpy as np
from matplotlib import pyplot as plt
from myphdlib.general.toolkit import smooth, stretch
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, Birch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import silhouette_score
from scipy.optimize import curve_fit as fitCurve
from scipy.signal import find_peaks as findPeaks

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

        return

    def loadPeths(
        self,
        hdf,
        normalize=True,
        baselineWindow=(-0.2, 0),
        responseWindow=(0, 0.3),
        minimumBaselineLevel=0.5,
        minimumResponseAmplitude=3,
        smoothingWindowSize=None,
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
                # peths['saccade'][iUnit, :] = rSaccade[iUnit]

        # Filter and smooth PETHs
        self.include = np.invert(exclude)
        self.peths = {
            'probe': peths['probe'][self.include, :],
            'saccade': peths['saccade'][self.include, :]
        }
        if smoothingWindowSize is not None:
            for k in self.peths.keys():
                self.peths[k] = smooth(self.peths[k], smoothingWindowSize, axis=1)

        return

    def fitPeths(
        self,
        event='probe',
        sortby='amplitude',
        tRange=(-0.2, 0.5),
        nx=None,
        returnFitCurves=False,
        minimumPeakHeight=0.15,
        maximumPeakWidth=0.1,
        ):
        """
        """

        peths = self.peths[event]
        nUnits = peths.shape[0]
        binIndices = np.where(np.logical_and(
            self.t >= tRange[0],
            self.t <= tRange[1]
        ))[0]
        if nx is None:
            tExpanded = self.t
        else:
            tExpanded = np.linspace(self.t.min(), self.t.max(), nx)
        fitCurves = list()
        fitParams = list()
        fillValues = {
            'a': np.nan,
            'b': np.nan,
            'c': np.nan,
        }

        #
        for iUnit in range(nUnits):

            #
            yTrue = peths[iUnit]
            peakIndices = list()
            for coef in (-1, 1):
                ySigned = yTrue[binIndices] * coef
                peakIndices_, peakProps = findPeaks(ySigned, height=minimumPeakHeight)
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
                np.repeat([[tRange[0], tRange[1]]], k, axis=0),
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
            yFit = gmm.predict(tExpanded)
            fitCurves.append(yFit)

            # Extract parameters
            A, B, C = np.split(gmm._popt[1:], 3)

            #
            if k == 1:
                sample = np.array([
                    A[0], fillValues['a'], fillValues['a'],
                    B[0], fillValues['b'], fillValues['b'],
                    C[0], fillValues['c'], fillValues['c'],
                ])
            elif k == 2:
                sample = np.array([
                    A[0], A[1], fillValues['a'],
                    B[0], B[1], fillValues['b'],
                    C[0], C[1], fillValues['c'],
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
            import pdb; pdb.set_trace()

            #
            sample[0:3] = sample[0:3][order] # Amplitude (Signed)
            sample[3:6] = sample[3:6][order] # Latency
            sample[6:9] = sample[6:9][order] # Width

            #
            fitParams.append(sample)

        #
        self.X = np.array(fitParams)

        if returnFitCurves:
            return np.array(fitCurves)
        
    def measureClusteringPerformance(
        self,
        kmin=3,
        kmax=30,
        ):
        """
        """


        # Mono-phasic PSTHs
        sampleIndices = np.where(np.isnan(self.X[:, :3]).sum(1) == 2)[0]
        xMonophasic = self.X[sampleIndices, 0:9:3]

        # Bi-phasic PSTHs
        sampleIndices = np.where(np.isnan(self.X[:, :3]).sum(1) == 1)[0]
        xBiphasic = np.concatenate([
            self.X[sampleIndices, 0:9:3],
            self.X[sampleIndices, 1:9:3],
        ], axis=1)

        # Multi-phasic PSTHs
        sampleIndices = np.where(np.isnan(self.X[:, :3]).sum(1) == 0)[0]
        xMultiphasic = np.concatenate([
            self.X[sampleIndices, 0:9:3],
            self.X[sampleIndices, 1:9:3],
            self.X[sampleIndices, 2:9:3],
        ], axis=1)
        
        #
        xSets = (
            xMonophasic,
            xBiphasic,
            xMultiphasic
        )

        #
        curves = list()
        for X in xSets:
            xScaled = MinMaxScaler().fit_transform(X)
            curve = list()
            for k in range(kmin, kmax + 1, 1):
                labels = AgglomerativeClustering(n_clusters=k).fit_predict(xScaled)
                curve.append(silhouette_score(xScaled, labels))
            curves.append(curve)

        return np.arange(kmin, kmax + 1, 1), np.array(curves)

    def cluster(
        self,
        clustersByType=(4, 6, 7),
        ):
        """
        """

        # Mono-phasic PSTHs
        iMonophasic = np.where(np.isnan(self.X[:, :3]).sum(1) == 2)[0]
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
        nUnits = self.peths['probe'].shape[0]
        labels = np.full(nUnits, np.nan)
        c = 0
        for X, i, k in zip(xSet, iSet, clustersByType):
            xScaled = MinMaxScaler().fit_transform(X)
            labelsWithinType = AgglomerativeClustering(n_clusters=k).fit_predict(xScaled)
            labels[i] = labelsWithinType + c
            # import pdb; pdb.set_trace()
            c += labelsWithinType.max() + 1

        return labels

    def saveClusterLabels(
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

    def plotPeths(
        self,
        plot='line',
        figsize=(3, 8.5),
        minimumClusterSize=15,
        order=None,
        z=None,
        ):
        """
        """

        labels, counts = np.unique(self.labels, return_counts=True)
        mask = counts >= minimumClusterSize
        labels = labels[mask]
        counts = counts[mask]
        if order is not None:
            labels = labels[order]
            counts = counts[order]
        if plot == 'heatmap':
            fig, axs = plt.subplots(nrows=labels.size, ncols=2, gridspec_kw={'height_ratios': counts})
        else:
            fig, axs = plt.subplots(nrows=labels.size, sharey=True, sharex=True)
        for i, l in enumerate(labels):
            color = f'C{int(i)}'
            if z is None:
                zProbe = self.peths['probe'][self.labels == l, :]
            else:
                zProbe = z[self.labels == l, :]
            zSaccade = self.peths['saccade'][self.labels == l, :]
            if plot == 'heatmap':
                axs[i, 0].set_ylabel(f'C{int(i + 1)}', va='center', rotation=0, labelpad=15)
                y = np.arange(zProbe.shape[0])
                shuffledIndices = np.arange(zProbe.shape[0])
                orderedIndices = np.argsort(np.array([np.argmax(np.abs(zi)) for zi in zProbe]))
                np.random.shuffle(shuffledIndices)
                axs[i, 0].pcolor(
                    self.t,
                    y,
                    zProbe[orderedIndices, :],
                    vmin=-0.7, vmax=0.7
                )
                axs[i, 1].pcolor(
                    self.t,
                    y,
                    zSaccade[orderedIndices, :],
                    vmin=-0.7, vmax=0.7
                )
                for ax in axs[i, :]:
                    ax.set_xticklabels([])
                axs[-1, 0].set_xlabel('Time from event (sec)')
            elif plot == 'line':
                axs[i].set_ylabel(f'C{int(i + 1)}', va='center', rotation=0, labelpad=15)
                y = zProbe.mean(0)
                e = zProbe.std(0)
                axs[i].plot(self.t, y, color=color)
                axs[i].plot(self.t, zSaccade.mean(0), color='k', alpha=0.5)
                axs[i].set_xticklabels([])
                axs[-1].set_xlabel('Time from event (sec)')
        if plot == 'heatmap':
            for ax in axs.flatten():
                ax.set_yticks([])
        elif plot == 'line':
            for ax in axs.flatten():
                ax.set_yticks([-1, 0, 1])
                ax.set_yticklabels([])
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.15)

        return fig, axs

    def plotSubspace(
        self,
        xy=None,
        **kwargs
        ):
        """
        """

        #
        fig, ax = plt.subplots()
        if xy is None:
            xy = PCA(n_components=2).fit_transform(self.X)
        for i, l in enumerate(np.unique(self.labels)):
            indices = np.where(self.labels == l)[0]
            ax.scatter(xy[indices, 0], xy[indices, 1], color=f'C{int(i)}', **kwargs)
            xc, yc = xy[indices].mean(0)
            ax.scatter(xc, yc, facecolor=f'C{int(i)}', edgecolor='k', marker='D', s=30, label=f'C{int(i + 1)}')
        ax.legend()

        return fig, ax