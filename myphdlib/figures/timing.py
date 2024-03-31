import h5py
import numpy as np
from scipy.stats import sem
from matplotlib import pyplot as plt
from myphdlib.figures.analysis import AnalysisBase, GaussianMixturesModel, g, findOverlappingUnits
from myphdlib.figures.modulation import BasicSaccadicModulationAnalysis
from myphdlib.general.toolkit import psth2

class SaccadicModulationTimingAnalysis(BasicSaccadicModulationAnalysis):
    """
    """

    def __init__(
        self,
        **kwargs
        ):
        """
        """

        super().__init__(**kwargs)

        self.latencies = None
        self.peths = { # Standardized PSTHs
            'extrasaccadic': None,
            'perisaccadic': None
        }
        self.terms = { # Raw PSTHs
            'rProbe': {
                'extrasaccadic': None,
                'perisaccadic': None
            },
            'rMixed': None,
            'rSaccade': None,
        }
        self.windows = None

        return

    def loadNamespace(self, hdf):
        """
        """
        m = findOverlappingUnits(self.ukeys, hdf)
        datasets = {
            'boostrap/sign': 'msigns',
            'boostrap/p': 'pvalues',
        }
        with h5py.File(hdf, 'r') as stream:
            for path, attribute in datasets.items():
                if path in stream:
                    ds = np.array(stream[path][m])
                    self.__setattr__(attribute, ds)
        super().loadNamespace(hdf)
        return

    def _plotModulationByProbeLatency(
        self,
        ax=None,
        iComp=0,
        fill=False,
        windowIndex=5,
        minimumResponseAmplitude=2,
        ):
        """
        """

        #
        if ax is None:
            fig, ax = plt.subplots()

        #
        nUnits, nBins, nWindows = self.peths['perisaccadic'].shape
        binCenters = np.mean(self.windows[:-1], axis=1)

        # Exclude small amplitude units
        m = self.params[:, 0] > minimumResponseAmplitude

        # Determine the sign of modulation for each curve
        labels = np.full(m.sum(), np.nan)
        for iUnit in np.where(m)[0]:
            yNormed = self.modulation[m, iComp, windowIndex] / self.params[iUnit, 0]
            p = self.pvalues[iUnit, windowIndex, iComp]
            if p < 0.05:
                if yNormed < 0:
                    labels[iUnit] = -1
                else:
                    labels[iUnit] = +1

        # Define the color for each curve
        colors = np.full([m.sum()], 'tab:gray')
        for i, l in enumerate(labels):
            if l == -1:
                colors[i] = 'tab:blue'
            elif l == 1:
                colors[i] = 'tab:red'

        # Collect all curves (normalized)
        samples = np.full([m.sum(), nWindows - 1], np.nan)
        for iWin in range(nWindows)[:-1]:
            yNormed = self.modulation[m, iComp, iWin] / self.params[m, 0]
            samples[:, iWin] = yNormed
    
        # Plot individual curves
        for i, ln in enumerate(samples):
            ax.plot(binCenters, np.clip(ln, -1, 1), color=colors[i], alpha=0.3, lw=0.5)

        # Plot average curves per label
        uniqueLabels = (-1, 0, 1)
        uniqueColors = ('tab:blue', 'tab:gray', 'tab:red')
        for i, l in enumerate(uniqueLabels):
            y = np.nanmean(samples[labels == l], axis=0)
            ax.plot(binCenters, y, color=uniqueColors[i])

        return

    def _plotModulationByPeakLatency(
        self,
        ax=None,
        a=0.05,
        iWindow=5,
        responseWindow=(-0.2, 0.5),
        runningWindowSize=0.02,
        binsize=0.01,
        minimumResponseAmplitude=2,
        averagingWindow=(0.05, 0.2),
        yrange=(-1, 1),
        **kwargs_
        ):
        """
        """

        # Keywords arguments for the scatter function
        kwargs = {
            'color': 'k',
            'marker': '.',
            'alpha': 0.7,
            's': 5,
        }
        kwargs.update(kwargs_)

        #
        if ax is None:
            fig, ax = plt.subplots()

        #
        leftEdges = np.arange(averagingWindow[0], averagingWindow[1], binsize)
        rightEdges = leftEdges + binsize
        binCenters = np.vstack([leftEdges, rightEdges]).T.mean(1)
        samples = [[] for iBin in range(binCenters.size)]
        X, Y = list(), list()

        #
        for iUnit in range(len(self.ukeys)):

            #
            if self.params[iUnit, 0] < minimumResponseAmplitude:
                continue
            if np.isnan(self.modulation[iUnit, :, iWindow]).all():
                continue
            if np.nanmin(self.pvalues[iUnit, iWindow, :]) > a:
                continue
            
            #
            y = self.modulation[iUnit, :, iWindow] / self.params[iUnit, 0]
            m = np.invert(np.isnan(y))
            y = np.atleast_1d(y[m])
            t = np.atleast_1d(self.latencies[iUnit, :y.size, iWindow])

            #
            for iComp, (dr, l) in enumerate(zip(y, t)):
                X.append(l)
                Y.append(dr)
                indices = np.where(np.logical_and(
                    l > leftEdges,
                    l <= rightEdges
                ))[0]
                if len(indices) == 1:
                    i = indices.item()
                    samples[i].append(dr)

        #
        X, Y = np.array(X), np.array(Y)
        ax.scatter(X, np.clip(Y, -1, 1), color='0.5', alpha=0.5, s=10, marker='.')
        m, b = np.polyfit(X, Y, 1)
        x1, x2 = X.min(), X.max()
        ln = np.array([x1, x2]) * m + b
        ax.plot([x1, x2], ln, color='k')
        # binned = np.array([np.nanmean(s) if len(s) != 0 else np.nan for s in samples])
        # ax.plot(binCenters, binned, color='k')
        ax.set_xlim(responseWindow)

        return

        binMeans = list()
        binCenters = list()
        for leftEdge in np.arange(0, responseWindow[1], runningWindowSize):
            rightEdge = leftEdge + runningWindowSize
            binCenter = np.mean([leftEdge, rightEdge])
            peakIndices = np.where(np.logical_and(
                X > leftEdge,
                X <= rightEdge
            ))[0]
            if len(peakIndices) < 1:
                binMeans.append(np.nan)
            else:
                binMeans.append(np.mean(Y[peakIndices]))
            binCenters.append(binCenter)

        #
        binCenters, binMeans = np.array(binCenters), np.array(binMeans)
        ax.plot(binCenters, binMeans, color='k')

        return binCenters, binMeans
    
    def _plotModulationByIntegratedLatency(
        self,
        ax=None,
        perisaccadicWindow=(-0.5, 0.5),
        interpolationWindow=(-0.35, 0.5),
        binsize=0.1,
        minimumResponseAmplitude=2,
        a=0.05,
        yrange=(-1, 1),
        **kwargs_
        ):
        """
        """

        #
        nt = int(round(np.diff(interpolationWindow).item(), 0) * 1000) + 1

        #
        kwargs = {
            'color': 'k',
            'marker': '.',
            's': 15,
            'alpha': 0.5,
        }
        kwargs.update(kwargs_)
        if ax is None:
            fig, ax = plt.subplots()

        #
        leftEdges = np.arange(perisaccadicWindow[0], perisaccadicWindow[1], binsize)
        rightEdges = leftEdges + binsize
        samples = [[] for i in range(leftEdges.shape[0])]

        #
        nUnits, nBins, nWindows = self.peths['perisaccadic'].shape
        nComponents = int(np.nanmax(self.k.flatten()))
        binCenters = np.mean(self.windows, axis=1)
        x = list()
        y = list()
        lines = list()
        for iUnit in range(nUnits):
            date, animal, cluster = self.ukeys[iUnit]
            if self.params[iUnit, 0] < minimumResponseAmplitude:
                continue
            for iComp in range(nComponents):
                if date == '2023-05-12' and animal == 'mlati7' and cluster == 224 and False:
                    color = 'r'
                    alpha = 1
                    lw = 1
                    zorder = 3
                else:
                    color = '0.5'
                    alpha = 0.2
                    lw = 0.5
                    zorder = -1
                ln = list()
                ps = list()
                t = list()
                for iWin in range(nWindows - 1):
                    ps.append(self.pvalues[iUnit, iWin, iComp])
                    l = self.latencies[iUnit, iComp, iWin] + binCenters[iWin]
                    t.append(l)
                    dr = self.modulation[iUnit, iComp, iWin]
                    dr /= self.params[iUnit, 0]
                    ln.append(dr)
                if np.sum(np.array(ps) < a) < 1:
                    continue
                ax.plot(t, np.clip(ln, *yrange), color=color, alpha=alpha, lw=lw, zorder=zorder)
                interpolated = np.interp(
                    np.linspace(*interpolationWindow, nt),
                    t,
                    ln,
                    left=np.nan,
                    right=np.nan
                )
                lines.append(interpolated)
                for dr, ti in zip(ln, t):
                    binIndices = np.where(np.logical_and(
                        ti > leftEdges,
                        ti <= rightEdges
                    ))[0]
                    if len(binIndices) == 1:
                        binIndex = binIndices.item()
                        samples[binIndex].append(dr)

        #
        binCenters = leftEdges + (binsize / 2)
        binMeans = list()
        for i, sample in enumerate(samples):
            if len(sample) == 0:
                binMeans.append(np.nan)
                continue
            binMeans.append(np.mean(sample))
        # ax.plot(binCenters, binMeans, color='k')
        ax.plot(
            np.linspace(*interpolationWindow, nt),
            np.nanmean(lines, axis=0),
            color='k'
        )

        return

    def plotModulationByLatency(
        self,
        figsize=(7, 3),
        referenceWindow=5,
        ):
        """
        """

        fig, axs = plt.subplots(ncols=3)
        self._plotModulationByProbeLatency(ax=axs[0])
        self._plotModulationByPeakLatency(ax=axs[1], iWindow=referenceWindow)
        self._plotModulationByIntegratedLatency(ax=axs[2], binsize=0.05)
        for ax in axs:
            ax.set_ylim([-1.1, 1.1])
        for ax in axs:
            ylim = ax.get_ylim()
            ax.vlines(0, *ylim, color='k', alpha=0.5)
            ax.set_ylim(*ylim)
            xlim = ax.get_xlim()
            ax.hlines(0, *xlim, color='k', alpha=0.5)
            ax.set_xlim(xlim)
        for ax in axs:    
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
        axs[1].set_xticks([-0.2, 0, 0.5])
        axs[0].set_xticks([-0.5, 0, 0.5])
        axs[0].set_xlabel('Time from saccade (s)')
        axs[1].set_xlabel('Time from probe (s)')
        axs[2].set_xlabel('Time from saccade (s)')
        axs[0].set_ylabel(r'Modulation ($\Delta R$)')
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, axs