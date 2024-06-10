import h5py
import numpy as np
import pathlib as pl
from scipy import stats
from scipy.interpolate import interp1d
from scipy.optimize import fmin
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
from myphdlib.figures.analysis import AnalysisBase, GaussianMixturesModel, g
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

        self.peths = { # Standardized PSTHs
            'extra': None,
            'peri': None
        }
        self.terms = {
            'rpp': None,
            'rps': None,
            'rs':  None,
        }
        self.features = {
            'm': None,
            's': None,
            'd': None
        }
        self.model = {
            'k': None,
            'fits': None,
            'peaks': None,
            'labels': None,
            'params1': None,
            'params2': None
        }
        self.templates = {
            'nasal': None,
            'temporal': None
        }
        self.tProbe = None
        self.tSaccade = None
        self.windows = None
        self.mi = None # Modulation index
        self.filter = None
        self.speed = None

        # Example neurons
        self.examples = (
            ('2023-05-24', 'mlati7', 337),
            ('2023-05-17', 'mlati7', 237),
        )

        return

    def _sortUnitsByPeakLatency(
        self,
        ):
        """
        """

        nUnits = len(self.ukeys)
        latencies = np.full(nUnits, np.nan)
        for iUnit in range(nUnits):
            mask = np.invert(np.isnan(self.params[iUnit, :]))
            if mask.sum() == 0:
                continue
            abcd = self.model['params'][iUnit, mask]
            abc, d = abcd[:-1], abcd[-1]
            A, B, C = np.split(abc, 3)
            latencies[iUnit] = B[0] # Latency of the largest component

        # Use the median to split units into slow and fast types
        threshold = np.nanmedian(latencies)
        self.speed = np.full(nUnits, np.nan)
        self.speed[latencies <  threshold] = -1
        self.speed[latencies >= threshold] =  1

        return

    def plotHeatmap(
        self,
        nBins=7,
        transform=True,
        intercepts=np.arange(-1, 1, 0.1),
        figsize=(3, 3),
        cmap='coolwarm',
        vrange=(-0.5, 0.5),
        xticks=np.array([0.05, 0.1, 0.2])
        ):
        """
        """

        fig, ax = plt.subplots()

        # Extract peak latencies
        windowIndices = np.arange(10)
        peakLatencies = np.full(len(self.ukeys), np.nan)
        for iUnit in range(len(self.ukeys)):
            params = self.model['params'][iUnit, :]
            abcd = params[np.invert(np.isnan(params))]
            if len(abcd) == 0:
                continue
            abc, d = abcd[:-1], abcd[-1]
            A, B, C = np.split(abc, 3)
            peakLatencies[iUnit] = B[0]
        peakLatencies = np.array(peakLatencies)

        # Define the bins that specify N quantiles
        leftEdges = np.array([np.nanpercentile(peakLatencies, i / nBins * 100) for i in range(nBins)])
        rightEdges = np.concatenate([leftEdges[1:], [np.nanmax(peakLatencies)]])
        binEdges = np.vstack([leftEdges, rightEdges]).T
        
        # Populate the heatmap
        Z = np.full([10, binEdges.shape[0]], np.nan)
        for i in windowIndices:
            mi = self.mi[:, i, 0] / self.model['params'][:, 0]
            for j in range(binEdges.shape[0]):
                leftEdge, rightEdge = binEdges[j]
                mask = np.vstack([
                    self.filter,
                    np.logical_and(
                        peakLatencies >= leftEdge,
                        peakLatencies <  rightEdge
                    )
                ]).all(0)
                Z[i, j] = np.nanmean(mi[mask])

        # Plot the heatmap
        x = np.concatenate([leftEdges, [rightEdges[-1]]])
        y = np.concatenate([self.windows[:, 0], [self.windows[-1, 1]]])
        if transform:
            tf = interp1d(x, np.arange(leftEdges.size + 1), kind='linear')
        else:
            tf = lambda x: x
        X, Y = np.meshgrid(tf(x), y)
        mesh = ax.pcolor(X, Y, Z, vmin=vrange[0], vmax=vrange[1], cmap=cmap)

        #
        xlim = (tf(x.min()), tf(x.max()))
        ylim = (y.min(), y.max())

        # Plot contours
        xTrans = tf(x)
        for y in intercepts:
            ax.plot(
                xTrans,
                -1 * x + y,
                color='k',
                lw=0.5
            )

        #
        yFit = list()
        for j in range(Z.shape[1]):
            y = Z[:, j]
            x = np.mean(self.windows, axis=1)
            p = np.polyfit(x, y, deg=9)
            f = np.poly1d(p)
            t = fmin(f, 0).item()
            yFit.append(t)
        yFit = np.array(yFit)
        xFit = tf(np.mean(binEdges, axis=1))
        ax.plot(
            xFit,
            yFit,
            color='w',
            linestyle='-',
            marker='o',
            markersize=5,
        )

        #
        ax.set_xticks(
            tf(xticks)
        )
        ax.set_xticklabels(xticks, rotation=45)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        #
        ax.set_xlabel('Response latency (s)')
        ax.set_ylabel('Saccade to probe latency (s)')
        fig.colorbar(mesh)
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, ax

    # TODO: Refactor this method
    def plotModulationByPeakLatency(
        self,
        a=0.05,
        windowIndices=None,
        responseWindow=(-0.2, 0.5),
        minimumResponseAmplitude=1,
        ylim=(-1.7, 1.7),
        figsize=(4, 3),
        examples=(
            ('2023-05-24', 'mlati7', 337),
            ('2023-05-17', 'mlati7', 237),
        ),
        plotExamples=False,
        **kwargs_
        ):
        """
        """

        # Keywords arguments for the scatter function
        kwargs = {
            'marker': '.',
            'alpha': 0.3,
            's': 5,
        }
        kwargs.update(kwargs_)

        #
        if True:
            fig, ax = plt.subplots()
            axs = [ax,]
            windowIndices = np.arange(10)
        elif windowIndices is None:
            fig, axs = plt.subplots(
                ncols=len(self.windows[:-1]),
                sharey=False,
                sharex=True
            )
            windowIndices = np.arange(10)
        else:
            fig, axs = plt.subplots(
                ncols=len(self.windows[np.array(windowIndices)]),
                sharey=False,
                sharex=True
            )

        #
        samples = {
            'fast': list(),
            'slow': list(),
            'all': list()
        }
        exampleCurves = [list(), list()]
        for i, windowIndex in enumerate(windowIndices):

            #
            X, Y, C, U = list(), list(), list(), list()

            #
            for j, (x, c, utype) in enumerate(zip([-0.1, 0.1], [plt.cm.Dark2(1), plt.cm.Dark2(2)], [-1, 1])):
                
                #
                m1 = np.vstack([
                    self.utypes == utype,
                    self.params[:, 0] >= minimumResponseAmplitude,
                    # self.pvalues[:, windowIndex, 0] < a,
                ]).all(0)
                
                #
                for iUnit in np.where(m1)[0]:
                    y = self.modulation[iUnit, 0, windowIndex] / self.params[iUnit, 0]
                    m2 = np.invert(np.isnan(self.params[iUnit]))
                    abcd = self.params[iUnit, m2]
                    abc, d = abcd[:-1], abcd[-1]
                    A, B, C_ = np.split(abc, 3)
                    l = B[0]
                    # X.append(B[0])
                    X.append(x + i)
                    Y.append(y)
                    C.append(c)
                    U.append(utype)

                    #
                    ukey = self.ukeys[iUnit]
                    for date, animal, cluster in examples:
                        if ukey[0] == date and ukey[1] == animal and ukey[2] == cluster:
                            exampleCurves[j].append(y)

            #
            X = np.array(X)
            Y = np.clip(np.array(Y), *ylim)
            U = np.array(U)
            # axs[0].scatter(
            #     Y,
            #     X,
            #     c=C,
            #     alpha=0.15,
            #     marker='.',
            #     s=5,
            # )

            #
            samples['fast'].append(np.vstack([X[U == -1], Y[U == -1]]).T)
            samples['slow'].append(np.vstack([X[U ==  1], Y[U ==  1]]).T)
            samples['all'].append(np.vstack([U, Y]).T)

        #
        y1, y2, y3 = list(), list(), list()
        for i in windowIndices:
            y1.append(np.mean(np.array(samples['fast'][i]), axis=0)[1])
            y2.append(np.mean(np.array(samples['slow'][i]), axis=0)[1])
            y3.append(np.mean(np.array(samples['all'][i]), axis=0)[1])

        axs[0].plot(
            windowIndices,
            y3,
            color='k',
            marker='D',
            markerfacecolor='k',
            markeredgecolor='k',
            markersize=4,
            label='Uncategorized'
        )
        axs[0].plot(
            windowIndices - 0.1,
            y1,
            color=plt.cm.Dark2(1),
            marker='D',
            markerfacecolor=plt.cm.Dark2(1),
            markeredgecolor='k',
            markersize=4,
            label='Fast',
        )
        axs[0].plot(
            windowIndices + 0.1,
            y2,
            color=plt.cm.Dark2(2),
            marker='D',
            markerfacecolor=plt.cm.Dark2(2),
            markeredgecolor='k',
            markersize=4,
            label='Slow',
        )
        colors = (
            plt.cm.Dark2(1),
            plt.cm.Dark2(2),
            'k',
        )
        for i, (k, o) in enumerate(zip(['fast', 'slow', 'all'], [-0.1, 0.1, 0])):
            for j in range(10):
                sample = samples[k][j][:, 1]
                y1, y2 = stats.t.interval(0.95, len(sample) - 1, loc=np.mean(sample), scale=stats.sem(sample))
                x = j + o
                axs[0].vlines(x, y1, y2, color=colors[i], zorder=-1)

        #
        if plotExamples:
            for i, (y, o) in enumerate(zip(exampleCurves, [-0.1, 0.1])):
                if len(y) != 0:
                    axs[0].plot(windowIndices[2:-2] + o, y[2:-2], color=colors[i], alpha=0.5)

        #
        axs[0].set_xlabel('Probe latency (sec)')
        axs[0].set_ylabel('MI')
        axs[0].legend()
        axs[0].set_xticks(windowIndices)
        axs[0].set_xticklabels(
            np.around(np.mean(self.windows[:-1], axis=1), 2),
            rotation=45
        )
        ylim = axs[0].get_ylim()
        ymax = np.max(np.abs(ylim))
        ylim = (-ymax, ymax)
        axs[0].set_ylim(ylim)
        xlim = axs[0].get_xlim()
        axs[0].vlines(4.5, *ylim, color='k', alpha=0.5, zorder=-1)
        axs[0].hlines(0, *xlim, color='k', alpha=0.5, zorder=-1)
        axs[0].set_xlim(xlim)
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()
        return fig, axs
    
    def plotPerisaccadicResponsesForExamples(
        self,
        figsize=(4, 3),
        ):
        """
        """

        fig, grid = plt.subplots(nrows=len(self.examples), sharex=True)
        cmap = plt.get_cmap('gist_rainbow', len(self.windows))
        for i, ukey in enumerate(self.examples):
            iUnit = self._indexUnitKey(ukey)
            for windowIndex in range(len(self.windows)):
                yFull = self.ns[f'ppths/pref/real/peri'][iUnit, :, windowIndex]
                tShort = np.linspace(0, 0.3, 100)
                yShort = np.interp(
                    tShort,
                    self.tProbe,
                    yFull
                )
                tShifted = tShort + self.windows[windowIndex].mean()
                grid[i].plot(
                    tShifted,
                    yShort,
                    color=cmap(windowIndex),
                    alpha=0.5,
                )

                #
                y1, y2 = grid[i].get_ylim()
                grid[i].scatter(
                    0 + self.windows[windowIndex].mean(),
                    # np.interp(0, self.tProbe, yFull) + self.windows[windowIndex].mean(),
                    y1,
                    color=cmap(windowIndex),
                    s=5,
                    clip_on=False,
                    zorder=3,
                )
                grid[i].set_ylim([y1, y2])

        #
        for ax in grid:
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
        grid[-1].set_xlabel('Latency from saccade to probe (sec)')
        grid[-1].set_ylabel('Firing rate (z-scored)')
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, grid
    
    def plotModulationByIntegratedLatency(
        self,
        componentIndex=0,
        perisaccadicWindow=(-0.5, 0.5),
        binsize=0.1,
        interpolationWindow=(-0.3, 0.45),
        nPointsForEvaluation=100,
        transform=False,
        yrange=(-3, 3),
        figsize=(3, 3),
        **kwargs_
        ):
        """
        """

        #
        if transform:
            yrange = (-1, 1)

        #
        kwargs = {
            'color': 'k',
            'marker': '.',
            's': 15,
            'alpha': 0.5,
        }
        kwargs.update(kwargs_)
        fig, ax = plt.subplots()

        #
        leftEdges = np.arange(perisaccadicWindow[0], perisaccadicWindow[1], binsize)
        rightEdges = leftEdges + binsize

        #
        nUnits, nBins, nWindows = self.ns['ppths/pref/real/peri'].shape
        windowCenters = np.mean(self.windows, axis=1)
        lines = list()
        for iUnit in range(nUnits):

            # Exclude enhanced or unmodulated units
            checks = np.vstack([
                self.ns['p/pref/real'][iUnit, :, componentIndex] < 0.05,
                self.ns['mi/pref/real'][iUnit, :, componentIndex] < 0
            ]).all(0)
            if np.any(checks):

                # Extract peak latency and amplitude for the largest component
                params = self.ns['params/pref/real/extra'][iUnit]
                abcd = np.delete(params, np.isnan(params))
                abc, d = abcd[:-1], abcd[-1]
                A, B, C = np.split(abc, 3)
                peakLatency = B[0]

                # Plot shifted curves
                y = list()
                t = list()
                for iWindow in range(nWindows):
                    t.append(windowCenters[iWindow] + peakLatency)
                    mi = self.ns['mi/pref/real'][iUnit, iWindow, 0]
                    if transform:
                        y.append(np.tanh(mi))
                    else:
                        y.append(mi)
                ax.plot(t, np.clip(y, *yrange), color='0.5', alpha=0.075, lw=0.5)

                # Interpolate
                interpolated = np.interp(
                    np.linspace(*interpolationWindow, nPointsForEvaluation),
                    t,
                    y,
                    left=np.nan,
                    right=np.nan
                )
                lines.append(interpolated)

        #
        ax.plot(
            np.linspace(*interpolationWindow, nPointsForEvaluation),
            np.nanmean(lines, axis=0),
            color='k'
        )
        # ax.fill_between(
        #     np.linspace(*interpolationWindow, nPointsForEvaluation),
        #     np.nanmean(lines, axis=0) - np.nanstd(lines, axis=0),
        #     np.nanmean(lines, axis=0) + np.nanstd(lines, axis=0),
        #     color='0.5',
        #     alpha=0.2
        # )
        xlim = ax.get_xlim()
        ax.vlines(0, *yrange, color='k', alpha=0.7, linestyle=':')
        ax.hlines(0, *xlim, color='k', alpha=0.7, linestyle=':')
        ax.set_xlim(xlim)
        ax.set_ylim(yrange)

        #
        ax.set_xlabel('Time from saccade (sec)')
        ax.set_ylabel('Modulation index (MI)')

        #
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, ax
