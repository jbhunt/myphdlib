import h5py
import numpy as np
import pathlib as pl
from scipy import stats
from scipy.interpolate import interp1d
from scipy.optimize import fmin, minimize_scalar
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
from myphdlib.figures.analysis import AnalysisBase, GaussianMixturesModel, g
from myphdlib.figures.modulation import BasicSaccadicModulationAnalysis
from myphdlib.general.toolkit import psth2
from matplotlib.gridspec import GridSpec

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
        modulation=-1,
        nBins=7,
        transform=True,
        intercepts=np.arange(-1, 1, 0.1),
        figsize=(3, 3),
        cmap='coolwarm',
        vrange=(-0.5, 0.5),
        xticks=np.array([0.05, 0.1, 0.2]),
        yticks=np.array([-0.5, -0.25, 0, 0.25, 0.5])
        ):
        """
        """

        #
        fig, ax = plt.subplots()

        #
        if modulation == -1:
            include = np.dstack([
                self.ns['mi/pref/real'][:, 4:7, 0] < 0,
                self.ns['p/pref/real'][:, 4:7, 0] < 0.05
            ]).all(-1).any(1)
        elif modulation == 1:
            include = np.dstack([
                self.ns['mi/pref/real'][:, 4:7, 0] > 0,
                self.ns['p/pref/real'][:, 4:7, 0] < 0.05,
            ]).all(-1).any(1)
        else:
            include = np.dstack([
                self.ns['p/pref/real'][:, 4:7, 0] >= 0.05,
            ]).all(-1).any(1)

        # Extract peak latencies
        windowIndices = np.arange(10)
        peakLatencies = np.full(len(self.ukeys), np.nan)
        for iUnit in np.where(include)[0]:
            # params = self.model['params'][iUnit, :]
            params = self.ns['params/pref/real/extra'][iUnit]
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
        for iWin in windowIndices:
            for j in range(binEdges.shape[0]):
                leftEdge, rightEdge = binEdges[j]
                mask = np.vstack([
                    include,
                    peakLatencies >= leftEdge,
                    peakLatencies <  rightEdge,
                ]).all(0)
                Z[iWin, j] = np.nanmean(self.ns['mi/pref/real'][mask, iWin, 0])

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
        # yFit = list()
        # for j in range(Z.shape[1]):
        #     y = Z[:, j]
        #     # x = np.mean(self.windows, axis=1)
        #     x = np.arange(y.size)
        #     f = np.poly1d(np.polyfit(x, y, deg=9))
        #     res = minimize_scalar(f, bounds=(y.min(), y.max()), method='bounded')
        #     t = np.interp(res.x, x, self.windows.mean(1))
        #     yFit.append(t)
        # yFit = np.array(yFit)
        
        #
        xFit = tf(np.mean(binEdges, axis=1))
        if modulation == -1:
            yFit = np.array([self.windows.mean(1)[np.argmin(z)] for z in Z.T])
        elif modulation == 1:
            yFit = np.array([self.windows.mean(1)[np.argmax(z)] for z in Z.T])
        else:
            yFit = np.full(self.windows.shape[0], np.nan)
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
        ax.set_yticks(yticks)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

        #
        ax.set_xlabel('Response latency (s)')
        ax.set_ylabel('Saccade to probe latency (s)')
        fig.colorbar(
            mesh,
            fraction=0.3,
            ticks=np.arange(-0.5, 0.5 + 0.25, 0.25)
        )
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, ax
    
    def plotPerisaccadicResponsesForExamples(
        self,
        responseWindow=(0, 0.3),
        tMargin=0.05,
        figsize=(5, 3),
        ):
        """
        """

        # fig, grid = plt.subplots(nrows=len(self.examples), sharex=True)
        # gs = GridSpec(len(self.examples), 10)
        # fig = plt.figure()
        # grid = np.array([
        #     [fig.add_subplot(gs[0, :7]), fig.add_subplot(gs[0, 7:])],
        #     [fig.add_subplot(gs[1, :7]), fig.add_subplot(gs[1, 7:])],
        # ], dtype=object)
        windowCenters = self.windows.mean(1)
        tLeft = (windowCenters[-1] + responseWindow[1]) - (windowCenters[0] - responseWindow[0])
        tRight = self.tProbe[-1] - self.tProbe[0]
        fig, grid = plt.subplots(nrows=2, ncols=2, gridspec_kw={'width_ratios': [tLeft, tRight]})
        cmap = plt.get_cmap('gist_rainbow', len(self.windows))
        for i, ukey in enumerate(self.examples):
            iUnit = self._indexUnitKey(ukey)
            for windowIndex in range(len(self.windows)):
                yFull = self.ns[f'ppths/pref/real/peri'][iUnit, :, windowIndex]
                tShort = np.linspace(*responseWindow, 100)
                yShort = np.interp(
                    tShort,
                    self.tProbe,
                    yFull
                )
                tShifted = tShort + self.windows[windowIndex].mean()
                grid[i, 0].plot(
                    tShifted,
                    yShort,
                    color=cmap(windowIndex),
                    alpha=0.7,
                )
                grid[i, 0].fill_between(
                    tShifted,
                    0,
                    yShort,
                    color=cmap(windowIndex),
                    alpha=0.1,
                )

                #
                y1, y2 = grid[i, 0].get_ylim()
                # grid[i, 0].scatter(
                #     0 + self.windows[windowIndex].mean(),
                #     # np.interp(0, self.tProbe, yFull) + self.windows[windowIndex].mean(),
                #     y1,
                #     color=cmap(windowIndex),
                #     s=5,
                #     clip_on=False,
                #     zorder=3,
                # )
                grid[i, 0].set_ylim([y1, y2])

            #
            y = self.ns[f'ppths/pref/real/extra'][iUnit, :]
            grid[i, 1].plot(self.tProbe, y, color='k', alpha=0.5)
            grid[i, 1].fill_between(
                self.tProbe,
                0,
                y,
                color='k',
                alpha=0.1
            )

        #
        for ax in grid.flatten():
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
        for i in range(len(self.examples)):
            ylim = [np.inf, -np.inf]
            for ax in grid[i, :]:
                y1, y2 = ax.get_ylim()
                if y1 < ylim[0]:
                    ylim[0] = y1
                if y2 > ylim[1]:
                    ylim[1] = y2
            for ax in grid[i, :]:
                ax.set_ylim(ylim)
        for ax in grid[:, 1].flatten():
            ax.set_yticklabels([])
        for ax in grid[0, :].flatten():
            ax.set_xticklabels([])
        for ax in grid[:, 0]:
            ax.set_xlim(windowCenters[0] - responseWindow[0] - tMargin, windowCenters[-1] + responseWindow[1] + tMargin)
        for ax in grid[:, 1]:
            ax.set_xlim(self.tProbe.min() - tMargin, self.tProbe.max() + tMargin)
        grid[1, 0].set_xlabel('Latency from saccade to probe (sec)')
        grid[1, 0].set_ylabel('Firing rate (z-scored)')
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.5)

        return fig, grid
    
    def plotModulationByIntegratedLatency(
        self,
        modulation=-1,
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

        #
        nUnits, nBins, nWindows = self.ns['ppths/pref/real/peri'].shape
        windowCenters = np.mean(self.windows, axis=1)
        lines = list()
        for iUnit in range(nUnits):

            # Exclude enhanced or unmodulated units
            if modulation == -1:
                checks = np.vstack([
                    self.ns['p/pref/real'][iUnit, 4:7, componentIndex] < 0.05,
                    self.ns['mi/pref/real'][iUnit, 4:7, componentIndex] < 0
                ]).all(0)
            elif modulation == 1:
                checks = np.vstack([
                    self.ns['p/pref/real'][iUnit, 4:7, componentIndex] < 0.05,
                    self.ns['mi/pref/real'][iUnit, 4:7, componentIndex] > 0
                ]).all(0)
            else:
                raise Exception('Not sure what to do here')
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
                # ax.plot(t, np.clip(y, *yrange), color='0.6', alpha=0.05, lw=1)
                # if np.random.choice([True, False], size=1):
                #    ax.scatter(t, np.clip(y, *yrange), color='k', alpha=0.05, s=5, marker='.', rasterized=True, clip_on=False)


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
        x1 = np.linspace(*interpolationWindow, nPointsForEvaluation)
        y1 = np.nanmean(lines, axis=0)
        ax.plot(
            x1,
            y1,
            color='k'
        )
        error = 1.96 * (np.nanstd(lines, axis=0) / np.sqrt(np.sum(np.invert(np.isnan(lines)), axis=0)))
        ax.fill_between(
            np.linspace(*interpolationWindow, nPointsForEvaluation),
            np.nanmean(lines, axis=0) - error,
            np.nanmean(lines, axis=0) + error,
            color='k',
            alpha=0.1,
            edgecolor=None
        )
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
