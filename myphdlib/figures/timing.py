import h5py
import numpy as np
import pathlib as pl
from scipy import stats
from scipy.interpolate import interp1d
from scipy.optimize import fmin, minimize_scalar, minimize, curve_fit
from scipy.ndimage import gaussian_filter
from matplotlib import pyplot as plt
from myphdlib.figures.analysis import AnalysisBase, GaussianMixturesModel, g
from myphdlib.figures.modulation import BasicSaccadicModulationAnalysis
from myphdlib.general.toolkit import psth2
from myphdlib.extensions.matplotlib import getIsoluminantRainbowColormap
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter1d
from matplotlib_venn import venn2
from mpl_toolkits.axes_grid1 import make_axes_locatable
from myphdlib.general.toolkit import ttest_1samp_with_weights
from scipy.stats import pearsonr

def weighted_quantiles(values, weights, quantiles=0.5, interpolate=False):

    i = values.argsort()
    sorted_weights = weights[i]
    sorted_values = values[i]
    Sn = sorted_weights.cumsum()

    if interpolate:
        Pn = (Sn - sorted_weights/2 ) / Sn[-1]
        return np.interp(quantiles, Pn, sorted_values)
    else:
        return sorted_values[np.searchsorted(Sn, quantiles * Sn[-1])]


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

        # Example neurons
        self.examples = (
            ('2023-05-24', 'mlati7', 337),
            ('2023-05-17', 'mlati7', 237),
            ('2023-05-15', 'mlati7', 408),
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
        nBins=10,
        windowIndices=(4, 5, 6),
        contourThreshold=0.4,
        transform=True,
        intercepts=np.arange(-1, 1, 0.1),
        figsize=(3.2, 3),
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
                self.ns['mi/pref/real'][:, windowIndices, 0] < 0,
                self.ns['p/pref/real'][:, windowIndices, 0] < 0.05
            ]).all(-1).any(1)
        elif modulation == 1:
            include = np.dstack([
                self.ns['mi/pref/real'][:, windowIndices, 0] > 0,
                self.ns['p/pref/real'][:, windowIndices, 0] < 0.05,
            ]).all(-1).any(1)
        else:
            include = np.dstack([
                self.ns['p/pref/real'][:, windowIndices, 0] >= 0.05,
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
                    np.invert(np.isnan(self.ns['mi/pref/real'][:, iWin, 0]))
                ]).all(0)
                W = self.ns['params/pref/real/extra'][mask, 0] # weight by amplitude
                Z[iWin, j] = np.average(
                    self.ns['mi/pref/real'][mask, iWin, 0],
                    weights=W
                )

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
        
        if modulation == -1:
            contourThreshold *= -1
        ax.contour(
            xTrans[:-1] + ((xTrans[1] - xTrans[0]) / 2),
            np.mean(self.windows, axis=1),
            Z,
            levels=(contourThreshold,),
            colors='w',
            linestyles='-'
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

        #
        xyz = list()
        for iUnit in np.where(include)[0]:
            x = peakLatencies[iUnit]
            for iWindow in range(len(self.windows)):
                y = self.windows[iWindow].mean()
                z = self.ns['mi/pref/real'][iUnit, iWindow, 0]
                xyz.append([x, y, z])
        xyz = np.array(xyz)

        return fig, ax, xyz
    
    def plotPerisaccadicResponsesForExamples(
        self,
        responseWindow=(0, 0.3),
        tMargin=0.05,
        cmap=None,
        figsize=(3, 3.5),
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
        fig, grid = plt.subplots(nrows=len(self.examples), ncols=2, gridspec_kw={'width_ratios': [tLeft, tRight]})
        grid = np.atleast_2d(grid)
        # cmap = plt.get_cmap('gist_rainbow', len(self.windows))
        if cmap is None:
            cmap = getIsoluminantRainbowColormap(len(self.windows))
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
            for ax in grid[i, :]:
                ax.relim()
                ax.autoscale()
            ylim = [np.inf, -np.inf]
            for ax in grid[i, :].flatten():
                y1, y2 = ax.get_ylim()
                if y1 < ylim[0]:
                    ylim[0] = y1
                if y2 > ylim[1]:
                    ylim[1] = y2
            for ax in grid[i, :].flatten():
                ax.set_ylim(ylim)
        for ax in grid[:, 1].flatten():
            ax.set_yticklabels([])
        if grid.shape[0] > 1:
            for ax in grid[0, :].flatten():
                ax.set_xticklabels([])
        for ax in grid[:, 0]:
            ax.set_xlim(windowCenters[0] - responseWindow[0] - tMargin, windowCenters[-1] + responseWindow[1] + tMargin)
        for ax in grid[:, 1]:
            ax.set_xlim(self.tProbe.min() - tMargin, self.tProbe.max() + tMargin)
        grid[-1, 0].set_xlabel('Latency from saccade to probe (sec)')
        grid[-1, 0].set_ylabel('Firing rate (SD)')
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.5)

        return fig, grid
    
    def plotModulationByIntegratedLatency(
        self,
        sign=-1,
        componentIndex=0,
        windowIndices=(4, 5, 6),
        minimumResponseAmplitude=0,
        allowOverlap=True,
        backgroundColor='k',
        interpolationWindow=(-0.3, 0.45),
        nPointsForEvaluation=30,
        transform=False,
        yrange=(-1, 1),
        figsize=(2.5, 3),
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
        responseAmplitudes = self.ns['params/pref/real/extra'][:, 0]

        #
        nUnits, nBins, nWindows = self.ns['ppths/pref/real/peri'].shape
        windowCenters = np.mean(self.windows, axis=1)
        lines = list()
        weights = list()
        subsets = [0, 0, 0]
        for iUnit in range(nUnits):

            #
            if abs(responseAmplitudes[iUnit]) < minimumResponseAmplitude:
                continue

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
            y = np.array(y)

            # Exclude noise units
            if np.all(y > 0.2) or np.all(y < -0.2):
                continue

            #
            isSuppressed = np.vstack([
                self.ns['p/pref/real'][iUnit, windowIndices, componentIndex] < 0.05,
                self.ns['mi/pref/real'][iUnit, windowIndices, componentIndex] < 0
            ]).all(0).any()
            isEnhanced = np.vstack([
                self.ns['p/pref/real'][iUnit, windowIndices, componentIndex] < 0.05,
                self.ns['mi/pref/real'][iUnit, windowIndices, componentIndex] > 0
            ]).all(0).any()

            #
            if isSuppressed and isEnhanced:
                subsets[1] += 1
            elif isSuppressed:
                subsets[0] += 1
            elif isEnhanced:
                subsets[2] += 1

            #
            if sign == -1 and isSuppressed == False:
                continue
            if sign == 1 and isEnhanced == False:
                continue

            #
            if all([sign == -1, allowOverlap == False, isEnhanced]):
                continue
            # if all([sign ==  1, allowOverlap == False, isSuppressed]):
            #     continue

            # Interpolate
            interpolated = np.interp(
                np.linspace(*interpolationWindow, nPointsForEvaluation),
                t,
                y,
                left=np.nan,
                right=np.nan
            )
            lines.append(interpolated)
            weights.append(abs(responseAmplitudes[iUnit]))

        #
        lines = np.array(lines)
        weights = np.array(weights)

        #
        x1 = np.linspace(*interpolationWindow, nPointsForEvaluation)
        y1 = list()
        iqr = list()
        for col in np.array(lines).T:
            indices = np.where(np.invert(np.isnan(col)))[0]
            yi = weighted_quantiles(
                col[indices],
                weights[indices],
                quantiles=0.5
            )
            e1 = weighted_quantiles(
                col[indices],
                weights=weights[indices],
                quantiles=0.25
            )
            e2 = weighted_quantiles(
                col[indices],
                weights=weights[indices],
                quantiles=0.75
            )
            y1.append(yi)
            iqr.append([e1, e2])
        iqr = np.array(iqr)
        y1 = np.array(y1)
        ax.plot(
            x1,
            y1,
            color='k'
        )
        ax.fill_between(
            x1,
            iqr[:, 0],
            iqr[:, 1],
            color=backgroundColor,
            alpha=0.15,
            edgecolor='none',
        )
        ax.vlines(0, *yrange, color='k', alpha=0.7, linestyle=':')
        ax.hlines(0, *interpolationWindow, color='k', alpha=0.7, linestyle=':')
        ax.set_xlim(interpolationWindow)
        ax.set_ylim(yrange)

        #
        divider = make_axes_locatable(ax)
        margin = divider.append_axes("bottom", size="5%", pad=0.05)
        p = list()
        for sample in lines.T:
            mask = np.invert(np.isnan(sample))
            # t, p_ = ttest_1samp_with_weights(sample[mask], 0, weights=weights[mask])
            # t, p_ = stats.ttest_1samp(sample[mask], 0)
            u, p_ = stats.wilcoxon(sample[mask])
            p.append(p_)
        p = np.array(p)
        X = np.linspace(*interpolationWindow, nPointsForEvaluation + 1)
        Y = np.array([0, 1])
        p[p < 0.001] = 0
        p[p >= 0.001] = 1
        margin.pcolor(X, Y, p.reshape(1, -1), cmap='binary_r')

        #
        ax.set_xticks([])
        margin.set_xticks([interpolationWindow[0], 0, interpolationWindow[1]])
        margin.set_xlim(interpolationWindow)
        margin.set_xlabel('Time from saccade (sec)')
        ax.set_ylabel('Modulation index (MI)')

        #
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, ax, subsets, np.array(lines)

    def plotModulationOverlapBySign(
        self,
        minimumResponseAmplitude=0,
        windowIndices=(4, 5, 6),
        componentIndex=0,
        cmap='coolwarm',
        subsets=None,
        figsize=(3, 1.5)
        ):
        """
        """

        #
        nUnits, nBins, nWindows = self.ns['ppths/pref/real/peri'].shape
        responseAmplitudes = self.ns['params/pref/real/extra'][:, 0]
        if subsets is None:
            subsets = [0, 0, 0]
        for iUnit in range(nUnits):

            #
            if abs(responseAmplitudes[iUnit]) < minimumResponseAmplitude:
                continue

            #
            y = list()
            for iWindow in range(nWindows):
                mi = self.ns['mi/pref/real'][iUnit, iWindow, 0]
                y.append(mi)
            y = np.array(y)
            if np.all(y > 0.2) or np.all(y < -0.2):
                continue

            # Exclude enhanced or unmodulated units
            isSuppressed = np.vstack([
                self.ns['p/pref/real'][iUnit, windowIndices, componentIndex] < 0.05,
                self.ns['mi/pref/real'][iUnit, windowIndices, componentIndex] < 0
            ]).all(0).any()
            isEnhanced = np.vstack([
                self.ns['p/pref/real'][iUnit, windowIndices, componentIndex] < 0.05,
                self.ns['mi/pref/real'][iUnit, windowIndices, componentIndex] > 0
            ]).all(0).any()

            #
            if subsets is None:
                if isSuppressed and isEnhanced:
                    subsets[1] += 1
                elif isSuppressed:
                    subsets[0] += 1
                elif isEnhanced:
                    subsets[2] += 1

        #
        f = plt.get_cmap(cmap, 3)
        fig, ax = plt.subplots()
        venn = venn2(
            subsets,
            set_labels=('', ''),
            set_colors=[f(0), f(2)],
            alpha=0.5,
            ax=ax,
        )

        #
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, ax

    def histProbeResponsePeakLatency(
        self,
        xrange=(0, 0.5),
        nbins=30,
        figsize=(1.5, 1)
        ):
        """
        """

        fig, ax = plt.subplots()
        sample = list()
        for iUnit in range(len(self.ukeys)):
            params = self.ns['params/pref/real/extra'][iUnit, :]
            abcd = np.delete(params, np.isnan(params))
            A, B, C = np.split(abcd[:-1], 3)
            sample.append(B[0])
        ax.hist(
            sample,
            range=xrange,
            bins=nbins,
            histtype='stepfilled',
            edgecolor='none',
            color='k'
        )

        #
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, ax

    def crossCorrelateModulation(
        self,
        minimumResponseAmplitude=5,
        windowIndices=(4, 5, 6)
        ):
        """
        """

        nUnits = len(self.ukeys)
        tRaw = self.windows.mean(1)
        tExpanded = np.linspace(tRaw.min(), tRaw.max(), 100)
        ccg = list()
        for iUnit in range(nUnits):

            #
            params = self.ns['params/pref/real/extra'][iUnit]
            abcd = np.delete(params, np.isnan(params))
            A, B, C = np.split(abcd[:-1], 3)

            #
            if len(B) < 2:
                continue
            if any([A[0] < minimumResponseAmplitude, A[1] < minimumResponseAmplitude]):
                continue
            
            #
            suppressed = np.vstack([
                self.ns['p/pref/real'][iUnit, windowIndices, 0] < 0.05,
                self.ns['mi/pref/real'][iUnit, windowIndices, 0] < 0
            ]).all(0)
            if any(suppressed) == False:
                continue

            #
            suppressed = np.vstack([
                self.ns['p/pref/real'][iUnit, windowIndices, 1] < 0.05,
                self.ns['mi/pref/real'][iUnit, windowIndices, 1] < 0
            ]).all(0)
            if any(suppressed) == False:
                continue

            #
            corr = list()
            for lag in np.linspace(-0.3, 0.3, 100):
                y1 = np.interp(
                    tExpanded,
                    tRaw,
                    self.ns['mi/pref/real'][iUnit, :, 0],
                    left=np.nan,
                    right=np.nan
                )
                y2 = np.interp(
                    tExpanded + lag,
                    tRaw,
                    self.ns['mi/pref/real'][iUnit, :, 1],
                    left=np.nan,
                    right=np.nan
                )
                m = np.logical_not(
                    np.logical_or(
                        np.isnan(y1),
                        np.isnan(y2)
                    )
                )
                r, p = pearsonr(y1[m], y2[m])
                corr.append(r)
            ccg.append(corr)

        return np.array(ccg)

    def scatterModulationByComponent(
        self,
        t=None,
        minimumResponseAmplitude=5,
        figsize=(2, 2),
        suppressPlot=False,
        ):
        """
        Create a scatterplot which relates modulation of the largest component
        to that of the second largest component
        """

        nUnits = len(self.ukeys)
        xy = list()
        for iUnit in range(nUnits):

            #
            params = self.ns['params/pref/real/extra'][iUnit]
            abcd = np.delete(params, np.isnan(params))
            A, B, C = np.split(abcd[:-1], 3)

            #
            if len(B) < 2:
                continue
            if any([A[0] < minimumResponseAmplitude, A[1] < minimumResponseAmplitude]):
                continue

            # Largest component
            y1 = self.ns['mi/pref/real'][iUnit, :, 0]
            t1 = self.windows.mean(1) + B[0]

            # Second largest component
            y2 = self.ns['mi/pref/real'][iUnit, :, 1]
            t2 = self.windows.mean(1) + B[1]

            #
            if t is None:
                binIndex = np.argmax(np.abs(y1))
                t = t1[binIndex]
            y = np.interp(t, t2, y2, left=np.nan, right=np.nan)
            x = y1[np.argmax(np.abs(y1))]
            if np.any(np.isnan([x, y])):
                continue
            xy.append([x, y])

        #
        xy = np.array(xy)
        xy = np.clip(xy, -3, 3)
        if len(xy) == 0:
            return None, None

        if suppressPlot:
            r, p = pearsonr(xy[:, 0], xy[:, 1])
            return r, p, xy

        #
        fig, ax = plt.subplots()
        ax.scatter(xy[:, 0], xy[:, 1], s=5, color='k', edgecolor='none', alpha=0.5, clip_on=False)
        ax.set_xlim([-3, 3])
        ax.set_ylim([-3, 3])
        ax.set_xlabel(r'$MI_{1}$')
        ax.set_ylabel(r'$MI_{2}$')
        ax.set_aspect('equal')
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, ax