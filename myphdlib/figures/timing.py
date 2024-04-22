import h5py
import numpy as np
from scipy import stats
from scipy.interpolate import interp1d
from scipy.optimize import fmin
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
        self.utypes = None

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

        #
        super().loadNamespace(hdf)

        return

    def sortUnitsByResponseLatency(
        self,
        threshold=0.1,
        ):
        """
        """

        nUnits = len(self.ukeys)
        latencies = np.full(nUnits, np.nan)
        for iUnit in range(nUnits):
            m = np.invert(np.isnan(self.params[iUnit, :]))
            if m.sum() == 0:
                continue
            abcd = self.params[iUnit, m]
            abc, d = abcd[:-1], abcd[-1]
            A, B, C = np.split(abc, 3)
            latencies[iUnit] = B[0]

        #
        self.utypes = np.full(nUnits, np.nan)
        self.utypes[latencies <  threshold] = -1
        self.utypes[latencies >= threshold] =  1

        return

    def plotHeatmap(
        self,
        minimumResponseAmplitude=1,
        figsize=(3, 3),
        nBins=7,
        cmap='coolwarm',
        transform=True,
        yIntercepts=(0, 0.1, 0.2, 0.3, 0.4, 0.5),
        vrange=(-0.5, 0.5),
        xticks=np.array([0.05, 0.1, 0.2])
        ):
        """
        """

        fig, ax = plt.subplots()
        windowIndices = np.arange(10)
        l = np.full(len(self.ukeys), np.nan)
        for iUnit in range(len(self.ukeys)):
            params = self.params[iUnit, :]
            abcd = params[np.invert(np.isnan(params))]
            if len(abcd) == 0:
                continue
            abc, d = abcd[:-1], abcd[-1]
            A, B, C = np.split(abc, 3)
            l[iUnit] = B[0]
        l = np.array(l)

        #
        leftEdges = np.array([np.nanpercentile(l, i / nBins * 100) for i in range(nBins)])
        rightEdges = np.concatenate([leftEdges[1:], [np.nanmax(l)]])
        binEdges = np.vstack([leftEdges, rightEdges]).T
        
        #
        Z = np.full([10, binEdges.shape[0]], np.nan)
        m1 = self.params[:, 0] >= minimumResponseAmplitude
        for i in windowIndices:
            mi = self.modulation[:, 0, i] / self.params[:, 0]
            for j in range(binEdges.shape[0]):
                leftEdge, rightEdge = binEdges[j]
                m2 = np.vstack([
                    m1,
                    np.logical_and(
                        l >= leftEdge,
                        l <  rightEdge
                    )
                ]).all(0)
                Z[i, j] = np.nanmean(mi[m2])

        #
        x = np.concatenate([leftEdges, [rightEdges[-1]]])
        y = np.concatenate([self.windows[:-1, 0], [self.windows[-2, 1]]])
        if transform:
            f = interp1d(x, np.arange(leftEdges.size + 1), kind='linear')
        else:
            f = lambda x: x
        X, Y = np.meshgrid(f(x), y)
        mesh = ax.pcolormesh(X, Y, Z, vmin=vrange[0], vmax=vrange[1], cmap=cmap)

        #
        # x_ = np.mean(binEdges, axis=1)
        x_ = np.linspace(np.nanmin(l), np.nanmax(l), nBins + 1)
        xt = f(x_)
        for yIntercept in yIntercepts:
            ax.plot(
                xt,
                -1 * x_ + yIntercept,
                color='k',
                lw=0.5
            )
        # ax.hlines(peakModulationOffset, f(np.nanmin(l)), f(np.nanmax(l)), color='k')

        #
        fit = list()
        for j in range(Z.shape[1]):
            y = Z[:, j]
            x = np.mean(self.windows[:-1], axis=1)
            p = np.polyfit(x, y, deg=9)
            f_ = np.poly1d(p)
            t = fmin(f_, 0)
            fit.append(t)
        x = f(np.mean(binEdges, axis=1))
        ax.plot(
            x,
            fit,
            color='w',
            linestyle='-',
            marker='o',
            markersize=5,
        )

        #
        ax.set_xticks(
            f(xticks)
        )
        ax.set_xticklabels(xticks, rotation=45)

        #
        ax.set_xlabel('Response latency (s)')
        ax.set_ylabel('Saccade to probe latency (s)')
        fig.colorbar(mesh)
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, ax, Z

    def plotUnitSurvivalByUnitType(
        self,
        arange=(0, 3),
        figsize=(4, 7)
        ):
        """
        """

        fig, axs = plt.subplots(nrows=2, sharey=True, sharex=True)
        utypes = (-1, 1)
        for i, ax in enumerate(axs):
            utype = utypes[i]
            curves = ([], [], [], [])
            for a in np.arange(*arange, 0.1):
                for j, label in enumerate([-1, 1, 2, 3]):
                    m = np.vstack([
                        self.params[:, 0] >= a,
                        self.utypes == utype,
                        self.labels.flatten() == label
                    ]).all(0)
                    n = m.sum()
                    curves[j].append(n)
            for j in range(len(curves)):
                ax.plot(np.arange(*arange, 0.1), curves[j])
        
        #
        axs[-1].legend(['Neg.', 'Mono.', 'Bi.', 'Multi.'])
        axs[-1].set_xlabel('Amplitude threshold')
        axs[-1].set_ylabel('# of units')
        axs[0].set_title('Slow-type', fontsize=10)
        axs[1].set_title('Fast-type', fontsize=10)
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[0])
        fig.tight_layout()

        return fig, axs

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
        ukeys=(
            ('2023-05-24', 'mlati7', 337),
            ('2023-05-17', 'mlati7', 237),
        ),
        colors=(
            plt.cm.Dark2(1),
            plt.cm.Dark2(2),
        ),
        figsize=(7, 2)
        ):
        """
        """

        fig, axs = plt.subplots(nrows=len(ukeys), ncols=len(self.windows) - 1, sharex=True)
        if len(ukeys) == 1:
            axs = axs.reshape(-1, 1)
        for j in np.arange(len(ukeys)):
            self.ukey = ukeys[j]
            for i in np.arange(len(self.windows) - 1):
                # axs[j, 9 - i].plot(
                #     self.t,
                #     self.peths['extra'][self.iUnit],
                #     color=colors[j],
                #     alpha=0.5
                #)
                axs[j, i].plot(
                    self.t,
                    self.peths['peri'][self.iUnit, :, i],
                    color=colors[j]
                )

        #
        for i in range(len(ukeys)):
            ylim = [np.inf, -np.inf]
            for j in range(axs.shape[1]):
                y1, y2 = axs[i, j].get_ylim()
                if y1 < ylim[0]:
                    ylim[0] = y1
                if y2 > ylim[1]:
                    ylim[1] = y2
            for j in range(axs.shape[1]):
                axs[i, j].set_ylim(ylim)
        for ax in axs.flatten():
            for sp in ('top', 'right', 'bottom', 'left'):
                ax.spines[sp].set_visible(False)
        for ax in axs[:, 1:].flatten():
            ax.set_yticks([])
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.15)
        return fig, axs

    
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
