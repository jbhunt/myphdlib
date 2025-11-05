from myphdlib.figures.analysis import AnalysisBase
import matplotlib.pyplot as plt
import numpy as np

class ReviewerFigure(AnalysisBase):
    """
    """

    def plotModulationByProbabilityValues(
        self,
        nbins=20,
        miRange=(-2, 2),
        figsize=(10, 4)
        ):
        """
        """

        cmap = plt.get_cmap('coolwarm', 3)
        blue, red = cmap(0), cmap(2)
        fig, axs = plt.subplots(ncols=10, nrows=3, sharex=True, gridspec_kw={'height_ratios': [1, 2, 1]})
        binEdges = np.logspace(-3, 0, nbins + 1)
        pmin = 10 ** -3
        pmax = 10 ** 0
        for j in range(self.windows.shape[0]):
            mi = np.clip(self.ns['mi/pref/real'][:, j, 0], *miRange)
            p = np.clip(self.ns['p/pref/real'][:, j, 0], pmin, pmax)
            axs[1, j].scatter(p, mi, marker='.', color='k', alpha=0.4, clip_on=False,
                rasterized=True, edgecolor='none', s=12)
            counts, binEdges_ = np.histogram(np.clip(p, pmin, pmax), bins=binEdges)
            countsNormed = counts / counts.sum()
            axs[0, j].bar(binEdges[:-1], countsNormed, np.diff(binEdges), facecolor='k', align='edge')

        #
        x = np.arange(0.001, 1, 0.001)
        for j in range(self.windows.shape[0]):
            n = list()
            mi = np.clip(self.ns['mi/pref/real'][:, j, 0], *miRange)
            p = self.ns['p/pref/real'][:, j, 0]
            for xi in x:
                ni = (
                    np.logical_and(mi < 0, p < xi).sum(),
                    np.logical_and(mi > 0, p < xi).sum()
                )
                n.append(ni)
            n = np.array(n)
            n = n / float(len(self.ukeys))
            axs[2, j].plot(x, n[:, 0], color=blue)
            axs[2, j].plot(x, n[:, 1], color=red)

        #
        ylim = [np.inf, -np.inf]
        for ax in axs[0, :]:
            y1, y2 = ax.get_ylim()
            if y1 < ylim[0]:
                ylim[0] = y1
            if y2 > ylim[1]:
                ylim[1] = y2
        for ax in axs[0, :]:
            ax.set_ylim(ylim)
        for ax in axs[0, 1:]:
            ax.set_yticks([])
        for ax in axs[1, :]:
            ax.set_ylim([-2.1, 2.1])
        for ax in axs[1, 1:]:
            ax.set_yticks([])
        # for ax in axs[2, :]:
        #     ax.set_ylim([0, 0.7])
        for ax in axs[2, 1:]:
            ax.set_yticks([])

        for i in range(3):
            for j in range(self.windows.shape[0]):
                axs[i, j].semilogx()
                axs[i, j].set_xlim([pmin, pmax])
                axs[i, j].set_xticks([0.001, 1.0])
                y1, y2 = axs[i, j].get_ylim()
                axs[i, j].vlines(0.05, y1, y2, color='k', linestyle=':', lw=1)
                axs[i, j].set_ylim([y1, y2])
        
        #
        axs[0, 0].set_ylabel('Frac. of units')
        axs[1, 0].set_ylabel('MI')
        axs[2, 0].set_xlabel('p-value')
        axs[2, 0].set_ylabel('Frac. of units')
        
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, axs