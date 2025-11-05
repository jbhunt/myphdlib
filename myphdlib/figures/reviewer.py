from myphdlib.figures.analysis import AnalysisBase
import matplotlib.pyplot as plt
import numpy as np

class ReviewerFigure(AnalysisBase):
    """
    """

    def plotModulationByProbabilityValues(
        self,
        figsize=(10, 4)
        ):
        """
        """

        fig, axs = plt.subplots(ncols=10, nrows=3, gridspec_kw={'height_ratios': [1, 2, 1]}, constrained_layout=True)
        for j in range(self.windows.shape[0]):
            mi = np.clip(self.ns['mi/pref/real'][:, j, 0], -1, 1)
            p = self.ns['p/pref/real'][:, j, 0]
            axs[1, j].scatter(p, mi, marker='.', color='k', s=5, alpha=0.3)
            axs[0, j].hist(p, range=(0, 1), bins=30, facecolor='k')

        #
        x = np.arange(0.001, 1, 0.001)
        for j in range(self.windows.shape[0]):
            n = list()
            mi = np.clip(self.ns['mi/pref/real'][:, j, 0], -1, 1)
            p = self.ns['p/pref/real'][:, j, 0]
            for xi in x:
                ni = (
                    np.logical_and(mi < 0, p < xi).sum(),
                    np.logical_and(mi > 0, p < xi).sum()
                )
                n.append(ni)
            n = np.array(n)
            n = n / float(len(self.ukeys))
            axs[2, j].plot(x, n[:, 0], color='b')
            axs[2, j].plot(x, n[:, 1], color='r')

        #
        for ax in axs[2, :].flatten():
            ax.semilogx()
            ax.set_xticks([0.01, 1.0])
            ax.vlines(0.05, 0, 1, color='k', alpha=0.3)
        for ax in axs[:2, :].flatten():
            ax.set_xticks([0, 0.5, 1])

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
            ax.set_ylim([-1.05, 1.05])
        for ax in axs[1, 1:]:
            ax.set_yticks([])
        for ax in axs[2, :]:
            ax.set_ylim([0, 0.7])
        for ax in axs[2, 1:]:
            ax.set_yticks([])
        
        #
        axs[0, 0].set_xlabel('p-value')
        axs[0, 0].set_ylabel('# of units')
        axs[1, 0].set_xlabel('p-value')
        axs[1, 0].set_ylabel('MI')
        axs[2, 0].set_xlabel('p-value')
        axs[2, 0].set_ylabel('Frac. of units')
        
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])

        return fig, axs