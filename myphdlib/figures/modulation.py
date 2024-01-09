import numpy as np
from matplotlib import pylab as plt

class SaccadicModulationByClusterFigure():
    """
    """

    def generate(
        self,
        sessions,
        alpha=0.05,
        probeDirection='left',
        figsize=(4, 6)
        ):
        """
        """

        self.data = {}

        for session in sessions:
            for unit in session.population:
                if np.isnan(unit.label):
                    continue
                if unit.label not in list(self.data.keys()):
                    self.data[unit.label] = list()
                p = unit.deltaResponseProbability[probeDirection]
                if np.isnan(p):
                    continue
                if 1 - p >= alpha:
                    continue
                self.data[unit.label].append(unit.deltaResponseValue[probeDirection])

        #
        for label in self.data.keys():
            self.data[label] = np.array(self.data[label])

        #
        medians = list()
        for label in self.data.keys():
            medians.append(np.median(self.data[label]))
        index = np.argsort(medians)
        labels = np.array(list(self.data.keys()))[index]

        #
        self.fig, ax = plt.subplots()
        for y, label in enumerate(labels):
            color = f'C{label}'
            ax.boxplot(
                self.data[label],
                positions=[y],
                patch_artist=True,
                vert=False,
                boxprops={'ec': 'k', 'fc': color},
                medianprops={'color': 'k'},
                showfliers=False,
                widths=[0.5]
            )
            ys = y + np.random.uniform(low=-0.15, high=0.15, size=self.data[label].size)
            ax.scatter(self.data[label], ys, color='k', s=3, alpha=0.5, zorder=3)

        #
        ax.set_yticks(np.arange(labels.size))
        ax.set_yticklabels(np.arange(labels.size)[index])
        xmax = np.max(np.abs(ax.get_xlim()))
        ax.set_xlim([-xmax, xmax])
        ymin, ymax = ax.get_ylim()
        ax.vlines(0, ymin, ymax, color='k')
        ax.set_ylim([ymin, ymax])
        ax.set_xlabel(r'$\Delta R$')
        ax.set_ylabel('Cluster #')
        self.fig.set_figwidth(figsize[0])
        self.fig.set_figheight(figsize[1])
        self.fig.tight_layout()

        return