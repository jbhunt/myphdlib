import numpy as np
from myphdlib.general.toolkit import psth2
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

class ClusteringAnalysis():
    """
    """

    def __init__(self):
        """
        """

        self._X = None
        self._model = None
        self._k = None
        self._fig = None

        return

    def _plotResults(
        self,
        figsize=(12, 4)
        ):
        """
        """


        #
        pca = PCA(n_components=2)
        xDecomposed = pca.fit_transform(self.X)
        x, y = xDecomposed[:, 0], xDecomposed[:, 1]
        c = [f'C{l}' for l in self.model.labels]

        #
        fig = plt.figure()
        gs = GridSpec(nrows=k, ncols=3)
        ax1 = fig.add_subplot(gs[:, 0])
        matrix = linkage(self.X, 'ward')
        R = dendrogram(
            matrix,
            link_color_func=lambda i: 'k',
            above_threshold_color='k',
            ax=ax1
        )
        ax2 = fig.add_subplot(gs[:, 1])
        ax2.scatter(x, y, c=c, s=3, alpha=0.7)

        #
        axs = list()
        for clusterIndex, clusterLabel in enumerate(np.unique(self.model.labels_)):
            ax = fig.add_subplot(gs[clusterIndex, 2])
            axs.append(ax)
            mask = self.model.labels_ == clusterLabel
            samples = self.X[mask]
            ax.pcolor(samples, vmin=-0.6, vmax=0.6, cmap='coolwarm')
            ymin, ymax = ax.get_ylim()
            curve = samples.mean(0)
            stretched = np.interp(curve, (curve.min(), curve.max()), (ymin * 0.9, ymax * 0.9))
            ax.plot(stretched, color='k')
            ax.set_ylim([ymin, ymax])

        #
        ax1.set_xticks([])
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        self._fig = fig

        return

    def _fitModel(
        self,
        sessions,
        k=7
        ):
        """
        """

        #
        self._k = k

        #
        for session in sessions:
            session.population.unfilter()

        samples = list()
        for session in sessions:
            if session.probeTimestamps is None:
                continue
            peths = {
                'left': session.load(f'population/peths/probe/left'),
                'right': session.load(f'population/peths/probe/right')
            }
            for unit in session.population:
                sample = np.zeros(peths['left'].shape[1])
                for probeDirection in ('left', 'right'):
                    peth = peths[probeDirection][unit.index]
                    if np.isnan(sample).all():
                        continue
                    if peth.max() > sample.max():
                        sample = peth
                if np.sum(sample) == 0:
                    continue
                samples.append(sample)
        self._X = np.array(samples)

        #
        self._model = AgglomerativeClustering(n_clusters=k).fit(self._X)
        
        return
    
    def run(
        self,
        sessions,
        ):
        """
        """

        self._fitModel(sessions)
        self._plotResults()

        return
    
    @property
    def X(self):
        return self._X
    
    @property
    def k(self):
        return self._k
    
    @property
    def model(self):
        return self._model
    
    @property
    def fig(self):
        return self._fig