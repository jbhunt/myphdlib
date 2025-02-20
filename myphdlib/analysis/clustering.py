import numpy as np
from myphdlib.general.toolkit import psth2
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_samples, silhouette_score
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.signal import find_peaks
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
        self._figs = None

        return

    def plotResults(
        self,
        figsize=(4, 4),
        t=None
        ):
        """
        """


        #
        pca = PCA(n_components=2)
        xDecomposed = pca.fit_transform(self.X)
        x, y = xDecomposed[:, 0], xDecomposed[:, 1]
        c = [f'C{l}' for l in self.model.labels_]
        coefs = silhouette_samples(self.X, self.model.labels_)
        a = np.interp(coefs, (coefs.min(), coefs.max()), (0.1, 0.9))

        #
        fig1 = plt.figure()
        ax1 = fig1.add_subplot()
        matrix = linkage(self.X, 'ward')
        R = dendrogram(
            matrix,
            link_color_func=lambda i: 'k',
            above_threshold_color='k',
            ax=ax1
        )
        ax1.set_xticks([])

        fig2 = plt.figure()
        ax2 = fig2.add_subplot()
        ax2.scatter(x, y, c=c, s=3)

        #
        for clusterLabel in np.unique(self.model.labels_):
            xc, yc = xDecomposed[self.model.labels_ == clusterLabel, :].mean(0)
            ax2.scatter(xc, yc, marker='+', color='k')

        #
        fig3 = plt.figure()
        gs = GridSpec(nrows=self.k, ncols=1)
        axs = list()
        if t is None:
            t_ = np.arange(self.X.shape[1])
        else:
            t_ = t
        for clusterIndex, clusterLabel in enumerate(np.unique(self.model.labels_)):
            ax = fig3.add_subplot(gs[clusterIndex])
            axs.append(ax)
            mask = self.model.labels_ == clusterLabel
            samples = self.X[mask]
            # peaks = np.array([find_peaks(np.abs(x), height=0.5)[0].min() for x in samples])
            # index = np.argsort(peaks)
            ax.pcolor(t_, np.arange(samples.shape[0]), samples, vmin=-1, vmax=1, cmap='coolwarm')
            ymin, ymax = ax.get_ylim()
            curve = samples.mean(0)
            stretched = np.interp(curve, (curve.min(), curve.max()), (ymax * 0.1, ymax * 0.9))
            ax.plot(t_, stretched, color='k')
            ax.set_ylim([ymin, ymax])

        #
        self._figs = [fig1, fig2, fig3]
        for fig in self.figs:
            fig.set_figwidth(figsize[0])
            fig.set_figheight(figsize[1])

        return

    def fitModel(
        self,
        sessions,
        preference='preferred',
        event='probe',
        k=None
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
            peths = session.load(f'peths/{event}/{preference}')
            for unit in session.population:
                sample = peths[unit.index]
                if np.isnan(sample).all():
                    continue
                else:
                    samples.append(sample)
        self._X = np.array(samples)

        #
        if k is None:
            models = list()
            scores = list()
            ks = np.arange(2, 16, 1)
            for k in ks:
                model = AgglomerativeClustering(n_clusters=k).fit(self.X)
                score = silhouette_score(self.X, model.labels_)
                models.append(model)
                scores.append(score)
            self._k = ks[np.argmin(scores)]
            self._model = models[np.argmin(scores)]
        else:
            ks = None
            scores = None
            self._k = k
            self._model = AgglomerativeClustering(n_clusters=self.k).fit(self.X)
        
        return ks, np.array(scores)
    
    def run(
        self,
        sessions,
        k=None,
        ):
        """
        """

        ks, scores = self.fitModel(sessions, k=k)
        self.plotResults()

        return ks, scores
    
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
    def figs(self):
        return self._figs