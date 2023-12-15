import numpy as np
from myphdlib.general.toolkit import psth2
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import linkage, dendrogram
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

def _normalizeResponseCurve(curve):
    """
    """

    index = np.argmax(np.abs(curve))
    if curve[index] < 0:
        xrange = (curve.min(), abs(curve.min()))
    elif curve[index] > 0:
        xrange = (-1 * curve.max(), curve.max())
    else:
        pass

    return np.interp(curve, xrange, (-1, 1))

def _stretchCurve(curve):
    """
    """

    return np.interp(curve, (curve.min(), curve.max()), (0, 1))

def _normalizeCurve2(w):
    bl = w[:20].mean()
    w -= bl
    pv = np.max([w.max(), abs(w.min())])
    return w / pv

class ClusteringAnalysis():
    """
    """

    def __init__(self):
        """
        """

        self.X = None

        return

    def plot(
        self,
        k=4,
        figsize=(12, 4)
        ):
        """
        """

        if self.X is None:
            raise Exception('Analysis incomplete')

        #
        pca = PCA(n_components=2)
        xDecomposed = pca.fit_transform(self.X)
        x, y = xDecomposed[:, 0], xDecomposed[:, 1]
        colors = [f'C{i}' for i in range(k)]

        #
        model = AgglomerativeClustering(n_clusters=k)
        model.fit(self.X)
        labels = model.labels_
        c = [f'C{l}' for l in labels]

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
        for clusterIndex, clusterLabel in enumerate(np.unique(model.labels_)):
            ax = fig.add_subplot(gs[clusterIndex, 2])
            axs.append(ax)
            mask = model.labels_ == clusterLabel
            samples = self.X[mask]
            a = np.clip(mask.sum() / mask.size + 0.1, 0, 1)
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

        return fig, model

    def run(
        self,
        sessions,
        ):
        """
        """

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
        self.X = np.array(samples)
        
        return