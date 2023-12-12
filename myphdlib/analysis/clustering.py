import numpy as np
from myphdlib.general.toolkit import psth2
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

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

class ClusteringAnalysis():
    """
    """

    def __init__(self):
        """
        """

        return

    def run(
        self,
        sessions,
        ):
        """
        """

        # Extract the shape of the visual response
        samples = list()
        raw = list()
        for session in sessions:
            if session.probeTimestamps is None:
                continue
            session.population.filter()
            for probeDirection in ('left',):
                responseCurves = session.load(f'population/peths/probe/{probeDirection}')
                for unit in session.population:
                    sample = responseCurves[unit.index]
                    normed = _normalizeResponseCurve(sample)
                    samples.append(normed)
                    raw.append(sample)

        #
        samples = np.array(samples)
        pca = PCA(n_components=3)
        x1 = pca.fit_transform(samples) # Decompose and scale
        scaler = MinMaxScaler(feature_range=(0, 1))
        x1 = scaler.fit_transform(x1)
        
        return x1, samples, np.array(raw), np.array(dr)