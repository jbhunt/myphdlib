import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering

def predictFunctionalGroup(
    sessions,
    k=7
    ):
    """
    """

    #
    for session in sessions:
        session.population.unfilter()

    samplesBySession = list()
    filtersBySession = list()
    for session in sessions:
        samplesFromSession = list()
        filtersFromSession = list()
        if session.probeTimestamps is None:
            samplesBySession.append(samplesFromSession)
            continue
        nUnits += session.population.count()
        peths = {
            'left': session.load(f'population/peths/probe/left'),
            'right': session.load(f'population/peths/probe/right')
        }
        nBins = peths['left'].shape[1]
        for unit in session.population:
            sample = np.zeros(peths['left'].shape[1])
            for probeDirection in ('left', 'right'):
                peth = peths[probeDirection][unit.index]
                if np.isnan(sample).all():
                    continue
                if peth.max() > sample.max():
                    sample = peth
            if np.sum(sample) == 0:
                samplesFromSession.append(np.full(nBins, np.nan))
                filtersFromSession.append(False)
            else:
                samplesFromSession.append(sample)
                filtersFromSession.append(True)
        samplesBySession.append(np.array(samplesFromSession))
        filtersBySession.append(np.array(filtersFromSession))

    return