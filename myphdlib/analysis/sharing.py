import numpy as np
import h5py

def collectResponseWaveforms(sessions, dst=None):
    """
    """

    peths = {
        'preferred': list(),
        'nonpreferred': list(),
    }
    clusters = list()

    for session in sessions:

        #
        masks = list()
        for preference in peths.keys():
            peths_ = session.load(f'peths/probe/{preference}')
            mask =  np.invert(np.isnan(peths_).all(1))
            masks.append(mask)
        mask = np.logical_and(*masks)

        #
        for unit in session.population[mask]:
            clusters.append(unit.cluster)
        
        #
        for preference in peths.keys():
            peths_ = session.load(f'peths/probe/{preference}')
            for peth in peths_[mask]:
                peths[preference].append(peth)
    clusters = np.array(clusters)
    for preference in peths.keys():
        peths[preference] = np.array(peths[preference])

    if dst is None:
        return peths, clusters

    else:
        with h5py.File(str(dst), 'w') as stream:
            for preference in peths.keys():
                value = peths[preference]
                dataset = stream.create_dataset(f'peths/{preference}', value.shape, value.dtype, data=value)
            dataset = stream.create_dataset(f'clusters', value.shape, value.dtype, data=value)