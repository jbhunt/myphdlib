import shutil
import numpy as np
import pathlib as pl

def makeDatasetForJesse(sessions, dst):
    """
    """

    for session in sessions:
        files = list()
        if 'saccadeOnsetTimestamps' not in session.keys:
            continue
        files.append(session.sessionFolderPath.joinpath('output.pickle'))
        files.append(
            session.sessionFolderPath.joinpath('ephys', 'sync_messages.txt')
        )
        files.append(
            session.sessionFolderPath.joinpath('ephys', 'continuous', 'Neuropix-PXI-100.0', 'spike_times.npy')
        )
        files.append(
            session.sessionFolderPath.joinpath('ephys', 'continuous', 'Neuropix-PXI-100.0', 'spike_clusters.npy')
        )
        dst2 = pl.Path(dst).joinpath(session.date.strftime('%Y-%m-%d'), session.animal)
        if dst2.exists() == False:
            dst2.mkdir(parents=True)
        for file in files:
            f1 = str(file)
            f2 = str(dst2.joinpath(file.name))
            shutil.copy(f1, f2)

    return

def addSpikeTimestampsToDataset(sessions, dst):
    """
    """

    for session in sessions:
        root = pl.Path(dst).joinpath(session.date.strftime('%Y-%m-%d'), session.animal)
        if root.exists() == False:
            continue
        spikesFolderPath = root.joinpath('spikes')
        if spikesFolderPath.exists() == False:
            spikesFolderPath.mkdir()
        for unit in session.spikeSortingResults:
            fname = str(spikesFolderPath.joinpath(f'neuron-{unit.cluster}.txt'))
            np.savetxt(fname, unit.timestamps, fmt='%.3f')

    return