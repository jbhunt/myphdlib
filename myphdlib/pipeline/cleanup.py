import h5py

pathsToKeep = {
    'barcodes',
    'barcodes/labjack',
    'barcodes/labjack/indices',
    'barcodes/labjack/trains',
    'barcodes/labjack/values',
    'barcodes/neuropixels',
    'barcodes/neuropixels/indices',
    'barcodes/neuropixels/trains',
    'barcodes/neuropixels/values',
    'epochs',
    'epochs/bn',
    'epochs/bn/hr',
    'epochs/bn/hr/hf',
    'epochs/bn/hr/lf',
    'epochs/bn/lr',
    'epochs/bn/lr/hf',
    'epochs/bn/lr/lf',
    'epochs/dg',
    'epochs/fs',
    'epochs/mb',
    'epochs/sn',
    'epochs/sn/post',
    'epochs/sn/pre',
    'frames',
    'frames/left',
    'frames/left/dropped',
    'frames/right',
    'frames/right/dropped',
    'labjack',
    'labjack/cameras',
    'labjack/cameras/missing',
    'labjack/cameras/timestamps',
    'labjack/matrix',
    'tfp',
    'tfp/b',
    'tfp/fp',
    'tfp/m',
    'tfp/xp',
    'stimuli',
    'stimuli/bn',
    'stimuli/bn/hr',
    'stimuli/bn/hr/hf',
    'stimuli/bn/hr/hf/fields',
    'stimuli/bn/hr/hf/grids',''
    'stimuli/bn/hr/hf/length',
    'stimuli/bn/hr/hf/missing',
    'stimuli/bn/hr/hf/timestamps',
    'stimuli/bn/hr/lf',
    'stimuli/bn/hr/lf/fields',
    'stimuli/bn/hr/lf/grids',
    'stimuli/bn/hr/lf/length',
    'stimuli/bn/hr/lf/missing',
    'stimuli/bn/hr/lf/timestamps',
    'stimuli/bn/lr',
    'stimuli/bn/lr/hf',
    'stimuli/bn/lr/hf/fields',
    'stimuli/bn/lr/hf/grids',
    'stimuli/bn/lr/hf/length',
    'stimuli/bn/lr/hf/missing',
    'stimuli/bn/lr/hf/timestamps',
    'stimuli/bn/lr/lf',
    'stimuli/bn/lr/lf/fields',
    'stimuli/bn/lr/lf/grids',
    'stimuli/bn/lr/lf/length',
    'stimuli/bn/lr/lf/missing',
    'stimuli/bn/lr/lf/timestamps',
    'stimuli/dg',
    'stimuli/dg/probe/motion',
    'stimuli/dg/probe/phase',
    'stimuli/dg/probe/timestamps',
    'stimuli/dg/probe/contrast',
    'stimuli/dg/grating/timestamps'
    'stimuli/dg/motion/timestamps',
    'stimuli/dg/isi/timestamps'
    'stimuli/fs',
    'stimuli/fs/coincident',
    'stimuli/fs/probes',
    'stimuli/fs/probes/timestamps',
    'stimuli/fs/saccades',
    'stimuli/fs/saccades/timestamps',
    'stimuli/mb',
    'stimuli/mb/orientation',
    'stimuli/mb/timestamps',
    'stimuli/sn',
    'stimuli/sn/post',
    'stimuli/sn/post/coords',
    'stimuli/sn/post/fields',
    'stimuli/sn/post/missing',
    'stimuli/sn/post/signs',
    'stimuli/sn/post/timestamps',
    'stimuli/sn/pre',
    'stimuli/sn/pre/coords',
    'stimuli/sn/pre/fields',
    'stimuli/sn/pre/missing',
    'stimuli/sn/pre/signs',
    'stimuli/sn/pre/timestamps',
    'spikes',
    'spikes/clusters',
    'spikes/timestamps',
    'saccades',
    'saccades/predicted',
    'saccades/predicted/left',
    'saccades/predicted/left/nasal',
    'saccades/predicted/left/nasal/indices',
    'saccades/predicted/left/nasal/indices/adjusted',
    'saccades/predicted/left/nasal/indices/uncorrected',
    'saccades/predicted/left/nasal/timestamps',
    'saccades/predicted/left/nasal/waveforms',
    'saccades/predicted/left/temporal',
    'saccades/predicted/left/temporal/indices',
    'saccades/predicted/left/temporal/indices/adjusted',
    'saccades/predicted/left/temporal/indices/uncorrected',
    'saccades/predicted/left/temporal/timestamps',
    'saccades/predicted/left/temporal/waveforms',
    'saccades/predicted/right',
    'saccades/predicted/right/nasal',
    'saccades/predicted/right/nasal/indices',
    'saccades/predicted/right/nasal/indices/adjusted',
    'saccades/predicted/right/nasal/indices/uncorrected',
    'saccades/predicted/right/nasal/timestamps',
    'saccades/predicted/right/nasal/waveforms',
    'saccades/predicted/right/temporal',
    'saccades/predicted/right/temporal/indices',
    'saccades/predicted/right/temporal/indices/adjusted',
    'saccades/predicted/right/temporal/indices/uncorrected',
    'saccades/predicted/right/temporal/timestamps',
    'saccades/predicted/right/temporal/waveforms',
    'saccades/putative',
    'saccades/putative/left',
    'saccades/putative/left/amplitudes',
    'saccades/putative/left/indices',
    'saccades/putative/left/waveforms',
    'saccades/putative/right',
    'saccades/putative/right/amplitudes',
    'saccades/putative/right/indices',
    'saccades/putative/right/waveforms',
    'saccades/training',
    'saccades/training/left',
    'saccades/training/left/X',
    'saccades/training/left/y',
    'saccades/training/right',
    'saccades/training/right/X',
    'saccades/training/right/y',
    'pose',
    'pose/corrected',
    'pose/decomposed',
    'pose/filtered',
    'pose/interpolated',
    'pose/missing',
    'pose/missing/left',
    'pose/missing/right',
    'pose/reoriented',
    'pose/uncorrected',
    'population',
    'population/filters',
    'population/filters/quality',
    'population/filters/visual',
    'population/metrics',
    'population/metrics/contamination',
    'population/metrics/stability',
    'population/metrics/zeta',
    'population/metrics/zeta/visual',
    'population/metrics/zeta/visual/latency/peak',
    'population/metrics/zeta/visual/latency/onset'
    'population/metrics/zeta/visual/pvalues',
    'population/metrics/zeta/motor/left/nasal/latency/peak',
    'population/metrics/zeta/motor/left/nasal/latency/onset',
    'population/metrics/zeta/motor/left/temporal/latency/peak',
    'population/metrics/zeta/motor/left/temporal/latency/onset',
    'population/metrics/zeta/motor/right/nasal/latency/peak',
    'population/metrics/zeta/motor/right/nasal/latency/onset',
    'population/metrics/zeta/motor/right/temporal/latency/peak',
    'population/metrics/zeta/motor/right/temporal/latency/onset',
    'population/uids',
}

def cleanupOutputFile(session):
    """
    Remove old datasets/groups
    """

    pathsToRemove = list()
    pathsInFile = list()
    with h5py.File(session.hdf, 'r') as file:
        file.visit(lambda name: pathsInFile.append(name))
    
    for path in pathsInFile:
        if path not in pathsToKeep:
            pathsToRemove.append(path)

    #
    for path in pathsToRemove:
        print(f'INFO[{session.animal}, {session.date}]: Removing "{path}" from output file')
        session.remove(path)

    return

def checkForMissingGroups(
    session,
    returnMissingPaths=False
    ):
    """
    """

    pathsInFile = list()
    with h5py.File(session.hdf, 'r') as file:
        file.visit(lambda name: pathsInFile.append(name))

    pathsNotFound = list()
    for path in pathsToKeep:
        if path not in pathsInFile:
            print(f'INFO[{session.animal}, {session.date}]: "{path}" path missing from output file')
            pathsNotFound.append(path)

    if returnMissingPaths:
        return pathsNotFound