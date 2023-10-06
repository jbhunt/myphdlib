import h5py

pathsToKeep = {

    # Labjack barcode data
    "barcodes",
    "barcodes/labjack",
    "barcodes/labjack/indices",
    "barcodes/labjack/trains",
    "barcodes/labjack/values",
    "barcodes/neuropixels",
    "barcodes/neuropixels/indices",
    "barcodes/neuropixels/trains",
    "barcodes/neuropixels/values",

    # Manually collected stimulus epochs
    "epochs",
    "epochs/bn",
    "epochs/bn/hr",
    "epochs/bn/hr/hf",
    "epochs/bn/hr/lf",
    "epochs/bn/lr",
    "epochs/bn/lr/hf",
    "epochs/bn/lr/lf",
    "epochs/dg",
    "epochs/fs",
    "epochs/mb",
    "epochs/sn",
    "epochs/sn/post",
    "epochs/sn/pre",

    # Dropped frames mask
    "frames",
    "frames/left",
    "frames/left/dropped",
    "frames/right",
    "frames/right/dropped",

    # Labjack data
    "labjack",
    "labjack/cameras",
    "labjack/cameras/missing",
    "labjack/cameras/timestamps",
    "labjack/matrix",

    # Data that maps onto single-/multi-unit data
    "population",

    #
    "population/clusters",

    # Masks
    "population/masks",
    "population/masks/hq",
    "population/masks/sr",
    "population/masks/vr",

    # Metrics
    "population/metrics",
    "population/metrics/ac", # Amplitude cutoff
    "population/metrics/gvr", # Greatest visual response
    "population/metrics/pr", # Presence ratio
    "population/metrics/rpvr", # Refractory period violation rate
    "population/metrics/ksl", # Kilosort label
    "population/metrics/pd", # Preferred direction
    "population/metrics/nd", # Null direction
    "population/metrics/dsi", # Direction-selectivity index
    "population/metrics/mi", # Saccadic modulatio index

    # PSTHs
    "population/psths",
    "population/psths/probe",
    "population/psths/probe/left",
    "population/psths/probe/right",
    "population/psths/saccade",
    "population/psths/saccade/nasal",
    "population/psths/saccade/temporal",

    # ZETA test data
    "population/zeta",
    "population/zeta/probe",
    "population/zeta/probe/left",
    "population/zeta/probe/left/latency",
    "population/zeta/probe/left/p",
    "population/zeta/probe/right",
    "population/zeta/probe/right/latency",
    "population/zeta/probe/right/p",
    "population/zeta/saccade",
    "population/zeta/saccade/nasal",
    "population/zeta/saccade/nasal/latency",
    "population/zeta/saccade/nasal/p",
    "population/zeta/saccade/temporal",
    "population/zeta/saccade/temporal/latency",
    "population/zeta/saccade/temporal/p",

    # Eye position data
    "pose",
    "pose/corrected",
    "pose/decomposed",
    "pose/filtered",
    "pose/interpolated",
    "pose/missing",
    "pose/missing/left",
    "pose/missing/right",
    "pose/reoriented",
    "pose/uncorrected",

    # Saccade data
    "saccades",

    # Predicted saccade data
    "saccades/predicted",

    # Left eye
    "saccades/predicted/left",

    # Nasal saccades
    "saccades/predicted/left/nasal",
    "saccades/predicted/left/nasal/indices",
    "saccades/predicted/left/nasal/indices/adjusted",
    "saccades/predicted/left/nasal/indices/uncorrected",
    "saccades/predicted/left/nasal/motion",
    "saccades/predicted/left/nasal/timestamps",
    "saccades/predicted/left/nasal/waveforms",

    # Temporal saccades
    "saccades/predicted/left/temporal",
    "saccades/predicted/left/temporal/indices",
    "saccades/predicted/left/temporal/indices/adjusted",
    "saccades/predicted/left/temporal/indices/uncorrected",
    "saccades/predicted/left/temporal/motion",
    "saccades/predicted/left/temporal/timestamps",
    "saccades/predicted/left/temporal/waveforms",

    # Right eye
    "saccades/predicted/right",

    # Nasal saccades
    "saccades/predicted/right/nasal",
    "saccades/predicted/right/nasal/indices",
    "saccades/predicted/right/nasal/indices/adjusted",
    "saccades/predicted/right/nasal/indices/uncorrected",
    "saccades/predicted/right/nasal/motion",
    "saccades/predicted/right/nasal/timestamps",
    "saccades/predicted/right/nasal/waveforms",

    # Temporal saccades
    "saccades/predicted/right/temporal",
    "saccades/predicted/right/temporal/indices",
    "saccades/predicted/right/temporal/indices/adjusted",
    "saccades/predicted/right/temporal/indices/uncorrected",
    "saccades/predicted/right/temporal/motion",
    "saccades/predicted/right/temporal/timestamps",
    "saccades/predicted/right/temporal/waveforms",

    # Unsigned saccade datasets (left eye)
    "saccades/predicted/left/unsigned/dop",
    "saccades/predicted/left/unsigned/ttp",

    # Unsigned saccade datasets (right eye)
    "saccades/predicted/right/unsigned/dop",
    "saccades/predicted/right/unsigned/ttp",

    # Putative saccade data
    "saccades/putative",
    "saccades/putative/left",
    "saccades/putative/left/amplitudes",
    "saccades/putative/left/indices",
    "saccades/putative/left/waveforms",
    "saccades/putative/right",
    "saccades/putative/right/amplitudes",
    "saccades/putative/right/indices",
    "saccades/putative/right/waveforms",
    "saccades/training",

    # Saccade classification training data
    "saccades/training/left",
    "saccades/training/left/X",
    "saccades/training/left/y",
    "saccades/training/right",
    "saccades/training/right/X",
    "saccades/training/right/y",

    # Extracellular spikes and clusters
    "spikes",
    "spikes/clusters",
    "spikes/timestamps",

    # Stimulus data
    "stimuli",

    # Binary noise stimuli
    "stimuli/bn",
    "stimuli/bn/hr",
    "stimuli/bn/hr/hf",
    "stimuli/bn/hr/hf/fields",
    "stimuli/bn/hr/hf/grids",
    "stimuli/bn/hr/hf/length",
    "stimuli/bn/hr/hf/missing",
    "stimuli/bn/hr/hf/timestamps",
    "stimuli/bn/hr/lf",
    "stimuli/bn/hr/lf/fields",
    "stimuli/bn/hr/lf/grids",
    "stimuli/bn/hr/lf/length",
    "stimuli/bn/hr/lf/missing",
    "stimuli/bn/hr/lf/timestamps",
    "stimuli/bn/lr",
    "stimuli/bn/lr/hf",
    "stimuli/bn/lr/hf/fields",
    "stimuli/bn/lr/hf/grids",
    "stimuli/bn/lr/hf/length",
    "stimuli/bn/lr/hf/missing",
    "stimuli/bn/lr/hf/timestamps",
    "stimuli/bn/lr/lf",
    "stimuli/bn/lr/lf/fields",
    "stimuli/bn/lr/lf/grids",
    "stimuli/bn/lr/lf/length",
    "stimuli/bn/lr/lf/missing",
    "stimuli/bn/lr/lf/timestamps",

    # Drifting grating stimulus
    "stimuli/dg",

    # Grating onset
    "stimuli/dg/grating",
    "stimuli/dg/grating/motion",
    "stimuli/dg/grating/timestamps",

    # Grating offset
    "stimuli/dg/iti",
    "stimuli/dg/iti/timestamps",

    # Grating motion
    "stimuli/dg/motion",
    "stimuli/dg/motion/timestamps",

    # Probe stimulus
    "stimuli/dg/probe",
    "stimuli/dg/probe/contrast",
    "stimuli/dg/probe/motion",
    "stimuli/dg/probe/phase",
    "stimuli/dg/probe/timestamps",
    "stimuli/dg/probe/dos",
    "stimuli/dg/probe/tts",

    # Fictive saccades stimulus
    "stimuli/fs",
    "stimuli/fs/coincident",
    "stimuli/fs/probes",
    "stimuli/fs/probes/timestamps",
    "stimuli/fs/saccades",
    "stimuli/fs/saccades/timestamps",

    # Moving bars stimulus
    "stimuli/mb",
    "stimuli/mb/offset",
    "stimuli/mb/offset/timestamps",
    "stimuli/mb/onset",
    "stimuli/mb/onset/timestamps",
    "stimuli/mb/orientation",

    # Sparse  noise stimulus
    "stimuli/sn",
    "stimuli/sn/pre",
    "stimuli/sn/pre/coords",
    "stimuli/sn/pre/fields",
    "stimuli/sn/pre/missing",
    "stimuli/sn/pre/signs",
    "stimuli/sn/pre/timestamps",
    "stimuli/sn/post",
    "stimuli/sn/post/coords",
    "stimuli/sn/post/fields",
    "stimuli/sn/post/missing",
    "stimuli/sn/post/signs",
    "stimuli/sn/post/timestamps",

    # Timestamping function parameters
    "tfp",
    "tfp/b",
    "tfp/fp",
    "tfp/m",
    "tfp/xp",
}

def removeObsoleteDatasets(
    session,
    dryrun=True,
    ):
    """
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
        if dryrun:
            continue
        session.remove(path)

    if dryrun:
        return pathsToRemove

def checkForMissingDatasets(
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