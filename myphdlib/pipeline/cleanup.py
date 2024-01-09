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
    "epochs/mb/pre",
    "epochs/mb/post",
    "epochs/sn",
    "epochs/sn/post",
    "epochs/sn/pre",
    "epochs/ng",

    # Dropped frames mask
    "frames",
    "frames/left",
    "frames/left/dropped",
    "frames/left/intervals",
    "frames/left/timestamps",
    "frames/right",
    "frames/right/dropped",
    "frames/right/intervals",
    "frames/right/timestamps",

    # Labjack data
    "labjack",
    "labjack/cameras",
    "labjack/cameras/missing",
    "labjack/cameras/timestamps",
    "labjack/matrix",
    "labjack/timespace",

    # Data that maps onto single-/multi-unit data
    "population",

    #
    "population/clusters",

    # Metrics
    "population/metrics",
    "population/metrics/ac", # Amplitude cutoff
    "population/metrics/pr", # Presence ratio
    "population/metrics/rpvr", # Refractory period violation rate
    "population/metrics/ksl", # Kilosort label
    "population/metrics/pd", # Preferred direction
    "population/metrics/nd", # Null direction
    "population/metrics/dsi", # Direction-selectivity index
    "population/metrics/lpi", # Luminance polarity index
    "population/metrics/vra", # Visual response amplitude
    "population/metrics/bsw", # Best spike waveform
    "population/metrics/vra/left",
    "population/metrics/vra/right",
    "population/metrics/mi", # Modulation index
    "population/metrics/mi/left",
    "population/metrics/mi/left/x",
    "population/metrics/mi/left/p",
    "population/metrics/mi",
    "population/metrics/mi/right",
    "population/metrics/mi/right/x",
    "population/metrics/mi/right/p",
    "population/metrics/dr", # Delta-response (i.e., MI numerator)
    "population/metrics/dr/left",
    "population/metrics/dr/left/x",
    "population/metrics/dr/left/p",
    "population/metrics/dr",
    "population/metrics/dr/right",
    "population/metrics/dr/right/x",
    "population/metrics/dr/right/p",

    # Saccade prediction related datasets
    'prediction',
    'prediction/saccades',
    'prediction/saccades/direction',
    'prediction/saccades/direction/X',
    'prediction/saccades/direction/y',
    'prediction/saccades/epochs',
    'prediction/saccades/epochs/X',
    'prediction/saccades/epochs/y',
    'prediction/saccades/epochs/z',

    # Saccade-related datasets
    'saccades',
    "saccades/putative",
    "saccades/putative/left",
    "saccades/putative/left/indices",
    "saccades/putative/left/waveforms",
    "saccades/putative/right",
    "saccades/putative/right/indices",
    "saccades/putative/right/waveforms",
    'saccades/predicted',
    'saccades/predicted/left',
    'saccades/predicted/left/direction',
    'saccades/predicted/left/epochs',
    'saccades/predicted/left/indices',
    'saccades/predicted/left/labels',
    'saccades/predicted/left/timestamps',
    'saccades/predicted/left/waveforms',
    'saccades/predicted/left/dop',
    'saccades/predicted/left/ttp',
    'saccades/predicted/right',
    'saccades/predicted/right/direction',
    'saccades/predicted/right/epochs',
    'saccades/predicted/right/indices',
    'saccades/predicted/right/labels',
    'saccades/predicted/right/timestamps',
    'saccades/predicted/right/waveforms',
    'saccades/predicted/right/dop',
    'saccades/predicted/right/ttp',

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
    "stimuli/fs/probe",
    "stimuli/fs/probe/timestamps",
    "stimuli/fs/probe/motion",
    "stimuli/fs/saccade",
    "stimuli/fs/saccade/timestamps",
    "stimuli/fs/saccade/motion",

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

    # Normalized PETHs for clustering
    "peths",
    "peths/probe",
    "peths/probe/preferred",
    "peths/probe/nonpreferred",
    "peths/saccade",
    "peths/saccade/preferred",
    "peths/saccade/nonpreferred",

    #
    "curves",
    "curves/rProbe",
    "curves/rProbe/left",
    "curves/rProbe/right",
    "curves/rMixed",
    "curves/rMixed/left",
    "curves/rMixed/right",
    "curves/rSaccade",
    "curves/rSaccade/left",
    "curves/rSaccade/right",
    "curves/rSaccadeUnshifted",
    "curves/rSaccadeUnshifted/nasal",
    "curves/rSaccadeUnshifted/temporal",

}

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

class CleanupProccessingMixin(object):
    """
    """

    def _removeObsoleteDatasets(
        self,
        dryrun=True,
        returnPaths=False
        ):
        """
        """

        pathsToRemove = list()
        pathsInFile = list()
        with h5py.File(self.hdf, 'r') as file:
            file.visit(lambda name: pathsInFile.append(name))
        
        for path in pathsInFile:
            if path not in pathsToKeep:
                pathsToRemove.append(path)

        #
        for path in pathsToRemove:
            self.log(f'Removing dataset: {path}')
            if dryrun:
                continue
            self.remove(path)

        if returnPaths:
            return pathsToRemove

    def _runCleanupModule(self, dryrun=True):
        """
        """

        self._removeObsoleteDatasets(dryrun)

        return