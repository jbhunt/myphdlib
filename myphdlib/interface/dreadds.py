import shutil
import numpy as np
import pathlib as pl
from myphdlib.general.labjack import filterPulsesFromPhotologicDevice
from myphdlib.interface.session import SessionBase
from myphdlib.pipeline.stimuli import StimuliProcessingMixin
from myphdlib.pipeline.events import EventsProcessingMixin
from myphdlib.pipeline.saccades import SaccadesProcessingMixin
from myphdlib.pipeline.spikes import SpikesProcessingMixin
from myphdlib.pipeline.activity import ActivityProcessingMixin
from myphdlib.pipeline.prediction import PredictionProcessingMixin
from myphdlib.extensions.matplotlib import placeVerticalLines
from myphdlib.general.algorithms import detectMissingEvents

def reformat(
    home
    ):
    """
    """

    if type(home) != pl.Path:
        home = pl.Path(home)

    print(f'Working on {home} ...')

    stimuliFolder = home.joinpath('stimuli')
    if stimuliFolder.exists() == False:
        stimuliFolder.mkdir()
    
    metadataFolder = stimuliFolder.joinpath('metadata')
    if metadataFolder.exists() == False:
        metadataFolder.mkdir()

    videosFolder = home.joinpath('videos')
    for file in videosFolder.iterdir():
        if 'Metadata' in file.name:
            shutil.copy2(
                file,
                metadataFolder
            )
        fileToCheck = metadataFolder.joinpath(file.name)
        if fileToCheck.exists():
            file.unlink()
        if file.name == 'notes.txt':
            file.unlink()
        if file.suffix in ('.h5', '.pickle'):
            file.unlink()

    notesFile = home.joinpath('notes.txt')
    if notesFile.exists():
        metadataFile = home.joinpath('metadata.txt')
        notesFile.replace(metadataFile)

    # Move the spike-sorting results to its own folder
    spikeSortingFolder = home.joinpath('ephys', 'sorting')
    if spikeSortingFolder.exists() == False:
        spikeSortingFolder.mkdir()

    ephysFolder = home.joinpath('ephys', 'continuous', 'Neuropix-PXI-100.0')
    if ephysFolder.exists() == False:
        raise Exception('Could not locate ephys folder')

    for src in ephysFolder.iterdir():
        if src.name in ('continuous.dat', 'timestamps.npy'):
            continue
        if src.name == 'temp_wh.dat':
            file.unlink()
            continue
        dst = spikeSortingFolder.joinpath(src.name)
        if dst.exists() == False:
            print(f'Copying {src.name} to spike sorting folder')
            shutil.copy2(
                src,
                spikeSortingFolder
            )
        if dst.exists():
            src.unlink()
    
    # Modify the metadata file
    notes = list()
    with open(home.joinpath('metadata.txt'), 'r') as stream:

        # Read
        metadata = dict()
        for line in stream.readlines():
            if line == '\n':
                continue
            if line.startswith('-'):
                notes.append(line)
                continue
            key, value = line.rstrip('\n').split(': ')
            if key.lower() == 'experiment':
                if value.lower() in ('cno', 'saline'):
                    metadata['Treatment'] = value
            metadata[key] = value
        metadata['Experiment'] = 'Dreadds'
        metadata['Cohort'] = '1'
        metadata['Hemisphere'] = 'Right'

        # Write
        with open(home.joinpath('metadata.txt'), 'w') as stream:
            for key, value in metadata.items():
                line = f'{key}: {value}\n'
                stream.write(line)

    #
    if len(notes) != 0:
        with open(home.joinpath('notes.txt'), 'w') as stream:
            for line in notes:
                stream.write(line)

    # Get rid of the old output and input files
    for filename in ('input.txt', 'output.pickle'):
        file = home.joinpath(filename)
        if file.exists():
            file.unlink()

    return

class StimuliProcessingMixinDreadds(StimuliProcessingMixin):
    """
    """

    def _identifyProtocolEpochs(self, xData=None):
        """
        """

        #
        if xData is None:
            M = self.load('labjack/matrix')
            lightSensorSignal = M[:, self.labjackChannelMapping['stimulus']]
            xData = np.around(placeVerticalLines(lightSensorSignal), 0).astype(int)
        
        #
        paths = (
            'epochs/sn/pre',
            'epochs/mb/pre',
            'epochs/dg',
            'epochs/sn/post',
            'epochs/mb/post',
            'epochs/ng'
        )

        #
        indices = np.hstack([
            xData[0:-1].reshape(-1, 1),
            xData[1:  ].reshape(-1, 1)
        ])

        #
        nEpochs = len(paths)
        if nEpochs != indices.shape[0]:
            raise Exception('User input does not match expected number of epochs')

        #
        for path, (start, stop) in zip(paths, indices):
            self.save(path, np.array([start, stop]))

        return

    def _detectMissingEventsDuringDriftingGratingProtocol(
        self,
        trialParameters,
        risingEdgeIndices,
        interBlockIntervalThresholdInSeconds=4,
        nBlocksExpected=60,
        maximumLagInSeconds=0.02,
        ):
        """
        """

        # 
        if self.cohort in (1,):
            eventMask = np.full(len(trialParameters['events']), True)

        # Expected timestamps
        eventTimestampsExpected = np.around(np.array(trialParameters['timestamps']).astype(float)[eventMask], 3)
        eventTimestampsExpected -= eventTimestampsExpected[0]
        nEventsExpected = eventTimestampsExpected.size

        # Observed timestamps
        eventTimestampsObserved = np.around(risingEdgeIndices / self.labjackSamplingRate, 3)
        eventTimestampsObserved -= eventTimestampsObserved[0]
        nEventsObserved = eventTimestampsObserved.size

        # Split observed timestamps into blocks
        peaks = np.where(
            np.diff(risingEdgeIndices) / self.labjackSamplingRate > interBlockIntervalThresholdInSeconds
        )[0]
        interBlockBoundaries = np.around(np.mean(np.vstack([
            risingEdgeIndices[peaks + 1], risingEdgeIndices[peaks]
        ]), axis=0), 0).astype(int)
        interBlockBoundaries = np.concatenate([[-np.inf], interBlockBoundaries, [np.inf]])
        blockBoundaries = np.hstack([
            interBlockBoundaries[0:-1].reshape(-1, 1),
            interBlockBoundaries[1:  ].reshape(-1, 1)
        ])
        eventTimestampsObservedByBlock = list()
        risingEdgeIndicesByBlock = list()
        for blockBoundary in blockBoundaries:
            eventIndicesInBlock = np.where(np.logical_and(
                risingEdgeIndices >= blockBoundary[0],
                risingEdgeIndices <= blockBoundary[1]
            ))[0]
            eventTimestampsObservedByBlock.append(
                eventTimestampsObserved[eventIndicesInBlock].tolist()
            )
            risingEdgeIndicesByBlock.append(
                risingEdgeIndices[eventIndicesInBlock].tolist()
            )

        # Split expected timestamps into blocks
        interBlockBoundaries = np.where(np.diff(eventTimestampsExpected) > interBlockIntervalThresholdInSeconds)[0] + 1
        interBlockBoundaries = np.concatenate([[0], interBlockBoundaries, [eventTimestampsExpected.size]])
        blockBoundaries = np.hstack([
            interBlockBoundaries[:-1].reshape(-1, 1),
            interBlockBoundaries[1: ].reshape(-1, 1)
        ])
        eventTimestampsExpectedByBlock = list()
        for start, stop in blockBoundaries:
            eventTimestampsExpectedByBlock.append(
                eventTimestampsExpected[start: stop].tolist()
            )

        #
        if len(eventTimestampsExpectedByBlock) != nBlocksExpected or len(eventTimestampsObservedByBlock) != nBlocksExpected:
            self.log(f'Failed to parse event timestamps into blocks for the drifting gratings stimulus', level='error')
            return False, risingEdgeIndices

        # Compare observed vs. expected timestamps block-by-block
        risingEdgeIndicesCorrected = list()
        resultAcrossBlocks = list()
        for iBlock in range(nBlocksExpected):

            #
            eventTimestampsExpectedInBlock = np.array(eventTimestampsExpectedByBlock[iBlock])
            nEventsExpectedInBlock = eventTimestampsExpectedInBlock.size
            eventTimestampsObservedInBlock = np.array(eventTimestampsObservedByBlock[iBlock])
            # eventTimestampsCorrectedInBlock = np.copy(eventTimestampsObservedInBlock)
            risingEdgeIndicesInBlock = np.copy(risingEdgeIndicesByBlock[iBlock]).astype(float)
            resultForBlock, missingEventsMask, insertionIndices = detectMissingEvents(
                eventTimestampsObservedInBlock,
                eventTimestampsExpectedInBlock
            ) 
            risingEdgeIndicesInBlock = np.insert(
                risingEdgeIndicesInBlock,
                insertionIndices,
                np.nan
            )
            nMissingEventsDetected = insertionIndices.size

            #
            if risingEdgeIndicesInBlock.size != nEventsExpectedInBlock:
                resultForBlock = False
            resultAcrossBlocks.append(resultForBlock)

            #
            if resultForBlock:
                if nMissingEventsDetected > 0:
                    self.log(f'{nMissingEventsDetected} missing events corrected for in block {iBlock + 1} of the drifting grating stimulus', level='info')
                for risingEdgeIndex in risingEdgeIndicesInBlock:
                    risingEdgeIndicesCorrected.append(risingEdgeIndex)
            else:
                self.log(f'Missing event correction failed for block {iBlock + 1} of the drifting grating stimulus', level='warning')
                for iEvent in range(nEventsExpectedInBlock):
                    risingEdgeIndicesCorrected.append(np.nan)

        return any(resultAcrossBlocks), risingEdgeIndicesCorrected

    def _processDriftingGratingProtocol(
        self,
        startLineIndex=6):
        """
        """

        self.log('Processing the drifting grating stimulus data', level='info')

        #
        if self.hasDataset('stimuli/dg'):
            self.remove('stimuli/dg')

        # Read the metadata file
        result = list(self.folders.stimuli.rglob('*driftingGratingMetadata*'))
        if len(result) != 1:
            self.log('Could not locate the drifting grating stimulus metadata', level='warning')
            return
        file = result.pop()
        with open(file, 'r') as stream:
            lines = stream.readlines()[startLineIndex:]
        trialParameters = {
            'events': list(),
            'motion': list(),
            'phase': list(),
            'contrast': list(),
            'timestamps': list()
        }
        trialParameterKeys = ('events', 'motion', 'contrast', 'phase', 'timestamps')
        for line in lines:
            contrast, phase = 1.0, np.nan
            event, motion, timestamp = line.rstrip('\n').split(', ')
            params = (event, motion, contrast, phase, timestamp)
            for key, value in zip(trialParameterKeys, params):
                trialParameters[key].append(value)

        # Load the labjack data
        M = self.load('labjack/matrix')
        start, stop = self.load('epochs/dg')
        signal = M[start: stop, self.labjackChannelMapping['stimulus']]

        # Check for data loss
        dataLossDetected = False
        if np.isnan(signal).sum() > 0:
            self.log('Data loss detected during the drifting grating stimulus', level='warning')
            dataLossDetected = True

        # Parse protocol events
        filtered = filterPulsesFromPhotologicDevice(signal, minimumPulseWidthInSeconds=0.03)
        risingEdgeIndices = np.where(np.diff(filtered) > 0.5)[0]

        nEventsExpected = np.array([1 for ev in trialParameters['events']
            if int(ev) == 3
        ]).sum()
        nEventsObserved = risingEdgeIndices.size
        if nEventsObserved != nEventsExpected: # TODO: Try to recover from missing events
            result, risingEdgeIndices = self._detectMissingEventsDuringDriftingGratingProtocol(
                trialParameters,
                risingEdgeIndices
            )
            if result == False:
                return
        probeOnsetTimestamps = self.computeTimestamps(risingEdgeIndices + start)
        gratingMotionDuringProbes = np.array(trialParameters['motion']).astype(int)
        self.save(f'stimuli/dg/probe/timestamps', probeOnsetTimestamps)
        self.save(f'stimuli/dg/probe/motion', gratingMotionDuringProbes)
        for eventName in ('grating', 'motion', 'iti'):
            self.save(f'stimuli/dg/{eventName}/timestamps', np.array([]).astype(float))

    # TODO: Code these methods
    def _processSparseNoiseProtocol(self):
        return

    def _processNoisyGratingProtocol(self):
        return

    def _runStimuliModule(
        self,
        ):
        """
        """

        self._processSparseNoiseProtocol()
        self._processDriftingGratingProtocol()
        self._processNoisyGratingProtocol()

        return

class DreaddsSession(
    StimuliProcessingMixinDreadds,
    SaccadesProcessingMixin,
    EventsProcessingMixin,
    SpikesProcessingMixin,
    ActivityProcessingMixin,
    PredictionProcessingMixin,
    SessionBase):
    """
    """

    labjackChannelMapping = {
        'barcode': 5,
        'cameras': 6,
        'stimulus': 7
    }