from myphdlib.interface.session import SessionBase
from myphdlib.pipeline.events import EventsProcessingMixin
from myphdlib.pipeline.saccades import SaccadesProcessingMixin
from myphdlib.general.labjack import filterPulsesFromPhotologicDevice
from myphdlib.pipeline.prediction import PredictionProcessingMixin
from myphdlib.pipeline.spikes import SpikesProcessingMixin
from myphdlib.pipeline.activity import ActivityProcessingMixin
import pathlib as pl
import numpy as np
import re
import pandas as pd

import code

class StimuliProcessingMixinNR1(
    ):
    """
    """

    def _calculatePulseIntervalTimestamps(
        self, 
        ):
        """
        Computes interval between pulses in the stimulus channel of the LabJack data!
        """

        #this if else clause will have to be edited to reflect hdf format
        if self.hasDataset('labjack/matrix') == False:
            raise Exception('Labjack data not extracted')
        else:
            labjackData = self.load('labjack/matrix')
        #defining the labjack data should also be edited to pull from hdf
        timestamps = labjackData[:, 0]
        if self.cohort == 1:
            TTLdata = labjackData[:, 8]
        # minimumPulseWidthInSeconds adjusted from 0.013 to check pulse count mismatch origin
            filtered = filterPulsesFromPhotologicDevice(labjackData[:, 8],
                minimumPulseWidthInSeconds=0.0469)
        elif self.cohort == 12:
            TTLdata = labjackData[:, 6]
        # minimumPulseWidthInSeconds adjusted from 0.013 to check pulse count mismatch origin
            filtered = filterPulsesFromPhotologicDevice(labjackData[:, 6],
                minimumPulseWidthInSeconds=0.0469)
        iPulses = np.where(np.diff(filtered) > 0.5)[0]
        iPulsesFall = np.where(np.diff(filtered) < -0.5)[0] # fall times, should be one for each.
        
        # Return pulseDurations for use in pulsewidth checking later :)
        # idk if adding it right away breaks something
        pulseDurations = np.subtract(iPulsesFall, iPulses)        

        iIntervals = np.where(np.diff(iPulses) > 16000)[0] #figure out threshold value 
        iIntervals2 = iPulses[iIntervals]
        pulseTimestamps = timestamps[iPulses]
        intervalTimestamps = (timestamps[iIntervals2] + 3)
        # DEBUG, remove - Sept 16th, 2024
        #code.interact(local=dict(globals(), **locals())) 
        return pulseTimestamps, intervalTimestamps, iPulses, pulseDurations

    def _createMetadataFileList(self, pulseTimestamps, intervalTimestamps
        ):
        """
        Creates chronological list of metadata files & asserts that the number of files equals the number of stimulus blocks
        """
        #need less hard coded way to enter mouse name and date
        #parentDirOptions = (
        #    self.home.joinpath('videos'),
        #    self.home.joinpath('stimuli', 'metadata')
        #)

        # List of required metadata files
        partialFileList = [
            'sparseNoiseMetadata-1.pkl',
            'driftingGratingMetadata-0.txt',
            'driftingGratingMetadata-1.txt',
            'driftingGratingMetadata-2.txt',
            'driftingGratingMetadata-3.txt',
            'driftingGratingMetadata-4.txt', 
            'driftingGratingMetadata-5.txt', 
            'driftingGratingMetadata-6.txt', 
            'driftingGratingMetadata-7.txt',
            'driftingGratingMetadata-8.txt',
            'driftingGratingMetadata-9.txt',
            'driftingGratingMetadata-10.txt',
            'driftingGratingMetadata-11.txt',
            'driftingGratingMetadata-12.txt',
            'driftingGratingMetadata-13.txt',
            'driftingGratingMetadata-14.txt', 
            'driftingGratingMetadata-15.txt', 
            'driftingGratingMetadata-16.txt', 
            'driftingGratingMetadata-17.txt',
            'driftingGratingMetadata-18.txt'
        ]
        parentDir = self.home.joinpath('videos')
        # Choose whichever directory has all the metadata files
        #for parentDir in parentDirOptions:
         #   if 'driftingGratingMetadata-0.txt' in list(parentDir.iterdir()):
          #      break

        fileList = []
        for f in partialFileList:
            fileList.append(parentDir.joinpath(f))
        print(len(fileList))
        print(intervalTimestamps.shape[0])
        # Quick check for file # mismatch w/ interval timestamps
        assert len(fileList) == intervalTimestamps.shape[0] + 1

        # pulseTimestampOffsets = np.diff(pulseTimestamps) <- for checking correctness in each combined event :)
        pulseTimestampOffsets = np.diff(pulseTimestamps)
        return fileList

    
    def _processDriftingGratingProtocol(
        self,
        file,
        fileList,
        eventIndex,
        metadataHolder,
        fileIndex,
        pulseTimestamps,
        intervalTimestamps
        ):
        """
        """
        print(file)
        with open(file, 'r') as stream:
            lines = stream.readlines()
        orientation = int(re.findall('\s\d*\s', lines[2]).pop().strip()) # TODO: Place this in the metadata holder
        velocity = float(re.findall(r'\d+\.\d+|\d+', lines[1])[0])
        contrast = float(re.findall(r'\d+\.\d+|\d+', lines[3]).pop(0))
        metadata = np.genfromtxt(file, skip_header = 5, delimiter=',')
        blockLength = metadata.shape[0]
        if fileIndex == 0:
            thisBlockBools = pulseTimestamps < intervalTimestamps[0]
        elif fileIndex == len(fileList) - 1:
            thisBlockBools = pulseTimestamps > intervalTimestamps[-1]
        else:
            thisBlockBools = np.logical_and((pulseTimestamps > intervalTimestamps[fileIndex-1]), \
                                (pulseTimestamps < intervalTimestamps[fileIndex]))
        print(thisBlockBools.sum())
        print(metadata.shape[0])
        # Mismatch detected - find the time window where the missed pulse will be.
        #                   Response - kick out pulse, continue?
        #                   More than one pulse missing, idk for now. - gnb and ab
        if thisBlockBools.sum() == metadata.shape[0]:
            pulseCountMismatch = False
        else:
            pulseCountMismatch = True


        # Additional data massaging if there's an issue
        # Situations:
        # - pulse dropped, yields count mismatch and time mismatch.
        # - pulse dropped, yields count mismatch, no timing mismatch (cannot easily identify timing)
        # - pulse dropped and pulse added, yields potential timing mismatch (worst, don't deal w/ this unless
        #           there is evidence that this is actually a problem)
        if pulseCountMismatch:
            # Timing conditions
            thisBlockPulses = pulseTimestamps[thisBlockBools]
            thisBlockDiffs = np.diff(thisBlockPulses)
            fromFileDiffs = np.diff(metadata[:,4])
            
            # Make sure the timing between pulses isn't too large
            timingThresh = 0.1
            #timingDifferences = np.zeros((np.min(fromFileDiffs.shape[0], thisBlockDiffs.shape[0])))

            prevPulseMissing = False
            missingPulsesOffset = 0
            for i in range(fromFileDiffs.shape[0]):
                if prevPulseMissing:
                    metadataHolder[eventIndex+i, :] = np.nan
                    #i feel like we need to add the metadata for this iteration too, not just the previous missing pulse
                    #metadataHolder[eventIndex+i+1, 0:5] = metadata[eventIndex+i+1, 0:5]
                    #metadataHolder[eventIndex+i+1, 5] = 0
                    #metadataHolder[eventIndex+i+1, 7] = orientation
                    prevPulseMissing = False
                else:
                    thisDiff = fromFileDiffs[i] - thisBlockDiffs[i - missingPulsesOffset] #moved subtracting offset to thisBlockDiffs isntead
                    # Dummy data for this row to keep things rolling
                    if thisDiff > timingThresh:
                        prevPulseMissing = True
                        missingPulsesOffset += 1
                    # True data populating the structure
                    metadataHolder[eventIndex+i, 0:5] = metadata[i, 0:5]
                    metadataHolder[eventIndex+i, 5]   = 0
                    metadataHolder[eventIndex+i, 7]   = orientation
                    metadataHolder[eventIndex+i, 8] = velocity
                    metadataHolder[eventIndex+i, 9] = contrast
                    #timingDifferences[it] = fromFileDiffs[it] - thisBlockDiffs[it]
            eventIndex += blockLength

        # no pulseCountMismatch, whole block is good. Load the whole kaboodle
        else:
            metadataHolder[eventIndex:eventIndex + blockLength, 0:5] = metadata

            # adding a column that is 0 if drifting grating, 1 if fictive saccade
            metadataHolder[eventIndex:eventIndex + blockLength, 5] = 0
            metadataHolder[eventIndex:eventIndex + blockLength, 7] = orientation
            metadataHolder[eventIndex:eventIndex + blockLength, 8] = velocity
            metadataHolder[eventIndex:eventIndex + blockLength, 9] = contrast
            eventIndex += blockLength

        return eventIndex, metadataHolder

    def _processSparseNoiseProtocol(
        self,
        file,
        fileList,
        eventIndex,
        metadataHolder,
        fileIndex,
        pulseTimestamps,
        intervalTimestamps,
        pulseDurations
        ):
        """
        """
        if self.cohort == 1:
            eventIndex = eventIndex + 340 #160 pulses per fictive saccade block
                    
        return eventIndex, metadataHolder

    def _parseMetadataHolder(self, metadataHolder, iPulses):
        """
        """

        # Compute drifting grating probe timestamps
        pulseIndex = np.where(metadataHolder[:, 0] == 3)[0]
        dgPulse = np.where(metadataHolder[:, 5] == 0)[0]
        matchingIndicesDG = np.intersect1d(pulseIndex, dgPulse)
        probeIndexDG = iPulses[matchingIndicesDG]
        probeTimestampsDG = self.computeTimestamps(probeIndexDG)
        self.save('stimuli/dg/probe/timestamps', probeTimestampsDG)

        # Computer fictive saccade probe timestamps
        #commented out to avoid error
        #pulseIndex = np.where(metadataHolder[:, 0] == 3)[0]
        #fsPulse = np.where(metadataHolder[:, 5] == 1)[0]
        #matchingIndicesFS = np.intersect1d(pulseIndex, fsPulse)
        #probeIndexFS = iPulses[matchingIndicesFS]
        #probeTimestampsFS = self.computeTimestamps(probeIndexFS)
        #self.save('stimuli/fs/probes/timestamps', probeTimestampsFS)

        # Compute timestamps of grating initialization (DG Only)
        pulseIndex = np.where(metadataHolder[:, 0] == 1)[0]
        gratingIndex = iPulses[pulseIndex]
        gratingTimestamps = self.computeTimestamps(gratingIndex)
        self.save('stimuli/dg/grating/timestamps', gratingTimestamps)

        #Compute timestamps of motion initialization (DG Only)
        pulseIndex = np.where(metadataHolder[:, 0] == 2)[0]
        motionIndex = iPulses[pulseIndex]
        motionTimestamps = self.computeTimestamps(motionIndex)
        self.save('stimuli/dg/motion/timestamps', motionTimestamps)

        #Compute timestamps for end of each block (DG Only)
        pulseIndex = np.where(metadataHolder[:, 0] == 4)[0]
        itiIndex = iPulses[pulseIndex]
        itiTimestamps = self.computeTimestamps(itiIndex)
        #import pdb; pdb.set_trace()
        self.save('stimuli/dg/iti/timestamps', itiTimestamps)

        #Compute fictive saccade timestamps
        #pulseIndex = np.where(metadataHolder[:, 0] == 5)[0]
        #saccadeIndex = iPulses[pulseIndex]
        #saccadeTimestamps = self.computeTimestamps(saccadeIndex)
        #self.save('stimuli/fs/saccades/timestamps', saccadeTimestamps)

        #Assign direction to DG probe
        pulseIndex = np.where(metadataHolder[:, 0] == 3)[0]
        dgPulse = np.where(metadataHolder[:, 5] == 0)[0]
        matchingIndicesDG = np.intersect1d(pulseIndex, dgPulse)
        probeDirectionDG = metadataHolder[matchingIndicesDG, 1]
        self.save('stimuli/dg/probe/motion', probeDirectionDG)

        #Assign direction to each drifting grating block
        pulseIndex = np.where(metadataHolder[:, 0] == 1)[0]
        blockDirection = metadataHolder[pulseIndex, 1]
        self.save('stimuli/dg/grating/motion', blockDirection)

        #Assign contrast to each drifting grating block
        pulseIndex = np.where(metadataHolder[:, 0] == 1)[0]
        blockContrast = metadataHolder[pulseIndex, 9]
        self.save('stimuli/dg/grating/contrast', blockContrast)

        #Assign velocity to each drifting grating block
        pulseIndex = np.where(metadataHolder[:, 0] == 1)[0]
        blockVelocity = metadataHolder[pulseIndex, 8]
        self.save('stimuli/dg/grating/velocity', blockVelocity)

    

    def _runStimuliModule(self):
        """
        """

        pulseTimestamps, intervalTimestamps, iPulses, pulseDurations = self._calculatePulseIntervalTimestamps()
        if self.cohort == 11:
            metadataHolder = self._processMetadataForMissingFS(pulseTimestamps)

        else:
            fileList = self._createMetadataFileList(pulseTimestamps, intervalTimestamps)
            eventIndex = 0
            #Columns in metadataHolder: 
            #0: Event Type - 1 = Grating Start, 2 = Motion Start, 3 = Probe, 4 = Grating End, 5 = Fictive Saccade
            #1: Motion Direction 
            #2: Probe Contrast (DG Only)
            #3: Probe Phase (DG Only)
            #4: Event Timestamps (DG Only)
            #5: Block Type - 0 = Drifting Grating, 1 = Fictive Saccade
            #6: Trial Type (FS Only) - 0 = Probe, 1 = Fictive Saccade, 2 = Both
            #7: Orientation (DG Only)
            #8: Velocity (DG Only)
            #9: Contrast (DG Only)
            metadataHolder = np.full((len(pulseTimestamps) + 100, 10), np.nan)
            for fileIndex, file in enumerate(fileList):

                # DG metadata
                if pl.Path(file).suffix == '.txt':
                    eventIndex, metadataHolder = self._processDriftingGratingProtocol( 
                        file, 
                        fileList,
                        eventIndex,
                        metadataHolder,
                        fileIndex,
                        pulseTimestamps,
                        intervalTimestamps
                    )

                # FS metadata
                elif pl.Path(file).suffix == '.pkl':
                    eventIndex, metadataHolder = self._processSparseNoiseProtocol(
                        file,
                        fileList,
                        eventIndex,
                        metadataHolder,
                        fileIndex,
                        pulseTimestamps,
                        intervalTimestamps,
                        pulseDurations
                    )

        self._parseMetadataHolder(metadataHolder, iPulses)
        #print(metadataHolder)
        #import sys
        #np.set_printoptions(threshold=sys.maxsize)
        return metadataHolder

class NR1Session(
    EventsProcessingMixin,
    SaccadesProcessingMixin,
    StimuliProcessingMixinNR1,
    PredictionProcessingMixin,
    SpikesProcessingMixin,
    ActivityProcessingMixin,
    SessionBase
    ):
    """
    """


    def __init__(self, sessionFolder, eye='left'):
        """
        """

        super().__init__(sessionFolder, eye=eye)
        if self.cohort == 1:
            self.labjackChannelMapping = {
            'barcode': 7,
            'cameras': 9,
            'stimulus': 8,
            }
        elif self.cohort == 12:
            self.labjackChannelMapping = {
            'barcode': 5,
            'cameras': 7,
            'stimulus': 6,
            }

        return
    
    @property 
    def leftEyePose(self):
        """
        """
        csv = list(self.home.joinpath('videos').glob('*.csv')).pop()
        return csv

    @property 
    def rightEyePose(self):
        """
        """
        
        return None

    @property
    def leftCameraTimestamps(self):
        """
        """
        leftCameraTimestamps = list(self.home.joinpath('videos').glob('*rightCam*')).pop()
        return leftCameraTimestamps

    @property
    def eventSampleNumbers(self):
        """
        """

        if self.cohort in [1, 11, 12]:
            file = self.folders.ephys.joinpath('events', 'Neuropix-PXI-100.ProbeA-AP', 'TTL', 'sample_numbers.npy')
        elif self.cohort in [2, 31]:
            file = self.folders.ephys.joinpath('events', 'Neuropix-PXI-100.0', 'TTL_1', 'timestamps.npy')
        elif self.cohort == 3:
            file = self.folders.ephys.joinpath('events', 'Neuropix-PXI-103.ProbeA-AP', 'TTL', 'sample_numbers.npy')
        if file.exists() == False: 
            raise Exception('Could not locate ephys event timestamps file')
        
        #
        eventSampleNumbers = np.load(file)

        return eventSampleNumbers

    @property
    def referenceSampleNumber(self):
        """
        """

        file = self.folders.ephys.joinpath('sync_messages.txt')
        if file.exists() == False:
            raise Exception('Could not locate the ephys sync messages file')
        
        #
        with open(file, 'r') as stream:
            referenceSampleNumber = None
            for line in stream.readlines():
                if self.cohort in [1, 11, 3, 12]:
                    pattern = '@.*30000.*Hz:.*\d*'
                elif self.cohort in [2, 31]:
                    pattern = 'start time:.*@'
                result = re.findall(pattern, line)
                if len(result) == 1:
                    if self.cohort in [1, 11, 3, 12]:
                        referenceSampleNumber = int(result.pop().rstrip('\n').split(': ')[-1])
                    elif self.cohort in [2, 31]:
                        referenceSampleNumber = int(result.pop().rstrip('@').split('start time: ')[1])
                    break
        
        #
        if referenceSampleNumber is None:
            raise Exception('Failed to parse sync messages file for first sample number')

        return referenceSampleNumber

    def pullManuallyLabeledSaccades(self, csvFile):
        """
        If we label saccades manually with Alon's GUI, run this function and use output to run determineGratingMotion
        """

        xl = csvFile
        xlfile = pd.read_excel(xl)
        saccadeFrames = xlfile.time
        frameTimestamps = self.load('frames/left/timestamps')
        saccadeTimes = frameTimestamps[saccadeFrames]
        saccadeDirection = xlfile.nasal_temporal
        self.save('saccades/predicted/left/manualtimes', saccadeTimes)
        self.save('saccades/predicted/left/manuallabels', saccadeDirection)

        return


    def _determineGratingMotionAssociatedWithEachSaccade(
        self,
        manualLabeling=False
        ):
        """
        """

        gratingMotionByBlock = self.load('stimuli/dg/grating/motion')
        contrastByBlock = self.load('stimuli/dg/grating/contrast')
        velocityByBlock = self.load('stimuli/dg/grating/velocity')
        if gratingMotionByBlock is None:
            self.log(f'Session missing processed data for the drifting grating stimulus', level='warning')
            return
        nBlocks = gratingMotionByBlock.size

        #
        for eye in ('left', 'right'):

            #
            if manualLabeling==False:
                saccadeEpochs = self.load(f'saccades/predicted/{eye}/timestamps')
                if saccadeEpochs is None or saccadeEpochs.shape[0] == 0:
                    continue
                saccadeOnsetTimestamps = saccadeEpochs[:, 0]
            elif manualLabeling==True:
                saccadeOnsetTimestamps = self.load(f'saccades/predicted/{eye}/manualtimes')
                if saccadeOnsetTimestamps is None or saccadeOnsetTimestamps.shape[0] == 0:
                    continue

            #
            gratingMotionBySaccade = list()
            contrastBySaccade = list()
            velocityBySaccade = list()

            #
            motionOnsetTimestamps = self.load('stimuli/dg/grating/timestamps')
            motionOffsetTimestamps = self.load('stimuli/dg/iti/timestamps')
            gratingEpochs = np.hstack([
                motionOnsetTimestamps.reshape(-1, 1),
                motionOffsetTimestamps.reshape(-1, 1)
            ])

            #
            nBlocks = gratingEpochs.shape[0]
            for saccadeOnsetTimestamp in saccadeOnsetTimestamps:
                searchResult = False
                for blockIndex in range(nBlocks):
                    gratingOnsetTimestamp, gratingOffsetTimestamp = gratingEpochs[blockIndex]
                    gratingMotion = gratingMotionByBlock[blockIndex]
                    gratingContrast = contrastByBlock[blockIndex]
                    gratingVelocity = velocityByBlock[blockIndex]
                    if gratingOnsetTimestamp <= saccadeOnsetTimestamp <= gratingOffsetTimestamp:
                        searchResult = True
                        break
                if searchResult:
                    gratingMotionBySaccade.append(gratingMotion)
                    contrastBySaccade.append(gratingContrast)
                    if gratingContrast == 0:
                        velocityBySaccade.append(0)
                    else:
                        velocityBySaccade.append(gratingVelocity)
                else:
                    gratingMotionBySaccade.append(0)
                    contrastBySaccade.append(-1)
                    velocityBySaccade.append(-1)

            #
            gratingMotionBySaccade = np.array(gratingMotionBySaccade)
            contrastBySaccade = np.array(contrastBySaccade)
            velocityBySaccade = np.array(velocityBySaccade)
            self.save(f'saccades/predicted/{eye}/gmds', gratingMotionBySaccade)
            self.save(f'saccades/predicted/{eye}/cbs', contrastBySaccade)
            self.save(f'saccades/predicted/{eye}/vbs', velocityBySaccade)

        return

    def _runSaccadesModule(self, pupilCenterName='center'):
        """
        """

        self._extractEyePosition(pupilCenterName = pupilCenterName)
        self._correctEyePosition()
        self._interpolateEyePosition()
        self._decomposeEyePosition()
        self._reorientEyePosition()
        self._filterEyePosition()
        self._detectPutativeSaccades()

        return

        