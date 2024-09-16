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

import code

class StimuliProcessingMixinDreadds2(
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
        TTLdata = labjackData[:, 6]
        # minimumPulseWidthInSeconds adjusted from 0.013 to check pulse count mismatch origin
        filtered = filterPulsesFromPhotologicDevice(labjackData[:, 6],
            minimumPulseWidthInSeconds=0.013)
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
            'driftingGratingMetadata-0.txt',
            'driftingGratingMetadata-1.txt',
            'fictiveSaccadeMetadata-0.pkl',
            'driftingGratingMetadata-2.txt',
            'driftingGratingMetadata-3.txt',
            'fictiveSaccadeMetadata-1.pkl',
            'driftingGratingMetadata-4.txt', 
            'driftingGratingMetadata-5.txt',
            'fictiveSaccadeMetadata-2.pkl', 
            'driftingGratingMetadata-6.txt', 
            'driftingGratingMetadata-7.txt',
            'fictiveSaccadeMetadata-3.pkl',
            'driftingGratingMetadata-8.txt',
            'driftingGratingMetadata-9.txt',
            'fictiveSaccadeMetadata-4.pkl'
        ]
        parentDir = self.home.joinpath('videos')
        # Choose whichever directory has all the metadata files
        #for parentDir in parentDirOptions:
         #   if 'driftingGratingMetadata-0.txt' in list(parentDir.iterdir()):
          #      break

        fileList = []
        for f in partialFileList:
            fileList.append(parentDir.joinpath(f))

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
        with open(file, 'r') as stream:
            lines = stream.readlines()
        orientation = int(re.findall('\s\d*\s', lines[2]).pop().strip()) # TODO: Place this in the metadata holder
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
                    #timingDifferences[it] = fromFileDiffs[it] - thisBlockDiffs[it]
            eventIndex += blockLength

        # no pulseCountMismatch, whole block is good. Load the whole kaboodle
        else:
            metadataHolder[eventIndex:eventIndex + blockLength, 0:5] = metadata

            # adding a column that is 0 if drifting grating, 1 if fictive saccade
            metadataHolder[eventIndex:eventIndex + blockLength, 5] = 0
            metadataHolder[eventIndex:eventIndex + blockLength, 7] = orientation
            eventIndex += blockLength

        return eventIndex, metadataHolder

    def _processFictiveSaccadesProtocol(
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
        if self.cohort == 11:
            eventIndex = eventIndex + 160 #160 pulses per fictive saccade block
        else:
            with open(file, 'rb') as f:
                metadataDict = np.load(f, allow_pickle=True)

            allEvents = metadataDict['events'] # np 1D array
            allTrials = metadataDict['trials'] # list
            blockLength = allEvents.shape[0]
            if fileIndex == 0:
                thisBlockBools = pulseTimestamps < intervalTimestamps[0]
            elif fileIndex == len(fileList) - 1:
                thisBlockBools = pulseTimestamps > intervalTimestamps[-1]
            else:
                thisBlockBools = np.logical_and((pulseTimestamps > intervalTimestamps[fileIndex-1]), \
                                    (pulseTimestamps < intervalTimestamps[fileIndex]))
            print(thisBlockBools.sum())
            #assert thisBlockBools.sum() == allEvents.shape[0], str(file) + " has pulse/event count mismatch!"
            if thisBlockBools.sum() == allEvents.shape[0]:
                pulseCountMismatch = False
            else:
                pulseCountMismatch = True
            
            # For pulse diagnosis and metadata cleaning
            thisBlockPulses = pulseTimestamps[thisBlockBools]
            thisBlockDiffs = np.diff(thisBlockPulses)
            thisBlockDiffs = np.append(thisBlockDiffs, 999) # arbitrary dist to next block

            pulseWidthThreshold = 100
            timeToNextPulseThreshold = 2.1
            
            # Starting pulse diagnosis 
            if pulseCountMismatch:
                pulseIndex = 0
                trialIndex = 0
                inBlockIter = 0
                missingPulsesOffset = 0
                pulseDurationOffset = 0
                while inBlockIter < 160: # Expecting 160 events for each block
                    eventType = allEvents[pulseIndex]
                    trialType = allTrials[trialIndex]
                    #the problem with using arrays that were calculated originally (pulseDurations & thisBlockDiffs)
                    #is that we have to correct every time we have an issue
                    pulseDuration = pulseDurations[eventIndex + inBlockIter - missingpulsesOffset - pulseDurationOffset]
                    pulseWidthViolation = (pulseDuration > pulseWidthThreshold)
                    timeToNextPulseViolation = thisBlockDiffs[inBlockIter - missingPulsesOffset - pulseDurationOffset] > timeToNextPulseThreshold
                    if pulseWidthViolation:
                        metadataHolder[eventIndex + inBlockIter, :] = np.nan
                        metadataHolder[eventIndex + inBlockIter + 1, :] = np.nan
                        # NaN this and the next entry since two signals have been fused and cannot be used. 
                        inBlockIter += 2
                        pulseIndex += 2
                        trialIndex += 1
                        pulseDurationOffset +=1
                        #as this is set up, we cannot check if we are missing a pulse directly after a pulse width violation
                        #maybe add this but i dont think it will be fatal msot of the time
                    else:
                        if timeToNextPulseViolation:
                              #because we know that there is a long time to next pulse, know it cannot be combined event
                              #were going to populate first because we will always populate this row regardless of whether the next pulse has issue
                            assert (trialType[2] == 'probe') or (trialType[2] == 'saccade'), \
                                    "Good error message!"
                            if trialType[2] == 'probe':
                                metadataHolder[eventIndex + inBlockIter, 0] = 3.0
                                metadataHolder[eventIndex + inBlockIter, 1] = allTrials[trialIndex][1]
                                metadataHolder[eventIndex + inBlockIter, 5] = 1
                                metadataHolder[eventIndex + inBlockIter, 6] = 0
                            elif trialType[2] == 'saccade':
                                metadataHolder[eventIndex + inBlockIter, 0] = 5.0
                                metadataHolder[eventIndex + inBlockIter, 1] = allTrials[trialIndex][1]
                                metadataHolder[eventIndex + inBlockIter, 5] = 1
                                metadataHolder[eventIndex + inBlockIter, 6] = 1
                            pulseIndex    += 1 
                            trialIndex    += 1
                            #if there is no direction change, populate next row with nan
                            if metadataDict[inBlockIter, 1] == metadataDict[inBlockIter + 1, 1]:
                                metadataHolder[eventIndex + inBlockIter + 1, :] = np.nan
                                inBlockIter += 1
                                pulseIndex += 1
                                trialIndex += 1
                                missingPulsesOffset += 1
                                #this does not work if the missing trial is a combined trial
                            inBlockIter += 1
                        else:
                            # Pulse (probably) has nothing wrong w/ it. Populate metadataHolder.
                            # Populate
                            if thisBlockDiffs[pulseIndex] > 0.4:
                                assert (trialType[2] == 'probe') or (trialType[2] == 'saccade'), \
                                    "Good error message!"
                                if trialType[2] == 'probe':
                                    metadataHolder[eventIndex + inBlockIter, 0] = 3.0
                                    metadataHolder[eventIndex + inBlockIter, 1] = allTrials[trialIndex][1]
                                    metadataHolder[eventIndex + inBlockIter, 5] = 1
                                    metadataHolder[eventIndex + inBlockIter, 6] = 0
                                elif trialType[2] == 'saccade':
                                    metadataHolder[eventIndex + inBlockIter, 0] = 5.0
                                    metadataHolder[eventIndex + inBlockIter, 1] = allTrials[trialIndex][1]
                                    metadataHolder[eventIndex + inBlockIter, 5] = 1
                                    metadataHolder[eventIndex + inBlockIter, 6] = 1
                                inBlockIter += 1
                                pulseIndex    += 1 
                                trialIndex    += 1
                            else:
                                assert trialType[2] == 'combined', "Invalid trialtype found : " + str(trialType[2])
                                assert pulseIndex != thisBlockPulses.shape[0] - 1, "Combined trial is last pulse, but only one found."
                                metadataHolder[eventIndex+ inBlockIter, 0] = 5.0
                                metadataHolder[eventIndex+ inBlockIter, 1] = allTrials[trialIndex][1]
                                metadataHolder[eventIndex+ inBlockIter, 5] = 1
                                metadataHolder[eventIndex+ inBlockIter, 6] = 2
                                metadataHolder[eventIndex + inBlockIter+ 1, 0] = 3.0
                                metadataHolder[eventIndex + inBlockIter+ 1, 1] = allTrials[trialIndex][1]
                                metadataHolder[eventIndex + inBlockIter+ 1, 5] = 1
                                metadataHolder[eventIndex + inBlockIter+ 1, 6] = 2
                                pulseIndex    += 2 # Two associated pulses within the index.
                                trialIndex    += 1
                               # eventIndex += 2
                                inBlockIter += 2
                eventIndex = eventIndex + 160

            else:
                # Iterate down pulses. Check the event type metadata of each one. Then, check distance from next.
                # Depending on metadata condition, different timiing allowances are permitted.
                pulseIndex = 0
                trialIndex = 0
                while pulseIndex < thisBlockPulses.shape[0]:
                    eventType = allEvents[pulseIndex]
                    trialType = allTrials[trialIndex]
                    if thisBlockDiffs[pulseIndex] > 0.4:
                        assert (trialType[2] == 'probe') or (trialType[2] == 'saccade'), \
                            "Good error message!"
                        if trialType[2] == 'probe':
                            metadataHolder[eventIndex, 0] = 3.0
                            metadataHolder[eventIndex, 1] = allTrials[trialIndex][1]
                            metadataHolder[eventIndex, 5] = 1
                            metadataHolder[eventIndex, 6] = 0
                        elif trialType[2] == 'saccade':
                            metadataHolder[eventIndex, 0] = 5.0
                            metadataHolder[eventIndex, 1] = allTrials[trialIndex][1]
                            metadataHolder[eventIndex, 5] = 1
                            metadataHolder[eventIndex, 6] = 1
                        eventIndex += 1
                        pulseIndex    += 1 
                        trialIndex    += 1
                    else:
                        assert trialType[2] == 'combined', "Invalid trialtype found : " + str(trialType[2])
                        assert pulseIndex != thisBlockPulses.shape[0] - 1, "Combined trial is last pulse, but only one found."
                        metadataHolder[eventIndex, 0] = 5.0
                        metadataHolder[eventIndex, 1] = allTrials[trialIndex][1]
                        metadataHolder[eventIndex, 5] = 1
                        metadataHolder[eventIndex, 6] = 2
                        metadataHolder[eventIndex + 1, 0] = 3.0
                        metadataHolder[eventIndex + 1, 1] = allTrials[trialIndex][1]
                        metadataHolder[eventIndex + 1, 5] = 1
                        metadataHolder[eventIndex + 1, 6] = 2
                        pulseIndex    += 2 # Two associated pulses within the index.
                        trialIndex    += 1
                        eventIndex += 2

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
        pulseIndex = np.where(metadataHolder[:, 0] == 3)[0]
        fsPulse = np.where(metadataHolder[:, 5] == 1)[0]
        matchingIndicesFS = np.intersect1d(pulseIndex, fsPulse)
        probeIndexFS = iPulses[matchingIndicesFS]
        probeTimestampsFS = self.computeTimestamps(probeIndexFS)
        self.save('stimuli/fs/probes/timestamps', probeTimestampsFS)

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
        self.save('stimuli/dg/iti/timestamps', itiTimestamps)

        #Compute fictive saccade timestamps
        pulseIndex = np.where(metadataHolder[:, 0] == 5)[0]
        saccadeIndex = iPulses[pulseIndex]
        saccadeTimestamps = self.computeTimestamps(saccadeIndex)
        self.save('stimuli/fs/saccades/timestamps', saccadeTimestamps)

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
    
    def _processMetadataForMissingFS(self, pulseTimestamps):
        """
        Process metadata for sessions that are missing fictive saccade metadata
        """
        file = self.home.joinpath('videos', 'driftingGratingMetadata.txt')
        metadata = np.genfromtxt(file, skip_header = 5, delimiter=',')
        metadataHolder = np.full((len(pulseTimestamps), 8), np.nan)
        print(len(metadata) + 800)
        print(metadataHolder.shape[0])
        if (len(metadata) + 800) != metadataHolder.shape[0]:
            self.log('Observed and expected number of events are not equal', level='error')
            return
        eventIndex = 0
        for i, row in enumerate(metadata):

            # Populate metadata holder
            metadataHolder[eventIndex, :5] = row
            metadataHolder[eventIndex, 5] = 0
            metadataHolder[eventIndex, 6] = np.nan
            metadataHolder[eventIndex, 7] = np.nan

            #
            if (i + 1) == len(metadata):
                break

            # Check time from current event to next event
            eventTimestampCurrent = row[-1]
            eventTimestampNext = metadata[i + 1, -1]

            # Check if entering FS block
            dt = eventTimestampNext - eventTimestampCurrent
            if dt > 50:
                eventIndex += 160

            # Increment event index
            eventIndex += 1

        return metadataHolder

    def _runStimuliModule(self):
        """
        """

        pulseTimestamps, intervalTimestamps, iPulses = self._calculatePulseIntervalTimestamps()
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
            metadataHolder = np.full((len(pulseTimestamps), 8), np.nan)
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
                    eventIndex, metadataHolder = self._processFictiveSaccadesProtocol(
                        file,
                        fileList,
                        eventIndex,
                        metadataHolder,
                        fileIndex,
                        pulseTimestamps,
                        intervalTimestamps
                    )

        self._parseMetadataHolder(metadataHolder, iPulses)
        return metadataHolder

class Dreadds2Session(
    EventsProcessingMixin,
    SaccadesProcessingMixin,
    StimuliProcessingMixinDreadds2,
    PredictionProcessingMixin,
    SpikesProcessingMixin,
    ActivityProcessingMixin,
    SessionBase
    ):
    """
    """

    labjackChannelMapping = {
        'barcode': 5,
        'cameras': 7,
        'stimulus': 6,
    }

    def __init__(self, sessionFolder, eye='left'):
        """
        """

        super().__init__(sessionFolder, eye=eye)

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

        if self.cohort in [1, 11]:
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
                if self.cohort in [1, 11, 3]:
                    pattern = '@.*30000.*Hz:.*\d*'
                elif self.cohort in [2, 31]:
                    pattern = 'start time:.*@'
                result = re.findall(pattern, line)
                if len(result) == 1:
                    if self.cohort in [1, 11, 3]:
                        referenceSampleNumber = int(result.pop().rstrip('\n').split(': ')[-1])
                    elif self.cohort in [2, 31]:
                        referenceSampleNumber = int(result.pop().rstrip('@').split('start time: ')[1])
                    break
        
        #
        if referenceSampleNumber is None:
            raise Exception('Failed to parse sync messages file for first sample number')

        return referenceSampleNumber

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

        