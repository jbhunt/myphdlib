import yaml
import numpy as np
import pandas as pd
import pathlib as pl
from myphdlib.interface.session import SessionBase
from myphdlib.general.labjack import loadLabjackData
from myphdlib.general.labjack import extractLabjackEvent
from myphdlib.extensions.deeplabcut import loadBodypartData
import scipy as sp
from matplotlib import pylab as plt

class GonogoSession(SessionBase):
    """
    """

    def __init__(self, sessionFolder):
        """
        """

        super().__init__(sessionFolder)

        return

    @property
    def fps(self):
        """
        """

        result = list(self.sessionFolderPath.joinpath('videos').glob('*_metadata.yaml'))
        if len(result) != 1:
            raise Exception('Could not locate video acquisition metadata file')
        with open(result.pop(), 'r')  as stream:
            metadata = yaml.full_load(stream)

        for key in metadata.keys():
            if key in ('cam1', 'cam2'):
                if metadata[key]['ismaster']:
                    fps = int(metadata[key]['framerate'])

        return fps
    @property
    def probeMetadata(self):
        """
        """

        result = list(self.sessionFolderPath.joinpath('videos').glob('*ProbeMetadata.txt'))
        if len(result) != 1:
            raise Exception('Could not locate the probe metadata')
        else:
            return result.pop()

    @property
    def rightCameraMovie(self):
        """
        """

        result = list(self.sessionFolderPath.joinpath('videos').glob('*_rightCam-0000.mp4'))
        if len(result) != 1:
            raise Exception('Could not locate the right camera movie')
        else:
            return result.pop()

    @property
    def leftEyePose(self):
        """
        """

        result = list(self.sessionFolderPath.joinpath('videos').glob('*pupilsizeFeb6shuffle1*'))
        if len(result) != 1:
            raise Exception('Could not locate the left eye pose estimate')
        else:
            return result.pop()
    @property
    def tonguePose(self):
        """
        """

        result = list(self.sessionFolderPath.joinpath('videos').glob('*licksNov3shuffle1*'))
        if len(result) != 1:
            raise Exception('Could not locate the tongue pose estimate')
        else:
            return result.pop()

    @property
    def rightCameraTimestamps(self):
        """
        """

        result = list(self.sessionFolderPath.joinpath('videos').glob('*rightCam_timestamps.txt'))
        if len(result) != 1:
            raise Exception('Could not locate the right camera timestamps')
        else:
            return result.pop()

    @property
    def leftCameraTimestamps(self):
        """
        """

        result = list(self.sessionFolderPath.joinpath('videos').glob('*leftCam_timestamps.txt'))
        if len(result) != 1:
            raise Exception('Could not locate the left camera timestamps')
        else:
            return result.pop()

    @property
    def labjackFolder(self):
        """
        """

        result = list(self.sessionFolderPath.joinpath('labjack').glob('*test*'))
        if len(result) != 1:
            raise Exception('Could not locate the Labjack folder')
        else:
            return result.pop()

    def _loadLabjackMatrix(self):
        """
        Load labjack matrix into memory and store as attribute
        """

        self._labjackMatrix = None
        labjackDirectory = session.labjackFolder
        labjackData = loadLabjackData(labjackDirectory)

        return labjackData

    def extractProbeTimestamps(self, session):
        """
        Extract timestamps of probes from Labjack data and return probe timestamps
        """
        labjackDirectory = session.labjackFolder
        labjackData = loadLabjackData(labjackDirectory)
        timestamps = labjackData[:, 0]
        probeOnset, probeIndices = extractLabjackEvent(labjackData, 6, edge = 'rising', pulseWidthRange = (20, 700))
        probeTimestamps = timestamps[probeIndices]
        self.probeTimestamps = probeTimestamps
        return probeTimestamps

    def extractFrameTimestamps(self, session):
        """
        Extract timestamps of frames from Labjack data and return frame timestamps
        """
        labjackDirectory = session.labjackFolder
        labjackData = loadLabjackData(labjackDirectory)
        timestamps = labjackData[:, 0]
        frameOnset, frameIndices = extractLabjackEvent(labjackData, 7, edge = 'both')
        frameTimestamps = timestamps[frameIndices]
        self.frameTimestamps = frameTimestamps
        return frameTimestamps 
          

    def extractLickTimestamps(self, session, frameTimestamps):
        """
        Extract timestamps of licks recorded by DLC and synchronized with Labjack data and return lick timestamps
        """
        csv = session.tonguePose
        spoutLikelihood = loadBodypartData(csv, bodypart='spout', feature='likelihood')
        M = np.diff(spoutLikelihood, n=1)
        M1 = M*-1
        peaks, _ = sp.signal.find_peaks(M1, height=0.9, threshold=None, distance=None, prominence=None, width=None, wlen=None, rel_height=0.5, plateau_size=None)
        frameIndex = np.arange(len(loadBodypartData(csv, bodypart='spout', feature='likelihood')))
        peakFrames = frameIndex[peaks]
        lickTimestamps = self.frameTimestamps[peakFrames]
        self.lickTimestamps = lickTimestamps
        return lickTimestamps

    def createLickRaster(self, probeTimestamps, lickTimestamps):
        """
        Find licks within a given range of each probe and plot in a raster, return plot
        """
        L = list()
        for probe in self.probeTimestamps:
            lickRelative = (self.lickTimestamps - probe)
            mask = np.logical_and(
                lickRelative > -2,
                lickRelative < 5,
            )
            lickRelativeFiltered = lickRelative[mask]
            L.append(lickRelativeFiltered)
        L1 = np.array(L)
        fig, ax = plt.subplots()
        font = {'size' : 15}
        plt.rc('font', **font)
        plt.gca().invert_yaxis()
        for rowIndex, row in enumerate(L1):
            x = row
            y0 = rowIndex - 0.5
            y1 = rowIndex + 0.5
            ax.vlines(x, y0, y1, color='k')
        ax.set_ylabel('Trial')
        ax.set_xlabel('Time from probe (sec)')
        fig.set_figheight(10)
        fig.set_figwidth(6)
        return fig

    def extractContrastValues(self, session):
        """
        Reads probe metadata file and zips array of contrast values with probe timestamps, returns zipped list of contrast values
        """
        metadata = session.probeMetadata
        fn = open(metadata, 'r'); # open the file
        allText = fn.readlines() # read the lines of text
        text = allText[1:] # remove the header
        contrastValues = []; # initialize
        indNum = 0; # initialize
        for lines in text: # loop through each line and extract the lick value
            indStart = lines.find(', ');
            indStop = lines.find(', ');
            contrastValues.append(lines[(indStart + 1):(indStop + 6)])
        contrastValues = np.array(contrastValues)
        self.contrastValues = contrastValues
        return contrastValues
    
    def sortUniqueContrasts(self, probeTimestamps, contrastValues):
        """
        Sorts the array of contrast values into a dictionary with 4 keys representing the unique contrast values, returns the dictionary
        """
        dictionary = dict() # Initialize an empty dictionary
        uniqueContrastValues = np.unique(self.contrastValues) # Find the unique constrast values
        for uniqueContrastValue in uniqueContrastValues: # Iterterate through the unique contrast values
            mask = self.contrastValues == uniqueContrastValue # Create a mask for each unique contrast value
            dictionary[uniqueContrastValue] = np.array(self.probeTimestamps)[mask]
        self.dictionary = dictionary
        return dictionary

    def createContrastRaster(self, probeTimestamps, lickTimestamps, dictionary):
        """
        Create a raster sorted by contrast, returns plot
        """
        list1 = list()
        list8 = list()
        list6 = list()
        list5 = list()
        listtemp = list()

        for key in self.dictionary:
            for probeTimestamp in self.dictionary[key]:
                lickRelative = (self.lickTimestamps - probeTimestamp)
                mask = np.logical_and(
                    lickRelative > min,
                    lickRelative < max,
                )
                lickRelativeFiltered = lickRelative[mask]
    
                listtemp.append(lickRelativeFiltered)

            if key == ' contrast4':
                list1 = listtemp
                array1 = np.array(list1)
                listtemp.clear()
            if key == ' contrast3':
                list8 = listtemp
                array8 = np.array(list8)
                listtemp.clear()
            if key == ' contrast2':
                list6 = listtemp
                array6 = np.array(list6)
                listtemp.clear()
            if key == ' contrast1':
                list5 = listtemp
                array5 = np.array(list5)
                listtemp.clear()
            
        fig, ax = plt.subplots()
        for rowIndex, row in enumerate(array1):
            x = row
            y0 = rowIndex - 0.5
            y1 = rowIndex + 0.5
            ax.vlines(x, y0, y1, color='k')
        ax.set_ylabel('Trial')
        ax.set_xlabel('Time (sec)')
        fig.set_figheight(10)
        fig.set_figwidth(6)
        for rowIndex, row in enumerate(array8):
            x = row
            y0 = rowIndex + (rowIndex1 - 0.5)
            y1 = rowIndex + (rowIndex1 + 0.5)
            ax.vlines(x, y0, y1, color='b')
        for rowIndex, row in enumerate(array6):
            x = row
            y0 = rowIndex + rowIndex1 + (rowIndex2 - 0.5)
            y1 = rowIndex + rowIndex1 + (rowIndex2 + 0.5)
            ax.vlines(x, y0, y1, color='g')
        for rowIndex, row in enumerate(array5):
            x = row
            y0 = rowIndex + rowIndex1 + rowIndex2 + (rowIndex3 - 0.5)
            y1 = rowIndex + rowIndex1 + rowIndex2 + (rowIndex3 + 0.5)
            ax.vlines(x, y0, y1, color='r')
        return fig

    def extractSaccadeTimestamps(self, session, frameTimestamps):
        """
        Extract indices of saccades and use frame timestamps to extract total saccade timestamps, not nasal vs temporal
        """
        res = session.read('saccadeClassificationResults')
        nasalIndices = res['left']['nasal']['indices']
        temporalIndices = res['left']['temporal']['indices']
        nasalSaccades = self.frameTimestamps[nasalIndices]
        temporalSaccades = self.frameTimestamps[temporalIndices]
        totalSaccades = np.concatenate((nasalSaccades, temporalSaccades))
        self.totalSaccades = totalSaccades
        return totalSaccades

    def createZippedList(self, probeTimestamps, totalSaccades):
        """
        Create a boolean variable to determine whether trial is perisaccadic and create zipped list of probetimestamps, contrast values, and boolean variable
        """
        perisaccadicProbeBool = list()
        for probe in self.probeTimestamps:
            saccadesRelative = (self.totalSaccades - probe)
            mask = np.logical_and(
                saccadesRelative > -0.05,
                saccadesRelative < 0.05
            )
            perisaccades = saccadesRelative[mask]
            if any(perisaccades):
                perisaccadicTrial = True
            else:
                perisaccadicTrial = False
            perisaccadicProbeBool.append(perisaccadicTrial)
    
        perisaccadicProbeBool = np.array(perisaccadicProbeBool)

        zipped3 = list(zip(self.probeTimestamps, self.contrastValues, perisaccadicProbeBool))
        self.zipped3 = zipped3
        self.perisaccadicProbeBool = perisaccadicProbeBool
        return zipped3, perisaccadicProbeBool

    def createPeriAndExtraSaccadicLists(self, perisaccadicProbeBool, probeTimestamps, contrastValues):
        """
        Based on boolean variable, separates probetimestamps and contrast values into zipped lists of perisaccadic and extrasaccadic trials
        """
        listPT = list()
        listCT = list()
        listPF = list()
        listCF = list()
        probeBoolIndex = 0
        for probeBool in self.perisaccadicProbeBool:
            if probeBool == True:
                listPT.append(self.probeTimestamps[probeBoolIndex])
                listCT.append(self.contrastValues[probeBoolIndex])
            else:
                listPF.append(self.probeTimestamps[probeBoolIndex])
                listCF.append(self.contrastValues[probeBoolIndex])
            probeBoolIndex = probeBoolIndex + 1

        zipTrue = zip(listCT, listPT)
        zipFalse = zip(listCF, listPF)
        self.listCT = listCT
        self.listPT = listPT
        self.listCF = listCF
        self.listPF = listPF
        self.zipTrue = zipTrue
        self.zipFalse = zipFalse
        return zipTrue, zipFalse, listCT, listPT, listCF, listPF

    def createPerisaccadicDictionary(self, listCT, listPT):
        """
        Compile probetimestamps from zipTrue (listPT) into a dictionary based on contrast values
        """
        dictionaryTrue = dict() # Initialize an empty dictionary
        uniqueContrastValuesTrue = np.unique(self.listCT) # Find the unique constrast values
        for uniqueContrastValueTrue in uniqueContrastValuesTrue: # Iterterate through the unique contrast values
            mask = np.array(self.listCT) == uniqueContrastValueTrue # Create a mask for each unique contrast value
            dictionaryTrue[uniqueContrastValueTrue] = np.array(self.listPT)[mask]
        self.dictionaryTrue = dictionaryTrue
        return dictionaryTrue


    def createExtrasaccadicDictionary(self, listCF, listPF):
        """
        Compile probetimestamps from zipFalse (listPF) into a dictionary based on contrast values
        """
        dictionaryFalse = dict() # Initialize an empty dictionary
        uniqueContrastValuesFalse = np.unique(self.listCF) # Find the unique constrast values
        for uniqueContrastValueFalse in uniqueContrastValuesFalse: # Iterterate through the unique contrast values
            mask = np.array(self.listCF) == uniqueContrastValueFalse # Create a mask for each unique contrast value
            dictionaryFalse[uniqueContrastValueFalse] = np.array(self.listPF)[mask]
        self.dictionaryFalse = dictionaryFalse
        return dictionaryFalse

    def calculateExtrasaccadicResponsePercentages(self, dictionaryFalse, lickTimestamps):
        """
        Calculate the percentage of response trials for each contrast in extrasaccadic trials
        """
        count1 = 0
        count6 = 0
        count5 = 0
        count4 = 0
        tempcount = 0
        percentage1 = 0
        percentage6 = 0
        percentage5 = 0
        percentage4 = 0

        for key in self.dictionaryFalse:
            for probeTimestamp in self.dictionaryFalse[key]:
                for lick in self.lickTimestamps:
                    lickRelative = (lick - probeTimestamp)
                    if lickRelative > 0 and lickRelative < 0.5:
                        tempcount = (tempcount + 1)
                        break
            if key == ' 0.80':
                count1 = tempcount
                tempcount = 0
                percentage1 = count1/len(self.dictionaryFalse[' 0.80'])
            if key == ' 0.60':
                count6 = tempcount
                tempcount = 0
                percentage6 = count6/len(self.dictionaryFalse[' 0.60'])
            if key == ' 0.55':
                count5 = tempcount
                tempcount = 0
                percentage5 = count5/len(self.dictionaryFalse[' 0.55'])
            if key == ' 0.50':
                count4 = tempcount
                tempcount = 0
                percentage4 = count4/len(self.dictionaryFalse[' 0.50'])
        percentList = (percentage4, percentage5, percentage6, percentage1)
        percentArrayExtrasaccadic = np.array(percentList)
        self.percentArrayExtrasaccadic = percentArrayExtrasaccadic
        self.percentage1 = percentage1
        return percentArrayExtrasaccadic, percentage1

    def calculatePerisaccadicResponsePercentages(self, dictionaryTrue, lickTimestamps):
        """
        Calculate the percentage of resposne trials for each contrast in perisaccadic trials
        """
        count1T = 0
        count6T = 0
        count5T = 0
        count4T = 0
        tempcountT = 0
        percentage1T = 0
        percentage6T = 0
        percentage5T = 0
        percentage4T = 0

        for key in self.dictionaryTrue:
            for probeTimestamp in self.dictionaryTrue[key]:
                for lick in self.lickTimestamps:
                    lickRelative = (lick - probeTimestamp)
                    if lickRelative > 0 and lickRelative < 0.5:
                        tempcountT = (tempcountT + 1)
                        break
            if key == ' 0.80':
                count1T = tempcountT
                tempcountT = 0
                percentage1T = count1T/len(self.dictionaryTrue[' 0.80'])
            if key == ' 0.60':
                count6T = tempcountT
                tempcountT = 0
                percentage6T = count6T/len(self.dictionaryTrue[' 0.60'])
            if key == ' 0.55':
                count5T = tempcountT
                tempcountT = 0
                percentage5T = count5T/len(self.dictionaryTrue[' 0.55'])
            if key == ' 0.50':
                count4T = tempcountT
                tempcountT = 0
                percentage4T = count4T/len(self.dictionaryTrue[' 0.50'])

        percentListT = (percentage4T, percentage5T, percentage6T, percentage1T)
        percentArrayPerisaccadic = np.array(percentListT)
        self.percentArrayPerisaccadic = percentArrayPerisaccadic
        return percentArrayPerisaccadic

    def calculateNormalizedResponseRateExtrasaccadic(self, percentArrayExtrasaccadic, percentage1):
        """
        Normalizes response rate for extrasaccadic trials by dividing percent response from all contrasts by percent response of the highest contrast
        """
        normalExtrasaccadic = self.percentArrayExtrasaccadic/self.percentage1
        self.normalExtrasaccadic = normalExtrasaccadic
        return normalExtrasaccadic

    def calculateNormalizedResponseRatePerisaccadic(self, percentArrayPerisaccadic, percentage1):
        """
        Normalizes response rate for perisaccadic trials by dividing percent response from all contrasts by percent response of the highest contrast of extrasaccadic trials, since we don't always have perisaccadic trials at highest contrast
        """
        normalPerisaccadic = self.percentArrayPerisaccadic/self.percentage1
        self.normalPerisaccadic = normalPerisaccadic
        return normalPerisaccadic
    
    def createPsychometricSaccadeCurve(self, normalExtrasaccadic, normalPerisaccadic):
        """
        Plot the normalized response rates for extrasaccadic (red) and perisaccadic (blue) trials
        """
        fig, ax = plt.subplots()
        plt.plot(['0%', '5%', '10%', '30%'], self.normalExtrasaccadic, color='r')
        plt.plot(['0%', '5%', '10%', '30%'], self.normalPerisaccadic, color='b')
        plt.ylim([0.0, 1.5])
        ax.set_ylabel('Fraction of Response Trials')
        ax.set_xlabel('Trials by Contrast Change')
        return fig
       
    def createPerisaccadicStimHistogram(self, totalSaccades, probeTimestamps):
        """
        Create histogram showing how many perisaccadic probes in a session
        """
        perisaccadicStimList = list()
        for saccade in self.totalSaccades:
            probeRelative = np.around(self.probeTimestamps - saccade, 2)
            mask = np.logical_and(
                probeRelative > -1,
                probeRelative < 1,
            )
            if mask.sum() == 1:
                probeRelativeFiltered = probeRelative[mask]
                perisaccadicStimList.append(probeRelativeFiltered)

        perisaccadicStimArray = np.array(perisaccadicStimList)
        fig, ax = plt.subplots()
        ax.hist(perisaccadicStimArray, range=(-0.1, 0.15), bins=10, facecolor='w', edgecolor='k')
        return fig

    def plotSaccadeWaveforms(self, session):
        """
        Plot nasal and temporal saccade individual and average waveforms
        """
        res = session.read('saccadeClassificationResults')
        nasalWaveforms = res['left']['nasal']['waveforms']
        temporalWaveforms = res['left']['temporal']['waveforms']
        plt.plot(nasalWaveforms.mean(0), color='k', alpha=1)
        plt.plot(temporalWaveforms.mean(0), color='k', alpha=1)
        for wave in nasalWaveforms:
            plt.plot(wave, color='b', alpha=0.05)
        for waveT in temporalWaveforms:
            plt.plot(waveT, color='r', alpha=0.05)
        return figure