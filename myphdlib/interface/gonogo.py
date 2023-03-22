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
          

    def extractLickTimestamps(self, session):
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

    def createLickRaster(self):
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
    
    def sortUniqueContrasts(self):
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

    def createContrastRaster(self):
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

    def createPsychometricSaccadeCurve(self):
        """
        Calculate the number of response trials for each contrast and plot it as psychometric curve, return plot
        """

        count1 = 0
        count6 = 0
        count5 = 0
        count4 = 0
        tempcount = 0

        for key in dictionary:
            for probeTimestamp in dictionary[key]:
                for lick in lickTimestamps:
                    lickRelative = (lick - probeTimestamp)
                    if lickRelative > 0 and lickRelative < responseTime:
                        tempcount = (tempcount + 1)
                        break
        if key == ' contrast4':
            count1 = tempcount
            tempcount = 0
        if key == ' contrast3':
            count6 = tempcount
            tempcount = 0
        if key == ' contrast2':
            count5 = tempcount
            tempcount = 0
        if key == ' contrast1':
            count4 = tempcount
            tempcount = 0       
        
        percentage1 = count1/len(dictionary[' contrast4'])
        percentage6 = count6/len(dictionary[' contrast3'])
        percentage5 = count5/len(dictionary[' contrast2'])
        percentage4 = count4/len(dictionary[' contrast1'])
        
        normal1 = percentage1/percentage1
        normal6 = percentage6/percentage1
        normal5 = percentage5/percentage1
        normal4 = percentage4/percentage1
        
        fig, ax = plt.subplots()
        plt.plot(['0%', '5%', '10%', '30%'], [normal4, normal5, normal6, normal1])
        plt.ylim([0.0, 1.5])
        ax.set_ylabel('Fraction of Response Trials')
        ax.set_xlabel('Trials by Contrast Change')
        return fig

        count1 = 0
        count6 = 0
        count5 = 0
        count4 = 0
        tempcount = 0

        for key in dictionary:
            for probeTimestamp in dictionary[key]:
                for lick in lickTimestamps:
                    lickRelative = (lick - probeTimestamp)
                    if lickRelative > 0 and lickRelative < responseTime:
                        tempcount = (tempcount + 1)
                        break
        if key == ' contrast4':
            count1 = tempcount
            tempcount = 0
        if key == ' contrast3':
            count6 = tempcount
            tempcount = 0
        if key == ' contrast2':
            count5 = tempcount
            tempcount = 0
        if key == ' contrast1':
            count4 = tempcount
            tempcount = 0       
        
        percentage1 = count1/len(dictionary[' contrast4'])
        percentage6 = count6/len(dictionary[' contrast3'])
        percentage5 = count5/len(dictionary[' contrast2'])
        percentage4 = count4/len(dictionary[' contrast1'])
        
        normal1 = percentage1/percentage1
        normal6 = percentage6/percentage1
        normal5 = percentage5/percentage1
        normal4 = percentage4/percentage1
        
        fig, ax = plt.subplots()
        plt.plot(['0%', '5%', '10%', '30%'], [normal4, normal5, normal6, normal1])
        plt.ylim([0.0, 1.5])
        ax.set_ylabel('Fraction of Response Trials')
        ax.set_xlabel('Trials by Contrast Change')
        return fig

    def test(self): return
    def test2(self): return