import numpy as np
import pandas as pd
from myphdlib.general.session import saveSessionData

def extractEyePosition(sessionObject, likelihoodThreshold=0.95):
    """
    Extract the raw eye position data
    """

    #
    nSamples = 0
    eyePositionLeft = None
    eyePositionRight = None

    #
    for side in ('left', 'right'):
        csv = sessionObject.dlcPoseEstimates[side]
        result = sessionObject.videosFolderPath.glob(f'*{side}Cam*.csv')
        if result:
            frame = pd.read_csv(csv, header=list(range(3)), index_col=0)
        else:
            continue
        frame = frame.sort_index(level=1, axis=1)
        network = frame.columns[0][0]
        x = np.array(frame[network, 'pupilCenter', 'x']).flatten()
        y = np.array(frame[network, 'pupilCenter', 'y']).flatten()
        l = np.array(frame[network, 'pupilCenter', 'likelihood']).flatten()
        x[l < likelihoodThreshold] = np.nan
        y[l < likelihoodThreshold] = np.nan
        coords = np.hstack([
            x.reshape(-1, 1),
            y.reshape(-1, 1)
        ])
        if side == 'left':
            eyePositionLeft = np.copy(coords)
        elif side == 'right':
            eyePositionRight = np.copy(coords)
        if coords.shape[0] > nSamples:
            nSamples = coords.shape[0]

    #
    eyePositionUncorrected = np.full([nSamples, 4], np.nan)
    if eyePositionLeft is not None:
        eyePositionUncorrected[:eyePositionLeft.shape[0], 0:2] = eyePositionLeft
    if eyePositionRight is not None:
        eyePositionUncorrected[:eyePositionRight.shape[0], 2:4] = eyePositionRight
    saveSessionData(sessionObject, 'eyePositionUncorrected', eyePositionUncorrected)

    return

def correctEyePosition(sessionObject, frameIntervalTolerance=0.1, nDroppedFramesAllowed=100000):
    """
    Search for and correct missing frames in the raw eye position data
    """

    #
    eyePositionUncorrected = sessionObject.load('eyePositionUncorrected')
    nSamples = eyePositionUncorrected.shape[0] + nDroppedFramesAllowed
    eyePositionCorrected = np.full([nSamples, 4])

    #
    expectedFrameInterval = 1 / sessionObject.fps
    thresholdFrameInterval = expectedFrameInterval + (expectedFrameInterval * frameIntervalTolerance)
    for camera in ('left', 'right'):

        #
        result = sessionObject.videosFolderPath.glob(f'*{camera}Cam_timestamps.txt')
        if result:
            frameIntervals = np.loadtxt(result.pop(), dtype=np.int64) / 1000000000
        else:
            continue
        frameTimestamps = np.full(frameIntervals.size + 1, np.nan)
        frameTimestamps[0] = 0
        frameTimestamps[1:] = np.cumsum(frameIntervals)

        #
        mask = frameIntervals > thresholdFrameInterval
        indices = np.where(mask)[0]
        dropped = list()
        for frameInterval in frameIntervals[mask]:
            nFramesDropped = round(frameInterval / expectedFrameInterval)
            dropped.append(nFramesDropped)

    return

def filterEyePosition(sessionObject):
    """
    """

    return

def decomposeEyePosition(sessionObject):
    """
    """

    return

def detectMonocularSaccades(sessionObject):
    """
    """
    
    return

def classifyMonocularSaccades(sessionObject):
    """
    """

    return

def detectConjugateSaccades(sessionObject):
    """
    """

    return

def labelPutativeSaccades():
    """
    """

    return

def createTrainingDataset():
    """
    """

    return

def identifySaccadeOnset():
    """
    """

    return