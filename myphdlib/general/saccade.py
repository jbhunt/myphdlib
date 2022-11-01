import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from scipy.stats import pearsonr
from scipy.signal import find_peaks as findPeaks
from myphdlib.general.toolkit import smooth
from myphdlib.general.session import saveSessionData

def extractEyePosition(sessionObject, likelihoodThreshold=0.95, pupilCenterName='pupilCenter'):
    """
    Extract the raw eye position data
    """

    #
    nSamples = 0
    eyePositionLeft = None
    eyePositionRight = None

    #
    for side in ('left', 'right'):
        result = list(sessionObject.videosFolderPath.glob(f'*{side}Cam*.csv'))
        if result:
            frame = pd.read_csv(result.pop(), header=list(range(3)), index_col=0)
        else:
            continue
        frame = frame.sort_index(level=1, axis=1)
        network = frame.columns[0][0]
        x = np.array(frame[network, pupilCenterName, 'x']).flatten()
        y = np.array(frame[network, pupilCenterName, 'y']).flatten()
        l = np.array(frame[network, pupilCenterName, 'likelihood']).flatten()
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

def correctEyePosition(sessionObject, pad=1e6):
    """
    Correct eye position data for missing/dropped frames
    """

    #
    eyePositionUncorrected = sessionObject.load('eyePositionUncorrected')
    nSamples = eyePositionUncorrected.shape[0] + int(pad)
    eyePositionCorrected = np.full([nSamples, 4], np.nan)

    #
    terminationIndex = 0
    expectedFrameInterval = 1 / sessionObject.fps * 1000 # In ms
    for camera in ('left', 'right'):

        #
        if camera == 'left':
            columnIndices = 0, 1
        else:
            columnIndices = 2, 3

        #
        result = list(sessionObject.videosFolderPath.glob(f'*{camera}Cam_timestamps.txt'))
        if result:
            frameIntervals = np.loadtxt(result.pop(), dtype=np.int64) / 1000000 # In ms
        else:
            continue

        # TODO: Figure out if I should save these data
        # frameTimestamps = np.full(frameIntervals.size + 1, np.nan)
        # frameTimestamps[0] = 0
        # frameTimestamps[1:] = np.cumsum(frameIntervals)

        #
        frameIndex = 0
        frameOffset = 0
        eyePositionUncorrected[0, columnIndices] = eyePositionUncorrected[0, columnIndices]
        for frameInterval in frameIntervals:
            frameOffset += round(frameInterval / expectedFrameInterval) - 1
            eyePositionCorrected[frameIndex + frameOffset, columnIndices] = eyePositionUncorrected[frameIndex, columnIndices]
            frameIndex += 1

        #
        if frameIndex + frameOffset > terminationIndex:
            terminationIndex = frameIndex + frameOffset

        #
        print(f'Info: {frameOffset} dropped frames detected in the {camera} camera recording')

    #
    eyePositionCorrected = eyePositionCorrected[:terminationIndex, :]
    saveSessionData(sessionObject, 'eyePositionCorrected', eyePositionCorrected)

    return

# TODO: Remove the constant motion of the eye
def stabilizeEyePosition(sessionObject):
    """
    """

    return

# TODO: Normlize eye position to some common measurement (e.g., eye width)
def normalizeEyePosition(sessionObject):
    """
    """

    return

def decomposeEyePosition(sessionObject, nNeighbors=5):
    """
    """

    eyePositionCorrected = sessionObject.load('eyePositionCorrected')
    eyePositionImputed = KNNImputer(n_neighbors=nNeighbors).fit_transform(eyePositionCorrected)
    eyePositionDecomposed = np.full_like(eyePositionImputed, np.nan)
    for columnIndices, X in zip([(0, 1), (2, 3)], np.split(eyePositionImputed, 2, axis=1)):
        eyePositionDecomposed[:, columnIndices] = PCA(n_components=2).fit_transform(X)

    saveSessionData(sessionObject, 'eyePositionDecomposed', eyePositionDecomposed)

    return

# TODO: Reorient eye position such that the decomposed signal matches the raw eye position
def reorientEyePosition(sessionObject, reflect='left'):
    """
    """

    eyePositionDecomposed = sessionObject.load('eyePositionDecomposed')
    eyePositionCorrected = sessionObject.load('eyePositionCorrected')
    eyePositionReoriented = np.full_like(eyePositionCorrected, np.nan)

    #
    iterable = zip(['left', 'left', 'right', 'right'], np.arange(4))
    for eye, columnIndex in iterable:
        if eye == reflect:
            coefficient = -1
        else:
            coefficient = +1
        signal3 = np.copy(eyePositionDecomposed[:, columnIndex])
        signal1 = eyePositionCorrected[:, columnIndex]
        indices = np.where(np.isnan(signal1))[0]
        signal1 = np.delete(signal1, indices)
        signal2 = eyePositionDecomposed[:, columnIndex]
        signal2 = np.delete(signal2, indices)
        r2, p =  pearsonr(signal1, signal2)
        if r2 > 0.1 and p < 0.05:
            pass
        elif r2 < -0.1 and p < 0.05:
            signal3 *= -1
        else:
            raise Exception()
        signal3 *= coefficient
        eyePositionReoriented[:, columnIndex] = signal3

    #
    saveSessionData(sessionObject, 'eyePositionReoriented', eyePositionReoriented)

    return

def filterEyePosition(sessionObject, t=25):
    """
    Filter eye position

    keywords
    --------
    t : int
        Size of time window for smoothing (in ms)
    """
    
    # Determine the nearest odd window size
    smoothingWindowSize = round(t / (1 / sessionObject.fps * 1000))
    if smoothingWindowSize % 2 == 0:
        smoothingWindowSize += 1

    # Filter
    eyePositionDecomposed = sessionObject.load('eyePositionDecomposed')
    eyePositionFiltered = np.full_like(eyePositionDecomposed, np.nan)
    for columnIndex in range(eyePositionDecomposed.shape[1]):
        eyePositionFiltered[:, columnIndex] = smooth(eyePositionDecomposed[:, columnIndex], smoothingWindowSize)

    # Save filtered eye position data
    saveSessionData(sessionObject, 'eyePositionFiltered', eyePositionFiltered)

    return

def detectMonocularSaccades(sessionObject, p=0.99, isi=0.05):
    """
    """


    eyePositionFiltered = sessionObject.load('eyePositionFiltered')
    distance = round(isi * sessionObject.fps)
    for eye, columnIndex in zip(['left', 'right'], [0, 2]):
        for coefficient in (+1, -1):
            signal = np.diff(eyePositionFiltered[:, columnIndex]) * coefficient
            threshold = np.percentile(signal, p * 100)
            peakIndices, peakProperties = findPeaks(signal, height=threshold, distance=distance)

    
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