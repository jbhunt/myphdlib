import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from scipy.stats import pearsonr
from scipy.signal import find_peaks as findPeaks
from myphdlib.general.toolkit import smooth, resample
from myphdlib.general.session import saveSessionData
from myphdlib.extensions.matplotlib import SaccadeLabelingGUI

def extractEyePosition(sessionObject, likelihoodThreshold=0.95, pupilCenterName='pupil-c'):
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
    eyePositionDecomposed = np.full_like(eyePositionCorrected, np.nan)

    #
    for columnIndices, X1 in zip([(0, 1), (2, 3)], np.split(eyePositionCorrected, 2, axis=1)):

        # Check for missing eye position data
        if np.isnan(X1).all(0).all():
            continue

        # Impute NaN values
        X2 = KNNImputer(n_neighbors=nNeighbors).fit_transform(X1)

        #
        eyePositionDecomposed[:, columnIndices] = PCA(n_components=2).fit_transform(X2)

    saveSessionData(sessionObject, 'eyePositionDecomposed', eyePositionDecomposed)

    return

def reorientEyePosition(sessionObject, reflect='left'):
    """
    """

    eyePositionDecomposed = sessionObject.load('eyePositionDecomposed')
    eyePositionCorrected = sessionObject.load('eyePositionCorrected')
    eyePositionReoriented = np.full_like(eyePositionCorrected, np.nan)

    #
    iterable = zip(['left', 'left', 'right', 'right'], np.arange(4))
    for eye, columnIndex in iterable:
        #
        if np.isnan(eyePositionDecomposed[:, columnIndex]).all():
            continue
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
        eyePositionReoriented[:, columnIndex] = signal3

    # TODO: Check that left and right eye position is anti-correlated

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
        if np.isnan(eyePositionDecomposed[:, columnIndex]).all():
            continue
        eyePositionFiltered[:, columnIndex] = smooth(eyePositionDecomposed[:, columnIndex], smoothingWindowSize)

    # Save filtered eye position data
    saveSessionData(sessionObject, 'eyePositionFiltered', eyePositionFiltered)

    return

# TODO: Align saccades to the onset of the saccade
# TODO: Record the saccade onset timestamp
def detectPutativeSaccades(
    sessionObject,
    p=0.992,
    isi=0.015,
    window=(-0.2, 0.2),
    alignment='before'
    ):
    """
    """

    #
    offset = (
        round(window[0] * sessionObject.fps),
        round(window[1] * sessionObject.fps)
    )

    #
    distance = round(isi * sessionObject.fps)
    eyePositionFiltered = sessionObject.load('eyePositionFiltered')
    saccadeWaveformsPutative = {
        'left': list(),
        'right': list()
    }
    for eye, columnIndex in zip(['left', 'right'], [0, 2]):

        # Check for NaN values
        if np.isnan(eyePositionFiltered[:, columnIndex]).all():
            saccadeWaveformsPutative[eye] = None
            continue
        
        #
        for coefficient in (+1, -1):
            velocity = np.diff(eyePositionFiltered[:, columnIndex]) * coefficient
            threshold = np.percentile(velocity, p * 100)
            peakIndices, peakProperties = findPeaks(velocity, height=threshold, distance=distance)
            for peakIndex in peakIndices:
                if alignment == 'before':
                    pass
                elif alignment == 'after':
                    peakIndex += 1
                s1 = peakIndex + offset[0]
                s2 = peakIndex + offset[1]
                saccadeWaveform = eyePositionFiltered[s1:s2, columnIndex]
                saccadeWaveformsPutative[eye].append(saccadeWaveform)

    #
    for eye in ('left', 'right'):
        saccadeWaveformsPutative[eye] = np.array(saccadeWaveformsPutative[eye])
    saveSessionData(sessionObject, 'saccadeWaveformsPutative', saccadeWaveformsPutative)
    
    return

# TODO: Exclude duplicate labeled saccades
def labelPutativeSaccades(sessionObject, nSamplesPerEye=30):
    """
    """

    saccadeWaveformsPutative = sessionObject.load('saccadeWaveformsPutative')
    saccadeWaveformsLabeled = {
        'left': {
            'X': None,
            'y': None,
        },
        'right': {
            'X': None,
            'y': None,
        }
    }

    #
    samples = list()
    for eye in ('left', 'right'):
        if saccadeWaveformsPutative[eye] is None:
            continue
        nSamples = saccadeWaveformsPutative[eye].shape[0]
        sampleIndices = np.random.choice(np.arange(nSamples), nSamplesPerEye)
        for sample in saccadeWaveformsPutative[eye][sampleIndices, :]:
            samples.append(sample)

    #
    gui = SaccadeLabelingGUI()
    gui.inputSamples(np.array(samples))
    while gui.isRunning():
        continue

    #
    xLeftEye, xRightEye = np.split(gui.xTrain, 2, axis=0)
    yLeftEye, yRightEye = np.split(gui.y, 2, axis=0)
    saccadeWaveformsLabeled['left']['X'] = xLeftEye
    saccadeWaveformsLabeled['right']['X'] = xRightEye
    saccadeWaveformsLabeled['left']['y'] = yLeftEye
    saccadeWaveformsLabeled['right']['y'] = yRightEye

    #
    saveSessionData(sessionObject, 'saccadeWaveformsLabeled', saccadeWaveformsLabeled)

    return

def classifyPutativeSaccades(factoryObject, classifierClass=MLPClassifier, **classifierKwargs):
    """
    """

    classifierKwargsDefaults = {
        'max_iter': 1000000
    }
    classifierKwargs.update(classifierKwargsDefaults)

    # Collect the training dataset
    X = list()
    y = list()
    for sessionObject in factoryObject:
        saccadeWaveformsLabeled = sessionObject.load('saccadeWaveformsLabeled')
        for eye in ('left', 'right'):
            samples = np.diff(saccadeWaveformsLabeled[eye]['X'], axis=1)
            if samples is None:
                continue
            labels = saccadeWaveformsLabeled[eye]['y']
            for sample, label in zip(samples, labels):
                X.append(sample)
                y.append(label)

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    # Init and fit the classifier
    clf = classifierClass(**classifierKwargs).fit(X, y.ravel())

    #
    for sessionObject in factoryObject:
        
        # Init the empty data container
        saccadeWaveformsClassified = {
            'left': {
                'nasal': list(),
                'temporal': list() 
            },
            'right': {
                'nasal': list(),
                'temporal': list()
            }
            
        }

        # Classify saccades
        for eye in ('left', 'right'):
            saccadeWaveformsPutative = sessionObject.load('saccadeWaveformsPutative')
            samples = np.diff(saccadeWaveformsPutative[eye], axis=1)
            if samples is None:
                continue
            labels = clf.predict(samples)

            for sampleIndex, (sample, label) in enumerate(zip(samples, labels)):
                waveform = saccadeWaveformsPutative[eye][sampleIndex, :]
                if label == -1:
                    saccadeWaveformsClassified[eye]['temporal'].append(waveform)
                elif label == +1:
                    saccadeWaveformsClassified[eye]['nasal'].append(waveform)
    
        # Save the results
        for eye in ('left', 'right'):
            for direction in ('nasal', 'temporal'):
                saccadeWaveformsClassified[eye][direction] = np.array(saccadeWaveformsClassified[eye][direction])
        saveSessionData(sessionObject, 'saccadeWaveformsClassified', saccadeWaveformsClassified)

    return

# TODO: Code this
def detectConjugateSaccades(sessionObject):
    """
    """

    return

def runAllModules(sessionObject):
    """
    """

    modules = (
        extractEyePosition,
        correctEyePosition,
        decomposeEyePosition,
        reorientEyePosition,
        filterEyePosition,
        detectPutativeSaccades,
    )

    for module in modules:
        module(sessionObject)

    return