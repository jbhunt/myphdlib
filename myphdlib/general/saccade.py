import time
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.decomposition import PCA, IncrementalPCA
from sklearn.neural_network import MLPClassifier
from scipy.stats import pearsonr
from scipy.signal import find_peaks as findPeaks
from scipy.optimize import curve_fit as fitCurve
from myphdlib.general.curves import relu
from myphdlib.general.toolkit import smooth, resample, interpolate, detectThresholdCrossing, DotDict
from myphdlib.general.session import saveSessionData
from myphdlib.extensions.matplotlib import SaccadeLabelingGUI

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
        print(f'INFO[animal={sessionObject.animal}, date={sessionObject.date}]: {frameOffset} dropped frames detected in the {camera} camera recording')

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

def decomposeEyePosition(sessionObject, nNeighbors=5, benchmark=False):
    """
    """

    eyePositionCorrected = sessionObject.load('eyePositionCorrected')
    eyePositionDecomposed = np.full_like(eyePositionCorrected, np.nan)
    missingDataMask = {
        'left': np.full(eyePositionCorrected.shape[0], np.nan),
        'right': np.full(eyePositionCorrected.shape[0], np.nan)
    }

    #
    if benchmark:
        t1 = time.time()

    #
    for columnIndices, X1, side in zip([(0, 2), (2, 4)], np.split(eyePositionCorrected, 2, axis=1), ('left', 'right')):

        # Check for missing eye position data
        if np.isnan(X1).all(0).all():
            continue

        # Impute NaN values
        X2 = SimpleImputer(missing_values=np.nan).fit_transform(X1)

        #
        mask = np.invert(np.isnan(X1).any(1))
        missingDataMask[side] = np.invert(mask)

        #
        eyePositionDecomposed[:, columnIndices[0]: columnIndices[1]] = PCA(n_components=2).fit_transform(X2)

    #
    if benchmark:
        t2 = time.time()
        elapsed = round((t2 - t1) / 60, 2)
        print(f'INFO: Decomposition took {elapsed} minutes')

    saveSessionData(sessionObject, 'eyePositionDecomposed', eyePositionDecomposed)
    saveSessionData(sessionObject, 'missingDataMask', missingDataMask)

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
            continue
            raise Exception('Could not determine correlation between raw and decomposed eye position')
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

def determineSaccadeOnset(wave, threshold, nBack=0):
    """
    """

    result = False
    nSamples = wave.size
    if wave.size % 2 == 0:
        stopIndex = int(nSamples / 2) + 1
    else:
        raise Exception('Waveform must have even number of samples')
    crossingIndices = detectThresholdCrossing(np.diff(wave[:stopIndex]), threshold)
    if crossingIndices.size == 0:
        return result, np.nan
    targetIndex = np.argmin(np.abs(crossingIndices - stopIndex))
    crossingIndex = crossingIndices[targetIndex] - nBack

    return True, crossingIndex

# TODO: Record saccade onset timesetamps
def detectPutativeSaccades(
    session,
    percentile=0.992,
    minimumISI=0.1,
    perisaccadicWindow=(-0.2, 0.2),
    alignment='before',
    enforceMinimumISI=True,
    smoothingWindowSize=(0.025)
    ):
    """
    """

    #
    if 'saccadeWaveformsPutative' in session.keys:
        session.removeDataEntry('saccadeWaveformsPutative')

    #
    offsets = np.array([
        round(perisaccadicWindow[0] * session.fps),
        round(perisaccadicWindow[1] * session.fps)
    ])

    # N samples across saccades waveforms
    nFeatures = offsets[1] - offsets[0]

    # Smoothing window size (in samples)
    wlen = round(smoothingWindowSize * session.fps)
    if wlen % 2 == 0:
        wlen += 1

    #
    # TODO: Reorganize the data container
    eyePositionFiltered = session.load('eyePositionReoriented')
    saccadeDetectionResults = {
        'waveforms': {
            'left': list(),
            'right': list(),
        },
        'indices': {
            'left': list(),
            'right': list(),
        },
        'amplitudes': {
            'left': list(),
            'right': list()
        }
    }

    for eye, columnIndex in zip(['left', 'right'], [0, 2]):

        # Check for NaN values
        if np.isnan(eyePositionFiltered[:, columnIndex]).all():
            for feature in ('waveforms', 'indices', 'amplitude'):
                saccadeDetectionResults[feature][eye] = None
            continue
        
        #
        for coefficient in (+1, -1):

            #
            velocity = smooth(
                np.diff(eyePositionFiltered[:, columnIndex]) * coefficient,
                wlen
            )
            threshold = np.percentile(velocity, percentile * 100)
            peakIndices, peakProperties = findPeaks(velocity, height=threshold, distance=None)

            #
            for peakIndex in peakIndices:

                # Determine alignment
                if alignment == 'before':
                    pass
                elif alignment == 'after':
                    peakIndex += 1

                # Extract saccade waveform
                s1, s2 = offsets + peakIndex
                saccadeWaveform = eyePositionFiltered[s1:s2, columnIndex]

                # Exclude incomplete saccades
                if saccadeWaveform.size != nFeatures:
                    continue

                saccadeAmplitude = velocity[peakIndex]
                saccadeDetectionResults['waveforms'][eye].append(saccadeWaveform)
                saccadeDetectionResults['indices'][eye].append(peakIndex)
                saccadeDetectionResults['amplitudes'][eye].append(saccadeAmplitude)

    #
    for eye in ('left', 'right'):
        if saccadeDetectionResults['waveforms'][eye] is None:
            continue
        else:
            sortedIndex = np.argsort(saccadeDetectionResults['indices'][eye])
            for feature in ('waveforms', 'indices', 'amplitudes'):
                saccadeDetectionResults[feature][eye] = np.array(saccadeDetectionResults[feature][eye])[sortedIndex]

    # Filter out saccades that violate the minimum ISI
    if enforceMinimumISI:
        for eye in ('left', 'right'):
            print(f'INFO[animal={session.animal}, date={session.date}]: Filtering saccades for the {eye} eye (n=?)', end='\r')
            while True:
                nSaccades = saccadeDetectionResults['waveforms'][eye].shape[0]
                interSaccadeIntervals = np.diff(saccadeDetectionResults['indices'][eye]) / session.fps
                if interSaccadeIntervals.min() >= minimumISI:
                    break
                print(f'INFO[animal={session.animal}, date={session.date}]: Filtering saccades for the {eye} eye (n={nSaccades})', end='\r')
                saccadeIndex = np.min(np.where(interSaccadeIntervals < minimumISI)[0])
                sampleIndex = saccadeDetectionResults['indices'][eye][saccadeIndex]
                a1 = saccadeDetectionResults['amplitudes'][eye][saccadeIndex]
                a2 = saccadeDetectionResults['amplitudes'][eye][saccadeIndex + 1]
                targetIndex = np.array([saccadeIndex, saccadeIndex + 1])[np.argmin([a1, a2])]
                for feature in ('waveforms', 'indices', 'amplitudes'):
                    saccadeDetectionResults[feature][eye] = np.delete(saccadeDetectionResults[feature][eye], targetIndex, axis=0)
            print(f'INFO[animal={session.animal}, date={session.date}]: Filtering saccades for the {eye} eye (n={nSaccades})')

    # Print INFO
    for eye in ('left', 'right'):
        if saccadeDetectionResults['waveforms'][eye] is None:
            continue
        else:
            nSaccades = saccadeDetectionResults['waveforms'][eye].shape[0]
            # print(f'INFO[animal={session.animal}, date={session.animal}]: {nSaccades} putative saccades detected in the {eye} eye')

    # Save results
    session.save('saccadeDetectionResults', saccadeDetectionResults)
    
    return

# TODO: Exclude duplicate labeled saccades
# TODO: Implement a method for NOT overwriting previously collected saccades waveforms
def labelPutativeSaccades(sessionObject, nSamplesPerEye=30):
    """
    """

    saccadeWaveformsPutative = sessionObject.load('saccadeDetectionResults')['waveforms']
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
            if saccadeWaveformsLabeled[eye]['X'] is None:
                continue
            samples = np.diff(saccadeWaveformsLabeled[eye]['X'], axis=1)
            labels = saccadeWaveformsLabeled[eye]['y']
            mask = np.invert(np.isnan(labels)).flatten()
            for sample, label in zip(samples[mask, :], labels[mask, :]):
                X.append(sample)
                y.append(label)

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    # Init and fit the classifier
    # TODO: grid search CV
    clf = classifierClass(**classifierKwargs).fit(X, y.ravel())

    #
    for session in factoryObject:

        #
        if 'saccadeWaveformsClassified' in session.keys:
            session.removeDataEntry('saccadeWaveformsClassified')
        
        # Init the empty data container
        saccadeClassificationResults = {
            'left': {
                'nasal': {
                    'indices': list(),
                    'waveforms': list()
                },
                'temporal': {
                    'indices': list(),
                    'waveforms': list()
                }
            },
            'right': {
                'nasal': {
                    'indices': list(),
                    'waveforms': list()
                },
                'temporal': {
                    'indices': list(),
                    'waveforms': list()
                }
            },
        }

        # Load the saccade detection results
        saccadeDetectionResults = session.load('saccadeDetectionResults')

        # Classify saccades
        for eye in ('left', 'right'):
            
            #
            if saccadeDetectionResults['waveforms'][eye] is None:
                continue
            samples = np.diff(saccadeDetectionResults['waveforms'][eye], axis=1)
            labels = clf.predict(samples)
            directions = list()
            for label in labels.flatten():
                if label == -1:
                    directions.append('temporal')
                elif label == 1:
                    directions.append('nasal')
                else:
                    directions.append(None)

            # Parse the putative saccades into classes
            for sampleIndex, (sample, direction) in enumerate(zip(samples, directions)):
                if direction is None:
                    continue
                saccadeOnsetIndex = saccadeDetectionResults['indices'][eye][sampleIndex]
                saccadeClassificationResults[eye][direction]['indices'].append(saccadeOnsetIndex)
                saccadeWaveform = saccadeDetectionResults['waveforms'][eye][sampleIndex, :]
                saccadeClassificationResults[eye][direction]['waveforms'].append(saccadeWaveform)
    
        # Save the results
        for eye in ('left', 'right'):
            for direction in ('nasal', 'temporal'):
                if len(saccadeClassificationResults[eye][direction]['indices']) == 0:
                    saccadeClassificationResults[eye][direction]['indices'] = None
                    saccadeClassificationResults[eye][direction]['waveforms'] = None
                else:
                    saccadeClassificationResults[eye][direction]['indices'] = np.array(saccadeClassificationResults[eye][direction]['indices'])
                    saccadeClassificationResults[eye][direction]['waveforms'] = np.array(saccadeClassificationResults[eye][direction]['waveforms'])
                    nSaccades = saccadeClassificationResults[eye][direction]['indices'].size
                    print(f'INFO[animal={session.animal}, date={session.date}]: {nSaccades} {direction} saccades classified for the {eye} eye')

        #
        session.save('saccadeClassificationResults', saccadeClassificationResults)

    return

# TODO: Code this
def detectConjugateSaccades(sessionObject):
    """
    """

    return

def determineSaccadeOnset(session, deviations=1, tolerance=0.025, baseline=0.1):
    """
    """

    saccadeClassificationResults = session.load('saccadeClassificationResults')

    for eye in ('left', 'right'):
        for direction in ('temporal', 'nasal'):

            #
            nSaccades = saccadeClassificationResults[eye][direction]['indices'].size
            saccadeClassificationResults[eye][direction]['indices2'] = np.full(nSaccades, 0, dtype=np.int64)

            #
            saccadeIndices = saccadeClassificationResults[eye][direction]['indices']
            saccadeWaveforms = saccadeClassificationResults[eye][direction]['waveforms']

            #
            mu = saccadeWaveforms[:, 0: int(np.ceil(baseline * session.fps))].flatten().mean()
            sigma = saccadeWaveforms[:, 0: int(np.ceil(baseline * session.fps))].flatten().std()
            threshold = mu + sigma * deviations

            #
            error = list()

            #
            for sampleIndex, (saccadeIndex, saccadeWaveform) in enumerate(zip(saccadeIndices, saccadeWaveforms)):

                # Get the frame index for the very first sample in the saccade waveform
                if saccadeWaveform.size % 2 != 0:
                    raise Exception('Saccade waveform must be an even number of samples')
                nFeatures = int(saccadeWaveform.size / 2)
                i0 = saccadeIndex - nFeatures

                # Determine the index to start from moving backwards
                velocity = np.diff(saccadeWaveform)
                i1 = round(((velocity.size - 1) / 2) - (tolerance * session.fps))
                i2 = round(((velocity.size - 1) / 2) + (tolerance * session.fps))
                startIndex = np.argmax(velocity[i1: i2]) + i1

                # Find the first threshold crossing
                crossingIndex = saccadeIndex
                for i in np.arange(0, startIndex)[::-1]:
                    vi = velocity[i]
                    if vi < threshold:
                        crossingIndex = int(i - 1 + i0)
                        dt = (crossingIndex - saccadeIndex) / session.fps
                        error.append(dt)
                        break

                #
                saccadeClassificationResults[eye][direction]['indices2'][sampleIndex] = crossingIndex

    #
    session.save('saccadeClassificationResults', saccadeClassificationResults)

    #
    error = np.array(error)
    print(f'INFO[animal={session.animal}, date={session.date}]: Mean saccade onset correction = {error.mean():.3f} seconds')

    return

def computeSaccadeTimestamps(session):
    """
    Compute the saccade onset timestamps
    """

    if 'timestampGeneratorParameters' not in session.keys:
        raise Exception(f'Session for {session.animal} on {session.date} has no timestamp generator')

    #
    saccadeOnsetTimestamps = {
        'left': {
            'nasal': None,
            'temporal': None
        },
        'right': {
            'nasal': None,
            'temporal': None
        }
    }

    # Get the indices for the sample after the beginning of the rising edges
    peakIndices, peakProps = findPeaks(
        np.abs(np.diff(session.load('exposureOnsetSignal'))),
        height=0.5,
    )

    #
    params = DotDict(session.load('timestampGeneratorParameters'))

    #
    saccadeClassificationResults = session.load('saccadeClassificationResults')
    for eye in ('left', 'right'):
        for direction in ('nasal', 'temporal'):
            saccadeOnsetIndices = saccadeClassificationResults[eye][direction]['indices2']
            sampleIndices = peakIndices[saccadeOnsetIndices]
            
            #
            timestamps = np.around(
                np.interp(sampleIndices, params.xp, params.fp) * params.m + params.b,
                3
            )
            saccadeOnsetTimestamps[eye][direction] = timestamps

    #
    session.save('saccadeOnsetTimestamps', saccadeOnsetTimestamps)

    return

def runAllModules(sessionObject):
    """
    """

    modules = (
        extractEyePosition,
        correctEyePosition,
        decomposeEyePosition,
        reorientEyePosition,
        detectPutativeSaccades,
    )

    for module in modules:
        module(sessionObject)

    return