import time
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import pearsonr
from scipy.signal import find_peaks as findPeaks
from scipy.optimize import curve_fit as fitCurve
from myphdlib.general.curves import relu
from myphdlib.general.toolkit import smooth, resample, interpolate, detectThresholdCrossing, DotDict
from myphdlib.general.session import saveSessionData
from myphdlib.extensions.matplotlib import SaccadeLabelingGUI

def extractEyePosition(
    session,
    likelihoodThreshold=0.99,
    pupilCenterName='pupilCenter',
    ):
    """
    Extract the raw eye position data
    """

    #
    nSamples = 0
    eyePositionLeft = None
    eyePositionRight = None

    #
    for side in ('left', 'right'):
        if side == 'left':
            csv = session.leftEyePose
        elif side == 'right':
            csv = session.rightEyePose
        if csv is not None:
            frame = pd.read_csv(csv, header=list(range(3)), index_col=0).sort_index(level=1, axis=1)
        else:
            continue
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
    # session.write(eyePositionUncorrected, 'eyePositionUncorrected')
    session.save('pose/uncorrected', eyePositionUncorrected)

    return

def correctEyePosition(session, pad=1e6):
    """
    Correct eye position data for missing/dropped frames
    """

    #
    # eyePositionUncorrected = session.eyePositionUncorrected
    eyePositionUncorrected = session.load('pose/uncorrected')
    nSamples = eyePositionUncorrected.shape[0] + int(pad)
    eyePositionCorrected = np.full([nSamples, 4], np.nan)

    #
    terminationIndex = 0
    expectedFrameInterval = 1 / session.fps * 1000 # In ms
    for camera in ('left', 'right'):

        #
        if camera == 'left':
            columnIndices = 0, 1
            frameIntervals = np.loadtxt(session.leftCameraTimestamps, dtype=np.int64) / 1000000 # In ms
        else:
            columnIndices = 2, 3
            frameIntervals = np.loadtxt(session.rightCameraTimestamps, dtype=np.int64) / 1000000 # In ms

        # TODO: Figure out if I should save these data
        # frameTimestamps = np.full(frameIntervals.size + 1, np.nan)
        # frameTimestamps[0] = 0
        # frameTimestamps[1:] = np.cumsum(frameIntervals)

        #
        frameIndex = 0
        frameOffset = 0
        missingFrames = 0
        eyePositionUncorrected[0, columnIndices] = eyePositionUncorrected[0, columnIndices]
        for frameInterval in frameIntervals:
            frameOffset += round(frameInterval / expectedFrameInterval) - 1
            if frameIndex >= eyePositionUncorrected.shape[0]:
                missingFrames += 1
            else:
                eyePositionCorrected[frameIndex + frameOffset, columnIndices] = eyePositionUncorrected[frameIndex, columnIndices]
            frameIndex += 1

        #
        if frameIndex + frameOffset > terminationIndex:
            terminationIndex = frameIndex + frameOffset

        #
        print(f'INFO[animal={session.animal}, date={session.date}]: {frameOffset} dropped frames detected in the {camera} camera recording')
        print(f'INFO[animal={session.animal}, date={session.date}]: {missingFrames} missing frames detected in the {camera} camera recording')

    #
    eyePositionCorrected = eyePositionCorrected[:terminationIndex, :]
    # session.write(eyePositionCorrected, 'eyePositionCorrected')
    session.save('pose/corrected', eyePositionCorrected)

    return

def interpolateEyePosition(
    session,
    maximumConsecutiveDroppedFrames=4
    ):
    """
    """

    eyePositionCorrected = session.load('pose/corrected')
    eyePositionInterpolated = np.copy(eyePositionCorrected)
    
    #
    for iColumn in range(eyePositionCorrected.shape[1]):

        #
        pose = eyePositionCorrected[:, iColumn]
        dropped = np.isnan(pose)
        windows = list()

        #
        iRow = 0
        while True:
            if iRow >= dropped.size:
                break
            flag = dropped[iRow]

            # Figure out how many frames were dropped
            if flag:
                nDroppedFrames = 0
                for flag_ in dropped[iRow:]:
                    if flag_ == False:
                        break
                    nDroppedFrames += 1

                #
                if nDroppedFrames <= maximumConsecutiveDroppedFrames:
                    if iRow + nDroppedFrames + 1 >= pose.size:
                        iRow += nDroppedFrames
                        continue
                    windows.append([
                        iRow - 1,
                        iRow + nDroppedFrames + 1
                    ])
                iRow += nDroppedFrames

            # Increment the counter
            else:
                iRow += 1

        # Interpolate over the windows of dropped frames
        for start, stop in windows:
            x = np.arange(start + 1, stop - 1, 1)
            xp = np.array([start, stop - 1])
            fp = np.array([pose[start], pose[stop - 1]])
            y = np.interp(x, xp, fp)
            eyePositionInterpolated[start + 1: stop - 1, iColumn] = y

    #
    session.save('pose/interpolated', eyePositionInterpolated)

    return

# TODO: Remove the constant motion of the eye
def stabilizeEyePosition(session):
    """
    """

    return

# TODO: Normlize eye position to some common measurement (e.g., eye width)
def normalizeEyePosition(session):
    """
    """

    return

def decomposeEyePosition(session, nNeighbors=5, benchmark=False):
    """
    """

    # eyePositionCorrected = session.eyePositionCorrected
    eyePositionInterpolated = session.load('pose/interpolated')
    eyePositionDecomposed = np.full_like(eyePositionInterpolated, np.nan)
    missingDataMask = {
        'left': np.full(eyePositionInterpolated.shape[0], np.nan),
        'right': np.full(eyePositionInterpolated.shape[0], np.nan)
    }

    #
    if benchmark:
        t1 = time.time()

    #
    for columnIndices, X1, side in zip([(0, 2), (2, 4)], np.split(eyePositionInterpolated, 2, axis=1), ('left', 'right')):

        # Check for missing eye position data
        if np.isnan(X1).all(0).all():
            missingDataMask[side] = np.full(eyePositionInterpolated.shape[0], True)
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

    #
    eyePositionDecomposed[missingDataMask['left'], 0:2] = np.nan
    eyePositionDecomposed[missingDataMask['right'], 2:4] = np.nan
    # session.write(eyePositionDecomposed, 'eyePositionDecomposed')
    # session.write(missingDataMask, 'missingDataMask')
    session.save('pose/decomposed', eyePositionDecomposed)
    session.save('pose/missing/left', missingDataMask['left'])
    session.save('pose/missing/right', missingDataMask['right'])

    return

def reorientEyePosition(session, reflect='left'):
    """
    """

    # eyePositionDecomposed = session.eyePositionDecomposed
    eyePositionDecomposed = session.load('pose/decomposed')
    # eyePositionCorrected = session.eyePositionCorrected
    eyePositionCorrected = session.load('pose/corrected')
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
        if r2 > 0.05 and p < 0.05:
            pass
        elif r2 < -0.05 and p < 0.05:
            signal3 *= -1
        else:
            raise Exception('Could not determine correlation between raw and decomposed eye position')
        eyePositionReoriented[:, columnIndex] = signal3

    # TODO: Check that left and right eye position is anti-correlated

    #
    # session.write(eyePositionReoriented, 'eyePositionReoriented')
    session.save('pose/reoriented', eyePositionReoriented)

    return

def filterEyePosition(session, t=25):
    """
    Filter eye position

    keywords
    --------
    t : int
        Size of time window for smoothing (in ms)
    """

    #
    # missingDataMask = session.missingDataMask
    missingDataMask = {
        'left': session.load('pose/missing/left'),
        'right': session.load('pose/missing/right')
    }
    
    # Determine the nearest odd window size
    smoothingWindowSize = round(t / (1 / session.fps * 1000))
    if smoothingWindowSize % 2 == 0:
        smoothingWindowSize += 1

    # Filter
    # eyePositionReoriented = session.eyePositionReoriented
    eyePositionReoriented = session.load('pose/reoriented')
    eyePositionFiltered = np.full_like(eyePositionReoriented, np.nan)
    for columnIndex in range(eyePositionReoriented.shape[1]):

        #
        if np.isnan(eyePositionReoriented[:, columnIndex]).all():
            continue

        # Interpolate missing values
        interpolated = interpolate(eyePositionReoriented[:, columnIndex])

        #
        smoothed = smooth(interpolated, smoothingWindowSize)

        #
        if columnIndex in (0, 1):
            smoothed[missingDataMask['left']] = np.nan
        else:
            smoothed[missingDataMask['right']] = np.nan

        #
        eyePositionFiltered[:, columnIndex] = smoothed

    # Save filtered eye position data
    # session.write(eyePositionFiltered, 'eyePositionFiltered')
    session.save('pose/filtered', eyePositionFiltered)

    return

# TODO: Record saccade onset timesetamps
def detectPutativeSaccades(
    session,
    percentile=0.992,
    minimumISI=0.05,
    perisaccadicWindow=(-0.2, 0.2),
    alignment='before',
    enforceMinimumISI=True,
    smoothingWindowSize=(0.025)
    ):
    """
    """

    #
    # if 'saccadeWaveformsPutative' in session.keys():
    #     session.removeDataEntry('saccadeWaveformsPutative')

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
    # eyePositionFiltered = session.read('eyePositionFiltered')
    eyePositionFiltered = session.load('pose/filtered')
    eyePositionImputed = np.full_like(eyePositionFiltered, np.nan)
    for iColumn, column in enumerate(eyePositionFiltered.T):

        # Skip over columns that are entirely NaNs
        if np.isnan(column).all():
            eyePositionImputed[:, iColumn] = column
            continue

        # Impute missing data
        eyePositionImputed[:, iColumn] = np.interp(
            np.arange(column.size),
            np.arange(column.size)[np.isfinite(column)],
            column[np.isfinite(column)]
        )
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
            print(f'WARNING: Eye position data missing for the {eye} eye')
            for feature in ('waveforms', 'indices', 'amplitudes'):
                saccadeDetectionResults[feature][eye] = None
            continue
        
        #
        for coefficient in (+1, -1):

            #
            velocity = smooth(
                np.diff(eyePositionImputed[:, columnIndex]) * coefficient,
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

            # Skip for missing data
            if saccadeDetectionResults['waveforms'][eye] is None:
                continue

            #
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

    # Print the results of saccade detection
    for eye in ('left', 'right'):
        if saccadeDetectionResults['waveforms'][eye] is None:
            continue
        else:
            nSaccades = saccadeDetectionResults['waveforms'][eye].shape[0]
            # print(f'INFO[animal={session.animal}, date={session.animal}]: {nSaccades} putative saccades detected in the {eye} eye')

    # Save results
    # session.write(saccadeDetectionResults, 'saccadeDetectionResults')
    for feature in saccadeDetectionResults.keys():
        for eye in ('left', 'right'):
            if saccadeDetectionResults[feature][eye] is None:
                continue
            path = f'saccades/putative/{eye}/{feature}'
            dataset = saccadeDetectionResults[feature][eye]
            if dataset is None:
                print(f'WARNING: No saccades detected for the {eye} eye')
                continue
            session.save(path, dataset)
    
    return

# TODO: Exclude duplicate labeled saccades
# TODO: Implement a method for NOT overwriting previously collected saccades waveforms

def labelPutativeSaccades(session, nSamplesPerEye=5):
    """
    """

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
        saccadeWaveformsPutative = session.load(f'saccades/putative/{eye}/waveforms')
        if saccadeWaveformsPutative is None:
            continue
        nSamples = saccadeWaveformsPutative.shape[0]
        sampleIndices = np.random.choice(np.arange(nSamples), nSamplesPerEye)
        for sample in saccadeWaveformsPutative[sampleIndices, :]:
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
    # session.write(saccadeWaveformsLabeled, 'saccadeWaveformsLabeled')
    for eye in ('left', 'right'):
        for key in ('X', 'y'):
            path = f'saccades/training/{eye}/{key}'
            value = saccadeWaveformsLabeled[eye][key]
            session.save(path, value)

    return

def _trainSaccadeClassifier(
    sessions,
    classifier='net',
    decompose=False,
    features='position'
    ):
    """
    """

    #
    pca = PCA(n_components=2)
    xTrain = list()
    xTest = list()
    yTrain = list()

    #
    for session in sessions:

        #
        saccadesExtracted = False
        for eye in ('left', 'right'):
            path = f'saccades/putative/{eye}/waveforms'
            if session.hasGroup(path):
                saccadesExtracted = True
                break

        # if 'saccadeDetectionResults' in session.keys():
        if saccadesExtracted:
            # saccadeDetectionResults = session.read('saccadeDetectionResults')
            for eye in ('left', 'right'):
                # saccadeWaveformsUnlabeled = saccadeDetectionResults['waveforms'][eye]
                saccadeWaveformsUnlabeled = session.load(f'saccades/putative/{eye}/waveforms')
                for sample in saccadeWaveformsUnlabeled:
                    if np.isnan(sample).any():
                        continue
                    if features == 'position':
                        xTest.append(sample)
                    elif features == 'velocity':
                        xTest.append(np.diff(sample))
        
        #
        saccadesLabeled = False
        for eye in ('left', 'right'):
            path = f'saccades/training/{eye}/X'
            if session.hasGroup(path):
                saccadesLabeled = True
                break

        if saccadesLabeled:
            # saccadeLabelingResults = session.read('saccadeWaveformsLabeled')
            for eye in ('left', 'right'):
                # saccadeWaveformsLabeled = saccadeLabelingResults[eye]
                X_ = session.load(f'saccades/training/{eye}/X')
                y_ = session.load(f'saccades/training/{eye}/y')
                iterable = zip(X_, y_)
                for sample, label in iterable:
                    if np.isnan(sample).any():
                        continue
                    if features == 'position':
                        xTrain.append(sample)
                    elif features == 'velocity':
                        xTrain.append(np.diff(sample))
                    yTrain.append(label.item())

    #
    xTrain = np.array(xTrain)
    xTest = np.array(xTest)
    yTrain = np.array(yTrain)
    nSamples = xTrain.shape[0]
    print(f'INFO[X]: {nSamples} samples collected for model training')

    # Decompose
    if decompose:
        pca.fit(np.vstack([xTrain, xTest]))
        xTrainDecomposed = pca.transform(xTrain)
        X = xTrainDecomposed
    else:
        X = xTrain

    # Fit
    if classifier == 'net':
        grid = {
            'hidden_layer_sizes': [
                (10 , 10 , 10 ),
                (100, 100, 100),
                (200, 200, 200),
                (300, 300, 300)
            ],
            'max_iter': [
                1000000,
            ]
        }
        net = MLPClassifier()
        search = GridSearchCV(net, grid)
        search.fit(X, yTrain.ravel())
        clf = search.best_estimator_
    
    #
    elif classifier == 'lda':
        lda = LinearDiscriminantAnalysis()
        lda.fit(X, yTrain.ravel())
        clf = lda

    return pca, clf, xTrain, yTrain

def classifyPutativeSaccades(
    sessionsToLabel,
    sessionsToTrainOn,
    classifier='net',
    decompose=False,
    features='velocity'
    ):
    """
    """

    #
    pca, clf, xTrain, yTrain = _trainSaccadeClassifier(
        sessionsToTrainOn,
        classifier,
        decompose,
        features
    )

    #
    for session in sessionsToLabel:
        
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

        # Classify saccades
        for eye in ('left', 'right'):

            #
            samples = list()
            waveforms = list()
            indices = list()
            saccadeWaveformsUnlabeled = session.load(f'saccades/putative/{eye}/waveforms')
            saccadeIndicesUnlabeled = session.load(f'saccades/putative/{eye}/indices')
            for index, waveform in enumerate(saccadeWaveformsUnlabeled):
                if np.isnan(waveform).any():
                    continue
                if features == 'position':
                    samples.append(waveform)
                elif features == 'velocity':
                    samples.append(np.diff(waveform))
                waveforms.append(waveform)
                indices.append(saccadeIndicesUnlabeled[index])
            
            #
            if decompose:
                samples = pca.transform(
                    np.array(samples)
                )
            else:
                samples = np.array(samples)
            if samples.size == 0:
                print(f'INFO[{session.animal}, {session.date}]: 0 {direction} saccades classified for the {eye} eye')
                continue
            else:
                labels = clf.predict(samples)

            #
            iterable = zip(
                waveforms,
                labels,
                indices
            )
            for waveform, label, index in iterable:
                if label == 0:
                    continue
                elif label == 1:
                    direction = 'nasal'
                elif label == -1:
                    direction = 'temporal'
                saccadeClassificationResults[eye][direction]['waveforms'].append(waveform)
                saccadeClassificationResults[eye][direction]['indices'].append(index) 
    
        # Save the results
        for eye in ('left', 'right'):
            for direction in ('nasal', 'temporal'):
                if len(saccadeClassificationResults[eye][direction]['indices']) == 0:
                    saccadeClassificationResults[eye][direction]['indices'] = None
                    saccadeClassificationResults[eye][direction]['waveforms'] = None
                    print(f'INFO[{session.animal}, {session.date}]: 0 {direction} saccades classified for the {eye} eye')
                else:
                    saccadeClassificationResults[eye][direction]['indices'] = np.array(saccadeClassificationResults[eye][direction]['indices'])
                    saccadeClassificationResults[eye][direction]['waveforms'] = np.array(saccadeClassificationResults[eye][direction]['waveforms'])
                    nSaccades = saccadeClassificationResults[eye][direction]['indices'].size
                    print(f'INFO[{session.animal}, {session.date}]: {nSaccades} {direction} saccades classified for the {eye} eye')

        #
        for eye in ('left', 'right'):
            for direction in ('nasal', 'temporal'):
                for feature in ('waveforms', 'indices'):
                    if feature == 'waveforms':
                        path = f'saccades/predicted/{eye}/{direction}/{feature}'
                    else:
                        path = f'saccades/predicted/{eye}/{direction}/indices/uncorrected'
                    dataset = saccadeClassificationResults[eye][direction][feature]
                    if dataset is None:
                        continue
                    session.save(path, dataset)
                    

    return

# TODO: Make this better; need a way of determining onset on a trial by trial basis
def determineSaccadeOnset(session, threshold=0.4, baseline=(0, 0.1), maximumCorrectionInFrames=10):
    """
    """

    for eye in ('left', 'right'):
        for direction in ('temporal', 'nasal'):

            #
            saccadeWaveforms = session.load(f'saccades/predicted/{eye}/{direction}/waveforms')
            if saccadeWaveforms is None:
                print(f'WARNING[{session.animal}, {session.date}]: No {direction} saccades detected in the {eye} eye')
                continue
            saccadeIndicesUncorrected = session.load(f'saccades/predicted/{eye}/{direction}/indices/uncorrected')
            saccadeIndicesCorrected = np.full_like(saccadeIndicesUncorrected, 0)

            #
            # mu = saccadeWaveforms[:, 0: int(np.ceil(baseline * session.fps))].flatten().mean()
            # sigma = saccadeWaveforms[:, 0: int(np.ceil(baseline * session.fps))].flatten().std()
            # threshold = mu + sigma * deviations

            #
            baselineWindowInFrames = (
                int(round(session.fps * baseline[0])),
                int(round(session.fps * baseline[1]))
            )
            baselineVelocitySample = np.abs(
                np.diff(saccadeWaveforms[baselineWindowInFrames[0]: baselineWindowInFrames[1]], axis=1).flatten()
            )

            #
            error = list()

            #
            pvalues = list()
            nPoints = int(saccadeWaveforms.shape[1] / 2)
            for saccadeWaveform in saccadeWaveforms[:, :nPoints]:
                row = list()
                for vi in np.diff(saccadeWaveform)[::-1]:
                    p = np.sum(baselineVelocitySample > abs(vi)) / baselineVelocitySample.size
                    row.append(p)
                pvalues.append(row)
            signal = np.mean(np.array(pvalues), axis=0)
            
            #
            frameOffset = 0
            for iFrame in range(signal.size):
                frameOffset -= 1
                if signal[iFrame] >= threshold:
                    break
            error.append(frameOffset / session.fps)

            #
            saccadeIndicesCorrected = saccadeIndicesUncorrected + frameOffset
            session.save(f'saccades/predicted/{eye}/{direction}/indices/adjusted', saccadeIndicesCorrected)

            # Figure out the timestamp for the frame of the saccade onset
            M = session.load('labjack/matrix')
            signal = M[:, session.labjackChannelMapping['cameras']]
            # frameTimestamps = session.computeTimestamps(np.where(np.abs(np.diff(signal)) > 0.5)[0])
            frameTimestamps = session.load('labjack/cameras/timestamps')
            saccadeOnsetTimestamps = frameTimestamps[saccadeIndicesCorrected]
            session.save(f'saccades/predicted/{eye}/{direction}/timestamps', saccadeOnsetTimestamps)


    #
    error = np.array(error)
    print(f'INFO[{session.animal}, {session.date}]: Mean saccade onset correction = {error.mean():.3f} seconds')

    return

def determineGratingMotionAssociatedWithEachSaccade(
    session,
    interBlockIntervalRange=(1, 10),
    interBlockIntervalStep=0.1
    ):
    """
    """

    #
    gratingMotionByBlock = session.load('stimuli/dg/grating/motion')
    if gratingMotionByBlock is None:
        session.log(f'Session missing processed data for the drifting grating stimulus', level='warning')
        return
    nBlocks = gratingMotionByBlock.size

    #
    for eye in ('left', 'right'):
        for saccadeDirection in ('nasal', 'temporal'):
            saccadeOnsetTimestamps = session.load(f'saccades/predicted/{eye}/{saccadeDirection}/timestamps')
            gratingMotionBySaccade = list()

            if session.cohort in (1, 2, 3):

                #
                probeOnsetTimestamps = session.load('stimuli/dg/probe/timestamps')
                gratingEpochs = list()
                interProbeIntervals = np.diff(probeOnsetTimestamps)

                #
                thresholdDetermined = False
                for interBlockIntervalThreshold in np.arange(interBlockIntervalRange[0], interBlockIntervalRange[1], interBlockIntervalStep):
                    lastProbeIndices = np.concatenate([
                        np.where(interProbeIntervals > interBlockIntervalThreshold)[0],
                        np.array([probeOnsetTimestamps.size - 1])
                    ])
                    if lastProbeIndices.size == nBlocks:
                        thresholdDetermined = True
                        break

                if thresholdDetermined == False:
                    session.log(f'Failed to determine the inter-block interval threshold', level='warning')
                    return

                firstProbeIndices = np.concatenate([
                    np.array([0]),
                    lastProbeIndices[:-1] + 1
                ])
                gratingEpochs = np.hstack([
                    probeOnsetTimestamps[firstProbeIndices].reshape(-1, 1),
                    probeOnsetTimestamps[lastProbeIndices].reshape(-1, 1)
                ])

            #
            elif session.cohort in (4,):

                #
                motionOnsetTimestamps = session.load('stimuli/dg/grating/timestamps')
                motionOffsetTimestamps = session.load('stimuli/dg/iti/timestamps')
                gratingEpochs = np.hstack([
                    motionOnsetTimestamps.reshape(-1, 1),
                    motionOffsetTimestamps.reshape(-1, 1)
                ])

            #
            else:
                session.log('Could not extract grating motion during {saccadeDirection} saccades in the {eye} for session in cohort {session.cohort}')
                session.save(f'saccades/predicted/{eye}/{saccadeDirection}/motion', np.array([]).astype(int))
                return

            #
            nBlocks = gratingEpochs.shape[0]
            for saccadeOnsetTimestamp in saccadeOnsetTimestamps:
                searchResult = False
                for blockIndex in range(nBlocks):
                    gratingOnsetTimestamp, gratingOffsetTimestamp = gratingEpochs[blockIndex]
                    gratingMotion = gratingMotionByBlock[blockIndex]
                    if gratingOnsetTimestamp <= saccadeOnsetTimestamp <= gratingOffsetTimestamp:
                        searchResult = True
                        break
                if searchResult:
                    gratingMotionBySaccade.append(gratingMotion)
                else:
                    gratingMotionBySaccade.append(0)

            #
            gratingMotionBySaccade = np.array(gratingMotionBySaccade)
            session.save(f'saccades/predicted/{eye}/{saccadeDirection}/motion', gratingMotionBySaccade)

    return