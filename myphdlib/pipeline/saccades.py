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

def extractEyePosition(session, likelihoodThreshold=0.99, pupilCenterName='pupilCenter'):
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
    session.write(eyePositionUncorrected, 'eyePositionUncorrected')

    return

def correctEyePosition(session, pad=1e6):
    """
    Correct eye position data for missing/dropped frames
    """

    #
    eyePositionUncorrected = session.eyePositionUncorrected
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
        try:
            eyePositionUncorrected[0, columnIndices] = eyePositionUncorrected[0, columnIndices]
        except:
            import pdb; pdb.set_trace()
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
    session.write(eyePositionCorrected, 'eyePositionCorrected')

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

    eyePositionCorrected = session.eyePositionCorrected
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
            missingDataMask[side] = np.full(eyePositionCorrected.shape[0], True)
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
    session.write(eyePositionDecomposed, 'eyePositionDecomposed')
    session.write(missingDataMask, 'missingDataMask')

    return

def reorientEyePosition(session, reflect='left'):
    """
    """

    eyePositionDecomposed = session.eyePositionDecomposed
    eyePositionCorrected = session.eyePositionCorrected
    eyePositionReoriented = np.full_like(eyePositionCorrected, np.nan)
    missingDataMask = session.missingDataMask

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
            raise Exception('Could not determine correlation between raw and decomposed eye position')
        eyePositionReoriented[:, columnIndex] = signal3

    # TODO: Check that left and right eye position is anti-correlated

    #
    session.write(eyePositionReoriented, 'eyePositionReoriented')

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
    missingDataMask = session.missingDataMask
    
    # Determine the nearest odd window size
    smoothingWindowSize = round(t / (1 / session.fps * 1000))
    if smoothingWindowSize % 2 == 0:
        smoothingWindowSize += 1

    # Filter
    eyePositionReoriented = session.eyePositionReoriented
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
    session.write(eyePositionFiltered, 'eyePositionFiltered')

    return

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
    if 'saccadeWaveformsPutative' in session.keys():
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
    eyePositionFiltered = session.read('eyePositionFiltered')
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
    session.write(saccadeDetectionResults, 'saccadeDetectionResults')
    
    return

# TODO: Exclude duplicate labeled saccades
# TODO: Implement a method for NOT overwriting previously collected saccades waveforms

def labelPutativeSaccades(session, nSamplesPerEye=30):
    """
    """

    saccadeWaveformsPutative = session.read('saccadeDetectionResults')['waveforms']
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
    session.write(saccadeWaveformsLabeled, 'saccadeWaveformsLabeled')

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
    y = list()

    #
    for session in sessions:

        #
        if 'saccadeDetectionResults' in session.keys():
            saccadeDetectionResults = session.read('saccadeDetectionResults')
            for eye in ('left', 'right'):
                saccadeWaveformsUnlabeled = saccadeDetectionResults['waveforms'][eye]
                if saccadeWaveformsUnlabeled is None:
                    continue
                for sample in saccadeWaveformsUnlabeled:
                    if np.isnan(sample).any():
                        continue
                    if features == 'position':
                        xTest.append(sample)
                    elif features == 'velocity':
                        xTest.append(np.diff(sample))
        
        #
        if 'saccadeWaveformsLabeled' in session.keys():
            saccadeLabelingResults = session.read('saccadeWaveformsLabeled')
            for eye in ('left', 'right'):
                saccadeWaveformsLabeled = saccadeLabelingResults[eye]
                iterable = zip(
                    saccadeWaveformsLabeled['X'],
                    saccadeWaveformsLabeled['y']
                )
                for sample, label in iterable:
                    if np.isnan(sample).any():
                        continue
                    if features == 'position':
                        xTrain.append(sample)
                    elif features == 'velocity':
                        xTrain.append(np.diff(sample))
                    y.append(label.item())

    #
    xTrain = np.array(xTrain)
    xTest = np.array(xTest)
    y = np.array(y)

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
        search.fit(X, y.ravel())
        clf = search.best_estimator_
    
    #
    elif classifier == 'lda':
        lda = LinearDiscriminantAnalysis()
        lda.fit(X, y.ravel())
        clf = lda

    return pca, clf

def classifyPutativeSaccades(
    sessions,
    classifier='net',
    decompose=False,
    features='position'
    ):
    """
    """

    #
    pca, clf = _trainSaccadeClassifier(
        sessions,
        classifier,
        decompose,
        features
    )

    #
    for session in sessions:

        #
        if 'saccadeWaveformsClassified' in session.keys():
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
        saccadeDetectionResults = session.read('saccadeDetectionResults')

        # Classify saccades
        for eye in ('left', 'right'):
            
            #
            if saccadeDetectionResults['waveforms'][eye] is None:
                continue

            #
            waveforms = list()
            indices = list()
            for index, waveform in enumerate(saccadeDetectionResults['waveforms'][eye]):
                if np.isnan(waveform).any():
                    continue
                if features == 'position':
                    waveforms.append(waveform)
                elif features == 'velocity':
                    waveforms.append(np.diff(waveform))
                indices.append(saccadeDetectionResults['indices'][eye][index])
            
            #
            if decompose:
                samples = pca.transform(
                    np.diff(np.array(waveforms), axis=1)
                )
            else:
                samples = np.array(waveforms)

            #
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
                else:
                    saccadeClassificationResults[eye][direction]['indices'] = np.array(saccadeClassificationResults[eye][direction]['indices'])
                    saccadeClassificationResults[eye][direction]['waveforms'] = np.array(saccadeClassificationResults[eye][direction]['waveforms'])
                    nSaccades = saccadeClassificationResults[eye][direction]['indices'].size
                    print(f'INFO[animal={session.animal}, date={session.date}]: {nSaccades} {direction} saccades classified for the {eye} eye')

        #
        session.write(saccadeClassificationResults, 'saccadeClassificationResults')

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
    session.write(saccadeClassificationResults, 'saccadeClassificationResults')

    #
    error = np.array(error)
    print(f'INFO[animal={session.animal}, date={session.date}]: Mean saccade onset correction = {error.mean():.3f} seconds')

    return

def process(session, pupilCenterName='pupilCenter'):
    """
    """

    modules_ = [
        extractEyePosition,
        correctEyePosition,
        decomposeEyePosition,
        reorientEyePosition,
        filterEyePosition,
        detectPutativeSaccades
    ]

    for module_ in modules_:
        if module_.__name__ == 'extractEyePosition':
            module_(session, pupilCenterName=pupilCenterName)
        else:
            module_(session)

    return
