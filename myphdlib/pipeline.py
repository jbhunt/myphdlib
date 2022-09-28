import os
import pathlib as pl
import re
import time

import numpy as np
import pandas as pd
import yaml
from scipy.signal import find_peaks as findPeaks
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import RobustScaler

from myphdlib import saccades
from myphdlib.deeplabcut import CONFIG, changeWorkingNetwork
from myphdlib.factory import SessionFactory
from myphdlib.labjack import extractLabjackEvent, loadLabjackData
from myphdlib.toolkit import interpolate, resample, smooth
from myphdlib.ffmpeg import reflectVideo

try:
    from deeplabcut import analyze_videos as analyzeVideos
except ImportError:
    # print('WARNING: DeepLabCut import failed')
    dlc = None

# TODO: Refactor the modules (read: functions) so that they accept a 
#       single session object instead of a dataset name.

datasetNames = (
    'Muscimol',
    'Suppression',
    'Concussion',
    'Realtime',
    'Dreadd',
)

cohortNames = (
    'musc',
    'pixel',
    'crush',
    'blast',
    'dreadd',
)

def iterateSessions(datasetName='Muscimol', checkValidity=False):
    """
    """

    factory = SessionFactory()

    global datasetNames
    if datasetName not in datasetNames:
        raise Exception('Invalid dataset folder')
    
    # Locate the external HDD
    user = os.environ['USER']
    externalHDDPath = None
    for folder in pl.Path(f'/media/{user}/').iterdir():
        # TODO: Figure out a way to handle multiple harddrives that match the "JH-DATA-*" pattern
        if bool(re.search('JH-DATA-\w{2,3}', folder.name)):
            externalHDDPath = folder
            break
    if externalHDDPath == None:
        raise Exception('Could not locate external HDD')

    # Find the dataset home folder
    datasetFolderPath = None
    for folder in externalHDDPath.iterdir():
        if bool(re.search(datasetName, folder.name)):
            datasetFolderPath = folder
            break
    if datasetFolderPath == None:
        raise Exception('Could not locate dataset folder')

    # 
    for firstLevel in datasetFolderPath.iterdir():
        if firstLevel.name.startswith('.'):
            continue
        animal, date = None, None
        if bool(re.search('\d{4}-\d{2}-\d{2}', firstLevel.name)):
            date = firstLevel.name
        else:
            for cohortName in cohortNames:
                if bool(re.search(f'{cohortName}\w*', firstLevel.name)):
                    animal = re.findall(f'{cohortName}\w*', firstLevel.name).pop()
                    if len(firstLevel.name) != len(animal):
                        session = firstLevel.name[-1]

        #
        if animal == None and date == None:
            continue

        #
        for secondLevel in firstLevel.iterdir():
            session = None
            if bool(re.search('\d{4}-\d{2}-\d{2}', secondLevel.name)):
                date = secondLevel.name
            else:
                for cohortName in cohortNames:
                    if bool(re.search(f'{cohortName}\w*', secondLevel.name)):
                        animal = re.findall(f'{cohortName}\w*', secondLevel.name).pop()
                        if len(secondLevel.name) != len(animal):
                            session = secondLevel.name[-1]

            #
            obj = factory.produce(str(secondLevel))
            
            if checkValidity:
                if obj.isValid():
                    yield obj, animal, date, session
            else:
                yield obj, animal, date, session

    return

def reflectVideoRecording(obj, eye='left'):
    """
    """

    result = [
        file for file in pl.Path(obj.videosFolder).glob(f'*{eye}Cam*.mp4')
            if 'reflected' not in file.name
    ]
    if len(result) == 0:
        raise Exception(f'No {eye} eye video recording detected')
    elif len(result) == 1:
        video = result.pop()
        filename = video.name.rstrip('.mp4')
        filename += ' (reflected).mp4'
        if video.parent.joinpath(filename).exists():
            return
        else:
            reflectVideo(str(video), suffix=' (reflected)')
    else:
        raise Exception(f'More than one recording detected for the {eye} eye (animal={obj.anima}, date={obj.date})')

    return

def processVideosWithDeeplabcut(obj, network='Gazer', overwrite=False):
    """
    """

    #
    videos = list()
    changeWorkingNetwork(network)

    #
    for eye in ('left', 'right'):
        
        # Look for existing DLC outputs and delete if desired
        result = list(pl.Path(obj.videosFolder).glob(f'*{eye}Cam*{network}*.csv'))

        #
        if len(result) == 1:
            if overwrite:
                for suffix in ['csv', 'h5', 'pickle']:
                    for file in pl.Path(obj.videosFolder).iterdir():
                        if bool(f'.*{eye}Cam.*{network}.*\.{suffix}', file.name):
                            file.unlink()
            else:
                continue

        #
        elif len(result) == 0:
            pass

        #
        else:
            raise Exception(f'More than one DLC score detected for the {eye} eye (animal={obj.animal}, date={obj.date})')

        # Find the video file
        result = list(pl.Path(obj.videosFolder).glob(f'*{eye}Cam*.mp4'))
        if eye == 'left':
            result = [
                file for file in result
                    if 'reflected' in file.name
            ]
        if len(result) == 1:
            videos.append(str(result.pop()))
        else:
            print(f'WARNING: Could not locate the {eye} eye video recording for {obj.animal} on {obj.date}')
        
    #
    if len(videos) != 0:
        analyzeVideos(CONFIG, videos, save_as_csv=True)

    return

def extractEyePosition(obj, likelihoodThreshold=0.99, maximumAttempts=3):
    """
    """

    #
    attemptCounter = 0
    result = False
    while True:
        if attemptCounter > maximumAttempts:
            print(f'WARNING: Could not locate both DeepLabCut scores for {obj.animal} on {obj.date}')
            break
        scores = list(pl.Path(obj.videosFolder).rglob('*.csv'))
        if len(scores) not in (1, 2):
            attemptCounter += 1
            time.sleep(1)
            continue
        else:
            result = True
            break

    if result == False:
        print(f'WARNING: Could not locate DLC output for {obj.animal} on {obj.date}')
        return

    #
    eyePositionLeft, eyePositionRight = np.full([1, 2], np.nan), np.full([1, 2], np.nan)
    for score in scores:
        frame = pd.read_csv(str(score), header=list(range(3)), index_col=0)
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
        if 'leftCam' in score.name:
            eyePositionLeft = coords
        else:
            eyePositionRight = coords

    nrows = eyePositionLeft.shape[0] if eyePositionLeft.shape[0] > eyePositionRight.shape[0] else eyePositionRight.shape[0]
    eyePosition = np.full([nrows, 4], np.nan)
    eyePosition[:eyePositionLeft.shape[0], 0:2] = eyePositionLeft
    eyePosition[:eyePositionRight.shape[0], 2:4] = eyePositionRight
    obj.save('eyePositionUncorrected', eyePosition)

    return

def correctEyePosition(
    obj,
    buffer=50000,
    ):
    """
    """

    #
    cameraTimestamps = list(pl.Path(obj.videosFolder).rglob('*Cam_timestamps.txt'))
    if len(cameraTimestamps) != 2:
        print(f'WARNING: Could not find camera timestamps for {obj.animal} on {obj.date}')
        return

    try:
        eyePositionUncorrected = obj.load('eyePositionUncorrected')
    except Exception as error:
        print(f'WARNING: Correction failed for {obj.animal} on {obj.date}: Could not load uncorrected eye position data')
        return

    eyePositionCorrected = np.full([eyePositionUncorrected.shape[0] + buffer, 4], np.nan)

    #
    terminationIndices = list()
    for file in cameraTimestamps:

        #
        if bool(re.search('.*leftCam.*', file.name)):
            side = 'left'
            columnIndices = 0, 2
            eyePosition = eyePositionUncorrected[:, columnIndices[0]: columnIndices[1]]
        elif bool(re.search('.*rightCam.*', file.name)):
            side = 'right'
            columnIndices = 2, 4
            eyePosition = eyePositionUncorrected[:, columnIndices[0]: columnIndices[1]]

        #
        if obj.frameCount.deeplabcut[side]['count'] is pd.NA:
            terminationIndices.append(0)
            continue

        # Count the number of frames
        if obj.frameCount.cameras[side]['count'] - obj.frameCount.deeplabcut[side]['count'] != 1:
            print(f'WARNING: Unaccounted missing frames were detected for {obj.animal} on {obj.date}')

        # Timestamps for each frame in the video
        intervals = np.loadtxt(file) / 1000000000
        timestamps = np.full(intervals.size + 1, np.nan)
        timestamps[0] = 0
        timestamps[1:] = np.cumsum(intervals)

        # TODO: Figure out why there is always one extra timestamp than the number of video frames and where it comes from

        #
        frameOffset = 0
        stopIndex = obj.frameCount.deeplabcut[side]['count']
        for frameIndex, timestamp in enumerate(timestamps):
            if frameIndex == stopIndex:
                break
            coordinate = eyePosition[frameIndex]
            if frameIndex != 0:
                interval = timestamp - timestamps[frameIndex - 1]
                frameCount = round(interval * obj.acquisitionFramerate)
                if frameCount > 1:
                    frameOffset += frameCount - 1
            try:
                eyePositionCorrected[frameIndex + frameOffset, columnIndices[0]: columnIndices[1]] = coordinate
            except:
                import pdb; pdb.set_trace()

        # Drop the NaN values on the tail end of the array
        terminationIndex = obj.frameCount.deeplabcut[side]['count'] + frameOffset
        terminationIndices.append(terminationIndex)

    # Determine where to terminate the eye position data
    if terminationIndices[0] != terminationIndices[1]:
        difference = abs(np.diff(terminationIndices).item())
        if terminationIndices[0] > terminationIndices[1]:
            terminationIndex = np.min(terminationIndices)
        else:
            print(f'WARNING: Correction failed for {obj.animal} on {obj.date}: Unequal frame counts (difference={difference} frames)')
            # TODO: Move the shorter array around such that the correlation is maximized
            terminationIndex = np.max(terminationIndices)
    else:
        difference = 0
        terminationIndex = np.unique(terminationIndices).item()

    # Drop the NaN tail and save the corrected eye position data
    eyePositionCorrected = eyePositionCorrected[:terminationIndex, :]
    obj.save('eyePositionCorrected', eyePositionCorrected)
    obj.save('videoAlignmentError', difference)

    return

def decomposeEyePosition(
    obj,
    reflectLeftEye=True,
    reflectRightEye=False,
    ):
    """
    """

    #
    try:
        eyePosition = obj.load('eyePositionCorrected')
    except Exception as error:
        print(f'WARNING: Could not load corrected eye position data')
        return

    eyePositionDecomposed = np.full(eyePosition.shape, np.nan)
    for columnIndexStart, columnIndexStop in ((0, 2), (2, 4)):

        # Interpolate NaN values
        X = eyePosition[:, columnIndexStart: columnIndexStop]
        X = interpolate(X, axis=1)

        # Decompose
        model = PCA(n_components=2).fit(X)
        Xt = model.transform(X)

        # Re-align with pixel coordinate system
        r, p = pearsonr(X[:, 0], Xt[:, 0])
        if r >= 0 and p < 0.05:
            coeff = +1
        else:
            coeff = -1
        Xt *= coeff

        # Reflect the left eye position
        if reflectLeftEye and columnIndexStart == 0:
            Xt *= -1
        if reflectRightEye and columnIndexStart == 2:
            Xt *= -1

        #
        eyePositionDecomposed[:, columnIndexStart: columnIndexStop] = Xt

    obj.save('eyePositionDecomposed', eyePositionDecomposed)

    return

def standardizeEyePosition(obj):
    """
    """

    scaler = RobustScaler()

    #
    if obj.eyePositionDecomposed is None:
        print(f'WARNING: Decomposed eye position is not available for {obj.animal} on {obj.date}')
        return

    eyePositionStandardized = scaler.fit_transform(obj.eyePositionDecomposed)
    obj.save('eyePositionStandardized', eyePositionStandardized)

    return

def detectMonocularSaccades(
    obj,
    targetFramerate=200,
    velocityThreshold=0.06,
    minimumSaccadeInterval=0.05,
    perisaccadicWindow=(-0.15, 0.15),
    smoothVelocity=True,
    smoothingWindowSize=7,
    ):
    """
    """

    startIndexOffset, stopIndexOffset = list(map(round, np.array(perisaccadicWindow) * obj.acquisitionFramerate))
    # saccadeWaveformSize = stopIndexOffset - startIndexOffset
    saccadeWaveformSize = abs(np.diff(list(map(round, np.array(perisaccadicWindow) * targetFramerate))).item())
    # stopIndexOffset += 1
            
    #
    try:
        eyePosition = obj.load('eyePositionStandardized')
    except Exception as error:
        print(f'WARNING: Failed to load decomposed eye position data')
        return

    #
    saccadeWaveforms = {
        'left': list(),
        'right': list(),
    }
    saccadeOnsetIndices = {
        'left': list(),
        'right': list(),
    }

    # Resample signal if necessary
    # if obj.acquisitionFramerate != targetFramerate:
    #     columns = list()
    #     for columnIndex, columnData in enumerate(eyePosition.T):
    #         column = resample(columnData, obj.acquisitionFramerate, targetFramerate)
    #         columns.append(column)
    #     eyePosition = np.vstack(columns).T

    #
    for eye, eyePositionMono in zip(['left', 'right'], np.split(eyePosition, 2, axis=1)):
        for coeff in (+1, -1):

            # Locate peaks in velocity
            velocity = np.diff(eyePositionMono[:, 0]) * coeff
            if smoothVelocity:
                velocity = smooth(velocity, smoothingWindowSize)
            peakIndices, peakProperties = findPeaks(
                velocity,
                height=velocityThreshold,
                distance=round(minimumSaccadeInterval * targetFramerate)
            )

            # TODO: Untransform the peak indices (if the signal was resampled)

            # Extract the saccade waveforms
            for peakIndex in peakIndices:
                saccadeOnsetIndices[eye].append(peakIndex + 1)
                saccadeWaveform = eyePositionMono[peakIndex + startIndexOffset: peakIndex + stopIndexOffset, 0]
                if saccadeWaveform.size == 0:
                    continue
                if obj.acquisitionFramerate != targetFramerate:
                    saccadeWaveform = resample(saccadeWaveform, obj.acquisitionFramerate, targetFramerate)
                if saccadeWaveform.size == saccadeWaveformSize:
                    saccadeWaveforms[eye].append(saccadeWaveform)

    # Cast lists to numpy arrays
    # TODO: Sort arrays by onset indices
    saccadeWaveforms['left'] = np.array(saccadeWaveforms['left'])
    saccadeWaveforms['right'] = np.array(saccadeWaveforms['right'])
    saccadeOnsetIndices['left'] = np.array(saccadeOnsetIndices['left'])
    saccadeOnsetIndices['right'] = np.array(saccadeOnsetIndices['right'])
    obj.save('saccadeWaveformsPutative', saccadeWaveforms)
    obj.save('saccadeOnsetIndicesPutative', saccadeOnsetIndices)

    return

def classifyMonocularSaccades(
    obj,
    targetLabelingSessions=['2022-06-27', '2022-06-28'],
    ):
    """
    """

    if hasattr(obj, 'saccadeWaveformsPutative') == False:
        print(f'WARNING: Session for {obj.animal} on {obj.date} has no putative saccades')
        return

    # TODO: Perform grid search to optimize classifier hyperparameters
    X, y = saccades.createTrainingDataset(targetLabelingSessions)
    clf = MLPClassifier(max_iter=1000000)
    clf.fit(X, y.ravel())
    
    #
    if hasattr(obj, 'saccadeWaveformsPutative'):

        #
        saccadeOnsetIndicesClassified = {
            'left': {
                'left': list(),
                'right': list(),
            },
            'right': {
                'left': list(),
                'right': list()
            } 
        }
        saccadeWaveformsClassified = {
            'left': {
                'left': list(),
                'right': list(),
            },
            'right': {
                'left': list(),
                'right': list()
            } 
        }

        #
        for side in ['left', 'right']:

            #
            saccadeWaveforms = obj.saccadeWaveformsPutative[side]
            saccadeOnsetIndicesPutative = obj.saccadeOnsetIndicesPutative[side]
            samples = np.diff(saccadeWaveforms, axis=1)
            labels = clf.predict(samples)

            #
            for sampleIndex, label in enumerate(labels.flatten()):

                # Skip noise
                if label == 0:
                    continue

                #
                saccadeWaveform = saccadeWaveforms[sampleIndex, :]
                saccadeOnsetIndex = saccadeOnsetIndicesPutative[sampleIndex]

                # Save the saccade onset index and waveform
                if label == -1:
                    direction = 'left'
                elif label == +1:
                    direction = 'right'
                saccadeOnsetIndicesClassified[side][direction].append(saccadeOnsetIndex)
                saccadeWaveformsClassified[side][direction].append(saccadeWaveform)

        # Cast to numpy arrays and save
        for container in (saccadeWaveformsClassified, saccadeOnsetIndicesClassified):
            for side in ('left', 'right'):
                for direction in ('left', 'right'):
                    container[side][direction] = np.array(container[side][direction])
        obj.save('saccadeOnsetIndicesClassified', saccadeOnsetIndicesClassified)
        obj.save('saccadeWaveformsClassified', saccadeWaveformsClassified)

    return

def identifyConjugateSaccades(obj):
    """
    """

    # TODO: Code this module

    return

def processLabjackData(obj):
    """
    """


    #
    if obj.labjackFolder == None:
        print(f'WARNING: Could not locate labjack folder for {obj.animal} on {obj.date}')
        return
    
    try:
        labjackData = loadLabjackData(obj.labjackFolder)
        obj.save('labjackData', labjackData)
    except:
        print(f'WARNING: Could not load labjack dat files for {obj.animal} on {obj.date}')
        return

    #
    timestampsArray = labjackData[:, 0]
    dt = timestampsArray[1] - timestampsArray[0]
    labjackSamplingRate = 1 / dt
    obj.save('labjackSamplingRate', labjackSamplingRate)

    #
    if hasattr(obj, '_identifyLabjackChannels'):
        obj._identifyLabjackChannels()

    #
    for channelName, channelIndex in obj.labjackChannelMapping.items():
        if channelName == 'Acquisition':
            edge = 'both'
        else:
            edge = 'rising'
        eventSignal, eventIndices = extractLabjackEvent(labjackData, channelIndex, edge)
        eventTimestamps = timestampsArray[eventIndices]
        obj.save(f'labjackIndices{channelName}', eventIndices)
        obj.save(f'labjackTimestamps{channelName}', eventTimestamps)

    #
    if hasattr(obj, '_identifyVisualProbes'):
        obj._identifyVisualProbes()

    #
    if hasattr(obj, '_identifySingleTrials'):
        obj._identifySingleTrials()

    # Saccade onset timestamps
    if hasattr(obj, 'saccadeOnsetIndicesClassified'):
        saccadeOnsetTimestamps = {'left': {'left': None, 'right': None}, 'right': {'left': None, 'right': None}}
        for targetEye in ('left', 'right'):
            for saccadeDirection in ('left', 'right'):
                labjackTimestampsAcquisition = obj.load('labjackTimestampsAcquisition')
                obj.saccadeOnsetIndicesClassified[targetEye][saccadeDirection]
                timestamps = labjackTimestampsAcquisition[obj.saccadeOnsetIndicesClassified[targetEye][saccadeDirection]]
                saccadeOnsetTimestamps[targetEye][saccadeDirection] = timestamps
        obj.save(f'saccadeOnsetTimestamps', saccadeOnsetTimestamps)

    return

def processDataset(
    datasetName='Realtime',
    allowInterrupt=True,
    ):
    """
    """

    for obj, animal, date, session in iterateSessions(datasetName):

        print(f'\n')
        print(f'Processing session from {obj.animal} on {obj.date}')
        print(f'--------------------------------------------------')

        modules = (
            reflectVideoRecording,
            processVideosWithDeeplabcut,
            extractEyePosition,
            correctEyePosition,
            decomposeEyePosition,
            standardizeEyePosition,
            detectMonocularSaccades,
            classifyMonocularSaccades,
            identifyConjugateSaccades,
            # processLabjackData
        )

        errors = list()
        for module in modules:
            print(f'Running module {module.__name__} ...')
            try:
                module(obj)
            except Exception as error:
                errors.append(error)
                if allowInterrupt:
                    raise error
                else:
                    print(f'ERROR: Broken pipeline: {error}')
                break

    return
