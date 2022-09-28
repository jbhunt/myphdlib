import os
import re

from requests import session
import numpy as np
import pandas as pd
import pathlib as pl
import subprocess as sp
from scipy import stats
from scipy import signal as sig
from skg.nsphere import nsphere_fit as fitSphere
from sklearn.decomposition import PCA
from myphdlib.toolkit import smooth
from myphdlib.ffmpeg import countVideoFrames
from myphdlib import labjack as lj

def getFrameTimestamps(sessionFolder, cameraNickname='leftCam', framerate=200, threshold=1.5):
    """
    """

    #
    homeFolderPath = pl.Path(sessionFolder)
    videosFolderPath = [
        folder for folder in homeFolderPath.iterdir()
            if folder.name in ['videos', 'Videos']
    ].pop()

    #
    frameIntervals = None
    for file in videosFolderPath.iterdir():
        if cameraNickname in str(file) and str(file).endswith('_timestamps.txt'):
            frameIntervals = np.loadtxt(str(file)) / 1000000
    if type(frameIntervals) != np.ndarray:
        raise Exception('Could not find camera timestamps file')
    testValue = 1 / framerate * threshold * 1000
    missingFrameIndices = np.where(frameIntervals > testValue)[0]
    # if missingFrameIndices.size > 0:
    #     print(f'{missingFrameIndices.size} dropped frames detected')

    #
    labjackFolder = str([
        folder for folder in homeFolderPath.iterdir()
            if folder.name in ['labjack', 'LabJack']
    ].pop())

    #
    data = lj.loadLabjackData(labjackFolder)
    pulseWidthRange = (
        np.around(1 / framerate - (1 / framerate * 0.05), 3).item(),
        np.around(1 / framerate + (1 / framerate * 0.05), 3).item()
    )
    signal, exposureOnsetIndices = lj.extractLabjackEvent(
        data,
        iev=6,
        edge='both',
        pulse_width_range=pulseWidthRange
    )
    labjackTimestamps = data[exposureOnsetIndices, 0]

    # Figure out where frames where dropped and how many
    forwardShift = 0
    droppedFrameCount = 0
    frameTimestamps = np.full(frameIntervals.size + 1, np.nan)
    frameTimestamps[0] = labjackTimestamps[0]
    for intervalIndex, frameInterval in enumerate(frameIntervals):
        frameIndex = intervalIndex + 1
        frameCountWithinInterval = round(frameInterval / 1000 * framerate)
        if frameCountWithinInterval > 1:
            droppedFrameCount += frameCountWithinInterval - 1
            forwardShift += frameCountWithinInterval - 1
        frameTimestamps[frameIndex] = labjackTimestamps[frameIndex + forwardShift]

    #
    extraFrameCount = labjackTimestamps.size - (frameTimestamps.size + droppedFrameCount)

    return frameTimestamps, droppedFrameCount, extraFrameCount

    # Double check that the number of timestamps computed is equal to the number
    # of frames in the video recording
    for file in videosFolderPath.iterdir():
        if cameraNickname in str(file) and str(file).endswith('-0000.mp4'):
            trueFrameCount = countVideoFrames(str(file))
            testFrameCount = frameTimestamps.size
            if trueFrameCount != testFrameCount:
                raise Exception(f'True frame count (n={trueFrameCount}) != computed frame count (n={testFrameCount})')

    return frameTimestamps

def detectDroppedFrames(homeFolder, cameraFramerate=60, errorMargin=5, blackflySamplingRate=1000000):
    """
    """

    #
    homeFolderPath = pl.Path(homeFolder)
    videosFolderPath = [
        folder for folder in homeFolderPath.iterdir()
            if folder.name in ['videos', 'Videos']
    ].pop()

    #
    for file in videosFolderPath.iterdir():
        if 'leftCam_timestamps' in str(file):
            leftCameraTimestamps = np.loadtxt(str(file))
        elif 'rightCam_timestamps' in str(file):
            rightCameraTimestamps = np.loadtxt(str(file))

    #
    frameCountDifference = abs(leftCameraTimestamps.size - rightCameraTimestamps.size)
    smallerArray = leftCameraTimestamps if leftCameraTimestamps.size < rightCameraTimestamps.size else rightCameraTimestamps
    smallerArraySide = 'left' if leftCameraTimestamps.size < rightCameraTimestamps.size else 'right'

    #
    missingFrameIndices = np.where(
        smallerArray / blackflySamplingRate > 1 / (cameraFramerate - errorMargin) * 1000
    )[0]

    #
    missingFrameIndices += 1
    missingFrameIndices.sort()

    if missingFrameIndices.size != frameCountDifference:
        raise Exception(f'Not solvable: frame difference={frameCountDifference} vs. dropped frames detected={missingFrameIndices.size}')

    return missingFrameIndices, smallerArraySide

def getEyePosition(
    homeFolder,
    likelihoodThreshold=0.95,
    process=True,
    smoothingWindowSize=7,
    pupilCenterBodypartLabel='pupil-c',
    targetEye='right',
    ):
    """
    """

    homeFolderPath = pl.Path(homeFolder)
    # videosFolderPath = [
    #       folder for folder in homeFolderPath.iterdir()
    #         if folder.name in ['videos', 'Videos']
    # ].pop()

    frameLeftEye, frameRightEye = None, None
    frameCount = np.inf
    for file in homeFolderPath.iterdir():
        if 'leftCam' in str(file) or 'left-eye-score' in str(file):
            if file.suffix == '.csv':
                frameLeftEye = pd.read_csv(str(file), index_col=0, header=list(range(4)))
                frameLeftEye = frameLeftEye.sort_index(level=1, axis=1)
                network = frameLeftEye.columns[0][0]
                if frameLeftEye.shape[0] < frameCount:
                    frameCount = frameLeftEye.shape[0]
        elif 'rightCam' in str(file) or 'right-eye-score' in str(file):
            if file.suffix == '.csv':
                frameRightEye = pd.read_csv(str(file), index_col=0, header=list(range(4)))
                frameRightEye = frameRightEye.sort_index(level=1, axis=1)
                network = frameRightEye.columns[0][0]
                if frameRightEye.shape[0] < frameCount:
                    frameCount = frameRightEye.shape[0]

    #
    unequalFrameCount = False
    eyePositionData = np.full([frameCount, 4], np.nan)
    for frame, side, (start, stop) in zip([frameLeftEye, frameRightEye], ['left', 'right'], [(0, 2), (2, 4)]):
        if type(frame) != pd.DataFrame:
            continue
        coords = np.hstack([
                np.array(frame[network, 'pupil-c', 'x']).reshape(-1, 1),
                np.array(frame[network, 'pupil-c', 'y']).reshape(-1, 1)
            ])
        likelihood = np.array(frame[network, 'pupil-c', 'likelihood']).flatten()
        subthresholdMask = likelihood < likelihoodThreshold
        coords[subthresholdMask, :] = np.array([np.nan, np.nan])

        # Fill in dropped frames with Nans
        if unequalFrameCount and side == smallerArraySide:
            offset = 0
            for frameIndex in missingFrameIndices:
                coords = np.insert(coords, frameIndex + offset, np.array([np.nan, np.nan]), axis=0)
                offset += 1

        eyePositionData[:frameCount, start: stop] = coords[:frameCount, :]

    #
    filtered = list()
    for column in eyePositionData.T:
        if np.sum(np.isnan(column)) < column.size:
            filtered.append(column)
    eyePositionData = np.array(filtered).T

    #
    if process:

        #
        pca = PCA(n_components=2)
        interpolated = np.array(pd.DataFrame(eyePositionData).interpolate(method='linear', axis=0))
        import pdb; pdb.set_trace()
        decomposed = np.full(eyePositionData.shape, np.nan)
        decomposed[:,  :2] = pca.fit_transform(interpolated[:,  :2])
        decomposed[:, 2:4] = pca.fit_transform(interpolated[:, 2:4])

        # Re-align PCs with the direction of the raw eye position
        # PC1 should be positively correlated with vertical eye position
        r, p = stats.pearsonr(decomposed[:, 0], interpolated[:, 1])
        if r < 0:
            decomposed[:, 0: 2] *= -1

        r, p = stats.pearsonr(decomposed[:, 2], interpolated[:, 3])
        if r < 0:
            decomposed[:, 2: 4] *= -1

        # Overwrite the eyePositionData array
        eyePositionData[:, :] = decomposed

        # Standardize (z-score)
        sigmaLeftEye, sigmaRightEye = np.nanstd(eyePositionData[:, 0]), np.nanstd(eyePositionData[:, 2])
        eyePositionData[:, 0] = (eyePositionData[:, 0] - np.nanmean(eyePositionData[:, 0])) / sigmaLeftEye
        eyePositionData[:, 1] = (eyePositionData[:, 1] - np.nanmean(eyePositionData[:, 1])) / sigmaLeftEye
        eyePositionData[:, 2] = (eyePositionData[:, 2] - np.nanmean(eyePositionData[:, 2])) / sigmaRightEye
        eyePositionData[:, 3] = (eyePositionData[:, 3] - np.nanmean(eyePositionData[:, 3])) / sigmaRightEye

        # Interpolate
        eyePositionData = np.array(pd.DataFrame(
            eyePositionData
        ).interpolate(method='linear', axis=0))

        # Unreflect left eye position data
        eyePositionData[:, :2] *= -1

        #
        eyePositionData = smooth(eyePositionData, smoothingWindowSize, axis=0)

    return eyePositionData

def getPupilArea(csv, likelihoodThreshold=0.95, nPoints=8, iPoint=0, standardize=True):
    """
    """

    frame = pd.read_csv(str(csv), index_col=0, header=list(range(4)))
    frame = frame.sort_index(level=1, axis=1)
    network = frame.columns[0][0]
    data = np.full([frame.shape[0], nPoints, 2], np.nan)
    for iPoint in np.arange(nPoints):
        xi = np.array(frame[network, f'pupil-{iPoint + 1}', 'x']).flatten()
        yi = np.array(frame[network, f'pupil-{iPoint + 1}', 'y']).flatten()
        li = np.array(frame[network, f'pupil-{iPoint + 1}', 'likelihood']).flatten()
        likelihoodMask = li <= likelihoodThreshold
        xi[likelihoodMask] = np.nan
        yi[likelihoodMask] = np.nan
        data[:, iPoint, 0] = xi
        data[:, iPoint, 1] = yi

    #
    pupilCenter = np.full([frame.shape[0], 2], np.nan)
    xi = np.array(frame[network, f'pupil-c', 'x']).flatten()
    yi = np.array(frame[network, f'pupil-c', 'y']).flatten()
    li = np.array(frame[network, f'pupil-c', 'likelihood']).flatten()
    likelihoodMask = li <= likelihoodThreshold
    xi[likelihoodMask] = np.nan
    yi[likelihoodMask] = np.nan
    pupilCenter[:, 0] = xi
    pupilCenter[:, 1] = yi

    area = np.full(frame.shape[0], np.nan)
    for iFrame, points in enumerate(data):
        if nPoints > 1:
            nanMask = np.isnan(points).all(1)
            r, c = fitSphere(points[~nanMask])
        elif nPoints == 1:
            r = np.linalg.norm(points[iPoint, :] - pupilCenter[iFrame, :])
        area[iFrame] = np.pi * r ** 2

    if standardize:
        area = (area - area.mean()) / area.std()

    return area

def getSaccadeOnsetIndices(homeFolder, saccadeDirection=None):
    """
    """

    homeFolderPath = pl.Path(homeFolder)
    result = list(homeFolderPath.rglob('*true-saccades-classification-results.npy'))
    if len(result) != 1:
        raise Exception('Cannot find saccade classificiation results')
    else:
        saccadeClassificationResults = np.load(result.pop())

    if saccadeDirection == 'ipsi':
        saccadeDirectionMask = saccadeClassificationResults[:, 2] == 3
    elif saccadeDirection == 'contra':
        saccadeDirectionMask = saccadeClassificationResults[:, 2] == 3
    elif saccadeDirection == None:
        saccadeDirectionMask = np.ones(saccadeClassificationResults.shape[0]).astype(bool)

    #
    saccadeOnsetIndices = saccadeClassificationResults[:, 0][saccadeDirectionMask]

    return saccadeOnsetIndices

import cv2 as cv

def getPupilArea(video, score, lowerBound=100, upperBound=200, maximumDistance=20):
    """
    """

    # sessionFolderPath = pl.Path(sessionFolder)
    # videosFolderPath = None
    # for folder in sessionFolderPath.iterdir():
    #     if folder.name in ['Videos', 'videos']:
    #         videosFolderPath = folder
    #         break

    frame = pd.read_csv(score, header=[0, 1, 2, 3], index_col=0)
    network = frame.columns[0][0]
    pupilCenterData = np.array(frame[network, 'pupil-c'])
    pupilCenter = pupilCenterData[:, :2]
    likelihood = pupilCenterData[:, -1]
    mask = likelihood < 0.9
    pupilCenter[mask, :] = np.array([np.nan, np.nan])
    cap = cv.VideoCapture(video)
    frameCount = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
    area = np.full(frameCount, np.nan)

    frameIndex = 0
    while True:
        result, image = cap.read()
        if result == False:
            break
        point = pupilCenter[frameIndex, :]
        blurred = cv.GaussianBlur(image, (3, 3), cv.BORDER_DEFAULT)
        edges = cv.Canny(blurred, lowerBound, upperBound)
        contours, hierarchy = cv.findContours(edges, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        contours = tuple([contour for contour in contours if contour.shape[0] >= 5])
        centroids = np.array([contour.mean(0).flatten() for contour in contours])
        # distances = np.linalg.norm(point - centroids)
        distances = np.array([np.linalg.norm(point - centroid) for centroid in centroids])
        closest = np.argmin(distances)
        if distances[closest] > maximumDistance:
            continue
        contour = contours[closest]
        if contour.shape[0] < 5:
            import pdb; pdb.set_trace()
        center, axes, theta = cv.fitEllipse(contour)
        ai = np.pi * axes[0] * axes[1]
        area[frameIndex] = ai
        frameIndex += 1

    return area

from .labjack import loadLabjackData

def countRecordedEvent(workingDirectory, excludeNoisyGrating=True):
    """
    """
    
    workingDirectoryPath = pl.Path(workingDirectory)
    totalEventCount = 0
    for file in workingDirectoryPath.iterdir():
        if 'Metadata' in file.name:
            with open(file, 'r') as stream:
                lines = stream.readlines()
            for lineIndex, line in enumerate(lines):
                if line == '\n':
                    break
            if 'sparseNoise' in file.name:
                eventCount = len(lines[lineIndex + 1:]) * 2
            elif 'noisyGrating' in file.name:
                eventCount = 0
                # eventCount = len(lines[lineIndex + 1:]) - 20
            else:
                eventCount = len(lines[lineIndex + 1:])
            totalEventCount += eventCount

    return totalEventCount
