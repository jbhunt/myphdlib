import os
import re
import numpy as np
import pandas as pd
import pathlib as pl
from myphdlib.toolkit import smooth
from myphdlib.backend import getEyePosition, getPupilArea, getSaccadeOnsetIndices

def detectThresholdCrossings(
    homeFolder,
    positionThreshold,
    timeout=1,
    targetEye='left'
    ):
    """
    """

    eyePosition = getEyePosition(homeFolder)
    # saccadeIndices = getSaccadeOnsetIndices(homeFolder, saccadeDirection=None)

    #
    leftEyePositiveThresholdCrossings = np.diff(eyePosition[:, 0] > positionThreshold, prepend=False)
    leftEyeNegativeThresholdCrossings = np.diff(eyePosition[:, 0] > -1 * positionThreshold, prepend=False)
    leftEyeThresholdCrossings = np.logical_or(
        leftEyePositiveThresholdCrossings,
        leftEyeNegativeThresholdCrossings,
    )

    #
    rightEyePositiveThresholdCrossings = np.diff(eyePosition[:, 0] > positionThreshold, prepend=False)
    rightEyeNegativeThresholdCrossings = np.diff(eyePosition[:, 0] > -1 * positionThreshold, prepend=False)
    rightEyeThresholdCrossings = np.logical_or(
        rightEyePositiveThresholdCrossings,
        rightEyeNegativeThresholdCrossings,
    )

    #
    if targetEye == None:
        thresholdCrossings = np.logical_or(
            leftEyeThresholdCrossings,
            rightEyeThresholdCrossings
        )
    elif targetEye == 'left':
        thresholdCrossings = leftEyeThresholdCrossings
    elif targetEye == 'right':
        thresholdCrossings = rightEyeThresholdCrossings

    #
    thresholdCrossingIndices = np.argwhere(thresholdCrossings)[:, 0]
    if thresholdCrossingIndices[0] == 0:
        thresholdCrossingIndices = np.delete(thresholdCrossingIndices, 0)

    # Get rid of threshold crossings that happen within the timeout period
    loss = None
    while True:
        intervals = np.diff(thresholdCrossingIndices)
        sizeBefore = thresholdCrossingIndices.size
        for counter, interval in enumerate(intervals):
            if interval <= round(200 * timeout):
                thresholdCrossingIndices = np.delete(thresholdCrossingIndices, counter + 1)
                break
        sizeAfter = thresholdCrossingIndices.size
        loss = sizeAfter - sizeBefore
        if loss == 0:
            break

    return thresholdCrossingIndices

def simulateRealtimeExperimentSingleSession(
    homeFolder,
    positionThresholds,
    perisaccadicWindow=(-0.05, 0.11),
    timeout=1,
    targetEye='left',
    deeplabcutBatchSize=5,
    deeplabcutBatchCount=2,
    ):
    """
    """

    saccadeOnsetIndices = getSaccadeOnsetIndices(homeFolder, saccadeDirection=None)
    perisaccadicProbeCounts = np.zeros(len(positionThresholds))

    #
    for arrayIndex, positionThreshold in enumerate(positionThresholds):

        #
        thresholdCrossingIndices = detectThresholdCrossings(
            homeFolder,
            positionThreshold,
            timeout=timeout,
            targetEye=targetEye,
        )

        #
        for thresholdCrossingIndex in thresholdCrossingIndices:

            # Select temporal delay
            projectorInputDelay = np.random.randint(round(200 * 0.03), high=round(200 * 0.045), size=1).item()
            poseEstimationDelay = np.random.randint(round(200 * 0.008), high=round(200 * 0.015), size=1).item()
            batchProcessingDelay = round(deeplabcutBatchSize * deeplabcutBatchCount)
            probeOnsetIndex = thresholdCrossingIndex + projectorInputDelay + poseEstimationDelay + batchProcessingDelay

            #
            t = np.array(probeOnsetIndex - saccadeOnsetIndices, dtype=float)
            closest = np.nanargmin(np.abs(t))
            latency = (probeOnsetIndex - saccadeOnsetIndices[closest]) / 200

            #
            if latency >= perisaccadicWindow[0] and latency <= perisaccadicWindow[1]:
                perisaccadicProbeCounts[arrayIndex] += 1

    return perisaccadicProbeCounts / saccadeOnsetIndices.size * 100

def simulateRandomProbePresentations(
    homeFolder,
    isiRange=(1, 3),
    timeout=1,
    trailingWindowSize=3,
    perisaccadicWindow=(-0.05, 0.11)
    ):
    """
    """

    eyePosition = getEyePosition(homeFolder)
    saccadeOnsetIndices = getSaccadeOnsetIndices(homeFolder, saccadeDirection=None)
    perisaccadicProbeCount = 0
    timeoutCountdown = 0
    isiCountdown = np.random.randint(round(200 * isiRange[0]), round(200 * isiRange[1]), 1).item()

    for frameIndex in range(trailingWindowSize + 1, eyePosition.shape[0], 1):

        if timeoutCountdown != 0:
            timeoutCountdown -= 1
            continue

        if isiCountdown != 0:
            isiCountdown -= 1
            continue

        #
        projectorInputDelay = np.random.randint(round(200 * 0.03), high=round(200 * 0.045), size=1).item()
        poseEstimationDelay = round(200 * 0.009)
        probeIndex = frameIndex + projectorInputDelay + poseEstimationDelay

        #
        t = np.array(probeIndex - saccadeOnsetIndices, dtype=float)
        closest = np.nanargmin(np.abs(t))
        latency = (probeIndex - saccadeOnsetIndices[closest]) / 200

        #
        if latency >= perisaccadicWindow[0] and latency <= perisaccadicWindow[1]:
            perisaccadicProbeCount += 1
            timeoutCountdown = round(200 * timeout)

        #
        isiCountdown = np.random.randint(round(200 * isiRange[0]), round(200 * isiRange[1]), 1).item()

    return perisaccadicProbeCount / saccadeOnsetIndices.size * 100

def simulateRealtimeExperimentMultiSession(
    rootFolder,
    positionThresholds,
    targetEye='left',
    targetDate=None,
    targetAnimal=None,
    ):
    """
    """

    trialCountsRandom = list()
    trialCountsTriggered = list()
    rootFolderPath = pl.Path(rootFolder)
    for date in rootFolderPath.iterdir():
        if bool(re.search('\d{4}-\d{2}-\d{2}', date.name)) == False:
            continue
        if targetDate != None and date.name != targetDate:
            continue
        for animal in date.iterdir():
            if bool(re.search('pixel\d{1}', animal.name)) == False:
                continue
            if targetAnimal != None and animal.name != targetAnimal:
                continue
            print(f'Info: Working on session from {animal.name} on {date.name}')
            try:
                trialCountsRandom.append(simulateRandomProbePresentations(str(animal)))
                trialCountsTriggered.append(simulateRealtimeExperimentSingleSession(str(animal), positionThresholds, targetEye=targetEye))
            except Exception as error:
                print(f'Warning: Simulation failed for session from {animal.name} on {date.name}')
                continue

    return np.array(trialCountsRandom), np.array(trialCountsTriggered)
