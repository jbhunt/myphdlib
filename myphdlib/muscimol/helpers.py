import re
import numpy as np
import pandas as pd
import pathlib as pl
from scipy.signal import find_peaks as findPeaks
from myphdlib.toolkit import smooth

def detectSaccades(homeFolder, threshold=0.5, minimumInterSaccadeInterval=0.1):
    """
    """

    homeFolderPath = pl.Path(homeFolder)
    condition = 'saline'
    for file in homeFolderPath.iterdir():
        if 'muscimol' in file.name:
            condition = 'muscimol'
    try:
        stimulusMetadataFile = list(homeFolderPath.rglob('*visual-stimuli-metadata*')).pop()
        leftEyePositionNumpyFile = list(homeFolderPath.rglob('*left-pupil-coords*')).pop()
    except:
        return condition, None, None
    leftEyePosition = np.load(leftEyePositionNumpyFile)
    mu = np.nanmean(leftEyePosition[:, 0])
    sigma = np.nanstd(leftEyePosition[:, 0])
    with open(str(stimulusMetadataFile), 'r') as stream:
        gratingOnsetIndices, gratingOffsetIndices = list(), list()
        lines = stream.readlines()
        print(len(lines))
        if len(lines) != 170:
            return condition, None, None
        for lineIndex, line in enumerate(lines):
            stimulusType, frameIndex = line.rstrip('\n').split(', ')
            if stimulusType == 'grating':
                gratingOnsetIndices.append(int(frameIndex))
                gratingOffsetIndices.append(
                    int(lines[lineIndex + 1].rstrip('\n').split(', ')[-1])
                )
        gratingIndices = np.array(list(zip(gratingOnsetIndices, gratingOffsetIndices)))

    #
    ipsi = True
    ipsiSaccadeWaveforms = list()
    contraSaccadeWaveforms = list()
    frameCount = round(120 * 0.5) * 2 + 1
    for gratingOnsetIndex, gratingOffsetIndices in gratingIndices:
        ipsi = not ipsi
        position = leftEyePosition[gratingOnsetIndex: gratingOffsetIndices, 0]
        velocity = smooth(np.diff(position) ** 2, 21)
        peaks, props = findPeaks(velocity, height=threshold, distance=round(120 * minimumInterSaccadeInterval))
        for peak in peaks:
            saccadeWaveform = position[peak - round(120 * 0.5): peak + 1 + round(120 * 0.5)]
            saccadeWaveformStandardized = (saccadeWaveform - mu) / sigma
            saccadeWaveformZeroed = saccadeWaveformStandardized - np.nanmean(saccadeWaveformStandardized[50:60])
            if len(saccadeWaveform) != frameCount:
                continue
            if ipsi:
                ipsiSaccadeWaveforms.append(saccadeWaveformZeroed)
            else:
                contraSaccadeWaveforms.append(saccadeWaveformZeroed)

    return condition, np.array(ipsiSaccadeWaveforms), np.array(contraSaccadeWaveforms)

def collectSaccadeWaveforms(rootFolder, targetAnimal='musc4'):
    """
    """

    multiSessionData = {
        'saline': {
            'ipsi': list(),
            'contra': list(),
        },
        'muscimol': {
            'ipsi': list(),
            'contra': list()
        }
    }
    rootFolderPath = pl.Path(rootFolder)
    for folder in rootFolderPath.iterdir():
        if bool(re.search('musc\d{1}', folder.name)) == False:
            continue
        if folder.name != targetAnimal:
            continue
        for homeFolder in folder.iterdir():
            condition, ipsi, contra = detectSaccades(str(homeFolder))
            if type(ipsi) != np.ndarray:
                continue
            for saccade in ipsi:
                multiSessionData[condition]['ipsi'].append(saccade)
            for saccade in contra:
                multiSessionData[condition]['contra'].append(saccade)
    
    return multiSessionData

def measureSaccadeAmpitude(rootFolder, targetAnimal='musc4'):
    """
    """

    data = {
        'ipsi': {
            'saline': list(),
            'muscimol': list(),
        },
        'contra': {
            'saline': list(),
            'muscimol': list()
        }
    }

    rootFolderPath = pl.Path(rootFolder)
    for folder in rootFolderPath.iterdir():
        if bool(re.search('musc\d{1}', folder.name)) == False:
            continue
        if folder.name != targetAnimal:
            continue
        for homeFolder in folder.iterdir():
            condition, ipsi, contra = detectSaccades(str(homeFolder))
            if type(ipsi) != np.ndarray:
                continue
            sample = list()
            for wave in ipsi:
                start = np.nanmean(wave[50: 60])
                end = np.nanmean(wave[62: 72])
                amp = abs(end - start)
                sample.append(amp)
            data['ipsi'][condition].append(sample)
            sample = list()
            for wave in contra:
                start = np.nanmean(wave[50: 60])
                end = np.nanmean(wave[62: 72])
                amp = abs(end - start)
                sample.append(amp)
            data['contra'][condition].append(sample)  

    return data       


def getDosage(rootFolder, targetAnimal, targetDate):
    """
    """

    rootFolderPath = pl.Path(rootFolder)
    logFilePath = list(rootFolderPath.rglob('*log.xlsx')).pop()
    frame = pd.read_excel(str(logFilePath), sheet_name=targetAnimal)
    rowIndex = np.where(frame['date'] == targetDate)[0].item()
    dosage = frame['dosage (mg/mL)'][rowIndex]

    return dosage

def computeDoseResponse(rootFolder, targetAnimal='musc4', endpointWindow=(65, 75)):
    """
    """

    data = {'ipsi': {0: list()}, 'contra': {0: list()}}
    rootFolderPath = pl.Path(rootFolder)
    for folder in rootFolderPath.iterdir():
        if bool(re.search('musc\d{1}', folder.name)) == False:
            continue
        if folder.name != targetAnimal:
            continue
        if folder == None:
            continue
        for homeFolder in folder.iterdir():
            condition, ipsi, contra = detectSaccades(str(homeFolder))
            if type(ipsi) != np.ndarray:
                continue
            ipsiSaccadeEndpoints = np.nanmean(ipsi[:, endpointWindow[0]: endpointWindow[1]], axis=1).flatten()
            contraSaccadeEndpoints = np.nanmean(contra[:, endpointWindow[0]: endpointWindow[1]], axis=1).flatten()
            if condition == 'muscimol':
                dosage = getDosage(rootFolder, targetAnimal, targetDate=homeFolder.name)
                if dosage not in data['ipsi'].keys():
                    data['ipsi'][dosage] = list()
                    data['contra'][dosage] = list()
            else:
                dosage = 0
            for endpoint in ipsiSaccadeEndpoints:
                data['ipsi'][dosage].append(endpoint)
            for endpoint in contraSaccadeEndpoints:
                data['contra'][dosage].append(endpoint)

    return data
