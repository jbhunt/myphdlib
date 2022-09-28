import os
import numpy as np
import pathlib as pl
from scipy.signal import find_peaks

from myphdlib.suppression1 import constants as const

# Number of samples in a single LabJack dat file
NSAMPLES = 12000

# Number of channels sampled by the LabJack
NCHANNELS = 9

class LabJackError(Exception):
    pass

def readDataFile(dat, line_length_range=(94, 100)):
    """
    Read a single labjack dat file into a numpy array
    """

    # read out the binary file
    with open(dat, 'rb') as stream:
        lines = stream.readlines()

    # Corrupted or empty dat files
    if len(lines) == 0:
        data = np.full((NSAMPLES, NCHANNELS), np.nan)
        return data

    #
    for iline, line in enumerate(lines):
        if line_length_range[0] <= len(line) <= line_length_range[1]:
            break

    # split into header and content
    header  = lines[:iline ]
    content = lines[ iline:]

    # extract data and convert to float or int
    nrows = len(content)
    ncols = len(content[0].decode().rstrip('\r\n').split('\t'))
    shape = (nrows, ncols)
    data = np.zeros(shape)
    for iline, line in enumerate(content):
        if len(line) > line_length_range[1] or len(line) < line_length_range[0]:
            continue
        elements = line.decode().rstrip('\r\n').split('\t')
        elements = [float(el) for el in elements]
        data[iline, :] = elements

    return np.array(data)

def loadLabjackData(labjack_folder, fileNumberRange=(None, None)):
    """
    Concatenate the dat files into a matrix of the shape N samples x N channels
    """

    # determine the correct sequence of files
    files = [
        str(file)
            for file in pl.Path(labjack_folder).iterdir() if file.suffix == '.dat'
    ]
    file_numbers = [int(file.rstrip('.dat').split('_')[-1]) for file in files]
    sort_index = np.argsort(file_numbers)

    # create the matrix
    data = list()
    for ifile in sort_index:
        if fileNumberRange[0] is not None:
            if ifile < fileNumberRange[0]:
                continue
        if fileNumberRange[1] is not None:
            if ifile > fileNumberRange[1]:
                continue
        dat = os.path.join(labjack_folder, files[ifile])
        mat = readDataFile(dat)
        for irow in range(mat.shape[0]):
            data.append(mat[irow,:].tolist())

    #
    return np.array(data)

def extractLabjackEvent(
    data,
    iev=0,
    edge='both',
    analog=False,
    pulseWidthRange=(None, None),
    ):
    """
    """

    #
    event = data[:, iev].flatten()

    # Rescale analog signal and binarize
    if analog:
        m = 1 / (event.max() - event.min())
        b = 0 - m * event.min()
        event = m * event + b
        event[event <  0.1] = 0
        event[event >= 0.1] = 1
        event = event.astype(int)

    # find all edges
    rising = find_peaks(np.diff(event), height=0.9)[0]
    falling = find_peaks(np.diff(event) * -1, height=0.9)[0]
    if rising.size == 0 or falling.size == 0:
        return event, np.array([])
    if rising[0] < falling[0]:
        firstEdge = 'rising'
    else:
        firstEdge = 'falling'
    edges  = np.sort(np.concatenate([rising, falling]))

    #
    exclude = list()
    if pulseWidthRange[0] != None:
        pulseWidths = np.abs(falling - rising)
        for pulseIndex in np.where(pulseWidths < pulseWidthRange[0])[0]:
            exclude.append(pulseIndex)

    if pulseWidthRange[1] != None:
        pulseWidths = np.abs(falling - rising)
        for pulseIndex in np.where(pulseWidths > pulseWidthRange[1])[0]:
            exclude.append(pulseIndex)

    exclude = np.unique(exclude)
    if len(exclude) > 0:
        mask = np.full(edges.size, 1).astype(bool)
        for pulseIndex in exclude:
            mask[round(pulseIndex * 2)] = False
            mask[round(pulseIndex * 2 + 1)] = False
        edges = edges[mask]

    # parse indice by the target edge
    if edge == 'rising':
        if firstEdge == 'rising':
            indices = edges[0::2]
        else:
            indices = edges[1::2]
    elif edge == 'falling':
        if firstEdge == 'rising':
            indices = edges[1::2]
        else:
            indices = edges[0::2]
    elif edge == 'both':
        indices = edges
    else:
        raise ValueError(f'Edge kwarg must be "rising", "falling", or "both"')

    # offset by 1 (this gives you the first index after the state transition)
    indices += 1

    return event, indices

def extractBarcodes(signal, minimumBarcodeInterval=3, bitSize=0.03, labjackSamplingRate=1000):
    """
    """

    # TODO: parameterize the 20 value below
    peaks, props = find_peaks(abs(np.diff(signal)), height=0.5)
    longIntervalIndices = np.where(np.diff(peaks) >= minimumBarcodeInterval * labjackSamplingRate)[0]
    pulseTrains = np.split(peaks, longIntervalIndices + 1)
    pulseTrainOnsetIndices = np.array([
        pulseTrain[0] + 20 for pulseTrain in pulseTrains
    ])

    #
    values = list()
    samplesPerBit = bitSize * labjackSamplingRate
    for pulseTrainOnsetIndex in pulseTrainOnsetIndices:
        pulseTrainOffsetIndex = int(pulseTrainOnsetIndex + samplesPerBit * 32)
        splits = np.split(signal[pulseTrainOnsetIndex: pulseTrainOffsetIndex], 32)
        string = ''.join(
            str(round(split.mean())) for split in splits[::-1]
        )
        value = int(string, 2)
        values.append(value)
    
    return np.array(values)
