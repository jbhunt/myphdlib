import os
import re
import numpy as np
import pathlib as pl

# TODO: Code this
def _determineMatrixShape(dat):
    """
    """

    return 12000, 9

def _readDataFile(dat):
    """
    Read a single labjack dat file into a numpy array
    """

    #
    nSamples, nChannels = _determineMatrixShape(dat)

    # read out the binary file
    with open(dat, 'rb') as stream:
        lines = stream.readlines()

    # Corrupted or empty dat files
    if len(lines) == 0:
        data = np.full((nSamples, nChannels), np.nan)
        return data

    #
    for iline, line in enumerate(lines):
        if bool(re.search('.*\t.*\t.*\t.*\t.*\t.*\t.*\r\n', line.decode())) and line.decode().startswith('Time') == False:
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
        elements = line.decode().rstrip('\r\n').split('\t')
        elements = [float(el) for el in elements]
        data[iline, :] = elements

    return np.array(data)

def _loadLabjackMatrix(labjack_folder, fileNumberRange=(None, None)):
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
        mat = _readDataFile(dat)
        for irow in range(mat.shape[0]):
            data.append(mat[irow,:].tolist())

    #
    return np.array(data)

def extractLabjackEvents(
    session,
    pulseWidthRange=(0.002, np.inf)
    ):
    """
    """

    if session.folders.labjack is None:
        raise Exception('Coulde not locate labjack folder')

    return

def estimateTimestampingFunction(
    session
    ):
    """
    """

    return