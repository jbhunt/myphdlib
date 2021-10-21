import os
import numpy as np
import pathlib as pl
from scipy.signal import find_peaks

from . import constants as const

class LabJackError(Exception):
    pass

def read_data_file(dat, line_length_range=(94, 100)):
    """
    Read a single labjack dat file into a numpy array
    """

    # read out the binary file
    with open(dat, 'rb') as stream:
        lines = stream.readlines()

    if len(lines) == 0:
        name = pl.Path(dat).name
        raise LabJackError(f'LabJack dat file is empty: {dat}')

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

def load_labjack_data(labjack_folder):
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
        dat = os.path.join(labjack_folder, files[ifile])
        mat = read_data_file(dat)
        for irow in range(mat.shape[0]):
            data.append(mat[irow,:].tolist())

    #
    return np.array(data)

def extract_labjack_event(
    data,
    iev=0,
    edge='both',
    analog=False,
    pulse_width_range=None,
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
        event[event <  0.5] = 0
        event[event >= 0.5] = 1
        event = event.astype(int)

    # find all edges
    rising  = find_peaks(np.diff(event), height=0.9)[0]
    falling = find_peaks(np.diff(event) * -1, height=0.9)[0]
    edges   = np.sort(np.concatenate([rising, falling]))

    # parse the pulses for a minimum width (in samples)
    if pulse_width_range is not None:
        minimum_pulse_width, maximum_pulse_width = pulse_width_range
        edges[::2] += 1 # add one to the rising edges
        pulse_widths = np.diff(edges)[::2]
        mask1 = pulse_widths >= minimum_pulse_width * const.SAMPLING_RATE_LABJACK

        #
        if maximum_pulse_width is not None:
            mask2 = pulse_widths <= maximum_pulse_width * const.SAMPLING_RATE_LABJACK
            index = np.where(np.logical_and(mask1, mask2) == True)[0]
        else:
            index = np.where(mask1 == True)[0]

        #
        filtered = list()
        for i in index:
            irising = int(i * 2)
            ifalling = int(i * 2 + 1)
            filtered.append(edges[irising])
            filtered.append(edges[ifalling])
        del edges
        edges = np.array(filtered)
        edges[::2] -= 1 # subtract one from the rising edges

    # parse indice by the target edge
    if edge == 'rising':
        indices = edges[::2]
    elif edge == 'falling':
        indices = edges[1::2]
    elif edge == 'both':
        indices = edges
    else:
        raise ValueError(f'Edge kwarg must be "rising", "falling", or "both"')

    # offset by 1 (this gives you the first index after the state transition)
    indices += 1

    return event, indices7

def parse_stimulus_pulses(data, iev, edge='both', onset_target_size=1120, threshold=0.99):
    """
    """

    # Extract just the rising edges
    lj_stim_sig, lj_stim_idxs = extract_labjack_event(data, iev, edge='rising')

    # Identify the true pulses
    ikeep = np.where(np.diff(lj_stim_idxs) > threshold * const.SAMPLING_RATE_LABJACK)[0] + 1
    if ikeep.size != onset_target_size:
        raise Exception(f'Parsing stimulus pulses failed (target={onset_target_size}, actual={ikeep.size})')

    # Return just the filtered rising edges
    if edge == 'rising':
        return lj_stim_idxs[ikeep]

    # Return the filtered rising and falling edges
    elif edge == 'both':
        lj_stim_sig, lj_stim_idxs = extract_labjack_event(data, iev, edge='both')
        out = list()
        for ipulse in ikeep:
            out.append(lj_stim_idxs[ipulse]) # Onset
            out.append(lj_stim_idxs[ipulse + 1]) # Offset
        return np.array(out)

def parse_sync_pulses(lj_sync_idxs, np_sync_idxs, pulse_width_range=(0.45, 0.55), method=1):
    """
    In the case that there are unequal numbers of sync pulses recorded by NPs
    and LJ, this function identifies the appropriate subset of sync pulses
    """

    # Target number of sync pulses
    target = lj_sync_idxs.size

    # Slices out the target number of pulses starting from the first full length pulse
    if method == 1:
        pulse_width_min, pulse_width_max = pulse_width_range
        diff = np.diff(np_sync_idxs)
        for i in np.arange(diff.size):
            v = diff[i]
            if pulse_width_min <= v / const.NEUROPIXELS_SAMPLING_RATE <= pulse_width_max:
                istart = i + 1
                break

        parsed = np_sync_idxs[istart: istart + target]
        error = abs(np_sync_idxs.size - target) * 0.5

    # Same as method 1 but going backwards starting from the last full length pulse
    elif method == 2:
        pulse_width_min, pulse_width_max = pulse_width_range
        diff = np.diff(np_sync_idxs)
        iend = None
        for i in np.arange(diff.size)[::-1]:
            v = diff[i]
            if pulse_width_min <= v / const.NEUROPIXELS_SAMPLING_RATE <= pulse_width_max:
                iend = i + 1
                break

        if iend is None:
            raise Exception('No endpoint detected')

        parsed = np_sync_idxs[iend - target: iend]
        error = abs(np_sync_idxs.size - target) * 0.5

    #
    elif method == 3:
        pass

    return parsed, error

def correct_labjack_indices(lj_event_idxs, lj_sync_idxs, np_sync_idxs):
    """
    """

    # convert indices to time
    lj_sync_idxs_refd = lj_sync_idxs - lj_sync_idxs[0]
    np_sync_idxs_refd = np_sync_idxs - np_sync_idxs[0]
    lj_sync_time = lj_sync_idxs_refd / const.SAMPLING_RATE_LABJACK
    np_sync_time = np_sync_idxs_refd / const.SAMPLING_RATE_NEUROPIXELS

    # compute the slope of the temporal drift
    signal = np.around((np_sync_time - lj_sync_time) * const.SAMPLING_RATE_LABJACK).astype(np.int64)
    x1, x2 = (lj_sync_idxs_refd[0], lj_sync_idxs_refd[-1])
    y1, y2 = (signal[0], signal[-1])
    slope = (y2 - y1) / (x2 - x1)

    # compute the offset
    lj_event_idxs_refd = lj_event_idxs - lj_event_idxs[0]
    offset = slope * lj_event_idxs_refd

    return np.around(lj_event_idxs + offset).astype(np.int64)

def convert_labjack_indices(lj_event_idxs, lj_sync_idxs, np_sync_idxs, version=1):
    """
    """

    if version == 1:
        corrected = correct_labjack_indices(lj_event_idxs, lj_sync_idxs, np_sync_idxs)
        timestamps = (corrected - lj_sync_idxs[0]) / const.SAMPLING_RATE_LABJACK
        np_event_idxs = np.around(timestamps * const.SAMPLING_RATE_NEUROPIXELS + np_sync_idxs[0]).astype(np.int64)

    elif version == 2:
        np_event_idxs = np.zeros(lj_event_idxs.size)
        for iev, lj_event_idx in enumerate(lj_event_idxs):
            zeroed = lj_sync_idxs - lj_event_idx
            imin = list(zeroed).index(max(zeroed[zeroed < 1]))
            offset = const.SAMPLING_RATE_NEUROPIXELS / const.SAMPLING_RATE_LABJACK * (lj_event_idx - lj_sync_idxs[imin])
            np_event_idx = np_sync_idxs[imin] + offset
            np_event_idxs[iev] = np_event_idx

    return np_event_idxs
