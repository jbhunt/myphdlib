import os
import sys
import numpy as np
import pathlib as pl
import subprocess as sp
from decimal import Decimal
from scipy.stats import pearsonr

def smooth(a, window_size=5, window_type='hanning', axis=1):
    """
    Smooth array by convolving with a sliding window

    keywords
    --------
    a
        Input array
    window_size
        Size of the smoothing window (in samples)
    window_type
        Type of window to use for smoothing
    axis
        Axis to smooth across for 2D arrays

    returns
    -------
    a2
        Smoothed array
    """

    if window_size < 3:
        raise ValueError('Window size must be odd and greater than or equal to 3')

    if not window_type in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError(f'Invalid window type: {window_type}')

    if window_type == 'flat':
        w = np.ones(window_size,'d')
    else:
        w = eval('np.' + window_type + f'({window_size})')


    # One dimension
    if  a.ndim == 1:

        if a.size < window_size:
            raise ValueError('Input array is smaller than the smoothing window')

        s = np.r_[a[window_size - 1: 0: -1], a, a[-2: -window_size - 1: -1]]
        smoothed = np.convolve(w / w.sum(), s, mode='valid')
        centered = smoothed[int((window_size - 1) / 2): int(-1 * (window_size - 1) / 2)]
        a2 = centered
        return a2

    # Two or more dimensions
    elif a.ndim == 2:

        if a.shape[axis] < window_size:
            raise ValueError(f'Size of input array along the {axis} axis is smaller than the smoothing window')

        a2 = np.full_like(a if axis == 1 else a.T, np.nan)

        # Iterate over rows or columns
        for i, slice in enumerate(a if axis == 1 else a.T):
            s = np.r_[slice[window_size - 1: 0: -1], slice, slice[-2: -window_size - 1: -1]]
            smoothed = np.convolve(w / w.sum(), s, mode='valid')
            centered = smoothed[int((window_size - 1) / 2): int(-1 * (window_size - 1) / 2)]
            a2[i] = centered

        return a2 if axis == 1 else a2.T

    else:
        raise ValueError('Smoothing not supported for arrays of 3 or more dimensions')

def psth(target_events, relative_events, binsize=0.01, window=(-0.5, 1), edges=None, return_relative_times=False):
    """
    Compute the peri-stimulus (event) time histogram

    keywords
    --------
    target_events
        Timestamps (in seconds) for the target event
    relative_events
        Timestamps (in seconds) for the relative event
    binsize
        Binsize (in seconds) for computing each histogram
    window
        Time window (in seconds) for computing each histogram
    edges: list or numpy.ndarray (optional)
        List of bin edges; useful for computing histograms with unequally sized
        bins

    returns
    -------
    edges
        Bin edges (in seconds)
    M
        Counts of the relative event for each target event (i.e. rows) for each
        time bin (i.e. columns)
    """

    # Linearly spaced bins
    if edges is None:

        # Check that the time range is evenly divisible by the binsize
        start, stop = np.around(window, 2)
        residual = (Decimal(str(stop)) - Decimal(str(start))) % Decimal(str(binsize))
        if residual != 0:
            raise ValueError('Time range must be evenly divisible by binsizes')

        # Compute the bin edges
        range = float(Decimal(str(stop)) - Decimal(str(start)))
        nbins = int(range / binsize) + 1
        edges = np.linspace(start, stop, nbins)

        #
        relative_times = list()

        # Create the histogram matrix M
        M = list()
        for iev, target_event in enumerate(target_events):
            referenced = relative_events - target_event
            within_window_mask = np.logical_and(referenced > start, referenced <= stop)
            for relative_time in referenced[within_window_mask]:
                relative_times.append(relative_time)
            counts, edges_ = np.histogram(referenced[within_window_mask], bins=edges)
            M.append(counts)

        M = np.array(M).astype(int)

    # Unequally sized bins
    else:

        # Make sure there is at least 1 bin
        if len(edges) < 2:
            raise ValueError('List of edges must specify at least 1 bin')

        # Compute bin edges and the range of the time window
        left_bin_edges  = edges[:-1]
        right_bin_edges = edges[1:]
        left_most_edge  = left_bin_edges[0]
        right_most_edge = right_bin_edges[-1]

        # Empty matrix
        M = np.zeros((target_events.size, len(edges) - 1), dtype=int)

        # For each target event ...
        for iev, target_event in enumerate(target_events):
            referenced = relative_events - target_event
            within_window_mask = np.logical_and(referenced > left_most_edge, referenced <= right_most_edge)

            # For each target bin ...
            for ibin, (left_bin_edge, right_bin_edge) in enumerate(zip(left_bin_edges, right_bin_edges)):

                # For each relative event in the target window
                for relative_event in referenced[within_window_mask]:

                    # Check if relative event is in the target bin
                    if left_bin_edge < relative_event <= right_bin_edge:
                        M[iev, ibin] += 1

            # Convert edges to numpy array
            if type(edges) != np.ndarray:
                edges = np.array(edges)

    if return_relative_times:
        return edges, np.array(relative_times)

    else:
        return edges, M

def detectThresholdCrossing(a, threshold, timeout=None):
    """
    Determine where a threshold was crossing in a time series (agnostic of
    direction and sign)
    """

    # Find all threshold crossings
    crossings = np.diff(a > threshold, prepend=False)
    crossings[0] = False
    indices = np.argwhere(crossings).flatten()

    # Get rid of crossings that happen in the refractory period
    if timeout != None:
        while True:
            intervals = np.diff(indices)
            sizeBeforeFiltering = indices.size
            for counter, interval in enumerate(intervals):
                if interval <= timeout:
                    indices = np.delete(indices, counter + 1)
                    break
            sizeAfterFiltering = indices.size
            loss = sizeAfterFiltering - sizeBeforeFiltering
            if loss == 0:
                break

    return indices

def interpolate(a, axis=0):
    """
    """

    if len(a.shape) == 1:
        a = a.reshape(1, -1)
    elif len(a.shape) == 2:
        pass
    else:
        raise Exception('Interpolation of arrays with > 2 dimensions not supported')

    M, N = a.shape

    b = np.copy(a)
    if axis == 0:
        for iM in range(M):
            mask = np.isnan(a[iM, :])
            x = np.where(mask)[0]
            xp = np.where(np.invert(mask))[0]
            fp = a[iM, np.invert(mask)]
            estimates = np.interp(x, xp, fp)
            b[iM, mask] = estimates
    elif axis == 1:
        for iN in range(N):
            mask = np.isnan(a[:, iN])
            x = np.where(mask)[0]
            xp = np.where(np.invert(mask))[0]
            fp = a[np.invert(mask), iN]
            estimates = np.interp(x, xp, fp)
            b[mask, iN] = estimates

    if b.shape[0] == 1:
        return b.flatten()
    else:
        return b

def resample(a, fs1, fs2):
    """
    """

    try:
        totalTime = 1 / fs1 * a.size
        N = round(totalTime * fs2)
        x = np.linspace(0, totalTime, N)
        xp = np.linspace(0, totalTime, a.size)
        b = np.interp(x, xp, a)
    except:
        import pdb; pdb.set_trace()

    return b

def inrange(value, lowerBound, upperBound):
    """
    """

    return np.logical_and(value >= lowerBound, value <= upperBound)
