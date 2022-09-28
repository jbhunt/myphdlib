import numpy as np
from decimal import Decimal
from scipy.interpolate import CubicSpline
from .. import toolkit as tk

def create_coincidence_mask(target_event_timestamps, relative_event_timestamps, window=(-0.05, 0.65)):
    """
    Create a mask for events which fall within a certain window of another event
    """

    coincidence_mask = list()

    for target_event_timestamp in target_event_timestamps:
        start = target_event_timestamp + window[0]
        stop = target_event_timestamp + window[1]

        coincidence = False
        for relative_event_timestamp in relative_event_timestamps:
            if start <= relative_event_timestamp <= stop:
                coincidence = True
                break

        coincidence_mask.append(coincidence)

    return np.array(coincidence_mask)

def coincident(target_event_timestamps, relative_event_timestamps, coincidence=1):
    """
    Return a mask which defines coincident events
    """

    mask = list()

    for target_event_timestamp in target_event_timestamps:

        #
        start = target_event_timestamp - coincidence
        stop = target_event_timestamp + coincidence

        flag = False
        for relative_event_timestamp in relative_event_timestamps:
            if start <= relative_event_timestamp <= stop:
                flag = True
                break

        mask.append(flag)

    return np.array(mask)

def coincident2(timestamps1, timestamps2, coincidence=(-1, 1)):
    """
    Return a mask which defines coincident events
    """

    mask = list()

    for ts1 in timestamps1:

        #
        start, stop = ts1 + np.array(coincidence)

        flag = False
        for ts2 in timestamps2:
            if start <= ts2 <= stop:
                flag = True
                break

        mask.append(flag)

    return np.array(mask)

def estimate_baseline_activity(session, unit, binsize=0.01, window=(-1, -0.5), stable_spiking_epochs=False):
    """
    """

    #
    if stable_spiking_epochs:
        epochs, stable = unit.load_stable_spiking_epochs()
        event_onset_timestamps = np.concatenate([
            stable['saccades']['ipsi'],
            stable['saccades']['contra'],
            stable['probes']['low'],
            stable['probes']['medium'],
            stable['probes']['high']
        ])

    #
    else:
        event_onset_timestamps = np.concatenate([
            session.saccade_onset_timestamps['ipsi'],
            session.saccade_onset_timestamps['contra'],
            session.probe_onset_timestamps['low'],
            session.probe_onset_timestamps['medium'],
            session.probe_onset_timestamps['high'],
        ])

    #
    event_onset_timestamps.sort()

    #
    edges, M = tk.psth(
        event_onset_timestamps,
        unit.timestamps,
        window=window,
        binsize=binsize
    )

    mu, sigma = M.flatten().mean() / binsize, M.flatten().std() / binsize

    return mu, sigma

def extract_response_amplitude(profile, response_window_mask, metric='peak'):
    """
    Extract the amplitude of a visual or motor response given the activity
    profile and a mask which slices out the response window
    """

    # Peak firing rate
    if metric == 'peak':
        n = profile[response_window_mask].size
        spline = CubicSpline(np.arange(n), profile[response_window_mask])
        interpolated = spline(np.linspace(0, n - 1, 100))
        amplitude = interpolated.max()

    #
    elif metric == 'mean':
        nbins = response_window_mask.size
        amplitude = profile[response_window_mask].sum() / nbins

    # Average firing rate across the response window
    elif metric == 'count':
        amplitude = profile[response_window_mask].sum()

    # Area under the curve of the response
    elif metric == 'auc':
        amplitude = np.trapz(profile[response_window_mask])

    return amplitude

def bin_and_average_data(data, around_saccade_window=(-1.5, 1.5), nbins=14, ndigits=3):
    """
    """
    binsize = np.around((around_saccade_window[1] - around_saccade_window[0]) / nbins, ndigits).item()
    edges = np.around(np.linspace(around_saccade_window[0], around_saccade_window[1] - binsize, nbins), ndigits)
    times = edges + binsize / 2
    heights, points = list(), list()
    for start in edges:
        stop = np.around(start + binsize, ndigits).item()
        mask = np.logical_and(
            data[:, 0] >= start,
            data[:, 0] <  stop,
        )
        heights.append(data[mask, 1].mean())
        points.append(data[mask, 1].tolist())

    return times, heights, points, binsize

def parse_visual_probes(session):
    """
    """

    result = {
        'ipsi':   {level: list() for level in ['low', 'medium', 'high']},
        'contra': {level: list() for level in ['low', 'medium', 'high']}
    }

    for level, probe_onset_timestamps in session.probe_onset_timestamps.items():
        for probe_onset_timestamp in probe_onset_timestamps:
            for direction in ['ipsi', 'contra']:
                for start, stop in zip(session.grating_motion_timestamps[direction], session.grating_offset_timestamps[direction]):
                    if start <= probe_onset_timestamp <= stop:
                        result[direction][level].append(probe_onset_timestamp)

    return result

import mplcursors
import matplotlib.pylab as plt

def score_stable_spiking_epochs(session, binsize=1, hanning_window_length=31, data=None):
    """
    """

    def annotate(sel):
        """
        """

        x, y = sel.target
        ylim = ax.get_ylim()
        l = ax.axvline(x, *ylim, color='r', alpha=0.8)
        sel.extras.append(l)
        ax.set_ylim(ylim)
        sel.annotation.set_text(f'{x:.2f} seconds')

    #
    fig, ax = plt.subplots()

    #
    if data is None:
        data = dict()

    #
    for unit in session.population:

        #
        if unit.uid in data.keys():
            continue

        #
        tmax = unit.timestamps.max()
        bins = np.arange(0, tmax, binsize)
        counts, edges = np.histogram(
            unit.timestamps,
            bins
        )

        #
        x = binsize / 2 + edges[:-1]
        y = tk.smooth(counts / binsize, hanning_window_length)
        ax.plot(x, y, color='k', alpha=0.8)

        #
        cursor = mplcursors.cursor(hover=True)
        cursor.connect('add', annotate)
        plt.show()

        #
        epochs = list()
        while True:
            item = input('Input: ')
            if item == 'done':
                break
            if item == 'skip':
                epochs.append([0, unit.timestamps.max()])
                break
            start, stop = item.split(', ')
            if start == 'start':
                start = unit.timestamps.min()
            if stop == 'end':
                stop = unit.timestamps.max()
            start, stop = map(float, [start, stop])
            epochs.append([start, stop])

        #
        data[unit.uid] = epochs
        ax.cla()

    return data
