import numpy as np
from decimal import Decimal

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

def estimate_baseline_activity(session, unit, binsize=0.01, exclusion_window=(-1, 1)):
    """
    """

    event_onset_timestamps = np.concatenate([
        session.saccade_onset_timestamps['ipsi'],
        session.saccade_onset_timestamps['contra'],
        session.probe_onset_timestamps['low'],
        session.probe_onset_timestamps['medium'],
        session.probe_onset_timestamps['high']
    ])
    event_onset_timestamps.sort()

    #
    residual = Decimal(str(np.around(unit.timestamps.max(), 3))) % Decimal(str(binsize))
    stop = float(Decimal(str(np.around(unit.timestamps.max(), 3))) + residual)
    bins = np.arange(0, stop + binsize, binsize)
    counts, edges = np.histogram(unit.timestamps, bins)
    pairs = np.array(list(zip(bins[:-1], bins[1:])))

    #

    masks = np.full((event_onset_timestamps.size, pairs.shape[0]), False).astype(bool)
    for irow, event_onset_timestamp in enumerate(event_onset_timestamps):
        lower = event_onset_timestamp + exclusion_window[0]
        upper = event_onset_timestamp + exclusion_window[1]
        mask = np.logical_or(
            pairs[:, 0] <= lower,
            pairs[:, 1] >= upper
        )
        masks[irow] = np.invert(mask)

    filter = np.invert(masks.any(axis=0))

    return counts[filter].mean(), counts[filter].std()
