import numpy as np
from scipy.signal import find_peaks

def createPeriEventTimeHistogram(
    session,
    eventTimestamps,
    responseWindow=(-0.3, 0.5),
    baselineWindow=(-8, -5),
    binsize=0.01,
    minimumPeakHeight=2,
    peakLatencyRange=(0.05, 0.3),
    ):
    """
    """

    uMask = session.filterUnits(
        utypes=('vr', 'vm', 'sr'),
        quality=('hq',)
    )
    nUnits = uMask.sum()
    include = np.full(nUnits, False)
    Z = list()
    for iUnit, unit in enumerate(session.population[uMask]):

        #
        t, z = unit.peth(
            eventTimestamps,
            responseWindow=responseWindow,
            baselineWindow=baselineWindow,
            binsize=binsize,
            standardize=True
        )
        Z.append(z)

        # Filtering
        peakIndices, peakProps = find_peaks(z, height=minimumPeakHeight - 0.5)
        if peakIndices.size == 0:
            continue
        firstPeakLatency = t[peakIndices[np.argmin(peakIndices)]]
        if firstPeakLatency < peakLatencyRange[0]:
            continue
        if firstPeakLatency > peakLatencyRange[1]:
            continue

        #
        include[iUnit] = True

    return np.array(Z), include

def summarizeEventRelatedActivity(
    sessions,
    responseWindow=(-0.3, 0.5),
    baselineWindow=(-8, -5),
    binsize=0.01,
    minimumPeakHeight=1.5,
    peakLatencyRangeForVisualResponses=(0.05, 0.2),
    peakLatencyRangeForSaccadeRelatedActivity=(-0.1, 0.2),
    ):
    """
    """

    #
    peakLatencyRanges = (
        peakLatencyRangeForVisualResponses,
        peakLatencyRangeForVisualResponses,
        peakLatencyRangeForSaccadeRelatedActivity,
        peakLatencyRangeForSaccadeRelatedActivity
    )

    #
    peths = {
        ('p', 'l'): list(),
        ('p', 'r'): list(),
        ('s', 'n'): list(),
        ('s', 't'): list(),
    }
    ukeys = list()

    #
    nSessions = len(sessions)
    for iSession, session in enumerate(sessions):

        #
        print(f'Working on session from {session.animal} on {session.date} ({iSession + 1}/{nSessions})')

        #
        if session.probeTimestamps is None:
            continue

        eventTimestamps = (
            session.probeTimestamps[session.filterProbes('es', probeDirections=(-1,))],
            session.probeTimestamps[session.filterProbes('es', probeDirections=(+1,))],
            session.saccadeTimestamps[session.filterSaccades('es', saccadeDirections=('n',))],
            session.saccadeTimestamps[session.filterSaccades('es', saccadeDirections=('t',))]
        )

        #
        heatmaps = list()
        filters = list()

        #
        for evt, plr in zip(eventTimestamps, peakLatencyRanges):
            Z, include = createPeriEventTimeHistogram(
                session,
                evt,
                responseWindow,
                baselineWindow,
                binsize,
                minimumPeakHeight,
                plr,
            )
            heatmaps.append(Z)
            filters.append(include)
        
        #
        filter_ = np.vstack(filters).any(0)

        #
        for iUnit, flag in enumerate(filter_):
            if flag:
                ukey = (
                    str(session.date),
                    session.animal,
                    session.population[iUnit].cluster
                )
                ukeys.append(ukey)

        #
        for Z, evk in zip(heatmaps, peths.keys()):
            for z in Z[filter_]:
                peths[evk].append(z)

    #
    for evk in peths.keys():
        peths[evk] = np.array(peths[evk])

    #
    return peths, ukeys