import numpy as np
from matplotlib import pyplot as plt
from myphdlib.general.toolkit import smooth

def measureSaccadeFrequencyDuringGratingMotion(
    session,
    ):
    """
    """

    motionOnsetTimestamps = session.load('stimuli/dg/motion/timestamps')
    motionOffsetTimestamps = session.load('stimuli/dg/iti/timestamps')
    motionEpochs = np.vstack([
        motionOnsetTimestamps,
        motionOffsetTimestamps,
    ]).T

    #
    intervals = list()
    frequency = list()

    #
    for start, stop in motionEpochs:
        saccadeIndices = np.where(np.logical_and(
            session.saccadeTimestamps >= start,
            session.saccadeTimestamps <= stop
        ))[0]
        n = saccadeIndices.size
        dt = stop - start
        f = n / dt
        frequency.append(f)
        for isi in np.diff(session.saccadeTimestamps[saccadeIndices]):
            intervals.append(isi)


    return np.array(frequency), np.array(intervals)

def plotInterSaccadeIntervalDistributions(
    sessions,
    animals=('mlati6', 'mlati7', 'mlati9', 'mlati10'),
    ):
    """
    """

    fig, ax = plt.subplots()
    cmap = plt.get_cmap('tab10')
    curves = list()
    for i, animal in enumerate(animals):
        alias = f'A{i + 1}'
        counts = list()
        for session in sessions:
            if session.animal != animal:
                continue
            if session.probeTimestamps is None:
                continue
            f, isi = measureSaccadeFrequencyDuringGratingMotion(
                session
            )
            y, x = np.histogram(
                isi,
                bins=50,
                range=(0, 10),
            )
            t = x[:-1] + 0.1
            counts.append(y / isi.size)
            # ax.plot(t, y / isi.size, color=cmap(i), alpha=0.1)

        #
        mu = np.mean(counts, axis=0)
        sd = np.std(counts, axis=0)
        ax.plot(t, mu, color=cmap(i), label=alias, alpha=0.5)
        # ax.fill_between(t, mu - sd, mu + sd, color=cmap(i), alpha=0.2)
        curves.append(mu)

    #
    ax.plot(t, np.mean(curves, axis=0), color='k')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('P(ISI)')
    ax.legend()

    return fig

def measurePerisaccadicTrialFrequency(session):
    """
    """

    data = {
        ('f1', 'left'): None,
        ('f1', 'right'): None,
        ('f2', 'left'): None,
        ('f2', 'right'): None
    }
    for probeMotion in (-1, 1):
        probeDirection = 'left' if probeMotion == -1 else 'right'
        nTrialsPerisaccadic = np.sum(session.filterProbes(
            trialType='ps',
            probeDirections=(probeMotion,)
        ))
        nTrialsExtrasaccadic = np.sum(session.filterProbes(
            trialType=None,
            probeDirections=(probeMotion,)
        ))
        data[('f1', probeDirection)] = round(nTrialsPerisaccadic / nTrialsExtrasaccadic, 2)
        data[('f2', probeDirection)] = nTrialsPerisaccadic

    return data

def plotTrialFrequencyByTrialType(
    sessions,
    animals=('mlati6', 'mlati7', 'mlati9', 'mlati10'),
    ):
    """
    """

    fig, ax = plt.subplots()
    cmap = plt.get_cmap('tab10')
    X1, X2 = list(), list()
    for i, animal in enumerate(animals):
        alias = f'A{i + 1}'
        x1, x2 = list(), list()
        for session in sessions:
            if session.animal != animal:
                continue
            if session.probeTimestamps is None:
                continue
            data = measurePerisaccadicTrialFrequency(session)
            x1.append(data[('f2', 'left')])
            x2.append(data[('f2', 'right')])
        
        #
        kwargs = {
            'boxprops': {'color': cmap(i), 'alpha': 0.5, 'lw': 1.5},
            'medianprops': {'color': cmap(i), 'alpha': 0.5, 'lw': 1.5},
            'capprops': {'color': cmap(i), 'alpha': 0.5, 'lw': 1.5},
            'whiskerprops': {'color': cmap(i), 'alpha': 0.5, 'lw': 1.5},
            'showfliers': False,
            'widths': [0.3]
        }
        ax.boxplot(
            x1,
            positions=[i],
            vert=False,
            **kwargs
        )
        ax.boxplot(
            x2,
            positions=[i + 4],
            vert=False,
            **kwargs
        )
        X1.append(np.mean(x1))
        X2.append(np.mean(x2))

    #
    xl = ax.get_xlim()
    y1, y2 = ax.get_ylim()
    ax.hlines(3.5, *xl, color='gray')
    ax.vlines(np.mean(X1), 3.5, y2, color='k')
    ax.vlines(np.mean(X2), y1, 3.5, color='k')
    ax.set_xlim(xl)
    ax.set_ylim([y1, y2])
    ax.set_yticks(range(len(animals)))
    ax.set_yticklabels([f'A{i}' for i in range(len(animals))])
    ax.set_xlabel('# of peri-saccadic trials/session')

    return fig

def plotEventTimingFigure(
    session,
    epochIndex=8
    ):
    """
    """

    #
    gratingTimestamps = session.load('stimuli/dg/grating/timestamps')
    gratingMotionDirections = session.load('stimuli/dg/grating/motion')
    gratingMotionTimestamps = session.load('stimuli/dg/motion/timestamps')
    itiTimestamps = session.load('stimuli/dg/iti/timestamps')
    eyePosition = session.load('pose/filtered')

    #
    epochs = list()
    for i in np.where(np.diff(gratingMotionDirections) != 0)[0]:
        epoch = (
            gratingTimestamps[i],
            gratingTimestamps[i + 2]
        )
        epochs.append(epoch)
    epoch = epochs[epochIndex]

    #
    fig, axs = plt.subplots(nrows=6, sharex=True, sharey=False)

    #
    xs, ys = [epoch[0]], [0, 0, 1, 1, 0, 0, -1, -1, 0]
    mask1 = np.logical_and(
        gratingMotionTimestamps > epoch[0],
        gratingMotionTimestamps < epoch[1]
    )
    mask2 = np.logical_and(
        itiTimestamps > epoch[0],
        itiTimestamps < epoch[1]
    )
    events = np.concatenate([
        gratingMotionTimestamps[mask1],
        itiTimestamps[mask2]
    ])
    events.sort()
    state = 0
    for i, t in enumerate(events):
        xs.append(t)
        xs.append(t)
    xs.append(epoch[1])
    ys.append(False)
    axs[0].plot(xs, ys, color='k')

    #
    for i, eye in enumerate(['left', 'right']):
        if eye == 'left':
            y = eyePosition[:, 0] * -1
        else:
            y = eyePosition[:, 2]
        t = session.load(f'frames/{eye}/timestamps')
        frameIndices = np.logical_and(
            t >= epoch[0],
            t <= epoch[1]
        )
        axs[i + 1].plot(t[frameIndices], smooth(y[frameIndices], 31), color='k')

    #
    for i, eye in enumerate(['left', 'right']):
        ax = axs[i + 3]
        mask = np.logical_and(
            session.saccadeTimestamps >= epoch[0],
            session.saccadeTimestamps <= epoch[1]
        )
        timestamps = session.saccadeTimestamps[mask]
        ax.vlines(timestamps, 0, 1, color='k', alpha=0.15)
        mask2 = session.filterSaccades('ps')
        mask3 = np.logical_and(mask, mask2)
        timestamps = session.saccadeTimestamps[mask3]
        ax.vlines(timestamps, 0, 1, color='k')

    #
    ax = axs[5]
    mask = np.logical_and(
        session.probeTimestamps >= epoch[0],
        session.probeTimestamps <= epoch[1]
    )
    ax.vlines(
        session.probeTimestamps[mask],
        0,
        1,
        color='k',
        alpha=0.15
    )
    ax.hlines(0, epoch[0], epoch[1], color='k', alpha=0.15)

    mask2 = session.filterProbes('ps')
    mask3 = np.logical_and(mask, mask2)
    ax.vlines(
        session.probeTimestamps[mask3],
        0,
        1,
        color='k'
    )

    return fig