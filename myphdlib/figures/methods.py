import h5py
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from myphdlib.general.toolkit import smooth
from sklearn.decomposition import PCA

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

class DataAcqusitionSummaryFigure():
    """
    """

    def generate(
        self,
        session,
        blockIndex=0,
        window=(0, 15),
        figsize=(9, 8)
        ):
        """
        """

        #
        gratingTimestamps = session.load('stimuli/dg/grating/timestamps')
        motionTimestamps = session.load('stimuli/dg/motion/timestamps')
        eyePosition = session.load('pose/filtered')
        t0 = gratingTimestamps[blockIndex] + window[0]
        t1 = motionTimestamps[blockIndex]
        t2 = gratingTimestamps[blockIndex] + window[1]

        #
        fig = plt.figure()
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        gs = GridSpec(12, 1)
        axs = list()
        for i in range(5):
            ax = fig.add_subplot(gs[i])
            axs.append(ax)
        ax = fig.add_subplot(gs[5:], rasterized=True)
        axs.append(ax)

        #
        axs[0].vlines(t1, 0, 1, color='k')

        #
        for columneIndex, eye, coefficient, ax in zip([0, 2], ['left', 'right'], [-1, 1], axs[1:3]):
            frameTimestamps = session.load(f'frames/{eye}/timestamps')
            frameIndices = np.where(
                np.logical_and(
                    frameTimestamps >= t0,
                    frameTimestamps <= t2
                )
            )[0]
            y = smooth(eyePosition[frameIndices, columneIndex] * coefficient, 15)
            t = frameTimestamps[frameIndices]
            ax.plot(t, y, color='k')

        #
        for ax, ev in zip(axs[3:], [session.saccadeTimestamps[:, 0], session.probeTimestamps]):
            eventIndices = np.where(np.logical_and(
                ev >= t0,
                ev <= t2
            ))[0]
            ax.vlines(ev[eventIndices], 0, 1, color='k')

        #
        session.population.filter(
            visualResponseAmplitude=None,
            visualResponseProbability=None,
        )
        for unitIndex, unit in enumerate(session.population):
            spikeIndices = np.where(np.logical_and(
                unit.timestamps >= t0,
                unit.timestamps <= t2
            ))
            axs[-1].vlines(unit.timestamps[spikeIndices], unitIndex + 0.2, unitIndex + 0.8, color='k', lw=1)  

        #
        for ax in axs[:5]:
            ax.set_xticks([])
        for ax in axs:
            xlim = (
                t0 + window[0] - 0.5,
                t0 + window[1] + 0.5
            )
            ax.set_xlim(xlim)
        y1 = np.min([*axs[1].get_ylim(), *axs[2].get_ylim()])
        y2 = np.max([*axs[1].get_ylim(), *axs[2].get_ylim()])
        for ax in axs[1:3]:
            ax.set_ylim([y1, y2])
        for ax in axs:
            for sp in ('left', 'right', 'top', 'bottom'):
                ax.spines[sp].set_visible(False)
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.2)          

        return fig

class PopulationSizesBeforeAndAfterFilteringFigure():
    """
    """

    def __init__(self):
        """
        """

        self.data = None

        return

    def generate(self, sessions, figsize=(2, 5)):
        """
        """

        nSessions = len(sessions)
        nUnitsBefore = np.full(nSessions, np.nan)
        nUnitsAfter = np.full(nSessions, np.nan)

        for sessionIndex, session in enumerate(sessions):
            session.population.unfilter()
            nUnitsBefore[sessionIndex] = session.population.count()
            session.population.filter()
            nUnitsAfter[sessionIndex] = session.population.count()

        #
        self.data = (
            nUnitsBefore,
            nUnitsAfter
        )
        
        #
        fig, ax = plt.subplots()
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        ax.boxplot(
            x=[nUnitsBefore, nUnitsAfter],
            positions=[0, 1],
            widths=0.5,
            medianprops={'color': 'k'},
            showfliers=False
        )
        fig.tight_layout()

        return fig, ax

class PopulationHeatmapBeforeAndAfterFilteringFigure():
    """
    """

    def __init__(self):
        """
        """

        self.data = None
        self.t = None

        return

    def generate(self, session, figsize=(5, 4), window=(-0.3, 0.5)):
        """
        """

        filtered = session.population.filter(
            returnMask=True,
            visualResponseAmplitude=None,
        )
        session.population.unfilter()
        R1, R2 = list(), list()

        #
        for unit, flag in zip(session.population, filtered):
            t, z = unit.peth(
                session.probeTimestamps,
                responseWindow=window,
                baselineWindow=(-1, -0.5),
                binsize=0.02,
                nRuns=30,
            )
            if np.isnan(z).all():
                continue
            R1.append(z)
            if flag:
                R2.append(z)

        #
        self.t = t

        #
        R1, R2 = smooth(np.array(R1), 9, axis=1), smooth(np.array(R2), 9, axis=1)
        self.data = (
            R1,
            R2
        )

        #
        fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True)
        ax1.pcolor(t, np.arange(R1.shape[0]), R1, vmin=-5, vmax=5, cmap='binary_r', rasterized=True)
        ax2.pcolor(t, np.arange(R2.shape[0]), R2, vmin=-5, vmax=5, cmap='binary_r', rasterized=True)

        #
        for ax in (ax1, ax2):
            for sp in ('top', 'right', 'bottom', 'left'):
                ax.spines[sp].set_visible(False)
            ax.set_xticks([])
            ax.set_yticks([])

        #
        ax1.set_xticks([0, 0.2])
        ax1.set_yticks([0, 20])
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, (ax1, ax2)
    
class ClusteringAnalysis():
    """
    """

    def __init__(
        self,
        ):
        """
        """

        self.data = None
        self.labels = None
        self.t = None

        return

    def loadData(
        self,
        hdf,
        baselineWindow=(-0.2, 0),
        responseWindow=(0, 0.3),
        minimumBaselineLevel=0.5,
        minimumResponseAmplitude=1,
        ):
        """
        """

        with h5py.File(hdf, 'r') as stream:
            labels = np.array(stream['unitLabel'])
            rProbeLeft = np.array(stream['rProbe/dg/left'])
            rProbeRight = np.array(stream['rProbe/dg/right'])
            # rProbeNormalized = np.array(stream['rProbe/dg/preferred'])
            # rSaccadeNormalized = np.array(stream['rSaccade/dg/preferred'])
            t = np.array(stream['rProbe/dg/left'].attrs['t'])


        #
        binIndicesForBaselineWindow = np.where(np.logical_and(
            t >= baselineWindow[0],
            t <= baselineWindow[1]
        ))[0]
        binIndicesForResponseWindow = np.where(np.logical_and(
            t >= responseWindow[0],
            t <= responseWindow[1]
        ))[0]

        #
        nUnits = rProbeLeft.shape[0]
        nBins = rProbeLeft.shape[1]
        exclude = np.full(nUnits, False)
        X = np.full([nUnits, nBins], np.nan)
        for iUnit in range(nUnits):

            #
            baselineLevelLeft = rProbeLeft[iUnit, binIndicesForBaselineWindow].mean()
            baselineLevelRight = rProbeRight[iUnit, binIndicesForBaselineWindow].mean()
            lowestBaselineLevel = np.max([
                baselineLevelLeft,
                baselineLevelRight
            ])
            if lowestBaselineLevel < minimumBaselineLevel:
                exclude[iUnit] = True

            #
            peakAmplitudeLeft = np.max(np.abs(rProbeLeft[iUnit, binIndicesForResponseWindow] - baselineLevelLeft))
            peakAmplitudeRight = np.max(np.abs(rProbeRight[iUnit, binIndicesForResponseWindow] - baselineLevelRight))
            greatestPeakAmplitude = np.max([
                peakAmplitudeLeft,
                peakAmplitudeRight
            ])
            if greatestPeakAmplitude < minimumResponseAmplitude:
                exclude[iUnit] = True

            #
            if exclude[iUnit]:
                continue
            if peakAmplitudeLeft > peakAmplitudeRight:
                x = (rProbeLeft[iUnit] - baselineLevelLeft) / peakAmplitudeLeft
            else:
                x = (rProbeRight[iUnit] - baselineLevelRight) / peakAmplitudeRight
            X[iUnit, :] = x

        include = np.logical_and(
            np.invert(exclude),
            np.invert(np.isnan(labels.flatten()))
        )

        # self.data = np.conatenate([rProbeNormalized, rSaccadeNormalized], axis=1)
        self.t = t
        self.data = X[include, :]
        self.labels = labels[include]

        return
    
    def plotResponseSubspaceByCluster(
        self,
        ):
        """
        """

        xy = PCA(n_components=3).fit_transform(self.data)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        colors = np.array([f'C{int(l)}' for l in self.labels])
        ax.scatter(xy[:, 0], xy[:, 1], zs=xy[:, 2], c=colors, s=2, alpha=1)

        return fig, ax
    
    def plotAverageResponsesbyCluster(
        self,
        figsize=(4, 8),
        clusterOrder=(1, 7, 8, 2, 3, 5, 6, 0, 4),
        ):
        """
        """

        nUnits = self.data.shape[0]
        clusterLabels, labelCounts = np.unique(self.labels.flatten(), return_counts=True)
        fig, axs = plt.subplots(nrows=clusterLabels.size, gridspec_kw={'height_ratios':labelCounts})
        for iCluster in range(clusterLabels.size):
            if clusterOrder is None:
                clusterLabel = clusterLabels[iCluster]
                clusterName = f'C{iCluster + 1}'
            else:
                clusterLabel = clusterLabels[np.array(clusterOrder)][iCluster]
                clusterName = f'C{int(clusterLabel + 1)}'
            clusterMask = self.labels.flatten() == clusterLabel
            unitIndices = np.arange(clusterMask.sum())
            np.random.shuffle(unitIndices)
            z = smooth(self.data[clusterMask, :][unitIndices, :], 5, axis=1)
            y = np.arange(clusterMask.sum())
            axs[iCluster].pcolor(self.t, y, z, vmin=-0.85, vmax=0.85)
            axs[iCluster].set_yticks([])
            axs[iCluster].set_ylabel(clusterName, rotation=0, va='center', ha='right')

        #
        for ax in axs[:-1]:
            ax.set_xticks([])
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.15)

        return fig, axs
