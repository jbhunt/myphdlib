import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from myphdlib.general.toolkit import smooth
from sklearn.decomposition import PCA

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

class SaccadeFrequencyAnalysis():

    def __init__(self):
        """
        """

        self.data = None

        return

    def measureSaccadeFrequency(
        self,
        sessions,
        ):
        """
        """

        saccadeFrequency = {
            'left': list(),
            'right': list(),
        }
        for gm, k in zip([-1, 1], ['left', 'right']):
            for session in sessions:
                if session.probeTimestamps is None:
                    continue
                motionOnsetTimestamps = session.load('stimuli/dg/motion/timestamps')
                motionOffsetTimestamps = session.load('stimuli/dg/iti/timestamps')
                gratingMotionByBlock = session.load('stimuli/dg/grating/motion')
                motionEpochs = np.vstack([
                    motionOnsetTimestamps,
                    motionOffsetTimestamps,
                ]).T
                duration = 0
                count = 0
                sample = list()
                for t1, t2 in motionEpochs[gratingMotionByBlock == gm, :]:

                    if np.isnan([t1, t2]).any():
                        continue

                    n = np.sum(np.logical_and(
                        session.saccadeTimestamps[:, 0] >= t1,
                        session.saccadeTimestamps[:, 0] <= t2
                    ))
                    dt = t2 - t1
                    count += n
                    duration += dt
                    sample.append(n / dt)
                saccadeFrequency[k].append(round(np.mean(sample), 2))

        #
        for k in saccadeFrequency:
            saccadeFrequency[k] = np.array(saccadeFrequency[k])

        self.data = saccadeFrequency

        return

    def plotFrequencyDistribution(
        self,
        gratingMotion='left',
        fRange=(0, 1),
        nBins=11,
        plot='line',
        figsize=(3, 2)
        ):
        """
        """

        fig, ax = plt.subplots()
        sample = self.data[gratingMotion]
        if plot == 'hist':
            ax.hist(
                sample,
                range=fRange,
                bins=nBins,
                color='k',
                alpha=0.1
            )
            ax.hist(
                sample,
                range=fRange,
                bins=nBins,
                color='k',
                histtype='step'
            )
        elif plot == 'line':
            counts, edges = np.histogram(
                sample,
                range=fRange,
                bins=nBins
            )
            x = edges[:-1] + (edges[1] - edges[0]) / 2
            ax.plot(x, counts, color='k', alpha=0.3)
            ax.fill_between(
                x,
                0,
                counts,
                color='k',
                alpha=0.1
            )

        ax.set_xlabel('Saccade frequency (Hz)')
        ax.set_ylabel('Number of sessions')
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, ax

class TrialFrequencyAnalysis():

    def __init__(self):
        """
        """

        self.data = None

        return

    def measureTrialFrequency(
        self,
        sessions,
        perisaccadicWindow=(-0.05, 0.1),
        ):
        """
        """

        self.data = {
            'f': list(),
            'n': list()
        }
        for session in sessions:
            if session.probeTimestamps is None:
                continue
            n = np.sum(np.logical_and(
                session.probeLatencies >= perisaccadicWindow[0],
                session.probeLatencies <= perisaccadicWindow[1]
            ))
            f = n / session.probeTimestamps.size * 100
            self.data['f'].append(f)
            self.data['n'].append(n)
        self.data = {
            'f': np.array(self.data['f']),
            'n': np.array(self.data['n'])
        }

        return

    def plotFrequencyDistribution(
        self,
        plot='line',
        fRange=(0, 15),
        nBins=15,
        figsize=(3, 2)
        ):
        """
        """

        fig, ax = plt.subplots()
        if plot == 'line':
            counts, edges = np.histogram(
                self.data['f'],
                range=fRange,
                bins=nBins
            )
            x = edges[:-1] + (edges[1] - edges[0]) / 2
            ax.plot(x, counts, color='k', alpha=0.3)
            ax.fill_between(
                x,
                0,
                counts,
                color='k',
                alpha=0.1
            )

        #
        ax.set_xlabel('Peri-saccadic trial rate (%)')
        ax.set_ylabel('Number of sessions')
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, ax

class SaccadeClassificationPerformanceAnalysis():
    
    def __init__(self):
        """
        """

        self.data = None

        return

    def measureClassificationPerformance(self, sessions):
        """
        """

        X = list()
        labels = list()
        for session in sessions:

            #
            saccadeLabels = session.load(f'saccades/predicted/{session.eye}/labels')
            saccadeWaveformsPredicted = session.load(f'saccades/predicted/{session.eye}/waveforms')
            saccadeWaveformsUnlabeled = session.load(f'saccades/putative/{session.eye}/waveforms')

            #
            for wf, l in zip(saccadeWaveformsPredicted, saccadeLabels):
                X.append(wf)
                labels.append(l)

            #
            # for wf in saccadeWaveformsUnlabeled:
            #     result = np.any(np.all(np.isin(saccadeWaveformsPredicted, wf), axis=1))
            #     if result:
            #         continue
            #     X.append(wf)
            #     labels.append(0)

        self.data = {
            'X': np.array(X),
            'y': np.array(labels),
            'xy': PCA(n_components=2).fit_transform(X)
        }

        return

    def plotClusteringPerformance(
        self,
        figsize=(4, 4),
        factor=0.9,
        colors=('k', 'k'),
        examples=(18571, 30060)
        ):
        """
        """

        fig, ax = plt.subplots()
        markers = ('D', 'D')
        for l, c, m in zip(np.unique(self.data['y']), colors, markers):
            mask = self.data['y'] == l
            ax.scatter(
                self.data['xy'][mask, 0],
                self.data['xy'][mask, 1],
                color=c,
                s=3,
                alpha=0.05,
                marker='.',
                rasterized=True
            )
            xc = self.data['xy'][mask, 0].mean()
            yc = self.data['xy'][mask, 1].mean()
            # ax.scatter(xc, yc, marker='+', color='w', s=100)

        #
        if examples is not None:
            for i, c in zip(examples, colors):
                ax.scatter(
                    self.data['xy'][i, 0],
                    self.data['xy'][i, 1],
                    marker='D',
                    edgecolor='w',
                    facecolor=c
                )
        
        #
        xmax = np.max(np.abs(self.data['xy'][:, 0]))
        ymax = np.max(np.abs(self.data['xy'][:, 1]))
        ax.set_xlim([-xmax * factor, xmax * factor])
        ax.set_ylim([-ymax * factor, ymax * factor])
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_aspect('equal')
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, ax
