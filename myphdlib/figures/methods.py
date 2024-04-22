import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from myphdlib.general.toolkit import smooth
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter as gaussianFilter
from scipy.ndimage import zoom
from scipy.interpolate import interp2d
import cv2 as cv
import pickle
from skimage.measure import find_contours
from scipy.signal import find_peaks
from myphdlib.figures.analysis import AnalysisBase

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

class PerisaccadicTrialFrequencyAnalysis(AnalysisBase):
    """
    """

    def __init__(self, **kwargs):
        """
        """

        super().__init__(**kwargs)
        self.saccadeFrequency = {
            'left': None,
            'right': None,
        }
        self.trialFrequency = {
            'f': None,
            'n': None
        }

        return

    def measureSaccadeFrequency(
        self,
        ):
        """
        """

        saccadeFrequency = {
            'left': list(),
            'right': list(),
        }
        for gm, k in zip([-1, 1], ['left', 'right']):
            for session in self.sessions:
                if session.probeTimestamps is None:
                    saccadeFrequency['left'].append(np.nan)
                    saccadeFrequency['right'].append(np.nan)
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

        self.saccadeFrequency = saccadeFrequency

        return

    def measureTrialFrequency(
        self,
        perisaccadicWindow=(-0.05, 0.1),
        ):
        """
        """

        self.trialFrequency = {
            'f': list(),
            'n': list()
        }
        for session in self.sessions:
            if session.probeTimestamps is None:
                self.trialFrequency['f'].append(np.nan)
                self.trialFrequency['n'].append(np.nan)
                continue
            n = np.sum(np.logical_and(
                session.probeLatencies >= perisaccadicWindow[0],
                session.probeLatencies <= perisaccadicWindow[1]
            ))
            f = n / session.probeTimestamps.size * 100
            self.trialFrequency['f'].append(f)
            self.trialFrequency['n'].append(n)
        for k in self.trialFrequency.keys():
            self.trialFrequency[k] = np.array(self.trialFrequency[k])

        return

    def run(
        self,
        ):
        """
        """

        self.measureSaccadeFrequency(),
        self.measureTrialFrequency()

        return

    def plotSaccadeFrequencyByTrialFrequency(
        self,
        figsize=(3, 3),
        **kwargs_
        ):
        """
        """

        kwargs = {
            'ec': 'none',
            'fc': 'k',
            'marker': 'o',
            'alpha': 1,
            's': 15,
            'lw': 1
        }
        kwargs.update(kwargs_)

        fig, ax = plt.subplots()
        for i in range(len(self.sessions)):
            x = np.mean([
                self.saccadeFrequency['left'][i],
                self.saccadeFrequency['right'][i]
            ])
            y = self.trialFrequency['n'][i]
            ax.scatter(x, y, **kwargs)

        #
        ax.set_xlim([0, ax.get_xlim()[1]])
        ax.set_ylim([0, ax.get_ylim()[1]])
        ax.set_xlabel('Saccade frequency (Hz)')
        ax.set_ylabel('# of peri-saccadic trials')
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

class ReceptiveFieldMappingDemonstrationFigure(AnalysisBase):
    """
    """

    def __init__(
        self,
        date='2023-07-05'
        ):
        """
        """
        super().__init__()
        self.x = None
        self.y = None
        self.date = date
        self.contours = None
        self.heatmaps = None
        return

    def extractContours(
        self,
        threshold=2,
        phase='on',
        smoothingKernelWidth=0.5,
        interpolationFactor=5,
        minimumContourArea=200,
        ):
        """
        """

        #
        self.contours = list()
        self.heatmaps = list()

        # Set the session by date
        for ukey in self.ukeys:
            if ukey[0] == self.date:
                self.ukey = ukey
                break

        #
        heatmaps = self.session.load(f'population/rf/{phase}')
        if heatmaps is None:
            return
        heatmapsFiltered = list()
        for ukey in self.ukeys:
            if ukey[0] != self.date:
                continue
            self.ukey = ukey
            hm = heatmaps[self.unit.index]
            if smoothingKernelWidth is not None:
                hm = gaussianFilter(hm, smoothingKernelWidth)
            if hm.max() < threshold:
                continue
            heatmapsFiltered.append(hm)

        # Interpolate
        heatmapsInterpolated = list()
        for hm in heatmapsFiltered:
            z = zoom(hm, interpolationFactor, grid_mode=False)
            heatmapsInterpolated.append(z)

        #
        nRows, nCols = heatmaps[0].shape
        self.xp = {
            'x': np.linspace(
                0.5,
                nCols * interpolationFactor + 0.5,
                nCols
            ),
            'y': np.linspace(
                0.5,
                nRows * interpolationFactor + 0.5,
                nRows
            ),
        }

        # Detect contours
        self.contours = list()
        for hm in heatmapsInterpolated:

            #
            grayscale = cv.threshold(hm, threshold, 255, cv.THRESH_BINARY)[-1].astype(np.uint8)
            contours, hierarchy = cv.findContours(
                grayscale,
                mode=cv.RETR_TREE,
                method=cv.CHAIN_APPROX_NONE
            )
            if len(contours) == 0:
                continue
            areas = np.array([cv.contourArea(contour) for contour in contours])
            index = np.argmax(areas)
            contour = contours[index]
            if areas[index] < minimumContourArea:
                continue
            self.contours.append(contour)

        #
        self.heatmaps = np.array(heatmapsInterpolated)

        return

    def extractGrid(
        self,
        ):
        """
        """

        folder = self.session.home.joinpath('stimuli', 'metadata')
        metadata = None
        for file in folder.iterdir():
            if 'sparseNoise' in file.name:
                with open(str(file), 'rb') as stream:
                    metadata = pickle.load(stream)
                break
                
        #
        if metadata is None:
            raise Exception('Could not locate sparse noise metadata file')

        #
        fieldCenters = np.unique(metadata['coords'], axis=0)
        self.fp = {
            'x': np.unique(fieldCenters[:, 0]),
            'y': np.unique(fieldCenters[:, 1])
        }

        return

    def plotReceptiveFields(
        self,
        figsize=(4, 3),
        cmap='gist_rainbow',
        fillContours=False
        ):
        """
        """
        
        #
        fig, ax = plt.subplots()
        nFields = len(self.contours)
        if nFields == 0:
            return fig, ax
        cm = plt.get_cmap(cmap, nFields)
        colors = [cm(i) for i in range(nFields)]

        #
        for i, contour in enumerate(self.contours):
            vertices = contour.reshape(-1, 2)
            xy = list()
            for iCol, letter in zip(range(2), ('x', 'y')):
                x1 = np.concatenate([
                    np.array([vertices[-1, iCol]]),
                    vertices[:, iCol],
                    np.array([vertices[0, iCol]])
                ]).astype(float)
                x2 = np.concatenate([x1[1:-1], np.array([x1[1]])])
                x3 = gaussianFilter(x2, 0.7)
                x4 = np.interp(
                    np.concatenate([x3, np.array([x3[0]])]),
                    self.xp[letter],
                    self.fp[letter]
                )
                xy.append(x4)
            xy = np.array(xy).T
            ax.plot(xy[:, 0], xy[:, 1], color=colors[i], alpha=0.3, lw=2.5)
            if fillContours:
                ax.fill(xy[:, 0], xy[:, 1], color=colors[i], alpha=0.05)

        #
        ax.vlines(0, -10, 10, color='k')
        ax.hlines(0, -10, 10, color='k')
        for sp in ('top', 'bottom', 'left', 'right'):
            ax.spines[sp].set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])

        #
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        ax.set_aspect('equal')
        fig.tight_layout()

        return fig, ax

class SaccadeDetectionDemonstrationFigure(AnalysisBase):
    """
    """

    def __init__(self, date='2023-07-05'):
        """
        """

        super().__init__()
        self.date = date
        self.y = None
        self.t = None
        self.wfs = None
        self.threshold = None
        self.peakIndices = None

        return

    def extractSaccadeWaveforms(
        self,
        height=0.5,
        distance=5,
        n=40,
        ):
        """
        """

        #
        self.peakIndices, peakProperties = find_peaks(np.diff(self.y), height=height, distance=distance)
        self.wfs = list()
        for peakIndex in self.peakIndices:
            wf = self.y[peakIndex - n: peakIndex + n + 1]
            self.wfs.append(wf)
        self.wfs = np.array(self.wfs)
        self.threshold = height

        return

    def extractEyePosition(
        self,
        blockIndex=32,
        buffer=3,
        window=None,
        ):
        """
        """

        # Set the session by date
        for ukey in self.ukeys:
            if ukey[0] == self.date:
                self.ukey = ukey
                break

        #
        gratingTimestamps = self.session.load('stimuli/dg/grating/timestamps')
        motionTimestamps = self.session.load('stimuli/dg/motion/timestamps')
        itiTimestamps = self.session.load('stimuli/dg/iti/timestamps')
        eyePosition = self.session.load('pose/filtered')
        t1 = motionTimestamps[blockIndex] - buffer
        if window is None:
            t2 = itiTimestamps[blockIndex] + buffer
        else:
            t2 = gratingTimestamps[blockIndex] + window

        #
        frameTimestamps = self.session.load(f'frames/{self.session.eye}/timestamps')
        frameIndices = np.where(
            np.logical_and(
                frameTimestamps >= t1,
                frameTimestamps <= t2
            )
        )[0]
        self.y = smooth(eyePosition[frameIndices, 0], 15)
        self.t = frameTimestamps[frameIndices]
        self.t -= motionTimestamps[blockIndex] # Zero the timestamps

        return

    def plot(
        self,
        figsize=(3, 4),
        ):
        """
        """

        #
        fig, axs = plt.subplots(nrows=3)
        axs[0].plot(self.t, self.y, color='k')
        dt = self.t[1] - self.t[0]
        axs[1].plot(self.t[:-1] + (0.5 * dt), np.diff(self.y), color='k')
        xlim = axs[0].get_xlim()
        tlim = self.t.min(), self.t.max()
        axs[1].scatter(self.t[self.peaksIndices], np.diff(self.y)[self.peakIndices], marker='x', color='r')
        axs[1].hlines(self.threshold, *tlim, color='r')
        axs[1].set_xlim(xlim)

        #
        dt = self.t[1] - self.t[0]
        hw = int((self.wfs.shape[1] - 1) / 2)
        t = (np.arange(self.wfs.shape[1]) - hw) * dt
        bl = self.wfs[:, :25].mean()
        for wf in self.wfs:
            axs[2].plot(t, wf - bl, color='0.8')
        axs[2].plot(t, self.wfs.mean(0) - bl, color='k')

        #
        axs[0].set_xticks([])
        for ax in axs:
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
        axs[0].set_ylabel('Position (pix)')
        axs[1].set_ylabel('Velocity (pix/s)')
        axs[1].set_xlabel('Time from motion onset (sec)')
        axs[2].set_ylabel('Position (pix)')
        axs[2].set_xlabel('Time from peak velocity (sec)')

        #
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, axs
    
class EyeVelocityDistribution(AnalysisBase):
    """
    """

    def __init__(
        self,
        **kwargs
        ):
        """
        """

        super().__init__(**kwargs)

        return
    
    def plotVelocityHistogram(
        self,
        animal='mlati6',
        date='2023-04-11',
        vrange=(-10, 10),
        nBins=200,
        figsize=(5, 3),
        colors=('r', 'b', '0.5'),
        labels=('Nasal', 'Temporal', 'Control')
        ):
        """
        """

        fig, ax = plt.subplots()
        sessionLocated = False
        for session in self.sessions:
            if str(session.date) == date and session.animal == animal:
                self._session = session
                sessionLocated = True
                break
        if sessionLocated == False:
            raise Exception(f'Could not locate session for {animal} on {date}')
        
        #
        saccadeWaveforms = {
            'nasal': self.session.load(f'saccades/predicted/{self.session.eye}/nasal/waveforms'),
            'temporal': self.session.load(f'saccades/predicted/{self.session.eye}/temporal/waveforms')
        }
        horizontalEyePosition = self.session.load('pose/filtered')[:, 0 if self.session.eye == 'left' else 2]
        velocity = {
            'nasal': list(),
            'temporal': list(),
            'control': np.diff(horizontalEyePosition)
        }
        for k in saccadeWaveforms.keys():
            for wf in saccadeWaveforms[k]:
                dx = np.diff(wf)
                i = np.argmax(np.abs(dx))
                v = dx[i]
                velocity[k].append(v)

        #
        for i, k in enumerate(velocity.keys()):
            counts, edges = np.histogram(velocity[k], range=vrange, bins=nBins)
            y = counts / counts.sum()
            x = edges[:-1] + ((edges[1] - edges[0]) / 2)
            ax.plot(x, y, color=colors[i], label=labels[i])
            ax.fill_between(x, 0, y, color=colors[i], alpha=0.2)

        #
        ax.legend()
        ax.set_xlabel('Velocity (pix/sec)')
        ax.set_ylabel('Probability')
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()
                
        return fig, ax