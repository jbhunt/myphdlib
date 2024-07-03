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

class ElectrodeMapFigure(AnalysisBase):
    """
    """

    def __init__(
        self,
        date='2023-07-05',
        **kwargs
        ):
        """
        """
        super().__init__(**kwargs)
        self._example = None
        for session in self.sessions:
            if str(session.date) == date:
                self._example = session
                break
        return

    @property
    def example(self):
        return self._example

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
        heatmaps = self.example.load(f'rf/{phase}')
        if heatmaps is None:
            return
        heatmapsFiltered = list()
        for unit in self.example.population:

            #
            ukey = (
                str(self.example.date),
                self.example.animal,
                unit.cluster
            )
            # if ukey not in self.ukeys:
            #     heatmapsFiltered.append(None)
            #    continue
            self.ukey = ukey

            #
            hm = heatmaps[self.unit.index]
            if smoothingKernelWidth is not None:
                hm = gaussianFilter(hm, smoothingKernelWidth)
            # if hm.max() < threshold:
            #     heatmapsFiltered.append(None)
            #     continue
            heatmapsFiltered.append(hm)

        # Interpolate
        heatmapsInterpolated = list()
        for hm in heatmapsFiltered:
            if hm is None:
                heatmapsInterpolated.append(None)
            else:
                z = zoom(hm, interpolationFactor, grid_mode=False)
                heatmapsInterpolated.append(z)

        # Detect contours
        contours = list()
        for hm in heatmapsInterpolated:

            if hm is None:
                contours.append(None)
                continue

            #
            grayscale = cv.threshold(hm, threshold, 255, cv.THRESH_BINARY)[-1].astype(np.uint8)
            cnts, hierarchy = cv.findContours(
                grayscale,
                mode=cv.RETR_TREE,
                method=cv.CHAIN_APPROX_NONE
            )
            if len(cnts) == 0:
                contours.append(None)
                continue
            areas = np.array([cv.contourArea(cnt) for cnt in cnts])
            index = np.argmax(areas)
            cnt = cnts[index]
            if areas[index] < minimumContourArea:
                contours.append(None)
                continue
            contours.append(cnt)

        #
        nRows, nCols = heatmaps[0].shape
        xp = {
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

        #
        folder = self.example.home.joinpath('stimuli', 'metadata')
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
        fp = {
            'x': np.unique(fieldCenters[:, 0]),
            'y': np.unique(fieldCenters[:, 1])
        }

        return contours, xp, fp

    def plotReceptiveFields(
        self,
        figsize=(4, 3),
        cmap='gist_rainbow',
        fillContours=False
        ):
        """
        """

        contours, xp, fp = self.extractContours()
        
        #
        fig, ax = plt.subplots()
        nFields = len(contours)
        if nFields == 0:
            return fig, ax
        cm = plt.get_cmap(cmap, nFields)
        colors = [cm(i) for i in range(nFields)]

        #
        for i, contour in enumerate(contours):
            if contour is None:
                continue
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
                    xp[letter],
                    fp[letter]
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

    def plotElectrodeMap(
        self,
        figsize=(15, 1.5)
        ):
        """
        """

        #
        contours, xp, fp = self.extractContours()

        #
        filepath = self.example.home.joinpath('ephys', 'sorting', 'manual', 'cluster_info.tsv')
        with open(filepath, 'r') as stream:
            lines = stream.readlines()

        depths = np.full(len(self.example.population), np.nan)
        for unit in self.example.population:
            for ln in lines[1:]:
                elements = ln.split('\t')
                cluster = int(elements[0])
                if cluster == unit.cluster:
                    depths[unit.index] = float(elements[6])

        # Plot contours at depth
        fig, ax = plt.subplots()
        cmap = lambda x: plt.cm.gist_rainbow(plt.Normalize(depths.min(), depths.max())(x))
        for contour, depth in zip(contours, depths):
            if contour is None:
                continue
            # ax.scatter(depth, np.random.uniform(low=-1, high=1, size=1), color='k')

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
                    xp[letter],
                    fp[letter]
                )
                xy.append(x4)
            xy = np.array(xy).T
            xy[:, 0] -= xy[:, 0].mean()
            xy[:, 0] += (0.5 * depth)
            xy[:, 1] -= xy[:, 1].mean()
            yoff = np.random.normal(loc=0, scale=30, size=1)
            xy[:, 1] += yoff
            ax.plot(xy[:, 0], xy[:, 1], color=cmap(depth), alpha=0.5, lw=1)
            ax.scatter(depth * 0.5, 0 + yoff, color=cmap(depth), alpha=0.5, s=10, edgecolor='none')

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
        saccadeLabels = self.session.load(f'saccades/predicted/{self.session.eye}/labels')
        saccadeWaveforms = {
            'nasal': self.session.load(f'saccades/predicted/{self.session.eye}/waveforms')[saccadeLabels == 1],
            'temporal': self.session.load(f'saccades/predicted/{self.session.eye}/waveforms')[saccadeLabels == -1]
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
    
class UnitFilteringPipelineFigure(AnalysisBase):
    """
    """

    def __init__(
        self,
        *args,
        **kwargs
        ):
        """
        """

        super().__init__(*args, **kwargs)
        self.counts = None

        return

    def measureUnitLoss(
        self
        ):
        """
        """

        #
        counts = np.full([len(self.sessions), 4], np.nan)

        #
        for i, session in enumerate(self.sessions):

            print(f'Working on session {i + 1} out of {len(self.sessions)} ...')

            #
            if session.probeTimestamps is None:
                continue

            #
            amplitudeCutoff = session.load('metrics/ac')
            presenceRatio = session.load('metrics/pr')
            isiViolations = session.load('metrics/rpvr')
            firingRate = session.load('metrics/fr')
            probabilityValues = np.nanmin(np.vstack([
                session.load('zeta/probe/left/p'),
                session.load('zeta/probe/right/p')
            ]), axis=0)
            responseLatency = np.nanmin(np.vstack([
                session.load('zeta/probe/left/latency'),
                session.load('zeta/probe/right/latency'),
            ]), axis=0)
            qualityLabels = session.load('metrics/ql')

            # Unfiltered count
            counts[i, 0] = len(session.population)

            # ZETA-test
            f1 = probabilityValues <= 0.01
            counts[i, 1] = f1.sum()

            # Manual spike-sorting
            if qualityLabels is None:
                include = np.full(len(session.population), True)
            else:
                exclude = np.full(len(session.population), False)
                exclude[np.logical_or(qualityLabels == 0, qualityLabels == 1)] = True
                include = np.invert(exclude)
            f2 = np.vstack([
                f1,
                include,
            ]).all(0)
            counts[i, 2] = f2.sum()

            # Quality  metrics
            f3 = np.vstack([
                f2,
                amplitudeCutoff <= 0.1,
                presenceRatio >= 0.9,
                isiViolations <= 0.5,
                firingRate >= 0.2,
                responseLatency >= 0.025,
            ]).all(0)
            counts[i, 3] = f3.sum()

        #
        self.counts = counts

        return
    
    def plotUnitLoss(
        self,
        figsize=(6, 3.5)
        ):
        """
        """

        if self.counts is None:
            return
        
        fig, axs = plt.subplots(ncols=2, sharex=True)
        axs[0].plot(np.arange(4), self.counts.sum(0), color='k', marker='o', markersize=4)
        y = self.counts.mean(0)
        e = np.std(self.counts, axis=0)
        axs[1].plot(np.arange(4), y, color='k', marker='o', markersize=4)
        # axs[1].fill_between(
        #     np.arange(4),
        #     y - e,
        #     y + e,
        #     color='k',
        #     alpha=0.15
        # )
        axs[1].vlines(np.arange(4), y - e, y + e, color='k')
        axs[0].set_ylabel('Total number of units')
        axs[1].set_ylabel('# of units per session')
        for ax in axs:
            ax.set_xticks(np.arange(4))
            ax.set_xticklabels(['Raw', 'ZETA', 'Phy', 'QC'])
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, axs

class ProbeLatencyAnalysis(AnalysisBase):
    """
    """

    def histProbeLatency(
        self,
        trange=(-3, 3),
        binsize=0.1,
        leftEdge=-0.5,
        rightEdge=0.5,
        figsize=(2.5, 1.5)
        ):
        """
        """

        n = int(np.diff(trange).item() * 1000 // (binsize * 1000))
        f = list()
        for session in self.sessions:
            if session.probeTimestamps is None:
                continue
            counts, edges = np.histogram(
                session.probeLatencies,
                range=trange,
                bins=n
            )
            f.append(counts)
        f = np.array(f)
        t = np.arange(trange[0], trange[1], binsize) + (binsize / 2)

        fig, ax = plt.subplots()

        y = f.mean(0)
        e = f.std(0)
        ax.step(
            t,
            y,
            where='mid',
            color='0.5',
        )
        # ax.plot(
        #     t,
        #     y,
        #     color='k',
        # )
        # ax.fill_between(
        #     t,
        #     y - e,
        #     y + e,
        #     color='k',
        #     alpha=0.1,
        #     edgecolor='none'
        # )
        ylim = ax.get_ylim()
        ax.vlines(
            [leftEdge, rightEdge],
            *ylim,
            color='k',
            linestyle=':'
        )
        # ax.fill_between(
        #     [leftEdge, rightEdge],
        #     ylim[0],
        #     ylim[1],
        #     color='r',
        #     alpha=0.1,
        #     edgecolor='none'
        # )
        ax.set_ylim(ylim)
        ax.set_xlim(trange)
        ax.set_xticks([-3, -1.5, 0, 1.5, 3])
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, ax