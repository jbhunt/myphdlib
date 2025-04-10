import numpy as np
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from myphdlib.general.toolkit import smooth, psth2
from sklearn.decomposition import PCA
from scipy.ndimage import gaussian_filter as gaussianFilter
from scipy.ndimage import zoom
from scipy.interpolate import interp2d
import cv2 as cv
import pickle
from skimage.measure import find_contours
from scipy.signal import find_peaks
from myphdlib.figures.analysis import AnalysisBase
from skimage.measure import EllipseModel
from myphdlib.extensions.matplotlib import getIsoluminantRainbowColormap, getCetI1Colormap

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
        smoothingKernelWidth=0.8,
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

    def plotElectrodeMap(
        self,
        depthScaling=0.8,
        widthScaling=3,
        receptiveFieldScaling=2,
        figsize=(10, 1.5),
        cmap=None,
        ):
        """
        """

        #
        contours, xp, fp = self.extractContours()

        #
        centroids = list()
        for contour in contours:
            if contour is None:
                centroids.append([np.nan, np.nan])
                continue
            centroidRaw = contour.mean(0).flatten()
            cx = np.interp(centroidRaw[0], xp['x'], fp['x'])
            cy = np.interp(centroidRaw[1], xp['y'], fp['y'])
            centroids.append([cx, cy])
        centroids = np.array(centroids)

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

        #
        filepath = self.example.home.joinpath('ephys', 'sorting', 'manual', 'spike_positions.npy')
        if filepath.exists() == False:
            raise Exception('Could not locate spike positions data')
        spikePositions = np.load(filepath)
        spikeClusters = self.example.load('spikes/clusters')
        uniqueClusters, index = np.unique(spikeClusters, return_index=True)
        unitCoords = spikePositions[index, :]
        unitCoords[:, 0] -= 40

        # Plot contours at depth
        fig, ax = plt.subplots()
        mask = np.array([False if cnt is None else True for cnt in contours])
        if cmap is None:
            cmap = plt.cm.gist_rainbow
        def f(x):
            i = np.interp(
                x,
                [-90, 90],
                [0, 256]
            )
            return cmap(int(round(i)))

        for iUnit, (contour, (xc, yc)) in enumerate(zip(contours, unitCoords)):
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
            # model = EllipseModel()
            # model.estimate(xy)
            # xy = model.predict_xy(np.linspace(0, 2 * np.pi, 100))

            #
            xy[:, 0] -= xy[:, 0].mean()
            xy[:, 1] -= xy[:, 1].mean()
            xy *= receptiveFieldScaling
            xy[:, 0] += (yc * depthScaling)
            xy[:, 1] += (xc * widthScaling)

            #
            ax.plot(xy[:, 0], xy[:, 1], color=f(centroids[iUnit, 0]), alpha=0.6, lw=1)
            ax.scatter(yc * depthScaling, xc * widthScaling, color=f(centroids[iUnit, 0]), alpha=0.6, s=7, edgecolor='none')

        #
        ax.vlines([0, 3800 * depthScaling], -40 * widthScaling, 40 * widthScaling, color='k', linestyle=':', lw=0.8)
        ax.hlines([-40 * widthScaling, 40 * widthScaling], 0, 3800 * depthScaling, color='k', linestyle=':', lw=0.8)
        ax.hlines(0, 0, 20 * receptiveFieldScaling, color='k')
        ax.vlines(0 - (20 * depthScaling), -40 * widthScaling,  (-40 * widthScaling) + (20 * widthScaling), color='k')
        ax.hlines(-40 * widthScaling - (5 * widthScaling), 0, 100 * depthScaling, color='k')

        #
        ax.set_xticks([0, 3800 * depthScaling])
        ax.set_xticklabels([0, 3800])
        ax.set_yticks([-40 * widthScaling, 0, 40 * widthScaling])
        ax.set_yticklabels([-40, 0, 40])
        ax.set_aspect('equal')
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, ax

class ExperimentSummaryFigure(ElectrodeMapFigure):
    """
    """

    def _plotSpikeRasters(
        self,
        ax,
        t0,
        window=(0, 15),
        cmap=None,
        ):
        """
        """

        #
        contours, xp, fp = self.extractContours()

        #
        centroids = list()
        for contour in contours:
            if contour is None:
                centroids.append([np.nan, np.nan])
                continue
            centroidRaw = contour.mean(0).flatten()
            cx = np.interp(centroidRaw[0], xp['x'], fp['x'])
            cy = np.interp(centroidRaw[1], xp['y'], fp['y'])
            centroids.append([cx, cy])
        centroids = np.array(centroids)

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

        #
        filepath = self.example.home.joinpath('ephys', 'sorting', 'manual', 'spike_positions.npy')
        if filepath.exists() == False:
            raise Exception('Could not locate spike positions data')
        spikePositions = np.load(filepath)
        spikeClusters = self.example.load('spikes/clusters')
        uniqueClusters, index = np.unique(spikeClusters, return_index=True)
        unitCoords = spikePositions[index, :]
        unitCoords[:, 0] -= 40

        #
        mask = np.array([False if cnt is None else True for cnt in contours])
        if cmap is None:
            cmap = plt.cm.gist_rainbow
        def f(x):
            i = np.interp(
                x,
                [-90, 90],
                [0, 256]
            )
            return cmap(int(round(i)))

        #
        unitCounter = 0
        unitIndices = np.argsort(unitCoords[:, 1])
        for iUnit in unitIndices:
            iUnit = int(iUnit) # NOTE: indexing the population won't work with numpy data types
            if contours[iUnit] is None:
                continue
            t, M, spikeTimestamps = psth2(
                np.array([t0,]),
                self.example.population[iUnit].timestamps,
                window=window,
                binsize=None,
                returnTimestamps=True
            )
            ax.vlines(
                spikeTimestamps,
                unitCounter - 0.45,
                unitCounter + 0.45,
                color=f(centroids[iUnit, 0]),
                linewidth=0.7,
                alpha=0.9,
                rasterized=True
            )
            unitCounter += 1

        return

    def plot(
        self,
        blockIndex=24,
        window=(-1, 15),
        figsize=(4.5, 3.5),
        cmap=None
        ):
        """
        """

        #
        gratingTimestamps = self.example.load('stimuli/dg/grating/timestamps')
        motionTimestamps = self.example.load('stimuli/dg/motion/timestamps')
        eyePosition = self.example.load('pose/filtered')
        t0 = gratingTimestamps[blockIndex]
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
        ax = fig.add_subplot(gs[5:], rasterized=False)
        axs.append(ax)

        #
        axs[0].vlines(t1 - t0, 0, 1, color='k')
        axs[0].vlines(t0 - t0, 0, 1, color='k')

        #
        for columneIndex, eye, coefficient, ax in zip([0, 2], ['left', 'right'], [-1, 1], axs[1:3]):
            frameTimestamps = self.example.load(f'frames/{eye}/timestamps')
            frameIndices = np.where(
                np.logical_and(
                    frameTimestamps >= t0 + window[0],
                    frameTimestamps <= t2
                )
            )[0]
            y = smooth(eyePosition[frameIndices, columneIndex] * coefficient, 15)
            t = frameTimestamps[frameIndices]
            ax.plot(t - t0, y, color='k')

        #
        for ax, ev in zip(axs[3:], [self.example.saccadeTimestamps[:, 0], self.example.probeTimestamps]):
            eventIndices = np.where(np.logical_and(
                ev >= t0,
                ev <= t2
            ))[0]
            ax.vlines(ev[eventIndices] - t0, 0, 1, color='k')

        #
        self._plotSpikeRasters(
            axs[-1],
            t0,
            window,
            cmap
        )

        #
        for ax in axs[:5]:
            ax.set_xticks([])
        for ax in axs:
            xlim = (
                window[0] - 0.5,
                window[1] + 0.5
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
        saccadeDirection='temporal',
        vrange=(-5, 5),
        nBins=200,
        ax=None,
        figsize=(5, 2),
        ):
        """
        """

        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        #
        samples = {
            'left': list(),
            'right': list(),
            'control': list()
        }
        if saccadeDirection == 'temporal':
            iterable = (
                ['left', 'temporal', 'left'],
                ['right', 'temporal', 'right']
            )
            flipDirection = 'right'
        elif saccadeDirection == 'nasal':
            iterable = (
                ['right', 'nasal', 'left'],
                ['left', 'nasal', 'right']
            )
            flipDirection = 'left'

        #
        for session in self.sessions:

            #
            sLeft, sRight, sCtrl = list(), list(), list()
        
            #
            for eye, d1, d2 in iterable:
                saccadeLabels = session.load(f'saccades/predicted/{eye}/labels')
                saccadeWaveforms = {
                    'nasal': session.load(f'saccades/predicted/{eye}/waveforms')[saccadeLabels == 1],
                    'temporal': session.load(f'saccades/predicted/{eye}/waveforms')[saccadeLabels == -1]
                }
                for wf in saccadeWaveforms[d1]:
                    dx = np.diff(wf)
                    i = np.argmax(np.abs(dx))
                    v = dx[i]
                    if d2 == flipDirection:
                        v *= -1
                    if d2 == 'left':
                        sLeft.append(v)
                    elif d2 == 'right':
                        sRight.append(v)
                    # samples[d2].append(v)
                nSaccades = int(round(np.mean([
                    saccadeWaveforms['nasal'].shape[0],
                    saccadeWaveforms['temporal'].shape[0]
                ]), 0))
                velocity = np.diff(session.load('pose/filtered')[:, 0 if eye == 'left' else 2])
                for v in np.random.choice(velocity, size=nSaccades, replace=False):
                    if d2 == flipDirection:
                        v *= -1
                    sCtrl.append(v)
                    # samples['control'].append(v)

                
            #
            samples['control'].append(sCtrl)
            samples['left'].append(sLeft)
            samples['right'].append(sRight)

        #
        freqs = {
            'control': list(),
            'left': list(),
            'right': list()
        }
        for i, k in enumerate(['left', 'control', 'right']):
            nSamples = len(samples[k])
            for iSample in range(nSamples):
                counts, edges = np.histogram(samples[k][iSample], range=vrange, bins=nBins)
                y = counts / counts.sum()
                freqs[k].append(y)
                x = edges[:-1] + ((edges[1] - edges[0]) / 2)
                continue
                ax.plot(
                    x,
                    y,
                    color='0.7',
                    lw=0.5,
                    alpha=1.0
                )
                # ax.step(
                #     x,
                #     y,
                #     where='mid',
                #     color='k'
                # )
            # ax.plot(x, y, color=colors[i], label=labels[i])
            # ax.fill_between(x, 0, y, color=colors[i], alpha=0.2)

        #
        for k in freqs.keys():
            y = np.array(freqs[k]).mean(0)
            e = np.std(freqs[k], axis=0)
            ax.plot(x, y, color='k')
            ax.fill_between(
                x,
                y - e,
                y + e,
                color='k',
                edgecolor='none',
                alpha=0.1
            )

        #
        y1, y2 = ax.get_ylim()
        ax.set_ylim([0, y2])

        #
        # ax.legend()
        ax.set_xlabel('Velocity (pix/sec)')
        ax.set_ylabel('Probability')
        if ax is None == False:
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
        trange=(-10, 10),
        binsize=0.1,
        leftEdge=-0.5,
        rightEdge=0.5,
        ax=None,
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
                np.abs(session.probeLatencies),
                range=trange,
                bins=n
            )
            f.append(counts)
        f = np.array(f)
        t = np.arange(trange[0], trange[1], binsize) + (binsize / 2)

        #
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure

        #
        # for y in f:
        #     ax.plot(t, y, color='0.7', alpha=1, lw=0.5)
        y = f.mean(0)
        ax.plot(t, y, color='k')
        e = np.std(f, axis=0)
        ax.fill_between(
            t,
            y - e,
            y + e,
            color='k',
            edgecolor='none',
            alpha=0.1
        )

        # y = f.mean(0)
        # e = f.std(0)
        # ax.step(
        #     t,
        #     y,
        #     where='mid',
        #    color='0.5',
        # )
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
        ax.set_xticks([-10, -5, 0, 5, 10])
        if ax is None:
            fig.set_figwidth(figsize[0])
            fig.set_figheight(figsize[1])
            fig.tight_layout()

        return fig, ax
    
class SaccadeDescriptionAnalysis(AnalysisBase):
    """
    """

    def __init__(self, *args, **kwargs):
        """
        """

        self.saccadeFeatures = {
            'label': list(),
            'velocity': list(),
            'startpoint': list(),
            'endpoint': list(),
            'amplitude': list(),
            'duration': list()
        }
        super().__init__(*args, **kwargs)

        return
    
    def _measurePeakVelocityForNoiseSaccades(self, session):
        """
        """

        saccadeWaveformsTrue = session.load(f'saccades/predicted/{session.eye}/waveforms')
        saccadeWaveformsPutative = session.load(f'saccades/putative/{session.eye}/waveforms')
        saccadeWaveformsFalse = list()
        for wf in saccadeWaveformsPutative:
            if wf not in saccadeWaveformsTrue:
                saccadeWaveformsFalse.append(wf)
        saccadeWaveformsFalse = np.array(saccadeWaveformsFalse)
        peakVelocity = list()
        for wf in saccadeWaveformsFalse:
            v = np.diff(wf) * session.fps
            vmax = np.interp(0.5, np.linspace(0, 1, v.size), v).item()
            peakVelocity.append(vmax)

        return np.array(peakVelocity)
    
    def _measureNullVelocityDistribution(self, session, n=30):
        """
        """

        poseEstimates = session.load('pose/filtered')
        if session.eye == 'left':
            eyePosition = poseEstimates[:, 0]
        elif session.eye == 'right':
            eyePosition = poseEstimates[:, 2]
        velocity = np.diff(eyePosition)
        sample = list()
        while len(sample) < n:
            observation = np.random.choice(velocity, size=1).item()
            if np.isnan(observation):
                continue
            sample.append(observation)

        return np.array(sample)

    def extractSaccadeFeatures(
        self,
        ):
        """
        """

        self.saccadeFeatures = {
            'label': list(),
            'velocity': list(),
            'startpoint': list(),
            'endpoint': list(),
            'amplitude': list(),
            'duration': list()
        }

        nSessions = len(self.sessions)
        for i, session in enumerate(self.sessions):
            end = '\r' if (i + 1) != nSessions else '\n'
            print(f'Working on session {i + 1} out of {nSessions}', end=end)

            #
            saccadeLabels = session.load(f'saccades/predicted/{session.eye}/labels')
            nSaccades = saccadeLabels.size
            saccadeWaveforms = session.load(f'saccades/predicted/{session.eye}/waveforms')
            saccadeEpochs = session.load(f'saccades/predicted/{session.eye}/epochs')
            saccadeTimestamps = session.load(f'saccades/predicted/{session.eye}/timestamps')

            # Validate session
            poseEstimates = session.load(f'pose/filtered')
            if session.eye == 'left': 
                eyePosition = poseEstimates[:, 0]
            elif session.eye == 'right':
                eyePosition = poseEstimates[:, 2]
            droppedFrames = session.load(f'frames/{session.eye}/dropped')
            nFramesRecorded = droppedFrames.size
            if nFramesRecorded > eyePosition.size:
                continue

            # Estimate null velocity
            peakVelocitiesNull = self._measureNullVelocityDistribution(session, n=nSaccades)
            for vmax in peakVelocitiesNull:
                self.saccadeFeatures['label'].append(0)
                self.saccadeFeatures['velocity'].append(vmax)
                for k in ('amplitude', 'duration', 'startpoint', 'endpoint'):
                    self.saccadeFeatures[k].append(np.nan)

            # Real saccades
            iterable = list(zip(
                saccadeLabels,
                saccadeWaveforms,
                saccadeEpochs[:, 0],
                saccadeEpochs[:, 1],
                saccadeTimestamps[:, 0],
                saccadeTimestamps[:, 1]
            ))
            for l, wf, f1, f2, t1, t2 in iterable:

                # Label
                self.saccadeFeatures['label'].append(l)

                # Velocity
                v = np.diff(wf) * session.fps
                vmax = np.interp(0.5, np.linspace(0, 1, v.size), v).item()
                self.saccadeFeatures['velocity'].append(vmax)

                # Startpoint
                p1 = np.interp(
                    f1,
                    np.arange(nFramesRecorded),
                    eyePosition[:nFramesRecorded]
                ).item()
                self.saccadeFeatures['startpoint'].append(p1)

                # Endpoint
                p2 = np.interp(
                    f2,
                    np.arange(nFramesRecorded),
                    eyePosition[:nFramesRecorded]
                ).item()
                self.saccadeFeatures['endpoint'].append(p2)

                # Amplitude
                a = abs(p2 - p1)
                self.saccadeFeatures['amplitude'].append(a)

                # Duration
                dt = t2 - t1
                self.saccadeFeatures['duration'].append(dt)

        #
        for k in self.saccadeFeatures.keys():
            self.saccadeFeatures[k] = np.array(self.saccadeFeatures[k])
        for k in self.saccadeFeatures.keys():
            self.ns[f'saccades/{k}'] = self.saccadeFeatures[k]

        return
    
    def plotHistograms(
        self,
        bins=100,
        keys=('velocity', 'amplitude', 'duration', 'startpoint', 'endpoint'),
        ranges=[(-1500, 1500), (0, 35), (0, 0.15), (-40, 40), (-40, 40)],
        figsize=(5, 8),
        ):
        """
        """

        fig, axs = plt.subplots(nrows=len(self.saccadeFeatures.keys()) - 1)
        saccadeLabels = self.ns['saccades/label']
        for i, k in enumerate(keys):
            binEdges = np.linspace(ranges[i][0], ranges[i][1], bins + 1)
            try:
                samples = [
                    self.ns[f'saccades/{k}'][saccadeLabels ==  0],
                    self.ns[f'saccades/{k}'][saccadeLabels == -1],
                    self.ns[f'saccades/{k}'][saccadeLabels ==  1]
                ]
                axs[i].hist(
                    samples[0],
                    bins=binEdges,
                    color='0.5',
                    alpha=0.3
                )
                axs[i].hist(
                    samples[1],
                    bins=binEdges,
                    color='r',
                    alpha=0.3
                )
                axs[i].hist(
                    samples[2],
                    bins=binEdges,
                    color='b',
                    alpha=0.3
                )
            except:
                import pdb; pdb.set_trace()

        #
        xlabels = (
            'Velocity (deg/s)',
            'Amplitude (deg)',
            'Duration (s)',
            'Startpoint (deg)',
            'Endpoint (deg)'
        )
        for i in range(len(axs)):
            axs[i].set_xlabel(xlabels[i])
            axs[i].set_ylabel('# of units')
        axs[0].legend(['Noise', 'Temp.', 'Nasal'])
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, axs

    def plotScatterplots(self):
        """
        """

        fig, axs = plt.subplots(ncols=2)
        nasalSaccades = self.saccadeFeatures['label'] == 1
        temporalSaccades = self.saccadeFeatures['label'] == -1
        noiseSaccades = self.saccadeFeatures['label'] == 0
        saccadeMasks = (
            nasalSaccades,
            temporalSaccades,
        )
        jitter = np.random.normal(loc=0, scale=0, size=noiseSaccades.sum())
        axs[0].scatter(
            np.full(noiseSaccades.sum(), 0) + jitter,
            self.saccadeFeatures['velocity'][noiseSaccades],
            marker='.',
            s=5,
            color='k',
            edgecolor='none',
            alpha=0.15
        )
        for m, c in zip(saccadeMasks, ['r', 'b']):
            axs[0].scatter(
                self.saccadeFeatures['amplitude'][m],
                self.saccadeFeatures['velocity'][m],
                marker='.',
                s=5,
                color=c,
                edgecolor='none',
                alpha=0.15
            )
            axs[1].scatter(
                self.saccadeFeatures['startpoint'][m],
                self.saccadeFeatures['endpoint'][m],
                marker='.',
                s=5,
                color=c,
                edgecolor='none',
                alpha=0.15
            )
        axs[0].hlines(0, -2, 35, color='k', linestyle=':', lw=1)
        axs[0].set_xlim([-2, 35])
        axs[0].set_ylim([-1350, 1500])
        axs[1].vlines(0, -30, 35, color='k', linestyle=':', lw=1)
        axs[1].hlines(0, -30, 35, color='k', linestyle=':', lw=1)
        axs[1].set_xlim([-30, 35])
        axs[1].set_ylim([-30, 35])
        axs[0].set_xlabel('Amplitude (deg)')
        axs[0].set_ylabel('Peak velocity (deg/s)')
        axs[1].set_xlabel('Startpoint (deg)')
        axs[1].set_ylabel('Endpoint (deg)')
        fig.tight_layout()

        return fig, axs