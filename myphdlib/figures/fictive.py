import h5py
import numpy as np
from scipy.signal import find_peaks as findPeaks
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
from myphdlib.figures.analysis import g
from myphdlib.figures.bootstrap import BootstrappedSaccadicModulationAnalysis
from myphdlib.general.toolkit import psth2

def convertGratingMotionToSaccadeDirection(
    gratingMotion=-1,
    referenceEye='left'
    ):
    """
    """

    saccadeDirection = None
    if referenceEye == 'left':
        if gratingMotion == -1:
            saccadeDirection = 'nasal'
        else:
            saccadeDirection = 'temporal'
    elif referenceEye == 'right':
        if gratingMotion == -1:
            saccadeDirection = 'temporal'
        else:
            saccadeDirection = 'nasal'

    return saccadeDirection

def convertSaccadeDirectionToGratingMotion(
    saccadeDirection,
    referenceEye='left'
    ):
    """
    """

    gratingMotion = None
    if referenceEye == 'left':
        if saccadeDirection == 'nasal':
            gratingMotion = -1
        else:
            gratingMotion = +1
    else:
        if saccadeDirection == 'nasal':
            gratingMotion = +1
        else:
            gratingMotion = -1

    return gratingMotion

class FictiveSaccadesAnalysis(BootstrappedSaccadicModulationAnalysis):
    """
    """

    def __init__(
        self,
        **kwargs,
        ):
        """
        """

        super().__init__(**kwargs)

        #
        self.examples = (

        )

        return

    def loadNamespace(self):
        """
        """
        super().loadNamespace()
        self.windows = np.array([
            [0, 0.1]
        ])
        return

    def _loadEventDataForSaccades(self):
        """
        """

        probeTimestamps = self.session.load('stimuli/fs/probe/timestamps')
        gratingMotion = self.session.load('stimuli/fs/saccade/motion')
        saccadeTimestamps = self.session.load('stimuli/fs/saccade/timestamps')

        #
        saccadeLatencies = np.full(saccadeTimestamps.size, np.nan)
        for iTrial in range(saccadeTimestamps.size):
            iProbe = np.argmin(np.abs(saccadeTimestamps[iTrial] - probeTimestamps))
            saccadeLatencies[iTrial] = saccadeTimestamps[iTrial] - probeTimestamps[iProbe]

        saccadeDirection = np.array([
            convertGratingMotionToSaccadeDirection(gm, self.session.eye)
                for gm in gratingMotion
        ])
        saccadeLabels = np.array([-1 if sd == 'temporal' else 1 for sd in saccadeDirection])
            
        return saccadeTimestamps, saccadeLatencies, saccadeLabels, gratingMotion

    def _loadEventDataForProbes(self):
        """
        """

        #
        probeTimestamps = self.session.load('stimuli/fs/probe/timestamps')
        gratingMotionDuringProbes = self.session.load('stimuli/fs/probe/motion')
        saccadeTimestamps = self.session.load('stimuli/fs/saccade/timestamps')
        gratingMotionDuringSaccades = self.session.load('stimuli/fs/saccade/motion')

        #
        probeLatencies = np.full(probeTimestamps.size, np.nan)
        saccadeLabels = np.full(probeTimestamps.size, np.nan)
        for iTrial in range(probeTimestamps.size):
            iSaccade = np.argmin(np.abs(probeTimestamps[iTrial] - saccadeTimestamps))
            probeLatencies[iTrial] = probeTimestamps[iTrial] - saccadeTimestamps[iSaccade]
            saccadeLabels[iTrial] = gratingMotionDuringSaccades[iSaccade] * -1

        return probeTimestamps, probeLatencies, saccadeLabels, gratingMotionDuringProbes

    def computeExtrasaccadicPeths(
        self,
        **kwargs
        ):
        """
        """

        kwargs['saccadeType'] = 'fictive'
        super().computeExtrasaccadicPeths(
            **kwargs
        )

        return

    def fitExtrasaccadicPeths(
        self,
        **kwargs
        ):
        """
        """

        kwargs['saccadeType'] = 'fictive'
        super().fitExtrasaccadicPeths(**kwargs)

        return

    def computeSaccadeResponseTemplates(
        self,
        perisaccadicWindow=(-0.2, 0.2),
        **kwargs
        ):
        """
        """

        kwargs['saccadeType'] = 'fictive'
        kwargs['perisaccadicWindow'] = perisaccadicWindow
        super().computeSaccadeResponseTemplates(**kwargs)

        return

    def computePerisaccadicPeths(
        self,
        **kwargs
        ):
        """
        """

        kwargs['saccadeType'] = 'fictive'
        super().computePerisaccadicPeths(**kwargs)

        return

    def fitPerisaccadicPeths(
        self,
        **kwargs
        ):
        """
        """

        kwargs['saccadeType'] = 'fictive'
        super().fitPerisaccadicPeths(**kwargs)
        
        return

    def resampleExtrasaccadicPeths(
        self,
        minimumTrialCount=10,
        rate=0.05,
        **kwargs
        ):
        """
        """

        kwargs['saccadeType'] = 'fictive'
        kwargs['minimumTrialCount'] = minimumTrialCount
        kwargs['rate'] = rate
        super().resampleExtrasaccadicPeths(**kwargs)

        return

    def generateNullSamples(
        self,
        **kwargs
        ):
        """
        """

        kwargs['saccadeType'] = 'fictive'
        super().generateNullSamples(**kwargs)

        return

    def computeProbabilityValues(
        self,
        **kwargs
        ):
        """
        """

        kwargs['saccadeType'] = 'fictive'
        super().computeProbabilityValues(**kwargs)

        return

    def plotAnalysisDemo(
        self,
        examples=(
            ('2023-07-20', 'mlati9', 73),
            ('2023-07-11', 'mlati10', 448),
            ('2023-07-11', 'mlati10', 434)
        ),
        responseWindow=(-0.2, 0.5),
        perisaccadicWindow=(-0.5, 0.5),
        figsize=(7, 4),
        **kwargs_
        ):
        """
        """

        kwargs = {
            'marker': '.',
            's': 3,
            'alpha': 0.3
        }
        kwargs.update(kwargs_)

        fig, axs = plt.subplots(ncols=5, nrows=len(examples), sharex=True)

        #
        for i in range(len(examples)):
            self.ukey = examples[i]
            probeTimestamps, probeLatencies, saccadeLabels, gratingMotionDuringProbes = self._loadEventDataForProbes()
            t, M, spikeTimestamps = psth2(
                probeTimestamps,
                self.unit.timestamps,
                window=responseWindow,
                binsize=None,
                returnTimestamps=True
            )
            
            #
            if trialIndices_.size == 0:
                plt.close(fig)
                print(f"Warning: Could not determine unit's preferred direction of motion")
                return None, None

            trialIndices = {
                'peri': trialIndices_,
                'extra': np.where(np.vstack([
                    np.logical_or(
                        probeLatencies < perisaccadicWindow[0],
                        probeLatencies > perisaccadicWindow[1],
                    ),
                    gratingMotionDuringProbes == self.ambc[self.iUnit, 1]
                ]).all(0))[0]
            }
            xy = {
                'peri': list(),
                'extra': list()
            }
            y = 0
            for k in ['extra', 'peri']:
                for iTrial in trialIndices[k]:
                    for x in spikeTimestamps[iTrial]:
                        xy[k].append([x, y])
                    y += 1
            for k in xy.keys():
                xy[k] = np.array(xy[k])

            if len(xy['extra']) != 0:
                axs[i, 0].scatter(xy['extra'][:, 0], xy['extra'][:, 1], color='k', **kwargs)
            if len(xy['peri']) != 0:
                axs[i, 0].scatter(xy['peri'][:, 0], xy['peri'][:, 1], color='r', **kwargs)

            #
            saccadeTimestamps, saccadeLatencies, saccadeLabels, gratingMotionDuringSaccades = self._loadEventDataForSaccades()
            trialIndices = np.where(np.logical_and(
                np.logical_or(
                    saccadeLatencies < perisaccadicWindow[1] * -1,
                    saccadeLatencies > perisaccadicWindow[0] * -1
                ),
                gratingMotionDuringSaccades == self.ambc[self.iUnit, 1]
            ))[0]
            averageLatency = np.mean(saccadeLatencies[
                np.logical_and(
                    saccadeLatencies >= perisaccadicWindow[1] * -1,
                    saccadeLatencies <= perisaccadicWindow[0] * -1
                )
            ])
            t, M, spikeTimestamps = psth2(
                saccadeTimestamps[trialIndices],
                self.unit.timestamps,
                window=responseWindow,
                binsize=None,
                returnTimestamps=True
            )
            xy = list()
            for iTrial in range(len(trialIndices)):
                # l = saccadeLatencies[trialIndices][iTrial]
                for x in spikeTimestamps[iTrial]:
                    xy.append([x + averageLatency, iTrial + y])
            xy = np.array(xy)
            if len(xy) != 0:
                axs[i, 0].scatter(xy[:, 0], xy[:, 1], color='b', **kwargs)

            #
            mu, sigma = self.ambc[self.iUnit, 2], self.ambc[self.iUnit, 3]
            axs[i, 1].plot(
                self.t,
                (self.terms['rMixed'][self.iUnit, :, 0] - mu) / sigma,
                color='k'
            )
            axs[i, 2].plot(
                self.t,
                self.terms['rSaccade'][self.iUnit, :, 0] / sigma,
                color='k'
            )
            #
            for k, j in zip(['peri', 'extra'], [3, 4]):
                if k == 'peri':
                    params = self.paramsRefit[self.iUnit, :]
                    yRaw = self.peths[k][self.iUnit, :, 0]
                else:
                    params = self.params[self.iUnit, :]
                    yRaw = self.peths[k][self.iUnit, :]
                abcd = params[np.invert(np.isnan(params))]
                abc, d = abcd[:-1], abcd[-1]
                A, B, C = np.split(abc, 3)
                yFit = g(self.t, A[0], B[0], C[0], d)
                axs[i, j].plot(
                    self.t,
                    yRaw,
                    color='0.5'
                ) 
                axs[i, j].plot(
                    self.t,
                    yFit,
                    color='k'
                )

            #
            ylim = [np.inf, -np.inf]
            for ax in axs[i, 1:].flatten():
                y1, y2 = ax.get_ylim()
                if y1 < ylim[0]:
                    ylim[0] = y1
                if y2 > ylim[1]:
                    ylim[1] = y2
            if ylim[1] < 5:
                ylim[1] = 5
            if abs(ylim[0]) < 5:
                ylim[0] = -5
            for ax in axs[i, 1:].flatten():
                ax.set_ylim(ylim)
                ax.set_yticks([-5, 0, 5])
            for ax in axs[i, 2:].flatten():
                ax.set_yticklabels([])

        #
        labels = (
            f'Raster',
            r'$R_{P}, R_{S}$',
            r'$R_{S}$',
            r'$R_{P (Peri)}$',
            r'$R_{P (Extra)}$'
        )
        for j, l in enumerate(labels):
            axs[0, j].set_title(l, fontsize=10)
        axs[-1, 0].set_xlabel('Time from probe (s)')
        
        #
        for ax in axs[:, 0].flatten():
            ax.set_ylabel('Trial #')
        for ax in axs[:, 1].flatten():
            ax.set_ylabel('FR (z-scored)')
        for ax in axs.flatten():
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
        for ax in axs[:, 0].flatten():
            for sp in ('bottom', 'left'):
                ax.spines[sp].set_visible(False)
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, axs

    def plotExamples(
        self,
        windowIndex=5,
        ):
        """
        """

        with h5py.File(self.hdf, 'r') as stream:
            pethsRealExtra = np.array(stream['clustering/peths/standard'])
            pethsRealPeri = np.array(stream['modulation/peths/peri'])

        fig, grid = plt.subplots(nrows=len(self.examples), ncols=2, sharey=True)
        if len(self.examples) == 1:
            grid = np.atleast_2d(grid)

        for i, ukey in enumerate(self.examples):
            iUnit = self._indexUnitKey(ukey)
            grid[i, 0].plot(pethsRealExtra[iUnit])
            grid[i, 0].plot(pethsRealPeri[iUnit, :, windowIndex])
            grid[i, 1].plot(self.peths['standard'][iUnit])
            grid[i, 1].plot(self.peths['peri'][iUnit, :, windowIndex])

        return fig, grid

    def plotModulationBySaccadeType(
        self,
        bounds=(-2, 2),
        windowIndex=5,
        figsize=(3, 3),
        ):
        """
        """

        fig, ax = plt.subplots()

        mask = np.vstack([
            self.filter,
            np.logical_or(
                self.p['real'][:, 0] < 0.05,
                self.p['fictive'][:, 0] < 0.05
            ),
        ]).all(0)
        x = np.clip(self.mi['real'][mask, windowIndex, 0] / self.model['params3'][mask, 0], *bounds)
        y = np.clip(self.mi['fictive'][mask, windowIndex, 0] / self.model['params1'][mask, 0], *bounds)
        c = np.full(x.size, 'k')
        
        ax.scatter(
            x,
            y,
            marker='.',
            s=10,
            c=c,
            alpha=0.7,
            clip_on=False,
        )
        ax.vlines(0, *bounds, color='k', alpha=0.5)
        ax.hlines(0, *bounds, color='k', alpha=0.5)
        ax.set_ylim(bounds)
        ax.set_xlim(bounds)
        ax.set_aspect('equal')
        ax.set_xlabel('Modulation Index (Real)')
        ax.set_ylabel('Modulation Index (Fictive)')
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, ax
