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
            ('2023-07-14', 'mlati9', 231),
            ('2023-05-12', 'mlati7', 222),
            ('2023-05-29', 'mlati7', 550),
        )
        self.colors = ('c', 'y', 'm')

        return

    def loadNamespace(self):
        """
        """
        super().loadNamespace()
        self.windows = np.array([
            [0, 0.2]
        ])
        return

    def _loadEventDataForSaccades(
        self,
        coincidenceWindow=(-0.1, 0.1),
        ):
        """
        """

        probeTimestamps = self.session.load('stimuli/fs/probe/timestamps')
        gratingMotion = self.session.load('stimuli/fs/saccade/motion')
        saccadeTimestamps = self.session.load('stimuli/fs/saccade/timestamps')

        #
        saccadeLatencies1 = np.full(saccadeTimestamps.size, np.nan)
        for iTrial in range(saccadeTimestamps.size):
            iProbe = np.argmin(np.abs(saccadeTimestamps[iTrial] - probeTimestamps))
            saccadeLatencies1[iTrial] = saccadeTimestamps[iTrial] - probeTimestamps[iProbe]

        saccadeDirection = np.array([
            convertGratingMotionToSaccadeDirection(gm, self.session.eye)
                for gm in gratingMotion
        ])
        saccadeLabels = np.array([-1 if sd == 'temporal' else 1 for sd in saccadeDirection])

        # Compute the latency from fictive saccades to real saccades
        saccadeLatencies2 = np.full(saccadeTimestamps.size, np.nan)
        saccadeTimestampsReal = np.delete(
            self.session.saccadeTimestamps[:, 0],
            np.isnan(self.session.saccadeTimestamps[:, 0])
        )
        for iTrial in range(saccadeTimestamps.size):
            iSaccade = np.argmin(np.abs(saccadeTimestamps[iTrial] - saccadeTimestampsReal))
            saccadeLatencies2[iTrial] = saccadeTimestamps[iTrial] - saccadeTimestampsReal[iSaccade]
        include = np.logical_or(
            saccadeLatencies2 < coincidenceWindow[0],
            saccadeLatencies2 > coincidenceWindow[1]
        )
            
        return saccadeTimestamps[include], saccadeLatencies1[include], saccadeLabels[include], gratingMotion[include]

    def _loadEventDataForProbes(
        self,
        coincidenceWindow=(-0.1, 0.1),
        ):
        """
        """

        #
        probeTimestamps = self.session.load('stimuli/fs/probe/timestamps')
        gratingMotionDuringProbes = self.session.load('stimuli/fs/probe/motion')
        saccadeTimestamps = self.session.load('stimuli/fs/saccade/timestamps')
        gratingMotionDuringSaccades = self.session.load('stimuli/fs/saccade/motion')

        # Compute latency from fictive saccade to probe stimulus
        probeLatenciesFictive = np.full(probeTimestamps.size, np.nan)
        saccadeLabels = np.full(probeTimestamps.size, np.nan)
        for iTrial in range(probeTimestamps.size):
            iSaccade = np.argmin(np.abs(probeTimestamps[iTrial] - saccadeTimestamps))
            probeLatenciesFictive[iTrial] = probeTimestamps[iTrial] - saccadeTimestamps[iSaccade]
            saccadeLabels[iTrial] = gratingMotionDuringSaccades[iSaccade] * -1

        # Compute the latency from real saccades to probe stimuli
        probeLatenciesReal = np.full(probeTimestamps.size, np.nan)
        saccadeTimestampsReal = np.delete(
            self.session.saccadeTimestamps[:, 0],
            np.isnan(self.session.saccadeTimestamps[:, 0])
        )
        for iTrial in range(probeTimestamps.size):
            iSaccade = np.argmin(np.abs(probeTimestamps[iTrial] - saccadeTimestampsReal))
            probeLatenciesReal[iTrial] = probeTimestamps[iTrial] - saccadeTimestampsReal[iSaccade]
        trialIndices = np.where(np.logical_or(
            probeLatenciesReal < coincidenceWindow[0],
            probeLatenciesReal > coincidenceWindow[1]
        ))[0]

        return probeTimestamps[trialIndices], probeLatenciesFictive[trialIndices], saccadeLabels[trialIndices], gratingMotionDuringProbes[trialIndices]

    def computeExtrasaccadicPeths(
        self,
        **kwargs
        ):
        """
        """

        kwargs['saccadeType'] = 'fictive'
        kwargs['perisaccadicWindow'] = (-0.2, 0.2)
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
        kwargs['trange'] = (0, 0.2)
        kwargs['tstep'] = 0.2
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
        minimumTrialCount=5,
        rate=50,
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

    def measureSaccadeResponseCorrelation(
        self,
        responseWindow=(-0.2, 0.5),
        ):
        """
        """

        nUnits = len(self.ukeys)
        R = np.full(nUnits, np.nan)
        P = np.full(nUnits, np.nan)
        binIndices = np.logical_and(
            self.tSaccade >= responseWindow[0],
            self.tSaccade <= responseWindow[1]
        )
        for iUnit in range(nUnits):
            session = self._getSessionFromUnitKey(self.ukeys[iUnit])
            saccadeDirection = convertGratingMotionToSaccadeDirection(
                self.preference[iUnit],
                session.eye,
            )
            pethReal = self.ns[f'psths/{saccadeDirection}/real'][iUnit, binIndices]
            pethFictive = self.ns[f'psths/{saccadeDirection}/fictive'][iUnit, binIndices]
            if any([np.isnan(pethReal).all(), np.isnan(pethFictive).all()]):
                continue
            r, p = pearsonr(pethReal, pethFictive)
            R[iUnit] = r
            P[iUnit] = p

        return R, P

    def measureSaccadeResponseAmplitude(
        self,
        responseWindow=(-0.2, 0.5),
        baselineWindow=(-1, -0.5),
        ):
        """
        """

        nUnits = len(self.ukeys)
        aReal = np.full(nUnits, np.nan)
        aFictive = np.full(nUnits, np.nan)
        binIndicesForResponse = np.logical_and(
            self.tSaccade >= responseWindow[0],
            self.tSaccade <= responseWindow[1]
        )
        binIndicesForBaseline = np.logical_and(
            self.tSaccade >= baselineWindow[0],
            self.tSaccade <= baselineWindow[1]
        )
        for iUnit in range(nUnits):
            session = self._getSessionFromUnitKey(self.ukeys[iUnit])
            saccadeDirection = convertGratingMotionToSaccadeDirection(
                self.preference[iUnit],
                session.eye,
            )
            frReal = self.ns[f'psths/{saccadeDirection}/real'][iUnit, binIndicesForResponse]
            frFictive = self.ns[f'psths/{saccadeDirection}/fictive'][iUnit, binIndicesForResponse]
            blReal = self.ns[f'psths/{saccadeDirection}/real'][iUnit, binIndicesForBaseline].mean()
            blFictive = self.ns[f'psths/{saccadeDirection}/fictive'][iUnit, binIndicesForBaseline].mean()
            yReal = frReal - blReal
            yFictive = frFictive - blFictive

            aReal[iUnit] = yReal[np.argmax(np.abs(yReal))]
            aFictive[iUnit] = yFictive[np.argmax(np.abs(yReal))]

        return aReal, aFictive

    def plotExamples(
        self,
        windowIndex=5,
        componentIndex=0,
        figsize=(5, 4),
        ):
        """
        """

        fig, grid = plt.subplots(nrows=len(self.examples), ncols=4)
        if len(self.examples) == 1:
            grid = np.atleast_2d(grid)

        #
        iterable = list(zip(
            ['extra', 'peri', 'extra', 'peri'],
            ['real', 'real', 'fictive', 'fictive'],
            ['k', 'k', 'r', 'r']
        ))

        for i, ukey in enumerate(self.examples):
            iUnit = self._indexUnitKey(ukey)
            for j, (trialType, saccadeType, color) in enumerate(iterable):

                #
                if trialType == 'peri':
                    if saccadeType == 'fictive':
                        peth = self.ns[f'ppths/pref/{saccadeType}/{trialType}'][iUnit, :, 0]
                        params = self.ns[f'params/pref/{saccadeType}/{trialType}'][iUnit, 0, componentIndex, :]
                    else:
                        peth = self.ns[f'ppths/pref/{saccadeType}/{trialType}'][iUnit, :, windowIndex]
                        params = self.ns[f'params/pref/{saccadeType}/{trialType}'][iUnit, windowIndex, componentIndex, :]
                else:
                    peth = self.ns[f'ppths/pref/{saccadeType}/{trialType}'][iUnit, :]
                    params = self.ns[f'params/pref/{saccadeType}/{trialType}'][iUnit, :]

                #
                grid[i, j].plot(self.tProbe, peth, color='0.8')

                #
                abcd = np.delete(params, np.isnan(params))
                if abcd.size == 0:
                    continue
                abc, d = abcd[:-1], abcd[-1]
                A, B, C = np.split(abc, 3)
                a, b, c = A[componentIndex], B[componentIndex], C[componentIndex]
                t2 = np.linspace(-15 * c, 15 * c, 100) + b
                y2 = g(t2, a, b, c, d)
                if saccadeType == 'fictive':
                    color = self.colors[i]
                grid[i, j].plot(t2, y2, color=color)

        #
        for axs in grid:
            ylim = [np.inf, -np.inf]
            for ax in axs:
                y1, y2 = ax.get_ylim()
                if y1 < ylim[0]:
                    ylim[0] = y1
                if y2 > ylim[1]:
                    ylim[1] = y2
            for ax in axs:
                ax.set_ylim(ylim)
        
        #
        for ax in grid[:, 1:].flatten():
            ax.set_yticklabels([])
        
        #
        for ax in grid.flatten():
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)

        #
        titles = (r'$R_{P (Extra, Real)}$', r'$R_{P (Peri, Real)}$', r'$R_{P (Extra, Fictive)}$', r'$R_{P (Peri, Fictive)}$')
        for j in range(4):
            grid[0, j].set_title(titles[j], fontsize=10)
        fig.supxlabel('Time from probe (sec)', fontsize=10)
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, grid

    def _classifyUnitsByAmplitude(
        self,
        minimumResponseAmplitude=2,
        ):
        """
        """

        nUnits = len(self.ukeys)
        uTypes = np.full(nUnits, np.nan)
        aReal, aFictive = self.measureSaccadeResponseAmplitude()
        for iUnit in range(nUnits):
            check = np.vstack([
                aReal[iUnit] > (-1 * minimumResponseAmplitude),
                aReal[iUnit] < minimumResponseAmplitude,
                aFictive[iUnit] > (-1 * minimumResponseAmplitude),
                aFictive[iUnit] < minimumResponseAmplitude
            ]).all(0)
            if check:
                utype = 3
            else:
                if aReal[iUnit] > minimumResponseAmplitude and aFictive[iUnit] > minimumResponseAmplitude:
                    utype = 2
                elif aReal[iUnit] < (-1 * minimumResponseAmplitude) and aFictive[iUnit] < (-1 * minimumResponseAmplitude):
                    utype = 2
                elif abs(aReal[iUnit]) > minimumResponseAmplitude and abs(aFictive[iUnit]) < minimumResponseAmplitude:
                    utype = 1
                elif abs(aReal[iUnit]) < minimumResponseAmplitude and abs(aFictive[iUnit]) > minimumResponseAmplitude:
                    utype = 4
                else:
                    utype = 4
            uTypes[iUnit] = utype

        return np.array(uTypes)

    def _classifyUnitsWithCorrelation(
        self,
        minimumResponseAmplitude=2,
        ):
        """
        """

        nUnits = len(self.ukeys)
        labels = np.full(nUnits, np.nan)
        R, P = self.measureSaccadeResponseCorrelation()
        aReal, aFictive = self.measureSaccadeResponseAmplitude()
        for iUnit in range(nUnits):
            check = np.vstack([
                aReal[iUnit] > (-1 * minimumResponseAmplitude),
                aReal[iUnit] < minimumResponseAmplitude,
                aFictive[iUnit] > (-1 * minimumResponseAmplitude),
                aFictive[iUnit] < minimumResponseAmplitude
            ]).all(0)
            if check:
                label = 3
            else:
                if R[iUnit] > 0 and P[iUnit] < 0.05:
                    label = 2
                elif R[iUnit] < 0 and P[iUnit] < 0.05:
                    label = 4
                elif P[iUnit] >= 0.05:
                    label = 1
                else:
                    label = 4
            labels[iUnit] = label

        return labels

    def scatterModulationBySaccadeType(
        self,
        bounds=(-3, 3),
        windowIndex=5,
        componentIndex=0,
        transform=True,
        figsize=(9, 3),
        ):
        """
        """

        #
        fig, axs = plt.subplots(ncols=4)

        #
        u = self._classifyUnitsByAmplitude()
        x = self.ns['mi/pref/real'][:, windowIndex, componentIndex]
        y = self.ns['mi/pref/fictive'][:, 0, componentIndex]
        if transform:
            x = np.tanh(x)
            y = np.tanh(y)
            bounds = (-1, 1)

        #
        for label, ax in zip(np.unique(u), axs):

            #
            mask = u == label

            #
            c = list()
            p = list()
            for iUnit in np.where(mask)[0]:
                pReal = self.ns['p/pref/real'][iUnit, windowIndex, componentIndex]
                pFictive = self.ns['p/pref/fictive'][iUnit, 0, componentIndex]
                p.append(np.min([pReal, pFictive]))
                if pReal < 0.05 and pFictive < 0.05:
                    c.append('xkcd:purple')
                elif pReal < 0.05 and pFictive >= 0.05:
                    c.append('xkcd:orange')
                elif pReal >= 0.05 and pFictive < 0.05:
                    c.append('xkcd:green')
                else:
                    c.append('0.8')
            c = np.array(c)
            order = np.argsort(p)[::-1]

            #
            ax.scatter(
                x[mask][order],
                y[mask][order],
                marker='.',
                s=15,
                c=c[order],
                alpha=0.7,
            )

        #
        for ax in axs:
            ax.vlines(0, *bounds, color='k', alpha=0.5, linestyle=':')
            ax.hlines(0, *bounds, color='k', alpha=0.5, linestyle=':')
            ax.set_ylim(bounds)
            ax.set_xlim(bounds)
            ax.set_aspect('equal')
            ax.set_xlabel('MI (Real)')
        axs[0].set_ylabel('MI (Fictive)')
        titles = ('Type 1', 'Type 2', 'Type 3', 'Type 4')
        for i in range(len(axs)):
            axs[i].set_title(titles[i], fontsize=10)
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, axs

    def scatterSaccadeResponseAmplitude(
        self,
        figsize=(4, 4)
        ):
        """
        """

        fig, ax = plt.subplots()
        aReal, aFictive = self.measureSaccadeResponseAmplitude()
        u = self._classifyUnitsByAmplitude()
        colors = ('0.8', 'y', 'm', 'c')
        alphas = (0.5, 0.7, 0.7, 0.7)
        for i, label in enumerate(np.unique(u)[::-1]):
            mask = u == label
            ax.scatter(
                aReal[mask],
                aFictive[mask],
                marker='.',
                s=15,
                color=colors[i],
                alpha=alphas[i],
            )
            
        ax.set_ylabel('Amplitude (Fictive)', fontsize=10)
        ax.set_xlabel('Amplitude (Real)', fontsize=10)
        ax.legend([
            'IV',
            'III',
            'II',
            'I'
        ])
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, ax

    def plotSaccadeResponseHeatmaps(
        self,
        responseWindow=(-0.2, 0.5),
        baselineWindow=(-1, -0.5),
        vrange=(-30, 30),
        cmap='coolwarm',
        figsize=(4, 6),
        ):
        """
        """

        fig, axs = plt.subplots(ncols=2, sharey=True)
        binIndicesForResponse = np.logical_and(
            self.tSaccade >= responseWindow[0],
            self.tSaccade <= responseWindow[1]
        )
        binIndicesForBaseline = np.logical_and(
            self.tSaccade >= baselineWindow[0],
            self.tSaccade <= baselineWindow[1]
        )
        nUnits = len(self.ukeys)
        data = {
            'real': np.full([nUnits, binIndicesForResponse.sum()], np.nan),
            'fictive': np.full([nUnits, binIndicesForResponse.sum()], np.nan),
        }
        for saccadeType in data.keys():
            for iUnit in range(len(self.ukeys)):
                session = self._getSessionFromUnitKey(self.ukeys[iUnit])
                saccadeDirection = convertGratingMotionToSaccadeDirection(
                    self.preference[iUnit],
                    session.eye,
                )
                fr = self.ns[f'psths/{saccadeDirection}/{saccadeType}'][iUnit, binIndicesForResponse]
                bl = self.ns[f'psths/{saccadeDirection}/{saccadeType}'][iUnit, binIndicesForBaseline].mean()
                data[saccadeType][iUnit, :] = (fr - bl) / self.factor[iUnit]

        #
        R, P = self.measureSaccadeResponseCorrelation()
        aReal, aFictive = self.measureSaccadeResponseAmplitude()
        amplitudeDifference = aFictive - aReal
        unitIndex = np.argsort(np.abs(R))
        # unitIndex = np.argsort(amplitudeDifference)
        axs[0].pcolor(self.tSaccade[binIndicesForResponse], np.arange(nUnits), data['real'][unitIndex], vmin=vrange[0], vmax=vrange[1], cmap=cmap)
        axs[1].pcolor(self.tSaccade[binIndicesForResponse], np.arange(nUnits), data['fictive'][unitIndex], vmin=vrange[0], vmax=vrange[1], cmap=cmap)

        #
        fig.supxlabel('Time from saccade', fontsize=10)
        fig.supylabel('Unit # (Sorted by correlation coefficient)', fontsize=10)
        axs[0].set_title('Real', fontsize=10)
        axs[1].set_title('Fictive', fontsize=10)
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, axs