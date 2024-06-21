import h5py
import numpy as np
from scipy.signal import find_peaks as findPeaks
from scipy.stats import pearsonr, sem
from matplotlib import pyplot as plt
from myphdlib.figures.analysis import g, convertGratingMotionToSaccadeDirection, convertSaccadeDirectionToGratingMotion
from myphdlib.figures.bootstrap import BootstrappedSaccadicModulationAnalysis
from myphdlib.general.toolkit import psth2
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec

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
            ('2023-07-20', 'mlati9', 329),
            ('2023-07-03', 'mlati10', 152),
        )

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

    def classifyUnitsByAmplitude(
        self,
        minimumResponseAmplitude=2,
        ratioThreshold=0.7,
        method='ratio',
        ):
        """
        """

        nUnits = len(self.ukeys)
        U = np.full(nUnits, np.nan)
        R = np.full(nUnits, np.nan)
        aReal, aFictive = self.measureSaccadeResponseAmplitude()
        for iUnit in range(nUnits):

            # Identify type III units (same for both methods)
            check = np.vstack([
                aReal[iUnit] > (-1 * minimumResponseAmplitude),
                aReal[iUnit] < minimumResponseAmplitude,
                aFictive[iUnit] > (-1 * minimumResponseAmplitude),
                aFictive[iUnit] < minimumResponseAmplitude
            ]).all(0)
            if check:
                utype = 3

            #
            else:

                #
                if method == 'Alon':
                    if aReal[iUnit] > minimumResponseAmplitude and aFictive[iUnit] > minimumResponseAmplitude:
                        utype = 1
                    elif aReal[iUnit] < (-1 * minimumResponseAmplitude) and aFictive[iUnit] < (-1 * minimumResponseAmplitude):
                        utype = 1
                    elif abs(aReal[iUnit]) > minimumResponseAmplitude and abs(aFictive[iUnit]) < minimumResponseAmplitude:
                        utype = 2
                    elif abs(aReal[iUnit]) < minimumResponseAmplitude and abs(aFictive[iUnit]) > minimumResponseAmplitude:
                        utype = 4
                    else:
                        utype = 4

                #
                elif method == 'ratio':
                    r = abs((aFictive[iUnit] / aReal[iUnit]) - 1)
                    R[iUnit] = r
                    if abs(aFictive[iUnit]) < minimumResponseAmplitude:
                        utype = 2
                    elif r <= ratioThreshold:
                        utype = 1
                    else:
                        utype = 4

            #
            U[iUnit] = utype

        #
        U = np.array(U)
        return U, R

    def classifyUnitsByCorrelation(
        self,
        minimumResponseAmplitude=2,
        responseWindow=(-0.5, 0.5),
        alpha=0.02,
        ):
        """
        """

        #
        nUnits = len(self.ukeys)
        U = np.full(nUnits, np.nan)
        R = np.full(nUnits, np.nan)
        P = np.full(nUnits, np.nan)

        #
        aReal, aFictive = self.measureSaccadeResponseAmplitude()

        #
        binIndicesForResponse = np.logical_and(
            self.tSaccade >= responseWindow[0],
            self.tSaccade <= responseWindow[1]
        )

        #
        for iUnit in range(nUnits):

            # Identify type III units (same for both methods)
            check = np.vstack([
                aReal[iUnit] > (-1 * minimumResponseAmplitude),
                aReal[iUnit] < minimumResponseAmplitude,
                aFictive[iUnit] > (-1 * minimumResponseAmplitude),
                aFictive[iUnit] < minimumResponseAmplitude
            ]).all(0)
            if check:
                utype = 3

            #
            else:
                session = self._getSessionFromUnitKey(self.ukeys[iUnit])
                saccadeDirection = convertGratingMotionToSaccadeDirection(
                    self.preference[iUnit],
                    session.eye
                )
                yReal = self.ns[f'psths/{saccadeDirection}/real'][iUnit, binIndicesForResponse]
                yFictive = self.ns[f'psths/{saccadeDirection}/fictive'][iUnit, binIndicesForResponse]
                if np.any([np.isnan(yReal).all(), np.isnan(yFictive).all()]):
                    utype = np.nan
                else:
                    r, p = pearsonr(yReal, yFictive)
                    R[iUnit] = r
                    P[iUnit] = p
                    if r > 0 and p < alpha:
                        utype = 1
                    elif r < 0 and p >= alpha:
                        utype = 4
                    elif p >= alpha:
                        utype = 2

            #
            U[iUnit] = utype

        return U, R

    def plotExamples(
        self,
        windowIndex=5,
        componentIndex=0,
        responseWindowForSaccades=(-0.5, 0.5),
        baselineWindowForSaccades=(-1, -0.5),
        colors=('c', 'm'),
        figsize=(5, 3),
        ):
        """
        """

        fig, grid = plt.subplots(nrows=len(self.examples), ncols=4)
        if len(self.examples) == 1:
            grid = np.atleast_2d(grid)

        #
        iterable = list(zip(
            ['extra', 'peri', 'peri'],
            ['real', 'real', 'fictive'],
            ['k', 'k', 'r']
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
                grid[i, j].plot(t2, y2, color='k')

        #
        binIndicesForResponse = np.logical_and(
            self.tSaccade >= responseWindowForSaccades[0],
            self.tSaccade <= responseWindowForSaccades[1]
        )
        binIndicesForBaseline = np.logical_and(
            self.tSaccade >= baselineWindowForSaccades[0],
            self.tSaccade <= baselineWindowForSaccades[1]
        )
        for i, ukey in enumerate(self.examples):
            iUnit = self._indexUnitKey(ukey)
            session = self._getSessionFromUnitKey(ukey)
            saccadeDirection = convertGratingMotionToSaccadeDirection(
                self.preference[iUnit],
                session.eye
            )
            for saccadeType, linestyle, color in zip(['real', 'fictive'], ['-', '-'], ['k', 'r']):
                psth = self.ns[f'psths/{saccadeDirection}/{saccadeType}'][iUnit, :]
                bl = psth[binIndicesForBaseline].mean()
                fr = (psth - bl) / self.factor[iUnit]
                grid[i, -1].plot(
                    self.tSaccade[binIndicesForResponse],
                    fr[binIndicesForResponse],
                    color=color,
                    alpha=0.5,
                    linestyle=linestyle
                )

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
        titles = (r'$R_{P (Extra, Real)}$', r'$R_{P (Peri, Real)}$', r'$R_{P (Peri, Fictive)}$', r'$R_{Saccade}$')
        for j in range(4):
            grid[0, j].set_title(titles[j], fontsize=10)
        fig.supxlabel('Time from probe (sec)', fontsize=10)
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, grid

    def plotAverageSaccadeResponseBySaccadeType(
        self,
        labels=(1, 2),
        responseWindow=(-0.2, 0.5),
        baselineWindow=(-1, -0.5),
        error='sem',
        figsize=(2, 4),
        axs=None,
        ):
        """
        """

        if axs is None:
            fig, axs = plt.subplots(nrows=len(labels), sharey=True)
        else:
            fig = axs[0].figure
        u, r = self.classifyUnitsByAmplitude()
        binIndicesForResponse = np.logical_and(
            self.tSaccade >= responseWindow[0],
            self.tSaccade <= responseWindow[1]
        )
        binIndicesForBaseline = np.logical_and(
            self.tSaccade >= baselineWindow[0],
            self.tSaccade <= baselineWindow[1]
        )
        result = list()
        for ax, label in zip(axs, labels):
            mask = np.vstack([
                u == label,
            ]).all(0)
            psths = {
                'real': np.full([mask.sum(), binIndicesForResponse.sum()], np.nan),
                'fictive': np.full([mask.sum(), binIndicesForResponse.sum()], np.nan),
            }
            for i, iUnit in enumerate(np.where(mask)[0]):
                session = self._getSessionFromUnitKey(self.ukeys[iUnit])
                saccadeDirection = convertGratingMotionToSaccadeDirection(
                    self.preference[iUnit],
                    session.eye,
                )
                for saccadeType in ('real', 'fictive'):
                    psth = self.ns[f'psths/{saccadeDirection}/{saccadeType}'][iUnit, :]
                    bl = psth[binIndicesForBaseline].mean()
                    fr = (psth[binIndicesForResponse] - bl)
                    if saccadeType == 'real':
                        ar = fr[np.argmax(np.abs(fr))]
                    fr /= ar
                    if fr[np.argmax(np.abs(fr))] < 0:
                        fr *= -1
                    psths[saccadeType][i] = fr
            result.append(psths)

            #
            ax.plot(
                self.tSaccade[binIndicesForResponse],
                np.nanmean(psths['real'], axis=0),
                color='k',
                alpha=0.5,
                label='Real'
            )
            if error == 'std':
                e = np.nanstd(psths['real'], axis=0)
            elif error == 'sem':
                e = sem(psths['real'], axis=0, nan_policy='omit')
            ax.fill_between(
                self.tSaccade[binIndicesForResponse],
                np.nanmean(psths['real'], axis=0) - e,
                np.nanmean(psths['real'], axis=0) + e,
                color='k',
                alpha=0.1,
            )
            ax.plot(
                self.tSaccade[binIndicesForResponse],
                np.nanmean(psths['fictive'], axis=0),
                color='r',
                alpha=0.5,
                label='Fictive'
            )
            if error == 'std':
                e = np.nanstd(psths['fictive'], axis=0)
            elif error == 'sem':
                e = sem(psths['fictive'], axis=0, nan_policy='omit')
            ax.fill_between(
                self.tSaccade[binIndicesForResponse],
                np.nanmean(psths['fictive'], axis=0) - e,
                np.nanmean(psths['fictive'], axis=0) + e,
                color='r',
                alpha=0.1,
            )

        #
        for ax in axs:
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
        axs[0].set_title('Type I', fontsize=10)
        axs[1].set_title('Type II', fontsize=10)
        axs[-1].legend()
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, axs

    def scatterModulationBySaccadeType(
        self,
        xylim=3,
        windowIndex=5,
        componentIndex=0,
        labels=(1, 2),
        colors=('xkcd:violet', 'xkcd:light blue', 'xkcd:light red', 'xkcd:gray'),
        figsize=(2.5, 4.5),
        ):
        """
        """

        #
        fig, axs = plt.subplots(nrows=len(labels))

        #
        u, r = self.classifyUnitsByAmplitude()
        x = np.clip(self.ns['mi/pref/real'][:, windowIndex, componentIndex], -1 * xylim, xylim)
        y = np.clip(self.ns['mi/pref/fictive'][:, 0, componentIndex], -1 * xylim, xylim)

        #
        for label, ax in zip(labels, axs):

            #
            mask = np.vstack([
                u == label,
                # self.ns['p/pref/real'][:, windowIndex, componentIndex] < 0.05,
                # self.ns['mi/pref/real'][:, windowIndex, componentIndex] < 0,
                np.invert(np.isnan(self.ns['mi/pref/fictive'][:, 0, componentIndex]))
            ]).all(0)

            #
            C = list()
            P = list()
            for iUnit in np.where(mask)[0]:

                # load MI and p-value
                pReal = self.ns['p/pref/real'][iUnit, windowIndex, componentIndex]
                # mReal = self.ns['mi/pref/real'][iUnit, windowIndex, componentIndex]
                pFictive = self.ns['p/pref/fictive'][iUnit, 0, componentIndex]
                # mFictive = self.ns['mi/pref/fictive'][iUnit, 0, componentIndex]

                # Real and fictive saccadic suppression
                if pReal < 0.05 and pFictive < 0.05:
                    C.append(colors[0])
                    P.append(pFictive)

                # Real saccdic suppresssion and fictive saccadic enhancement
                elif pReal < 0.05 and pFictive >= 0.05:
                    C.append(colors[1])
                    P.append(pFictive)

                # Real saccadic suppression and no modulation for fictive saccades
                elif pReal >= 0.05 and pFictive < 0.05:
                    C.append(colors[2])
                    P.append(pFictive)

                #
                elif pReal >= 0.05 and pFictive >= 0.05:
                    C.append(colors[3])
                    P.append(pFictive)

            #
            C = np.array(C)
            order = np.argsort(P)[::-1]

            #
            ax.scatter(
                x[mask][order],
                y[mask][order],
                marker='.',
                s=15,
                c=C[order],
                alpha=0.5,
                clip_on=False,
            )

            # Indicate examples
            for i, iUnit in enumerate(np.where(mask)[0]):
                if self._matchUnitKey(self.ukeys[iUnit], self.examples):
                    ax.scatter(
                        x[iUnit],
                        y[iUnit],
                        color=C[i],
                        edgecolor='k',
                        marker='D',
                        zorder=3,
                    )

            #
            div = make_axes_locatable(ax)
            tax = div.append_axes('top', size='20%', pad=0.05)
            samples = list()
            for c in colors:
                samples.append(np.clip(x[mask][C == c], -1 * xylim, xylim))
            tax.hist(
                samples,
                color=('0.0', '0.0', 'w', 'w'),
                # color=(colors[0], colors[1], 'w', 'w'),
                histtype='barstacked',
                range=(-1 * xylim, xylim),
                bins=30,
            )
            tax.hist(
                np.concatenate([sample for sample in samples]),
                edgecolor='k',
                histtype='step',
                range=(-1 * xylim, xylim),
                bins=30,
            )
            for sp in ('top', 'right'):
                tax.spines[sp].set_visible(False)
            tax.set_xticks([])
            tax.set_xlim([-1 * xylim, xylim])

        #
        for ax in axs:
            ax.vlines(0, -1 * xylim, xylim, color='k', alpha=0.5, linestyle=':')
            ax.hlines(0, -1 * xylim, xylim, color='k', alpha=0.5, linestyle=':')
            ax.set_ylim([-1 * xylim, xylim])
            ax.set_xlim([-1 * xylim, xylim])
            ax.set_aspect('equal')
            ax.set_ylabel('MI (Fictive)')
        axs[1].set_xlabel('MI (Real)')
        axs[1].set_yticklabels([])
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
        u, r = self.classifyUnitsByAmplitude()
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