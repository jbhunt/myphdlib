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
from matplotlib.colors import LinearSegmentedColormap
from cmcrameri import cm

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
            ('2023-07-21', 'mlati10', 47),
            ('2023-07-05', 'mlati9', 206)
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

            #
            mi1 = self.ns['mi/pref/fictive'][iUnit, 0, 0]
            mi2 = self.ns['mi/null/fictive'][iUnit, 0, 0]
            if all(np.isnan([mi1, mi2])):
                continue

            # 
            check = np.vstack([
                aReal[iUnit] > (-1 * minimumResponseAmplitude),
                aReal[iUnit] < minimumResponseAmplitude,
                aFictive[iUnit] > (-1 * minimumResponseAmplitude),
                aFictive[iUnit] < minimumResponseAmplitude
            ]).all(0)
            if check:
                utype = 2

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
                        utype = 2
                    else:
                        utype = 2

                #
                elif method == 'ratio':
                    r = (aFictive[iUnit] / aReal[iUnit]) - 1
                    R[iUnit] = r
                    if abs(aFictive[iUnit]) < minimumResponseAmplitude:
                        utype = 2
                    elif r < 0 and abs(r) <= ratioThreshold:
                        utype = 1
                    elif r > 0:
                        utype = 1
                    else:
                        utype = 2

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
                utype = 2

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
                        utype = 2
                    elif p >= alpha:
                        utype = 2

            #
            U[iUnit] = utype

        return U, R

    def _plotExamples(
        self,
        grid,
        windowIndex=5,
        componentIndex=0,
        responseWindowForSaccades=(-0.2, 0.5),
        baselineWindowForSaccades=(-1, -0.5),
        ):
        """
        """

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
                grid[i, 0].plot(
                    self.tSaccade[binIndicesForResponse],
                    fr[binIndicesForResponse],
                    color=color,
                    alpha=0.5,
                    linestyle=linestyle
                )

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
                grid[i, j + 1].plot(self.tProbe, peth, color='0.8')

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
                    color = 'r'
                else:
                    color = 'k'
                grid[i, j + 1].plot(t2, y2, color=color)

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
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)
        
        #
        for ax in grid.flatten():
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            ax.set_xticks([-0.2, 0, 0.5])
            ax.set_xlim(responseWindowForSaccades)

        #
        # titles = (r'$R_{P (Extra)}$', r'$R_{P (Peri, Real)}$', r'$R_{P (Peri, Fictive)}$', r'$R_{Saccade}$')
        # for j in range(4):
        #     grid[0, j].set_title(titles[j], fontsize=10)
                

        return

    def _plotAverageSaccadeResponseBySaccadeType(
        self,
        axs=None,
        labels=(1, 2),
        responseWindow=(-0.2, 0.5),
        baselineWindow=(-1, -0.5),
        centeredWindow=(-0.2, 0.2),
        parametric=True,
        figsize=None
        ):
        """
        """

        u, r = self.classifyUnitsByAmplitude(method='ratio')
        binIndicesForResponse = np.logical_and(
            self.tSaccade >= responseWindow[0],
            self.tSaccade <= responseWindow[1]
        )
        binIndicesForBaseline = np.logical_and(
            self.tSaccade >= baselineWindow[0],
            self.tSaccade <= baselineWindow[1]
        )
        tCentered = np.linspace(*centeredWindow, 31)
        result = list()
        if axs is None:
            fig, axs = plt.subplots(ncols=2, sharex=True, sharey=True)
        for ax, label in zip(axs, labels):
            mask = np.vstack([
                u == label,
            ]).all(0)
            psths = {
                'real': np.full([mask.sum(), tCentered.size], np.nan),
                'fictive': np.full([mask.sum(), tCentered.size], np.nan),
            }
            for i, iUnit in enumerate(np.where(mask)[0]):

                #
                session = self._getSessionFromUnitKey(self.ukeys[iUnit])
                unit = session.population.indexByCluster(self.ukeys[iUnit][-1])
                bins = np.arange(0, 1000 + 0.01, 0.01)
                counts, edges = np.histogram(unit.timestamps, bins=bins)
                fr = counts / 0.01
                sigma = fr.std()

                #
                saccadeDirection = convertGratingMotionToSaccadeDirection(
                    self.preference[iUnit],
                    session.eye,
                )

                #
                for saccadeType in ('real', 'fictive'):
                    psth = self.ns[f'psths/{saccadeDirection}/{saccadeType}'][iUnit, :]
                    zeroed = psth[binIndicesForResponse] - psth[binIndicesForBaseline].mean()
                    iPeak = np.argmax(np.abs(zeroed))
                    tPeak = self.tSaccade[binIndicesForResponse][iPeak]
                    if zeroed[iPeak] < 0:
                        coef = -1
                    else:
                        coef = +1

                    #
                    standardized = zeroed / sigma # Stand
                    if saccadeType == 'real':
                        scalingFactor = np.abs(standardized).max()
                    scaled = standardized / scalingFactor

                    #
                    centered = np.interp(
                        tCentered + tPeak,
                        self.tSaccade[binIndicesForResponse],
                        scaled,
                        left=np.nan,
                        right=np.nan
                    )
                    signed = centered * coef
                    psths[saccadeType][i] = signed
            result.append(psths)

            #
            for saccadeType, color in zip(['real', 'fictive'], ['k', 'r']):
                if parametric:
                    y1 = np.nanmean(psths[saccadeType], axis=0)
                    error = np.array([
                        sem(sample[np.invert(np.isnan(sample))])
                            for sample in psths[saccadeType].T
                    ])
                    y2 = y1 - error
                    y3 = y1 + error
                else:
                    y1 = np.nanmedian(psths[saccadeType], axis=0)
                    y2 = np.nanpercentile(psths[saccadeType], 25, axis=0)
                    y3 = np.nanpercentile(psths[saccadeType], 75, axis=0)
                ax.plot(
                    tCentered,
                    y1,
                    color=color,
                    alpha=0.5,
                    label=saccadeType,
                )
                ax.fill_between(
                    tCentered,
                    y2,
                    y3,
                    color=color,
                    alpha=0.1,
                    edgecolor='none',
                )

        #
        for ax in axs:
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
        n = np.sum(u == 1)
        axs[0].set_ylabel(f'Type I (n={n})', fontsize=10)
        n = np.sum(u == 2)
        axs[1].set_ylabel(f'Type II (n={n})', fontsize=10)
        
        ylim = [np.inf, -np.inf]
        for ax in axs:
            y1, y2 = ax.get_ylim()
            if y1 < ylim[0]:
                ylim[0] = y1
            if y2 > ylim[1]:
                ylim[1] = y2
        for ax in axs:
            ax.set_ylim(ylim)
            ax.set_xticks([centeredWindow[0], 0, centeredWindow[1]])
            ax.set_xlim(centeredWindow)

        #
        fig = axs[0].figure
        if figsize is not None:
            fig.set_figwidth(figsize[0])
            fig.set_figheight(figsize[1])
            fig.tight_layout()

        return fig, axs
    
    def plotEventRelatedActivity(
        self,
        figsize=(6, 5)
        ):
        """
        """

        fig = plt.figure()
        gs = gridspec.GridSpec(2, 5)
        axs1 = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[1, 0])
        ]
        self._plotAverageSaccadeResponseBySaccadeType(
            axs1
        )
        axs2 = np.array([
            [fig.add_subplot(gs[0, j]) for j in range(1, 5, 1)],
            [fig.add_subplot(gs[1, j]) for j in range(1, 5, 1)],
        ])
        self._plotExamples(axs2)
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.subplots_adjust(hspace=0.3)
        axs = np.hstack([np.array(axs1).reshape(-1, 1), axs2])

        return fig, axs

    def scatterModulationBySaccadeType(
        self,
        xylim=3,
        windowIndex=5,
        componentIndex=0,
        labels=(1, 2),
        figsize=(2.5, 4.5),
        ):
        """
        """

        colorspace = [
            (0, '#ff8080'),
            (0.5, '#8c4d4d'),
            (1, '#808080')

        ]
        cmap = LinearSegmentedColormap.from_list(
            'kr',
            colorspace,
            3
        )        

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
                np.invert(np.isnan(self.ns['mi/pref/fictive'][:, 0, componentIndex])),
                np.invert(np.isnan(self.ns['mi/pref/real'][:, windowIndex, componentIndex]))
            ]).all(0)

            #
            C = list()
            P = list()
            L = list()
            for iUnit in np.where(mask)[0]:

                # load MI and p-value
                pReal = self.ns['p/pref/real'][iUnit, windowIndex, componentIndex]
                pFictive = self.ns['p/pref/fictive'][iUnit, 0, componentIndex]

                # Real and fictive saccadic suppression
                if pReal < 0.05 and pFictive < 0.05:
                    c = list(cmap(1))
                    c[-1] = 0.5
                    C.append(c)
                    P.append(pFictive)
                    L.append(1)

                # Real saccdic suppresssion and fictive saccadic enhancement
                elif pReal < 0.05 and pFictive >= 0.05:
                    c = list(cmap(0))
                    c[-1] = 0.5
                    C.append(c)
                    P.append(pFictive)
                    L.append(0)

                # Real saccadic suppression and no modulation for fictive saccades
                elif pReal >= 0.05 and pFictive < 0.05:
                    c = list(cmap(2))
                    c[-1] = 0.5
                    C.append(c)
                    P.append(pFictive)
                    L.append(2)

                #
                elif pReal >= 0.05 and pFictive >= 0.05:
                    C.append(np.zeros(4))
                    P.append(pFictive)
                    L.append(3)

            #
            C = np.array(C)
            L = np.array(L)
            order = np.concatenate([
                np.where(L == 3)[0],
                np.where(L == 2)[0],
                np.where(L == 0)[0],
                np.where(L == 1)[0],
            ])

            #
            ax.scatter(
                x[mask][order],
                y[mask][order],
                marker='.',
                s=40,
                edgecolor='none',
                c=C[order],
                clip_on=False,
                rasterized=True
            )

            # Indicate examples
            for i, iUnit in enumerate(np.where(mask)[0]):
                if self._matchUnitKey(self.ukeys[iUnit], self.examples):
                    ax.scatter(
                        x[iUnit],
                        y[iUnit],
                        color='k',
                        edgecolor='none',
                        marker='.',
                        zorder=3,
                        s=40
                    )

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
    
    def contourModulationBySaccadeType(
        self,
        labels=(1, 2),
        windowIndex=5,
        componentIndex=0,
        xylim=3,
        nLevels=10,
        nBins=20,
        fmax=None,
        fmin=None,
        cmap='viridis',
        ):
        """
        """

        #
        fig, axs = plt.subplots(ncols=len(labels))

        #
        u, r = self.classifyUnitsByAmplitude()
        x = np.clip(self.ns['mi/pref/real'][:, windowIndex, componentIndex], -1 * xylim, xylim)
        y = np.clip(self.ns['mi/pref/fictive'][:, 0, componentIndex], -1 * xylim, xylim)

        #
        for ax, label in zip(axs, labels):
            mask = np.vstack([
                u == label,
                np.invert(np.isnan(x)),
                np.invert(np.isnan(y))
            ]).all(0)
            Z, X, Y = np.histogram2d(
                x[mask],
                y[mask],
                range=[[-xylim, xylim], [-xylim, xylim]],
                bins=nBins,
            )
            X1 = X[:-1] + ((X[1] - X[0]) / 2)
            Y1 = Y[:-1] + ((Y[1] - Y[0]) / 2)            
            zNormed = Z / Z.sum()
            if fmax is None:
                levels = np.linspace(fmin, zNormed.max(), nLevels)
            else:
                levels = np.linspace(fmin, fmax, nLevels)
            ax.contourf(
                X1,
                Y1,
                zNormed,
                levels=levels,
                cmap=cmap,
                # colors='k'
            )
            ax.vlines(0, -xylim, xylim, color='w', linestyle=':')
            ax.hlines(0, -xylim, xylim, color='w', linestyle=':')
            ax.set_aspect('equal')
            ax.set_xlim([-xylim, xylim])
            ax.set_ylim([-xylim, xylim])

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
    
    def plotUnitTypeByVisualResponseComplexity(
        self,
        figsize=(1.5, 3.5)
        ):
        """
        """

        fig, ax = plt.subplots()
        u, r = self.classifyUnitsByAmplitude()
        k = list()
        for iUnit in range(len(self.ukeys)):
            params = self.ns['params/pref/real/extra'][iUnit]
            abcd = np.delete(params, np.isnan(params))
            abc = abcd[:-1]
            A, B, C = np.split(abc, 3)
            ki = A.size
            if A[0] < 0:
                ki *= -1
            k.append(ki)
        k = np.array(k)
        combos = np.vstack([
            self.labels,
            u
        ]).T
        combos = np.delete(
            combos,
            np.isnan(combos).any(1),
            axis=0
        )
        uniqueCombos, counts = np.unique(combos, axis=0, return_counts=True)

        #
        data = np.full(uniqueCombos.shape, np.nan)
        for i, complexity in enumerate(np.unique(self.labels)):
            for j, utype in enumerate(np.unique(u)):
                test = np.array([complexity, utype])
                index = np.where([np.array_equal(row, test) for row in uniqueCombos])[0]
                count = counts[index]
                data[i, j] = count
        data[:, 0] /= np.nansum(data[:, 0])
        data[:, 1] /= np.nansum(data[:, 1])
        data = np.around(data, 2)

        #
        for (complexity, utype), count in zip(uniqueCombos, counts):
            rowIndices = np.where(uniqueCombos[:, 1] == utype)[0]
            n = counts[rowIndices].sum()
            s = count / n * 300
            ax.scatter(utype, complexity, s=s, color='k', marker='o', edgecolor='none')

        ax.scatter([1, 2], [0, 0], s=1 * 300, marker='o', color='k', alpha=0.1, edgecolor='none')

        #
        ax.set_yticks(np.arange(-1, 3 + 1, 1))
        ax.set_xlabel('Type')
        ax.set_ylabel('Signed complexity (k)')
        ax.set_xlim([0.5, 2.5])
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, (ax,), data, uniqueCombos, counts.reshape(4, -1)

    def computeModulationFrequency(
        self,
        windowIndex=5,
        componentIndex=0,
        label=1,
        ):
        """
        """

        #
        u, r = self.classifyUnitsByAmplitude()
        table = np.full([3, 3], 0)
        for iUnit in np.where(u == label)[0]:

            #
            miReal = self.ns['mi/pref/real'][iUnit, windowIndex, componentIndex]
            miFictive = self.ns['mi/pref/fictive'][iUnit, 0, componentIndex]
            pReal = self.ns['p/pref/real'][iUnit, windowIndex, componentIndex]
            pFictive = self.ns['p/pref/fictive'][iUnit, 0, componentIndex]

            #
            if np.isnan([miReal, miFictive, pReal, pFictive]).any():
                continue

            # Suppressed (real) x suppressed (fictive)
            if np.all([miReal < 0, pReal < 0.05, miFictive < 0, pFictive < 0.05]):
                table[0, 0] += 1

            # Enhanced (real) x suppressed (fictive)
            elif np.all([miReal > 0, pReal < 0.05, miFictive < 0, pFictive < 0.05]):
                table[0, 1] += 1

            # Unmodulated (real) x suppressed (fictive)
            elif np.all([pReal >= 0.05, miFictive < 0, pFictive < 0.05]):
                table[0, 2] += 1

            # Suppressed (real) x enhanced (fictive)
            elif np.all([miReal < 0, pReal < 0.05, miFictive > 0, pFictive < 0.05]):
                table[1, 0] += 1

            # Enhanced (real) x enhanced (fictive)
            elif np.all([miReal > 0, pReal < 0.05, miFictive > 0, pFictive < 0.05]):
                table[1, 1] += 1

            # Unodulated (real) x enhanced (fictive)
            elif np.all([pReal >= 0.05, miFictive > 0, pFictive < 0.05]):
                table[1, 2] += 1

            # Suppressed (real) x unmodulated (fictive)
            elif np.all([miReal < 0, pReal < 0.05, pFictive >= 0.05]):
                table[2, 0] += 1

            # Enhanced (real) x unmodulated (fictive)
            elif np.all([miReal > 0, pReal < 0.05, pFictive >= 0.05]):
                table[2, 1] += 1

            # Unmodulated (real) x unmodulated (fictive)
            elif np.all([True, pReal >= 0.05, pFictive >= 0.05]):
                table[2, 2] += 1

        return table