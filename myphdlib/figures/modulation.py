import h5py
import numpy as np
import pathlib as pl
from scipy.signal import find_peaks as findPeaks
from scipy.stats import pearsonr
from matplotlib import pylab as plt
from myphdlib.general.toolkit import psth2
from myphdlib.figures.analysis import GaussianMixturesModel, g
from myphdlib.figures.clustering import GaussianMixturesFittingAnalysis

class BasicSaccadicModulationAnalysis(GaussianMixturesFittingAnalysis):
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
        self.windows = None

        #
        self.examples = (
            ('2023-07-05', 'mlati9', 288), # Enhanced
            ('2023-07-20', 'mlati9', 337), # No modulation
            ('2023-07-07', 'mlati9', 206), # Suppressed
        )

        return

    def _computeResponseTerms(
        self,
        ukey=None,
        responseWindow=(-0.2, 0.5),
        baselineWindow=(-3.0, -2.0),
        perisaccadicWindow=(-0.05, 0.1),
        binsize=0.01,
        smoothingKernelWidth=0.01,
        saccadeType='real',
        ):
        """
        """

        #
        t, nTrials, nBins = psth2(
            np.zeros(1),
            np.zeros(1),
            window=responseWindow,
            binsize=binsize,
            returnShape=True
        )

        #
        if ukey is not None:
            self.ukey = ukey

        # Compute peri-saccadic response
        probeTimestamps, probeLatencies, saccadeLabels, gratingMotionDuringProbes = self._loadEventDataForProbes()
        trialIndicesPerisaccadic = np.vstack([
            probeLatencies >= perisaccadicWindow[0],
            probeLatencies <= perisaccadicWindow[1],
            gratingMotionDuringProbes == self.preference[self.iUnit]
        ]).all(0)

        # NOTE: This might fail with not enough spikes in the peri-saccadic window
        try:
            t_, rMixed = self.unit.kde(
                probeTimestamps[trialIndicesPerisaccadic],
                responseWindow=responseWindow,
                binsize=binsize,
                sigma=smoothingKernelWidth
            )
        except:
            return (
                np.full(nBins, np.nan),
                np.full(nBins, np.nan),
            )

        # Compute latency-shifted saccade response
        iterable = zip(
            probeLatencies[trialIndicesPerisaccadic],
            saccadeLabels[trialIndicesPerisaccadic]
        )
        rSaccade = list()
        rBaseline = list()
        binIndicesBaseline = np.logical_and(self.tSaccade >= baselineWindow[0], self.tSaccade <= baselineWindow[1])
        for probeLatency, saccadeLabel in iterable:
            saccadeDirection = 'temporal' if saccadeLabel == -1 else 'nasal'
            fp = self.ns[f'psths/{saccadeDirection}/{saccadeType}'][self.iUnit]
            x = t + probeLatency
            fr = np.interp(x, self.tSaccade, fp, left=np.nan, right=np.nan)
            rSaccade.append(fr)
            bl = fp[binIndicesBaseline].mean()
            rBaseline.append(bl)

        rSaccade = np.nanmean(np.array(rSaccade) - np.array(rBaseline).reshape(-1, 1), 0)

        return rMixed, rSaccade

    def computeSaccadeResponseTemplates(
        self,
        responseWindow=(-0.2, 0.5),
        perisaccadicWindow=(-0.2, 0.2),
        binsize=0.01,
        pad=3,
        gaussianKernelWidth=0.01,
        saccadeType='real',
        ):
        """
        """

        tSaccade, nTrials, nBins = psth2(
            np.zeros(1),
            np.zeros(1),
            window=(responseWindow[0] - pad, responseWindow[1] + pad),
            binsize=binsize,
            returnShape=True
        )
        if self.tSaccade is None:
            self.tSaccade = tSaccade
        nUnits = len(self.ukeys)

        #
        for saccadeDirection in ('nasal', 'temporal'):
            self.ns[f'psths/{saccadeDirection}/{saccadeType}'] = np.full([nUnits, nBins], np.nan)

        #
        for ukey in self.ukeys:

            #
            self.ukey = ukey

            #
            end = None if self.iUnit + 1 == nUnits else '\r'
            print(f'Copmuting saccade response templates for unit {self.iUnit + 1} out of {nUnits}', end=end)
            
            # Set the unit (and session)
            self.ukey = self.ukeys[self.iUnit]


            #
            saccadeTimestamps, saccadeLatencies, saccadeLabels, gratingMotion = self._loadEventDataForSaccades()

            # Compute saccade response templates
            for saccadeLabel, saccadeDirection in zip([-1, 1], ['temporal', 'nasal']):
                trialIndices = np.where(np.vstack([
                    np.logical_or(
                        saccadeLatencies < perisaccadicWindow[1] * -1,
                        saccadeLatencies > perisaccadicWindow[0] * -1,
                    ),
                    saccadeLabels == saccadeLabel,
                    gratingMotion == self.preference[self.iUnit]
                    # gratingMotion == self.features['d'][self.iUnit]
                ]).all(0))[0]
                if trialIndices.size == 0:
                    continue
                try:
                    tSaccade, fr = self.unit.kde(
                        saccadeTimestamps[trialIndices],
                        responseWindow=(responseWindow[0] - pad, responseWindow[1] + pad),
                        binsize=binsize,
                        sigma=gaussianKernelWidth
                    )
                except:
                    continue

                #
                self.ns[f'psths/{saccadeDirection}/{saccadeType}'][self.iUnit] = fr

        return

    def computePerisaccadicPeths(
        self,
        trange=(-0.5, 0.5),
        tstep=0.1,
        responseWindow=(-0.2, 0.5),
        baselineWindow=(-0.2, 0),
        binsize=0.01,
        saccadeType='real',
        ):
        """
        """

        #
        tProbe, nTrials, nBins = psth2(
            np.zeros(1),
            np.zeros(1),
            window=responseWindow,
            binsize=binsize,
            returnShape=True
        )
        if self.tProbe is None:
            self.tProbe = tProbe

        #
        leftEdges = np.arange(trange[0], trange[1], tstep)
        rightEdges = leftEdges + tstep
        self.windows = np.vstack([leftEdges, rightEdges]).T
        if saccadeType == 'real':
            self.ns['globals/windows'] = self.windows

        #
        nUnits = len(self.ukeys)
        nBins = self.tProbe.size
        nWindows = self.windows.shape[0]

        for probeDirection in ('pref', 'null'):
            self.ns[f'ppths/{probeDirection}/{saccadeType}/peri'] = np.full([nUnits, nBins, nWindows], np.nan)
        for term in ('extra', 'mixed', 'saccade', 'peri'):
            for probeDirection in ('pref', 'null'):
                self.ns[f'terms/{probeDirection}/{saccadeType}/{term}'] = np.full([nUnits, nBins, nWindows], np.nan)

        #
        for ukey in self.ukeys:

            #
            self.ukey = ukey

            #
            end = '\r' if self.iUnit + 1 != nUnits else None
            print(f'Computing PETHs for unit {self.iUnit + 1} out of {nUnits}', end=end)

            #
            for probeDirection in ('pref', 'null'):

                #
                for iWin, perisaccadicWindow in enumerate(self.windows):

                    #
                    rMixed, rSaccade = self._computeResponseTerms(
                        ukey=self.ukeys[self.iUnit],
                        responseWindow=responseWindow,
                        perisaccadicWindow=perisaccadicWindow,
                    )

                    # Standardize the PETHs
                    baselineLevel = self.ns[f'stats/{probeDirection}/{saccadeType}/extra'][self.iUnit][0]
                    scalingFactor = self.factor[self.iUnit]
                    yResidual = np.clip(rMixed - rSaccade, 0, np.inf)
                    yStandard = (yResidual - baselineLevel) / scalingFactor

                    # Correct for baseline shift
                    binIndices = np.where(np.logical_and(
                        self.tProbe >= baselineWindow[0],
                        self.tProbe < baselineWindow[1]
                    ))
                    yCorrected = yStandard - yStandard[binIndices].mean()

                    #
                    self.ns[f'terms/{probeDirection}/{saccadeType}/mixed'][self.iUnit, :, iWin] = rMixed
                    self.ns[f'terms/{probeDirection}/{saccadeType}/saccade'][self.iUnit, :, iWin] = rSaccade
                    self.ns[f'terms/{probeDirection}/{saccadeType}/peri'][self.iUnit, :, iWin] = yCorrected

                    #
                    self.ns[f'ppths/{probeDirection}/{saccadeType}/peri'][self.iUnit, :, iWin] = yCorrected

        return

    def _fitPerisaccadicPeth(
        self,
        peth,
        probeDirection='pref',
        saccadeType='real',
        maximumAmplitudeShift=200,
        ):
        """
        """

        #
        paramsExtra = self.ns[f'params/{probeDirection}/{saccadeType}/extra'][self.iUnit]
        if np.isnan(paramsExtra).all():
            return None, None
        abcd = np.delete(paramsExtra, np.isnan(paramsExtra))
        abc, d1 = abcd[:-1], abcd[-1]
        A1, B1, C1 = np.split(abc, 3)
        k = A1.size

        #
        kmax = (paramsExtra.size - 1) // 3
        dr = np.full(kmax, np.nan)
        paramsPeri = np.full([kmax, paramsExtra.size], np.nan)

        #
        for iComp in np.arange(k):

            # Refit
            amplitudeBoundaries = np.vstack([A1 - 0.001, A1 + 0.001]).T
            amplitudeBoundaries[iComp, 0] -= maximumAmplitudeShift
            amplitudeBoundaries[iComp, 1] += maximumAmplitudeShift
            bounds = np.vstack([
                [[d1 - 0.001, d1 + 0.001]],
                amplitudeBoundaries,
                np.vstack([B1 - 0.001, B1 + 0.001]).T,
                np.vstack([C1 - 0.001, C1 + 0.001]).T,
            ]).T
            p0 = np.concatenate([
                np.array([d1]),
                A1,
                B1,
                C1
            ])
            gmm = GaussianMixturesModel(k)
            gmm.fit(self.tProbe, peth, p0, bounds)

            # Store the re-fit
            A2 = np.split(gmm._popt[1:], 3)[0]
            B2 = np.split(gmm._popt[1:], 3)[1]
            C2 = np.split(gmm._popt[1:], 3)[2]
            d2 = gmm._popt[0]
            paramsPeri[iComp, :abc.size + 1] = np.concatenate([
                A2,
                B2,
                C2,
                np.array([d2,])
            ])
            #
            
            dr[iComp] = A2[iComp] - A1[iComp]

        return dr, paramsPeri
    
    def fitPerisaccadicPeths(
        self,
        maximumAmplitudeShift=200,
        saccadeType='real',
        applyCorrection=False,
        ):
        """
        """

        #
        nUnits, nBins, nWindows = self.ns[f'ppths/pref/{saccadeType}/peri'].shape
        nParams = self.ns[f'params/pref/{saccadeType}/extra'].shape[1]
        nComponents = int((nParams - 1) / 3)
        for probeDirection in ('pref', 'null'): 
            self.ns[f'mi/{probeDirection}/{saccadeType}'] = np.full([nUnits, nWindows, nComponents], np.nan)
            self.ns[f'params/{probeDirection}/{saccadeType}/peri'] = np.full([nUnits, nWindows, nComponents, nParams], np.nan)

        #
        for ukey in self.ukeys:

            #
            self.ukey = ukey
            end = '\r' if self.iUnit + 1 != nUnits else None
            print(f'Re-fitting peri-saccadic PETHs for unit {self.iUnit + 1} out of {nUnits}', end=end)

            #
            for probeDirection in ('pref', 'null'):

                # Extra-saccadic parameters
                paramsExtra = self.ns[f'params/{probeDirection}/{saccadeType}/extra'][self.iUnit]
                paramsExtra = np.delete(paramsExtra, np.isnan(paramsExtra))
                if paramsExtra.size == 0:
                    continue
                abc = paramsExtra[:-1]
                responseAmplitudes, B, C = np.split(abc, 3)

                # For each peri-saccadic window
                for iWin in range(nWindows):

                    # Load the peri-saccadic PPTH
                    peth = self.ns[f'ppths/{probeDirection}/{saccadeType}/peri'][self.iUnit, :, iWin]
                    if np.isnan(peth).all():
                        continue

                    # Compute the changes in response amplitude for each component
                    responseDeltas, paramsPeri = self._fitPerisaccadicPeth(
                        peth=peth,
                        probeDirection=probeDirection,
                        maximumAmplitudeShift=maximumAmplitudeShift,
                        saccadeType=saccadeType
                    )
                    if all([responseDeltas is None, paramsPeri is None]):
                        continue

                    # Compute the modulation index (normalized to amplitude)
                    responseDeltas = np.delete(responseDeltas, np.isnan(responseDeltas))
                    mi = responseDeltas / responseAmplitudes
                    self.ns[f'mi/{probeDirection}/{saccadeType}'][self.iUnit, iWin, :mi.size] = mi
                    self.ns[f'params/{probeDirection}/{saccadeType}/peri'][self.iUnit, iWin, :, :] = paramsPeri

        # Correct the sign of modulation for the null probe based on the sign
        # of the response to the preferred probe
        if applyCorrection:
            self._correctModulationIndexForNullProbes(
                saccadeType,
            )

        return

    def _correctModulationIndexForNullProbes(
        self,
        saccadeType='real',
        maximumComponentDistance=0.01,
        ):
        """
        Reverses the sign of modulation for null probes if the responses to the
        preferred and null probes have opposite signs
        """

        nUnits, nBins, nWindows = self.ns[f'ppths/pref/{saccadeType}/peri'].shape
        corrected = np.copy(self.ns[f'mi/null/{saccadeType}'])

        #
        for ukey in self.ukeys:

            #
            self.ukey = ukey

            # Extra-saccadic params for the preferred direction
            paramsPrefExtra = self.ns[f'params/pref/{saccadeType}/extra'][self.iUnit]
            paramsPrefExtra = np.delete(paramsPrefExtra, np.isnan(paramsPrefExtra))
            aPrefExtra, bPrefExtra, cPrefExtra = np.split(paramsPrefExtra[:-1], 3)

            # Extra-saccadic params for the null direction
            paramsNullExtra = self.ns[f'params/null/{saccadeType}/extra'][self.iUnit]
            paramsNullExtra = np.delete(paramsNullExtra, np.isnan(paramsNullExtra))
            aNullExtra, bNullExtra, cNullExtra = np.split(paramsNullExtra[:-1], 3)

            #
            for iComp in range(aNullExtra.size):

                # Lookup the closest component in the extra-saccadic response to the preferred probe
                closest = np.argmin(np.abs(bNullExtra[iComp] - bPrefExtra))
                if closest > maximumComponentDistance:
                    continue

                #
                flipSign = any([
                    aNullExtra[iComp] < 0 and aPrefExtra[closest] > 0,
                    aNullExtra[iComp] > 0 and aPrefExtra[closest] < 0
                ])

                #
                if flipSign:
                    for iWindow in range(nWindows):
                        corrected[self.iUnit, iWindow, iComp] *= -1

        #
        self.ns[f'mi/null/{saccadeType}'] = corrected

        return

    def plotSaccadeAccountingPerformance(
        self,
        examples=(
            ('2023-05-12', 'mlati7', 163),
            ('2023-05-29', 'mlati7', 366)
        ),
        figsize=(9, 2)
        ):
        """
        """

        nUnits = len(self.ukeys)
        nWindows = len(self.windows)
        R, P = np.full([nUnits, nWindows], np.nan), np.full([nUnits, nWindows], np.nan)
        for iUnit in range(nUnits):
            rProbe = self.ns['ppths/pref/real/extra'][iUnit]
            baselineLevel, scalingFactor = self.ns['stats/pref/real/extra'][iUnit, :]
            rProbe = (rProbe + baselineLevel) * scalingFactor

            for iWindow in range(nWindows):
                rMixed = self.ns['terms/pref/real/mixed'][iUnit, :, iWindow]
                rSaccade = self.ns['terms/pref/real/saccade'][iUnit, :, iWindow]
                rTest = rMixed - rProbe
                # rTest -= rTest[binIndices].mean()
                if np.any([np.isnan(rProbe).all(), np.isnan(rSaccade).all(), np.isnan(rTest).all()]):
                    continue
                r, p = pearsonr(rSaccade, rTest)
                R[iUnit, iWindow] = r
                P[iUnit, iWindow] = p

        #
        fig, axs = plt.subplots(ncols=2)
        y = np.dstack([P < 0.05, R > 0]).all(-1).sum(0) / P.shape[0]
        x = self.windows.mean(1)
        axs[0].plot(x, y, color='k', marker='o', markersize=5)

        #
        samples = list()
        for iWindow in range(nWindows):
            mask = np.logical_and(
                P[:, iWindow] < 0.05,
                R[:, iWindow] > 0
            )
            sample = R[mask, iWindow]
            samples.append(sample)
        y = np.array([np.nanmean(sample) for sample in samples])
        e = np.array([np.nanstd(sample) for sample in samples])
        axs[1].plot(x, y, color='k', marker='o', markersize=5)
        # axs[1].vlines(x, y - e, y + e, color='k')
        axs[1].fill_between(
            x,
            y - e,
            y + e,
            color='k',
            alpha=0.1,
            edgecolor=None,
        )
        colors = ('xkcd:orange', 'xkcd:purple')
        counter = 0
        for iUnit in range(nUnits):
            ukey = self.ukeys[iUnit]
            if self._matchUnitKey(ukey, examples):
                axs[1].plot(
                    x,
                    R[iUnit],
                    color=colors[counter],
                    marker='o',
                    markersize=5,
                    clip_on=False,
                )
                counter += 1

        #
        for ax in axs:
            ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5])
            ax.set_xticklabels(
                np.around(ax.get_xticks(), 2),
                rotation=45
            )
            ax.set_xlim([
                self.windows.min(),
                self.windows.max()
            ])
            ax.set_yticks([0, 0.5, 1])
            ax.set_yticklabels(['0', '', '1'])
            ax.set_ylim([0, 1])
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            ax.fill_between(
                [0, 0.1],
                0,
                1,
                color='r',
                alpha=0.1,
                edgecolor=None,
            )
        axs[0].set_xlabel('Time from saccade (sec)')
        axs[0].set_ylabel('Frac. of units')
        axs[1].set_ylabel('Corr. Coeff.')
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, ax

    def plotLatencySortedRasterplot(
        self,
        ukey,
        responseWindow=(-0.5, 0.5),
        figsize=(2, 3),
        alpha=0.5,
        **kwargs_
        ):

        """
        """

        #
        kwargs = {
            'marker': '.',
            's': 15
        }
        kwargs.update(kwargs_)

        #
        self.ukey = ukey

        #
        nWindows = len(self.windows)
        nTrialsPerWindow = list()
        for window in self.windows:
            gratingMotion = self.preference[self.iUnit]
            trialIndices = np.where(np.vstack([
                self.session.gratingMotionDuringProbes == gratingMotion,
                self.session.probeLatencies > window[0],
                self.session.probeLatencies <= window[1]
            ]).all(0))[0]
            nTrialsPerWindow.append(trialIndices.size)

        #
        fig, axs = plt.subplots(
            nrows=nWindows,
            gridspec_kw={'height_ratios': nTrialsPerWindow},
            sharex=True
        )

        #
        for iWin, window in enumerate(self.windows):
            gratingMotion = self.preference[self.iUnit]
            trialIndices = np.where(np.vstack([
                self.session.gratingMotionDuringProbes == gratingMotion,
                self.session.probeLatencies > window[0],
                self.session.probeLatencies <= window[1]
            ]).all(0))[0]
            latencySortedIndex = np.argsort(self.session.probeLatencies[trialIndices])[::-1]
            t, M, spikeTimestamps = psth2(
                self.session.probeTimestamps[trialIndices],
                self.unit.timestamps,
                window=responseWindow,
                binsize=None,
                returnTimestamps=True
            )
            x = list()
            y = list()
            for i, trialIndex in enumerate(latencySortedIndex):
                nSpikes = spikeTimestamps[trialIndex].size
                for t in np.atleast_1d(spikeTimestamps[trialIndex]):
                    x.append(t)
                for r in np.full(nSpikes, i):
                    y.append(r)
            x = np.array(x)
            y = np.array(y)

            #
            axs[iWin].scatter(x, y, rasterized=True, color='k', alpha=alpha, **kwargs, ec='none')
            # axs[iWin].vlines(x, y - 0.5, y + 0.5, color='k', alpha=1, lw=1)

            # Indicate the time of saccade onset
            x, y = list(), list()
            for i, t in enumerate(self.session.probeLatencies[trialIndices][latencySortedIndex]):
                x.append(-1 * t)
                y.append(i)
            x, y = np.array(x), np.array(y)
            axs[iWin].scatter(
                x,
                y,
                rasterized=True,
                color='m',
                ec='none',
                **kwargs
            )
            # axs[iWin].vlines(x, y - 0.5, y + 0.5, rasterized=True, color='m', alpha=1, lw=1)

            # TODO: Indicate the time of probe onset
            y = np.arange(trialIndices.size)
            x = np.full(y.size, 0)
            axs[iWin].scatter(x, y, rasterized=True, color='c', **kwargs, ec='none')
            # axs[iWin].vlines(x, y - 0.5, y + 0.5, rasterized=True, color='c', alpha=1, lw=1)

        #
        for ax in axs[:-1]:
            ax.set_yticks([])
            ax.set_xticks([])
        axs[-1].set_yticks([30,])
        for ax in axs.flatten():
            for sp in ('top', 'right', 'left', 'bottom'):
                ax.spines[sp].set_visible(False)
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()
        fig.subplots_adjust(hspace=0.1)

        return fig, axs

    def _plotResponseTerms(
        self,
        ukey=None,
        axs=None,
        iWindow=0,
        colors=(
            'k',
            'k'
        ),
        ):
        """
        """

        #
        if self.ukey is not None:
            self.ukey = ukey
        if self.iUnit is None:
            raise Exception('Could not locate example unit')

        #
        mu, sigma = self.ns[f'stats/pref/real/extra'][self.iUnit, 0], self.factor[self.iUnit]

        #
        rMixed = self.ns[f'terms/pref/real/mixed'][self.iUnit, :, iWindow]
        rSaccade = self.ns[f'terms/pref/real/saccade'][self.iUnit, :, iWindow]
        rProbeExtra = self.ns[f'ppths/pref/real/extra'][self.iUnit]
        rProbePeri = self.ns[f'ppths/pref/real/peri'][self.iUnit, :, iWindow]

        #
        params = self.ns['params/pref/real/extra'][self.iUnit, :]
        params = params[np.invert(np.isnan(params))]
        abc, d = params[:-1], params[-1]
        A, B, C = np.split(abc, 3)
        a, b, c = A[0], B[0], C[0]
        rProbeExtraFit = g(self.tProbe, a, b, c, d)

        #
        params = self.ns['params/pref/real/peri'][self.iUnit, iWindow, 0, :]
        params = params[np.invert(np.isnan(params))]
        if params.size == 0:
            return
        abc, d = params[:-1], params[-1]
        A, B, C = np.split(abc, 3)
        a, b, c = A[0], B[0], C[0]
        rProbePeriFit = g(self.tProbe, a, b, c, d)

        #
        if axs is None:
            fig, axs = plt.subplots(ncols=4, sharey=True)
        axs[0].plot(self.tProbe, (rMixed - mu) / sigma, color='k')
        axs[1].plot(self.tProbe, (rSaccade) / sigma, color='k')
        axs[2].plot(self.tProbe, rProbePeri, color=colors[1])

        return

    def plotAnalysisDemo(
        self,
        ukey=('2023-05-12', 'mlati7', 163),
        windowIndices=None,
        figsize=(5, 5),
        **kwargs_
        ):
        """
        """

        #
        kwargs = {
            'alpha': 0.3,
            's': 3,
        }
        kwargs.update(kwargs_)

        #
        if windowIndices is None:
            windowIndices = np.arange(self.windows.shape[0])
        elif type(windowIndices) in (list, tuple):
            windowIndices = np.array(windowIndices)

        # Set the unit key
        self.ukey = ukey
        if self.iUnit is None:
            raise Exception('Could not locate unit')

        # Create the subplots
        nWindows = len(windowIndices)
        fig, grid = plt.subplots(
            nrows=nWindows,
            ncols=3,
        )

        # For each row, plot the raster and the trial-averages PSTHs for each term
        for i, axs, w in zip(windowIndices, grid, self.windows[windowIndices, :]):
            self._plotResponseTerms(
                ukey=ukey,
                axs=axs,
                iWindow=i,
            )

        # Set the y-axis limits for the tiral-averaged responses
        ylim = [np.inf, -np.inf]
        for ax in grid.flatten():
            y1, y2 = ax.get_ylim()
            if y1 < ylim[0]:
                ylim[0] = y1
            if y2 > ylim[1]:
                ylim[1] = y2
        yticks = grid[0, 0].get_yticks()
        for ax in grid.flatten():
            ax.set_ylim(ylim)
            ax.set_yticks(yticks)
            for sp in ('top', 'right', 'bottom', 'left'):
                ax.spines[sp].set_visible(False)

        #
        for ax in grid[:, 1:].flatten():
            ax.set_yticklabels([])
        for ax in grid[:-1, 0].flatten():
            ax.set_yticklabels([])
        for ax in grid[:-1, :].flatten():
            ax.set_xticklabels([])

        #
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])

        return fig, grid

    def plotExamplePeths(
        self,
        figsize=(2.5, 3.5),
        windowIndex=5,
        yticks=(0, 50),
        xticks=(-0.2, 0, 0.2),
        xrange=(-0.2, 0.2),
        ):
        """
        """

        #
        fig, grid = plt.subplots(nrows=len(self.examples), ncols=2, sharex=True)
        if len(self.examples) == 1:
            grid = np.atleast_2d(grid)
        k = (self.ns[f'params/pref/real/extra'].shape[1] - 1) // 3
        cmap = plt.get_cmap('rainbow', k)

        #
        for i in range(len(self.examples)):

            #
            ukey = self.examples[i]
            if ukey not in self.ukeys:
                continue
            self.ukey = ukey

            # Plot the raw PETHs
            rProbePeri = self.ns[f'ppths/pref/real/peri'][self.iUnit, :, windowIndex]
            rProbeExtra = self.ns[f'ppths/pref/real/extra'][self.iUnit, :]

            # Plot the fit for the largest component of the response
            paramsExtra = self.ns[f'params/pref/real/extra'][self.iUnit, :]
            paramsPeri = self.ns[f'params/pref/real/peri'][self.iUnit, windowIndex, 0, :]

            #
            for j, params in enumerate([paramsExtra, paramsPeri]):

                #
                params = np.delete(params, np.isnan(params))
                if len(params) == 0:
                    continue
                abc, d = params[:-1], params[-1]
                A, B, C = np.split(abc, 3)
                a, b, c = A[0], B[0], C[0]

                #
                t2 =  np.linspace(*xrange, 100) + B[0]
                yPeri = np.interp(t2, self.tProbe, rProbePeri)
                yExtra = np.interp(t2, self.tProbe, rProbeExtra)
                grid[i, 1].plot(t2- B[0], yPeri, 'k', alpha=0.1)
                grid[i, 0].plot(t2 - B[0], yExtra, color='k', alpha=0.1)

                #
                t1 = np.linspace(-15 * C[0], 15 * C[0], 100) + B[0]
                yFit = g(t1, a, b, c, d)
                color = 'k' if j == 0 else 'g'
                grid[i, j].plot(t1 - B[0], yFit, color=color, zorder=3, linestyle='-')

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
        for ax in grid.flatten():
            ax.set_yticks(yticks)
            ax.set_xticks(xticks)
        for ax in grid[:, 1].flatten():
            ax.set_yticks([])
            ax.spines['left'].set_visible(False)

        #
        for ax in grid.flatten():
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)

        #
        for ax in grid[-1, :].flatten():
            ax.set_xticklabels(xticks, rotation=45)

        #
        fig.supxlabel('Time from response peak (sec)', fontsize=10)
        fig.supylabel('FR (z-scored)', fontsize=10)
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, grid

    def plotModulationIndexByFiringRate(
        self,
        ):
        """
        """

        x, y = list(), list()
        for session in self.sessions:
            firingRate = session.load('metrics/fr')
            for ukey in self.ukeys:
                date, animal, cluster = ukey
                if date != session.date or animal != session.animal:
                    continue
                unit = session.indexByCluster(cluster)
                x.append(firingRate[unit.index])
                iUnit = self._indexUnitKey(ukey)
                y.append(self.mi[iUnit])

        #
        fig, ax = plt.subplots()
        ax.scatter(x, y, color='k', marker='.', s=15, alpha=0.5)

        return fig, ax
