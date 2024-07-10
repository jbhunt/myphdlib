import h5py
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks as findPeaks
from matplotlib.colors import LinearSegmentedColormap
from myphdlib.general.toolkit import psth2
from myphdlib.figures.modulation import BasicSaccadicModulationAnalysis
from myphdlib.figures.analysis import GaussianMixturesModel, convertGratingMotionToSaccadeDirection
from itertools import product
from scipy.stats import spearmanr
from multiprocessing import Pool, cpu_count
from joblib import delayed, Parallel

def _generateNullSampleForSingleUnit(
    peths,
    params1,
    tProbe,
    nComponents,
    iUnit,
    nUnits,
    maximumAmplitudeShift=100,
    maxfev=1000,
    nRuns=None
    ):
    """
    Worker function (for parallelization) that refits the peri-saccadic PPTHs for a single unit
    """

    #
    # end = '\r' if iUnit + 1 != nUnits else '\n'
    end = '\n'
    print(f'Generating null sample for unit {iUnit + 1} out of {nUnits}', end=end, flush=True)
    nBins, nRuns_ = peths.shape
    if nRuns is None:
        nRuns = nRuns_
    sample = np.full([nRuns, nComponents], np.nan)

    #
    if np.isnan(params1).all():
        return iUnit, sample

    #
    abcd = np.delete(params1, np.isnan(params1))
    abc, d = abcd[:-1], abcd[-1]
    A1, B1, C1 = np.split(abc, 3)

    #
    for iRun in range(nRuns):

        #
        peth = peths[:, iRun]
        if np.isnan(peth).all():
            continue

        #
        mi = np.full(nComponents, np.nan)

        #
        params2 = np.full([nComponents, params1.size], np.nan)
        for iComp in range(A1.size):

            # Refit
            amplitudeBoundaries = np.vstack([A1 - 0.001, A1 + 0.001]).T
            amplitudeBoundaries[iComp, 0] -= maximumAmplitudeShift
            amplitudeBoundaries[iComp, 1] += maximumAmplitudeShift
            bounds = np.vstack([
                [[d - 0.001, d + 0.001]],
                amplitudeBoundaries,
                np.vstack([B1 - 0.001, B1 + 0.001]).T,
                np.vstack([C1 - 0.001, C1 + 0.001]).T,
            ]).T
            p0 = np.concatenate([
                np.array([d]),
                A1,
                B1,
                C1
            ])
            gmm = GaussianMixturesModel(A1.size, maxfev=maxfev)

            # Try to fit
            try:
                gmm.fit(tProbe, peth, p0, bounds)
            except:
                continue

            # Save the refit parameters
            d, abc = gmm._popt[0], gmm._popt[1:]
            params2[iComp, :abc.size + 1] = np.concatenate([
                abc,
                np.array([d,])
            ])

            # Compute modulation index (Normalized to response amplitude)
            A2 = np.split(gmm._popt[1:], 3)[0]
            mi[iComp] = (A2[iComp] - A1[iComp]) / A1[iComp]

        #
        sample[iRun] = mi

    return iUnit, sample

class BootstrappedSaccadicModulationAnalysis(BasicSaccadicModulationAnalysis):
    """
    """

    def __init__(self, **kwargs):
        """
        """

        super().__init__(**kwargs)

        #
        self.examples = (
            ('2023-07-05', 'mlati9', 288), # Enhanced
            ('2023-07-20', 'mlati9', 337), # No modulation
            ('2023-07-07', 'mlati9', 206), # Suppressed
        )

        return

    def resampleExtrasaccadicPeths(
        self,
        nRuns=100,
        rate=0.03,
        minimumTrialCount=5,
        responseWindow=(-0.2, 0.5),
        baselineWindow=(-0.2, 0),
        perisaccadicWindow=(-0.5, 0.5),
        binsize=0.01,
        smoothingKernelWidth=0.01,
        buffer=1,
        saccadeType='real',
        ):
        """
        """

        t, nTrials, nBins = psth2(
            np.zeros(1),
            np.zeros(1),
            window=responseWindow,
            binsize=binsize,
            returnShape=True
        )
        nUnits = len(self.ukeys)
        for probeDirection in ('pref', 'null'):
            self.ns[f'ppths/{probeDirection}/{saccadeType}/resampled'] = np.full([nUnits, nBins, nRuns], np.nan)

        #
        for iUnit in range(nUnits):

            #
            self.ukey = self.ukeys[iUnit]
            end = '\r' if iUnit + 1 != nUnits else None
            print(f'Re-sampling PETHs for unit {iUnit + 1} out of {nUnits}', end=end)

            #
            for probeDirection in ('pref', 'null'):

                # Load baseline FR
                # bl = self.ns[f'stats/{probeDirection}/{saccadeType}/extra'][iUnit, 0]

                # Load event data
                probeTimestamps, probeLatencies, saccadeLabels, gratingMotion = self._loadEventDataForProbes()

                # Extra-saccadic trial indices
                trialIndicesExtrasaccadic = np.where(np.vstack([
                    gratingMotion == self.preference[self.iUnit],
                    np.logical_or(
                        probeLatencies < perisaccadicWindow[0],
                        probeLatencies > perisaccadicWindow[1]
                    )
                ]).all(0))[0]
                if trialIndicesExtrasaccadic.size == 0:
                    continue

                # Total number of trials in the target direction
                if probeDirection == 'pref':
                    nTrialsTotal = np.sum(gratingMotion == self.preference[self.iUnit])
                else:
                    nTrialsTotal = np.sum(gratingMotion != self.preference[self.iUnit])

                # Number of trials specified
                if type(rate) == int and rate >= 1:
                    nTrialsForResampling = rate

                # Fraction of trials specified
                elif type(rate) in (float, int) and rate < 1:
                    nTrialsForResampling = int(round(nTrialsTotal * rate, 0))

                # Check if minimum trial count is met
                if minimumTrialCount is not None and nTrialsForResampling < minimumTrialCount:
                    nTrialsForResampling = minimumTrialCount

                # Compute relative spike timestamps
                responseWindowBuffered = (
                    responseWindow[0] - buffer,
                    responseWindow[1] + buffer
                )
                t, M, spikeTimestamps = psth2(
                    probeTimestamps,
                    self.unit.timestamps,
                    window=responseWindowBuffered,
                    binsize=binsize,
                    returnTimestamps=True
                )

                #
                for iRun in range(nRuns):

                    # Use kernel density estimation (takes a long time)
                    trialIndicesResampled = np.random.choice(
                        trialIndicesExtrasaccadic,
                        size=nTrialsForResampling,
                        replace=False
                    )

                    # Use KDE
                    sample = np.concatenate([spikeTimestamps[i] for i in trialIndicesResampled])
                    try:
                        t, fr = self.unit.kde(
                            probeTimestamps[trialIndicesResampled],
                            responseWindow=responseWindow,
                            binsize=binsize,
                            sigma=smoothingKernelWidth,
                            sample=sample,
                            nTrials=nTrialsForResampling,
                        )
                        bl = np.mean(fr[np.logical_and(t >= baselineWindow[0], t <= baselineWindow[1])])
                    except:
                        continue

                    # Standardize PSTH
                    self.ns[f'ppths/{probeDirection}/{saccadeType}/resampled'][iUnit, :, iRun] = (fr - bl) / self.factor[iUnit]

        return

    def generateNullSamples(
        self,
        nRuns=None,
        maximumAmplitudeShift=200,
        saccadeType='real',
        parallelize=True,
        nProcesses=30,
        chunksize=1,
        maxfevs=100,
        backend='multiprocessing',
        ):
        """
        """

        #
        nUnits, nBins, nRuns_ = self.ns[f'ppths/pref/{saccadeType}/resampled'].shape
        if nRuns is None:
            nRuns = nRuns_
        nComponents = int((self.ns[f'params/pref/{saccadeType}/extra'].shape[1] - 1) // 3)

        #
        for probeDirection in ('pref', 'null'):

            #
            samples = np.full([nUnits, nRuns, nComponents], np.nan)

            #
            arguments = list()
            for iUnit in range(nUnits):
                arguments.append((
                    self.ns[f'ppths/pref/{saccadeType}/resampled'][iUnit],
                    self.ns[f'params/pref/{saccadeType}/extra'][iUnit],
                    self.tProbe,
                    nComponents,
                    iUnit,
                    nUnits,
                    maximumAmplitudeShift,
                    maxfevs,
                    nRuns,
                ))

            # Parallelize the bootstrap
            if parallelize:
                if backend == 'multiprocessing':
                    with Pool(nProcesses, maxtasksperchild=1) as pool:
                        result = pool.starmap(
                            _generateNullSampleForSingleUnit,
                            arguments,
                            chunksize=chunksize,
                        )
                elif backend == 'joblib':
                    result = Parallel(n_jobs=-1)(delayed(_generateNullSampleForSingleUnit)(
                        args for args in arguments
                    ))
                for iUnit, sample in result:
                    samples[iUnit] = sample

            # Serial processing
            else:
                for args in arguments:
                    iUnit, sample = _generateNullSampleForSingleUnit(*args)
                    samples[iUnit] = sample

            #
            self.ns[f'samples/{probeDirection}/{saccadeType}'] = samples

        return

    def computeProbabilityValues(
        self,
        saccadeType='real',
        ):
        """
        """

        #
        nUnits, nBins, nRuns = self.ns[f'ppths/pref/{saccadeType}/resampled'].shape
        nWindows = len(self.windows)
        nComponents = int((self.ns[f'params/pref/{saccadeType}/extra'].shape[1] - 1) // 3)
        for probeDirection in ('pref', 'null'):
            self.ns[f'p/{probeDirection}/{saccadeType}'] = np.full([nUnits, nWindows, nComponents], np.nan)

        #
        for ukey in self.ukeys:

            #
            self.ukey = ukey
            end = '\n' if self.iUnit + 1 == nUnits else '\r'
            print(f'Computing p-values for unit {self.iUnit + 1} out of {nUnits}', end=end)

            #
            for probeDirection in ('pref', 'null'):

                #
                for iWindow in range(nWindows):

                    #
                    for iComp in range(nComponents):

                        # Determine the test value
                        mi = self.ns[f'mi/{probeDirection}/{saccadeType}'][self.iUnit, iWindow, iComp]
                        if np.isnan(mi).item() == True:
                            continue

                        # Load the null sample
                        sample = self.ns[f'samples/{probeDirection}/{saccadeType}'][self.iUnit, :, iComp]
                        mask = np.invert(np.isnan(sample))
                        if mask.sum() == 0:
                            continue

                        # Compute the probability of a more extreme measurement
                        p = np.sum(np.abs(sample[mask]) > np.abs(mi)) / mask.sum()
                        self.ns[f'p/{probeDirection}/{saccadeType}'][self.iUnit, iWindow, iComp] = p

        return
    
    def histModulationIndices(
        self,
        targetIndex=5,
        componentIndex=0,
        minimumProbeResponseAmplitude=0,
        alpha=0.01,
        nbins=30,
        cmap='coolwarm',
        xrange=(-3, 3),
        xticks=(-3, 0, 3),
        figsize=(7, 1.5),
        ):
        """
        """

        #
        fig, axs = plt.subplots(ncols=len(self.windows), sharex=True)

        #
        for windowIndex, ax in zip(np.arange(len(self.windows)), axs):
            
            # Collect the subsets of unit
            samples = list()
            include = self.ns['params/pref/real/extra'][:, 0] >= minimumProbeResponseAmplitude
            for sign in (-1, 1, 0):
                sample = list()
                for iUnit in range(len(self.ukeys)):
                    if include[iUnit] == False:
                        continue
                    mi = np.clip(self.ns['mi/pref/real'][iUnit, windowIndex, componentIndex], *xrange)
                    p = self.ns['p/pref/real'][iUnit, windowIndex, componentIndex]
                    if sign == -1:
                        if mi < 0 and p < alpha:
                            sample.append(mi)
                    elif sign == 0:
                        if p >= 0.05:
                            sample.append(mi)
                    elif sign == 1:
                        if mi > 0 and p < alpha:
                            sample.append(mi)
                samples.append(sample)

            #
            f = plt.get_cmap(cmap, 3)
            # binCounts_, binEdges, patchList = ax.hist(
            #     samples,
            #     bins=nbins,
            #     histtype='barstacked',
            #     color=[f(i) for i in (0, 2, 1)],
            #     range=xrange
            # )
            binCounts_, binEdges = np.histogram(
                np.concatenate(samples),
                bins=nbins,
                range=xrange
            )

            ax.hist(
                np.concatenate(samples),
                bins=nbins,
                color='0.8',
                range=xrange,
                histtype='stepfilled',
                edgecolor='none'
            )
            ax.hist(
                samples[0],
                bins=nbins,
                color=f(0),
                range=xrange,
                histtype='stepfilled',
                edgecolor='none'
            )
            ax.hist(
                samples[1],
                bins=nbins,
                color=f(2),
                range=xrange,
                histtype='stepfilled',
                edgecolor='none'
            )

            #
            if windowIndex != 0:
                ax.set_yticks([])
                ax.set_xticklabels([])
                ax.spines['left'].set_visible(False)
            else:
                ax.set_yticks([0, 200])
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
            if xticks is not None:
                ax.set_xticks(xticks)
            ax.set_xlim(*xrange)

            #
            if windowIndex != targetIndex:
                continue

            #
            binCenters = binEdges[:-1] + ((binEdges[1] - binEdges[0]) / 2)
            leftEdges = binEdges[:-1]
            rightEdges = binEdges[1:]

            #
            for ukey in self.examples:
                iUnit = self._indexUnitKey(ukey)
                if iUnit is None:
                    continue
                mi = np.clip(self.ns['mi/pref/real'][iUnit, windowIndex, componentIndex], *xrange)
                if np.isnan(mi):
                    continue
                binIndex = int(np.where(np.logical_and(
                    mi >= leftEdges,
                    mi <  rightEdges
                ))[0].item())
                ax.scatter(
                    binCenters[binIndex],
                    binCounts_[binIndex] + 10,
                    marker='v',
                    color='k',
                    s=20
                )

        #
        ylim = [np.inf, -np.inf]
        for ax in axs:
            y1, y2 = ax.get_ylim()
            if y1 < ylim[0]:
                ylim[0] = y1
            if y2 > ylim[1]:
                ylim[1] = y2
        for ax in axs:
            ax.set_ylim(ylim)

        # axs[0].set_ylabel('# of units')
        # axs[0].set_xlabel(f'Modulation index (MI)')
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.1)

        return fig, ax

    def plotFractionModulatedByProbeLatency(
        self,
        alpha=0.01,
        componentIndex=0,
        minimumProbeResponseAmplitude=0,
        xrange=(-3, 3),
        yrange=(0, 1),
        cmap='coolwarm',
        figsize=(3, 2),
        ):
        """
        """

        #
        y = np.full([3, len(self.windows)], np.nan)
        f = plt.get_cmap(cmap, 3)

        #
        for windowIndex in np.arange(len(self.windows)):
            samples = list()
            include = self.ns['params/pref/real/extra'][:, 0] >= minimumProbeResponseAmplitude
            nUnits = include.sum()
            for sign in (-1, 0, 1):
                sample = list()
                for iUnit in range(len(self.ukeys)):
                    if include[iUnit] == False:
                        continue
                    mi = np.clip(self.ns['mi/pref/real'][iUnit, windowIndex, componentIndex], *xrange)
                    p = self.ns['p/pref/real'][iUnit, windowIndex, componentIndex]
                    if sign == -1:
                        if mi < 0 and p < alpha:
                            sample.append(mi)
                    elif sign == 0:
                        if p >= 0.05:
                            sample.append(mi)
                    elif sign == 1:
                        if mi > 0 and p < alpha:
                            sample.append(mi)
                samples.append(sample)
            for i, sample in zip(range(3), samples):
                freq = len(sample) / nUnits
                y[i, windowIndex] = freq
        
        #
        fig, ax = plt.subplots()
        t = self.windows.mean(1)
        for i in range(3):
            ax.plot(t, y[i, :], color=f(i))
        ax.set_ylim(yrange)
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, ax
    
    def histModulationIndexByUnitType(
        self,
        responseWindow=(-0.2, 0.5),
        baselineWindow=(-1, -0.5),
        windowIndex=5,
        componentIndex=0,
        minmumResponseAmplitude=3,
        figsize=(3, 5)
        ):
        """
        """

        #
        fig, ax = plt.subplots()
        cmap = plt.get_cmap('coolwarm', 3)

        # Type units based on the saccade response, 0 = negative, 1 = no response, 2 = positive
        nUnits = len(self.ukeys)
        saccadeResponseTypes = np.full(nUnits, np.nan)
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
            fr = self.ns[f'psths/{saccadeDirection}/real'][iUnit, binIndicesForResponse]
            bl = self.ns[f'psths/{saccadeDirection}/real'][iUnit, binIndicesForBaseline].mean()
            y = fr - bl
            tv = y[np.argmax(np.abs(y))]
            if tv <= -minmumResponseAmplitude:
                label = -1
            elif tv >= minmumResponseAmplitude:
                label = 1
            else:
                label = 0
            saccadeResponseTypes[iUnit] = label        

        # Type units based on the response to the probe, 0 = negative, 1 = positive
        probeResponseTypes = np.full(nUnits, -1)
        probeResponseTypes[self.labels != -1] = 1

        # Type units based on unique combinations of probe and saccade responses
        uniqueResponseCombos = np.unique(np.vstack([
            probeResponseTypes,
            saccadeResponseTypes
        ]).T, axis=0)
        mi = self.ns['mi/pref/real'][:, windowIndex, componentIndex]
        p = self.ns['p/pref/real'][:, windowIndex, componentIndex]
        for x, (l1, l2) in enumerate(uniqueResponseCombos):
            unitIndices = np.where(np.logical_and(
                probeResponseTypes == l1,
                saccadeResponseTypes == l2
            ))[0]
            N = np.full(3, np.nan)
            N[0] = np.sum(np.logical_and(
                mi[unitIndices] < 0,
                p[unitIndices] < 0.05
            ))
            N[1] = np.sum(p[unitIndices] >= 0.05)
            N[2] = np.sum(np.logical_and(
                mi[unitIndices] > 0,
                mi[unitIndices] < 0.05
            ))
            N /= N.sum()
            ax.bar(
                x,
                N,
                width=0.8,
                color=[cmap(i) for i in range(3)],
                bottom=np.concatenate([[0], np.cumsum(N)[:-1]])
            )

        #
        ax.set_xticks(np.arange(uniqueResponseCombos.shape[0]))
        ax.set_xticklabels(uniqueResponseCombos, rotation=45)
        ax.set_ylim([0, 1.02])
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, (ax,)

    def plotComplexityByModulation(
        self,
        iComp=0,
        iWindow=5,
        minimumResponseAmplitude=2,
        binsize=None,
        xrange=(-2, 2),
        a=0.05,
        figsize=(4, 2.5),
        ):
        """
        """

        fig, ax = plt.subplots()
        m = np.logical_and(
            self.params[:, 0] >=  minimumResponseAmplitude,
            self.pvalues[:, iWindow, iComp] < a
        )
        if binsize is None:
            x = self.k.flatten()[m]
            y = self.modulation[m, iComp, iWindow] / self.params[m, iComp]
            ax.scatter(y, x)
        else:
            leftEdges = np.around(np.arange(xrange[0], xrange[1], binsize), 2)
            rightEdges = np.around(leftEdges + binsize, 2)
            binCenters = np.mean(np.vstack([leftEdges, rightEdges]).T, axis=1)
            y = list()
            x = list()
            iterable = zip(
                self.modulation[m, iComp, iWindow] / self.params[m, iComp],
                self.k.flatten()[m]
            )
            for i, (dr, k) in enumerate(iterable):
                if np.isnan(dr):
                    y.append(np.nan)
                    x.append(np.nan)
                    continue
                if np.clip(dr, *xrange) == -1:
                    y.append(binCenters[0])
                else:
                    binIndex = np.where(np.logical_and(
                        np.clip(dr, *xrange) > leftEdges,
                        np.clip(dr, *xrange) <= rightEdges
                    ))[0].item()
                    y.append(binCenters[binIndex])
                x.append(k)
            
            x = np.array(x)
            y = np.array(y)
            for k, dr in product(np.unique(x), np.unique(y)):
                n = np.sum(np.logical_and(x == k, y == dr))
                ax.scatter(dr, k, s=4 * n, color='k')
            r, p = spearmanr(y, x)

        #
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
        ax.set_yticks(np.arange(5) + 1)
        ax.set_xticks([-1, 0, 1])
        ax.set_ylim([0.5, 5.5])
        ax.set_ylabel('# of components')
        ax.set_xlabel(r'$\Delta R$')
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, ax, r, p

    def plotAveragePerisaccadicVisualResponse(
        self,
        sign=-1,
        windowIndices=(4, 5, 6),
        responseWindow=(-0.25, 0.25),
        minimumResponseAmplitude=0,
        figsize=(5, 3)
        ):
        """
        """

        #
        if windowIndices is None:
            windowIndices = np.arange(self.windows.shape[0])

        #
        fig, axs = plt.subplots(ncols=len(windowIndices) + 1)
        axs = np.atleast_1d(axs)

        #
        if sign == -1:
            suppressed = np.dstack([
                self.ns['mi/pref/real'][:, windowIndices, 0] < 0,
                self.ns['p/pref/real'][:, windowIndices, 0] < 0.05,
            ]).all(-1).any(1)
            include = np.vstack([
                suppressed,
                self.ns['params/pref/real/extra'][:, 0] >=  minimumResponseAmplitude
            ]).all(0)
        elif sign == 1:
            enhanced = np.dstack([
                self.ns['mi/pref/real'][:, windowIndices, 0] > 0,
                self.ns['p/pref/real'][:, windowIndices, 0] < 0.05,
            ]).all(-1).any(1)
            include = np.vstack([
                enhanced,
                self.ns['params/pref/real/extra'][:, 0] >=  minimumResponseAmplitude
            ]).all(0)
        else:
            raise Exception('Sign must be +1 or -1')

        #
        for i, (ax, windowIndex) in enumerate(zip(axs[1:], windowIndices)):
            R = {
                'peri': list(),
                'extra': list()
            }
            for iUnit, flag in enumerate(include):
                if flag == False:
                    continue

                #
                params = self.ns['params/pref/real/extra'][iUnit]
                abcd = np.delete(params, np.isnan(params))
                A, B, C = np.split(abcd[:-1], 3)
                tCentered = np.linspace(*responseWindow, 100) + B[0]
                yExtra = np.interp(
                    tCentered,
                    self.tProbe,
                    self.ns['ppths/pref/real/extra'][iUnit]
                )
                scalingFactor = np.max(np.abs(yExtra))
                yExtra = yExtra / scalingFactor

                #
                params = self.ns['params/pref/real/peri'][iUnit, windowIndex, 0]
                abcd = np.delete(params, np.isnan(params))
                A, B, C = np.split(abcd[:-1], 3)
                tCentered = np.linspace(*responseWindow, 100) + B[0]
                yPeri = np.interp(
                    tCentered,
                    self.tProbe,
                    self.ns['ppths/pref/real/peri'][iUnit, :, windowIndex] / scalingFactor
                )
                R['peri'].append(yPeri)
                R['extra'].append(yExtra)

            # Cast to numpy array
            for k in R.keys():
                R[k] = np.array(R[k])

            #
            tCentered = np.linspace(*responseWindow, 100)
            y = np.nanmean(R['peri'], axis=0)
            e = 2.33 * np.std(R['peri'], axis=0) / np.sqrt(R['peri'].shape[0])
            # e = np.std(R['peri'], axis=0)
            ax.plot(tCentered, y, color='r')
            ax.fill_between(tCentered, y + e, y - e, color='r', alpha=0.15, edgecolor='none')

            #
            if i == 0:
                y = np.nanmean(R['extra'], axis=0)
                e = 2.33 * np.std(R['extra'], axis=0) / np.sqrt(R['extra'].shape[0])
                # e = np.std(R['extra'], axis=0)
                axs[0].plot(tCentered, y, color='k')
                axs[0].fill_between(tCentered, y + e, y - e, color='k', alpha=0.15, edgecolor='none')

        #
        ylim = [np.inf, -np.inf]
        for ax in axs:
            y1, y2 = ax.get_ylim()
            if y1 < ylim[0]:
                ylim[0] = y1
            if y2 > ylim[1]:
                ylim[1] = y2
        for ax in axs:
            ax.set_ylim(ylim)
            ax.hlines(1, *responseWindow, color='k', linestyle=':')
            ax.set_xlim(responseWindow)
            ax.set_xticks([responseWindow[0], 0, responseWindow[1]])
            ax.set_xticklabels(ax.get_xticks(), rotation=45)

        #
        axs[0].set_ylabel('FR (normed)')
        if len(axs) > 1:
            for ax in axs[1:]:
                ax.set_yticklabels([])

        #
        fig.supxlabel('Time from response peak (sec)', fontsize=10)
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, axs
    