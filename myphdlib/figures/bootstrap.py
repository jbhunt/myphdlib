import h5py
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks as findPeaks
from matplotlib.colors import LinearSegmentedColormap
from myphdlib.general.toolkit import psth2
from myphdlib.figures.modulation import BasicSaccadicModulationAnalysis
from myphdlib.figures.analysis import GaussianMixturesModel
import seaborn as sns
import pandas as pd
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
    

    # TODO: Indicate which significantly modulated units have little or no saccade response
    def histModulationIndices(
        self,
        windowIndex=5,
        componentIndex=0,
        minimumProbeResponseAmplitude=0,
        maximumSaccadeResponseAmplitude=2,
        transform=False,
        nbins=30,
        colors=('b', 'r', '0.8'),
        xrange=(-3, 3),
        xticks=None,
        figsize=(4, 2),
        ):
        """
        """

        # Collect the subsets of unit
        samples = list()
        include = self.ns['params/pref/real/extra'][:, 0] >= minimumProbeResponseAmplitude
        for sign in (-1, 1, 0):
            sample = list()
            for iUnit in range(len(self.ukeys)):
                if include[iUnit] == False:
                    continue
                mi = self.ns['mi/pref/real'][iUnit, windowIndex, componentIndex]
                if transform:
                    mi = np.tanh(mi)
                    xrange = (-1, 1)
                else:
                    mi = np.clip(mi, *xrange)
                p = self.ns['p/pref/real'][iUnit, windowIndex, componentIndex]
                if sign == -1:
                    if mi < 0 and p < 0.05:
                        sample.append(mi)
                elif sign == 0:
                    if p >= 0.05:
                        sample.append(mi)
                elif sign == 1:
                    if mi > 0 and p < 0.05:
                        sample.append(mi)
            samples.append(sample)

        #
        fig, ax = plt.subplots()
        binCounts_, binEdges, patchList = ax.hist(
            samples,
            bins=nbins,
            histtype='barstacked',
            range=xrange
        )
        for i, patches in enumerate(patchList):
            for patch in patches:
                patch.set_facecolor(colors[i])

        binCounts = binCounts_.max(0)
        binCenters = binEdges[:-1] + ((binEdges[1] - binEdges[0]) / 2)
        leftEdges = binEdges[:-1]
        rightEdges = binEdges[1:]

        #
        for ukey in self.examples:
            iUnit = self._indexUnitKey(ukey)
            if iUnit is None:
                continue
            mi = np.tanh(self.ns['mi/pref/real'][iUnit, windowIndex, componentIndex])
            if np.isnan(mi).item():
                continue
            binIndex = np.where(np.logical_and(
                mi >= leftEdges,
                mi <  rightEdges
            ))[0].item()
            ax.scatter(
                binCenters[binIndex],
                binCounts[binIndex] + 10,
                marker='v',
                color='k',
                s=20
            )

        #
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)

        #
        if xticks is not None:
            ax.set_xticks(xticks)
        ax.set_ylabel('# of units')
        ax.set_xlabel(f'Modulation index (MI)')
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, ax

    def plotModulationDistributionsWithHistogram(
        self,
        alpha=0.05,
        windowIndex=5,
        componentIndex=0,
        figsize=(3, 2),
        colorspace=('b', 'r', 'w'),
        minimumResponseAmplitude=0,
        nBins=20,
        labels=(1, 2, 3, -1),

        ):
        """
        """

        #
        if minimumResponseAmplitude is None:
            minimumResponseAmplitude = 0

        #
        cmap = LinearSegmentedColormap.from_list('mycmap', colorspace, N=3)
        fig, ax = plt.subplots()

        #
        mi = self.ns[f'mi/pref/real']
        paramsExtra = self.ns[f'params/pref/real/extra']
        pvalues = self.ns[f'p/pref/real']
        polarity = np.array([
            -1 if mi[i, windowIndex, componentIndex] < 0 else 1
                for i in range(len(self.ukeys))
        ])
        # polarity[np.isnan(mi[:, windowIndex, componentIndex])]
        samples = ([], [], [])
        for i, l in enumerate(labels):
            for sign in [-1, 1]:
                mask = np.vstack([
                    polarity == sign,
                    self.labels == l,
                    np.abs(paramsExtra[:, 0]) >= minimumResponseAmplitude,
                ]).all(0)
                for dr, p, iUnit in zip(mi[mask, windowIndex, componentIndex], pvalues[mask, windowIndex, componentIndex], np.arange(len(self.ukeys))[mask]):
                    if l == -1:
                        dr *= -1
                    if p < alpha:
                        if sign == -1:
                            samples[0].append(dr)
                        else:
                            samples[1].append(dr)
                    else:
                        samples[2].append(dr)

        #
        xmin, xmax = xrange
        ax.hist(
            [np.clip(sample, xmin, xmax) for sample in samples],
            range=(xmin, xmax),
            bins=nBins,
            histtype='barstacked',
            color=cmap(np.arange(3)),
        )
        binCounts_, binEdges, patches = ax.hist(
            [np.clip(sample, xmin, xmax) for sample in samples],
            range=(xmin, xmax),
            bins=nBins,
            histtype='barstacked',
            facecolor='none',
            edgecolor='k',
        )
        binCounts = binCounts_.max(0)
        binCenters = binEdges[:-1] + ((binEdges[1] - binEdges[0]) / 2)
        leftEdges = binEdges[:-1]
        rightEdges = binEdges[1:]

        #
        for ukey in self.examples:
            iUnit = self._indexUnitKey(ukey)
            if iUnit is None:
                continue
            dr = mi[iUnit, windowIndex, componentIndex]
            if np.isnan(dr).item():
                continue
            binIndex = np.where(np.logical_and(
                dr >= leftEdges,
                dr <  rightEdges
            ))[0].item()
            ax.scatter(
                binCenters[binIndex],
                binCounts[binIndex] + 10,
                marker='v',
                color='k',
                s=20
            )

        #
        ax.set_xticks(xticks)
        ax.set_xlabel(f'Modulation index (MI)')
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, ax

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