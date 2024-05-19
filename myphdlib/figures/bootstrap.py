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

def _generateNullSampleForSingleUnit(
    peths,
    params1,
    tProbe,
    nComponents,
    iUnit,
    nUnits,
    maximumAmplitudeShift=100,
    maxfev=1000,
    ):
    """
    Worker function (for parallelization) that refits the peri-saccadic PPTHs for a single unit
    """

    #
    end = '\r' if iUnit + 1 != nUnits else '\n'
    print(f'Generating null sample for unit {iUnit + 1} out of {nUnits}', end=end, flush=True)
    nBins, nRuns = peths.shape
    sample = np.full([nRuns, nComponents], np.nan)

    #
    if np.isnan(params1).all():
        return sample

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

        # Example neurons
        self.examples = (
            ('2023-07-12', 'mlati9', 710),
            ('2023-07-20', 'mlati9', 337),
            ('2023-05-26', 'mlati7', 336)
        )

        return

    def resampleExtrasaccadicPeths(
        self,
        nRuns=100,
        rate=0.05,
        minimumTrialCount=10,
        responseWindow=(-0.2, 0.5),
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
                bl = self.ns[f'stats/{probeDirection}/{saccadeType}/extra'][iUnit, 0]

                # Load event data
                probeTimestamps, probeLatencies, saccadeLabels, gratingMotion = self._loadEventDataForProbes( )

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
                nTrialsTotal = np.sum(gratingMotion == self.preference[self.iUnit])

                # Determine the number of trials to sample
                nTrialsForResampling = int(round(nTrialsTotal * rate, 0))
                if minimumTrialCount is not None and nTrialsForResampling < minimumTrialCount:
                    nTrialsForResampling = minimumTrialCount
                if trialIndicesExtrasaccadic.size < nTrialsForResampling:
                    raise Exception('Not enough extra-saccadic trials to re-sample')

                # Compute relative spike timestamps
                responseWindowBuffered = (
                    responseWindow[0] - buffer,
                    responseWindow[1] + buffer
                )
                t, M, spikeTimestamps = psth2(
                    probeTimestamps[trialIndicesExtrasaccadic],
                    self.unit.timestamps,
                    window=responseWindowBuffered,
                    binsize=binsize,
                    returnTimestamps=True
                )

                #
                for iRun in range(nRuns):

                    # Use kernel density estimation (takes a long time)
                    trialIndices = np.random.choice(
                        np.arange(trialIndicesExtrasaccadic.size),
                        size=nTrialsForResampling,
                        replace=False
                    )

                    # Use KDE
                    sample = np.concatenate([spikeTimestamps[i] for i in trialIndices])
                    try:
                        t, fr = self.unit.kde(
                            probeTimestamps[trialIndices],
                            responseWindow=responseWindow,
                            binsize=binsize,
                            sigma=smoothingKernelWidth,
                            sample=sample,
                            nTrials=nTrialsForResampling,
                        )
                    except:
                        continue

                    # Standardize PSTH
                    self.ns[f'ppths/{probeDirection}/{saccadeType}/resampled'][iUnit, :, iRun] = (fr - bl) / self.factor[iUnit]

        return

    def generateNullSamples(
        self,
        nRuns=None,
        maximumAmplitudeShift=100,
        saccadeType='real',
        parallelize=True,
        nProcesses=30,
        chunksize=1,
        maxfevs=100,
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
            args = list()
            for iUnit in range(nUnits):
                args.append((
                    self.ns[f'ppths/pref/{saccadeType}/resampled'][iUnit],
                    self.ns[f'params/pref/{saccadeType}/extra'][iUnit],
                    self.tProbe,
                    nComponents,
                    iUnit,
                    nUnits,
                    maximumAmplitudeShift,
                    maxfevs,
                ))
            if parallelize:
                with Pool(nProcesses, maxtasksperchild=1) as pool:
                    result = pool.starmap(
                        _generateNullSampleForSingleUnit,
                        args,
                        chunksize=chunksize,
                    )
                for iUnit, sample in result:
                    samples[iUnit] = sample
            else:
                for iUnit_, el in enumerate(args):
                    end = '\r' if iUnit + 1 != nUnits else '\n'
                    print(f'Generating null sample for unit {iUnit_ + 1} out of {nUnits}', end=end, flush=True)
                    iUnit, sample = _generateNullSampleForSingleUnit(*el)
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

                        #
                        sample = self.ns[f'samples/{probeDirection}/{saccadeType}'][self.iUnit, :, iComp]
                        mi = self.ns[f'mi/{probeDirection}/{saccadeType}'][self.iUnit, iWindow, iComp]
                        mask = np.invert(np.isnan(sample))
                        if mask.sum() == 0:
                            continue
                        p = np.sum(np.abs(sample[mask]) > np.abs(mi)) / mask.sum()
                        self.ns[f'p/{probeDirection}/{saccadeType}'][self.iUnit, iWindow, iComp] = p

        return

    def plotModulationDistributionsWithHistogram(
        self,
        a=0.05,
        figsize=(4, 2),
        colorspace=('k', 'k', 'w'),
        minimumResponseAmplitude=2,
        nBins=20,
        labels=(1, 2, 3, -1),
        normalize=True,
        xrange=(-2, 2),
        iWindow=5,
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
        polarity = np.array([
            -1 if self.mi[i, iWindow, 0] < 0 else 1
                for i in range(len(self.ukeys))
        ])
        polarity[np.isnan(self.mi[:, iWindow, 0])]
        samples = ([], [], [])
        for i, l in enumerate(labels):
            for sign in [-1, 1]:
                mask = np.vstack([
                    polarity == sign,
                    self.model['labels'] == l,
                    np.abs(self.model['params1'][:, 0]) >= minimumResponseAmplitude,
                ]).all(0)
                for dr, p, iUnit in zip(self.mi[mask, iWindow, 0], self.p[mask, 0], np.arange(len(self.ukeys))[mask]):
                    if l == -1:
                        dr *= -1
                    if normalize:
                        dr /= self.model['params1'][iUnit, 0]
                    if p < a:
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
            mi = self.mi[iUnit, iWindow, 0] / self.model['params1'][iUnit, 0]
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
    
    def plotSurvivalByAmplitudeThreshold(
        self,
        arange=np.arange(0, 3.1, 0.1),
        windowIndex=5,
        figsize=(3, 6)
        ):
        """
        """

        fig, axs = plt.subplots(nrows=3, sharex=True, sharey=True)
        nUnitsTotal = list()
        nUnitsSuppressed = list()
        nUnitsEnhanced = list()

        #
        for a in arange:
            nUnitsTotal.append(np.sum(self.params[:, 0] > a))
            m = np.logical_and(
                self.pvalues[:, windowIndex, 0] < 0.05,
                self.params[:, 0] >= a
            )
            nUnitsSuppressed.append(np.sum(
                self.modulation[m, 0, 5] < 0
            ))
            nUnitsEnhanced.append(np.sum(
                self.modulation[m, 0, 5] > 0
            ))

        #
        axs[0].plot(arange, nUnitsTotal, color='k')
        axs[1].plot(arange, nUnitsSuppressed, color='k')
        axs[2].plot(arange, nUnitsEnhanced, color='k')
        titles = (
            '# of total units',
            '# of suppressed units',
            '# of enhanced units'
        )
        for i, ax in enumerate(axs):
            ax.set_title(titles[i], fontsize=10)
            ax.set_ylabel('# of units')
        axs[-1].set_xlabel('Amplitude threshold')

        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, axs