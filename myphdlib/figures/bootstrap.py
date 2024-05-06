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
from multiprocessing import Pool

class BoostrappedSaccadicModulationAnalysis(BasicSaccadicModulationAnalysis):
    """
    """

    def __init__(self, iw=5, **kwargs):
        """
        """

        super().__init__(**kwargs)

        # Window index
        self.iw = iw

        #
        self.peths = {
            'resampled': None,
        }
        self.p = {
            'real': None,
        }
        self.samples = {
            'real': None
        }
        self.features = {
            'm': None,
            's': None,
            'd': None,
        }
        self.model = {
            'k': None,
        }
        self.windows = None
        self.filter = None
        self.mi = {
            'real': None,
        }
        self.tProbe = None

        # Example neurons
        self.examples = (
            ('2023-07-12', 'mlati9', 710),
            ('2023-07-20', 'mlati9', 337),
            ('2023-05-26', 'mlati7', 336)
        )

        return

    def saveNamespace(
        self,
        nUnitsPerChunk=100,
        ):
        """
        """

        datasets = {
            'bootstrap/p': self.p['real'],
        }

        #
        mask = self._intersectUnitKeys(self.ukeys)
        with h5py.File(self.hdf, 'a') as stream:
            for path, attr in datasets.items():
                if attr is None:
                    continue
                if path in stream:
                    del stream[path]
                data = np.full([mask.size, *attr.shape[1:]], np.nan)
                data[mask] = attr
                ds = stream.create_dataset(
                    path,
                    dtype=data.dtype,
                    shape=data.shape,
                    data=data
                )

        #
        self._saveLargeDataset(
            self.hdf,
            path='bootstrap/peths',
            dataset=self.peths['resampled'],
            nUnitsPerChunk=nUnitsPerChunk,
        )

        #
        self._saveLargeDataset(
            self.hdf,
            path='bootstrap/samples',
            dataset=self.samples['real'],
            nUnitsPerChunk=nUnitsPerChunk
        )

        return

    def loadNamespace(
        self,
        ):
        """
        """

        #
        mask = self._intersectUnitKeys(self.ukeys)
        datasets = {
            'bootstrap/peths': ('peths', 'resampled'),
            'bootstrap/samples': ('samples', 'real'),
            'bootstrap/p': ('p', 'real'),
            'clustering/features/m': ('features', 'm'),
            'clustering/features/s': ('features', 's'),
            'clustering/features/d': ('features', 'd'),
            'clustering/model/k': ('model', 'k'),
            'clustering/model/params': ('model', 'params1'),
            'clustering/model/labels': ('model', 'labels'),
            'clustering/filter': ('filter', None),
            'modulation/windows': ('windows', None),
            'modulation/mi': ('mi', 'real'),
        }
        with h5py.File(self.hdf, 'r') as stream:
            for path, (attr, key) in datasets.items():
                if path in stream:
                    if 'windows' in path.split('/'): # Do not apply unit mask
                        ds = np.array(stream[path])
                    else:
                        ds = np.array(stream[path][mask])
                    if len(ds.shape) == 2 and ds.shape[-1] == 1:
                        ds = ds.flatten()
                    if key is None:
                        self.__setattr__(attr, ds)
                    else:
                        self.__getattribute__(attr)[key] = ds

            #
            ds = stream['clustering/peths/standard']
            if 't' in ds.attrs.keys():
                self.tProbe = np.array(ds.attrs['t'])

        return

    def downsampleExtrasaccadicPeths(
        self,
        nRuns=30,
        responseWindow=(-0.2, 0.5),
        binsize=0.01,
        smoothingKernelWidth=0.01,
        buffer=1,
        rate=None,
        minimumTrialCount=1,
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
        self.peths['resampled'] = np.full([nUnits, nBins, nRuns], np.nan)
        for iUnit in range(nUnits):

            #
            end = '\r' if iUnit + 1 != nUnits else None
            print(f'Re-sampling PETHs for unit {iUnit + 1} out of {nUnits}', end=end)

            #
            self.ukey = self.ukeys[iUnit]
            mu, sigma = self.features['m'][iUnit], self.features['s'][iUnit]

            #
            perisaccadicWindow = self.windows[self.iw]
            trialIndicesPerisaccadic, probeTimestamps, probeLatencies, saccadeLabels, gratingMotion = self._loadEventDataForProbes(
                perisaccadicWindow
            )

            # Extra-saccadic trial indices
            trialIndicesExtrasaccadic = np.where(np.vstack([
                gratingMotion == self.features['d'][self.iUnit],
                np.logical_or(
                    probeLatencies < perisaccadicWindow[0],
                    probeLatencies > perisaccadicWindow[1]
                )
            ]).all(0))[0]
            if trialIndicesExtrasaccadic.size == 0:
                continue

            #
            if rate is None:
                nTrialsForResampling = trialIndicesPerisaccadic.size
                if nTrialsForResampling == 0:
                    continue
        
            else:
                nTrialsForResampling = int(round(trialIndicesExtrasaccadic.size * rate, 0))
            
            #
            if minimumTrialCount is not None and nTrialsForResampling < minimumTrialCount:
                nTrialsForResampling = minimumTrialCount

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

                # Use raw PSTH
                # sample = list()
                # for iTrial in trialIndices:
                #     for ts in spikeTimestamps[iTrial]:
                #         sample.append(ts)
                # sample = np.array(sample)

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
                self.peths['resampled'][iUnit, :, iRun] = (fr - mu) / sigma

        return
    
    def _generateNullSample(
        self,
        iUnit,
        ):
        """
        """

        ukey = self.ukeys[iUnit]
        peths = self.peths['resampled'][iUnit]
        nRuns = peths.shape[0]
        sample = np.full(nRuns, np.nan)
        for iRun in range(nRuns):
            peth = peths[iRun]
            dr, params2 = super()._fitPerisaccadicPeth(
                ukey,
                peth=peth
            )
            sample[iRun] = dr

        return sample

    def generateNullSamples(
        self,
        nRuns=None,
        useFilter=True,
        parallelize=True,
        key='real',
        ):
        """
        """

        #
        nUnits, nBins, nRuns_ = self.peths['resampled'].shape
        if nRuns is None:
            nRuns = nRuns_
        nComponents = int(np.nanmax(self.model['k']))
        samples = np.full([nUnits, nRuns, nComponents], np.nan)

        #
        if parallelize:
            with Pool(None) as pool:
                samples_ = pool.map(
                    self._generateNullSample,
                    np.arange(len(self.ukeys))
                )
            samples = np.array(samples_)

        #
        else:
            for iUnit in range(nUnits):

                #
                if useFilter and self.filter[iUnit] == False:
                    continue

                #
                self.ukey = self.ukeys[iUnit]
                end = '\r' if iUnit + 1 != nUnits else None
                print(f'Generating null samples for unit {iUnit + 1} out of {nUnits}', end=end)

                #
                sample = np.full([nRuns, nComponents], np.nan)
                for iRun in range(nRuns):
                    peth = self.peths['resampled'][iUnit, :, iRun]
                    if np.isnan(peth).all():
                        continue
                    dr, params2 = super()._fitPerisaccadicPeth(
                        self.ukey,
                        peth=peth
                    )
                    sample[iRun, :] = dr

                #
                samples[iUnit, :, :] = sample

        #
        self.samples[key] = samples

        return

    def computeProbabilityValues(
        self,
        key='real',
        ):
        """
        """

        nUnits, nBins, nRuns = self.peths['resampled'].shape
        nComponents = int(np.nanmax(self.model['k']))
        self.p[key] = np.full([nUnits, nComponents], np.nan)

        for ukey in self.ukeys:

            #
            self.ukey = ukey
            end = '\n' if self.iUnit + 1 == nUnits else '\r'
            print(f'Computing p-values for unit {self.iUnit + 1} out of {nUnits}', end=end)
            for iComp in range(nComponents):
                sample = self.samples[key][self.iUnit, :, iComp]
                tv = self.mi[key][self.iUnit, self.iw, iComp]
                mask = np.invert(np.isnan(sample))
                p = np.sum(np.abs(sample[mask]) > np.abs(tv)) / mask.sum()
                self.p[key][self.iUnit, iComp] = p

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