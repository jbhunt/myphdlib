import h5py
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import find_peaks as findPeaks
from matplotlib.colors import LinearSegmentedColormap
from myphdlib.general.toolkit import psth2
from myphdlib.figures.modulation import BasicSaccadicModulationAnalysis
from myphdlib.figures.analysis import findOverlappingUnits, GaussianMixturesModel
import seaborn as sns
import pandas as pd
from itertools import product
from scipy.stats import spearmanr

class BoostrappedSaccadicModulationAnalysis(BasicSaccadicModulationAnalysis):
    """
    """

    def __init__(self):
        """
        """

        super().__init__()

        self.pethsResampled = None
        self.pvalues = None
        self.samples = None
        self.msign = None

        return

    def saveNamespace(
        self,
        hdf,
        nUnitsPerChunk=100,
        ):
        """
        """

        datasets = {
            'bootstrap/p': self.pvalues,
            'bootstrap/sign': self.msign,
        }

        #
        m = findOverlappingUnits(self.ukeys, hdf)

        with h5py.File(hdf, 'a') as stream:

            #
            for path, attribute in datasets.items():
                if attribute is None:
                    continue
                if path in stream:
                    del stream[path]
                data = np.full([m.size, *attribute.shape[1:]], np.nan)
                data[m] = attribute
                ds = stream.create_dataset(
                    path,
                    dtype=data.dtype,
                    shape=data.shape,
                    data=data
                )

        #
        self._saveLargeDataset(
            hdf,
            path='bootstrap/peths',
            dataset=self.pethsResampled,
            nUnitsPerChunk=nUnitsPerChunk,
        )

        #
        self._saveLargeDataset(
            hdf,
            path='bootstrap/samples',
            dataset=self.samples,
            nUnitsPerChunk=nUnitsPerChunk
        )

        return

    def loadNamespace(
        self,
        hdf
        ):
        """
        """

        #
        m = findOverlappingUnits(self.ukeys, hdf)
        datasets = {
            'bootstrap/peths': 'pethsResampled',
            'bootstrap/samples': 'samples',
            'bootstrap/sign': 'msigns',
            'bootstrap/p': 'pvalues',
        }
        with h5py.File(hdf, 'r') as stream:
            for path, attribute in datasets.items():
                if path in stream:
                    ds = np.array(stream[path][m])
                    self.__setattr__(attribute, ds)

        super().loadNamespace(hdf)

        return

    def downsampleExtrasaccadicPeths(
        self,
        nRuns=30,
        responseWindow=(-0.2, 0.5),
        perisaccadicWindow=(0, 0.1),
        binsize=0.01,
        smoothingKernelWidth=0.01,
        buffer=1,
        ):
        """
        """

        nUnits, nBins = self.peths['extra'].shape
        self.pethsResampled = np.full([nUnits, nBins, nRuns], np.nan)
        for iUnit in range(nUnits):

            #
            end = '\r' if iUnit + 1 != nUnits else None
            print(f'Re-sampling PETHs for unit {iUnit + 1} out of {nUnits}', end=end)

            #
            self.ukey = self.ukeys[iUnit]
            mu, sigma = self.ambc[iUnit, 2], self.ambc[iUnit, 3]

            #
            trialIndicesPerisaccadic, probeTimestamps, probeLatencies, saccadeLabels, gratingMotion = self._loadEventDataForProbes(
                perisaccadicWindow
            )
            nTrialsForResampling = trialIndicesPerisaccadic.size
            if nTrialsForResampling == 0:
                continue

            # Extra-saccadic trial indices
            trialIndicesExtrasaccadic = np.where(np.vstack([
                gratingMotion == self.ambc[self.iUnit, 1],
                np.logical_or(
                    probeLatencies < perisaccadicWindow[0],
                    probeLatencies > perisaccadicWindow[1]
                )
            ]).all(0))[0]
            if trialIndicesExtrasaccadic.size == 0:
                continue

            # Compute relative spike timestamps
            responseWindowBuffered = (
                responseWindow[0] - buffer,
                responseWindow[1] + buffer
            )
            t, M, spikeTimestamps = psth2(
                self.session.probeTimestamps[trialIndicesExtrasaccadic],
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
                    size=nTrialsForResampling
                )
                # sample = list()
                # for iTrial in trialIndices:
                #     for ts in spikeTimestamps[iTrial]:
                #         sample.append(ts)
                # sample = np.array(sample)
                sample = np.concatenate([spikeTimestamps[i] for i in trialIndices])
                try:
                    t, fr = self.unit.kde(
                        self.session.probeTimestamps[trialIndices],
                        responseWindow=responseWindow,
                        binsize=binsize,
                        sigma=smoothingKernelWidth,
                        sample=sample,
                        nTrials=nTrialsForResampling,
                    )
                except:
                    continue

                # Standardize PSTH
                self.pethsResampled[iUnit, :, iRun] = (fr - mu) / sigma

        return

    def refitExtrasaccadicPeth(
        self,
        ukey=None,
        iRun=0,
        **kwargs_
        ):
        """
        Re-fit the resampled extra-saccadic PETHs
        """

        #
        kwargs = {
            'minimumPeakHeight': 0.15,
            'maximumPeakHeight': 1,
            'minimumPeakProminence': 0.05,
            'minimumPeakWidth': 0.001,
            'maximumPeakWidth': 0.02,
            'minimumPeakLatency': 0,
            'initialPeakWidth': 0.001,
            'maximumLatencyShift': 0.003,
            'maximumBaselineShift': 0.001,
            'maximumAmplitudeShift': 0.001 
        }
        kwargs.update(kwargs_)

        #
        if ukey is None:
            self.ukey = ukey
        for iUnit in range(len(self.ukeys)):
            date, animal, cluster = self.ukeys[iUnit]
            if date == self.ukey[0] and animal == self.ukey[1] and cluster == self.ukey[2]:
                break

        #
        y = self.pethsResampled[iUnit ,:, iRun]

        # Load parameters estimated for extra-saccadic PSTH
        params = self.params[iUnit]
        if np.isnan(params).all():
            return
        abcd = params[np.invert(np.isnan(params))]
        abc, d = abcd[:-1], abcd[-1]
        A1, B1, C1 = np.split(abc, 3)
        k = A1.size
        order = np.argsort(np.abs(A1))[::-1]
        A1 = A1[order]
        B1 = B1[order]
        C1 = C1[order]

        #
        peakIndices = list()
        for coef in (-1, 1):
            peakIndices_, peakProperties = findPeaks(
                coef * y,
                height=kwargs['minimumPeakHeight'],
                prominence=kwargs['minimumPeakProminence']
            )
            if peakIndices_.size == 0:
                continue
            for iPeak in range(peakIndices_.size):

                # Exclude peaks detected before the stimulus  onset
                if self.t[peakIndices_[iPeak]] <= 0:
                    continue

                #
                peakIndices.append(peakIndices_[iPeak])
        peakIndices = np.array(peakIndices)

        # Find the peak most similar to the original largest peak
        peakIndex = peakIndices[np.argmin(np.abs(self.t[peakIndices] - B1[0]))]

        # Compute peak properties
        peakAmplitude = y[peakIndex]
        peakLatency = self.t[peakIndex]
        peakAmplitudes = A1
        peakAmplitudes[0] = peakAmplitude
        peakLatencies = B1
        peakLatencies[0] = peakLatency
        peakWidths = C1
        peakWidths[0] = kwargs['initialPeakWidth']

        # Define the parameter boundaries
        amplitudeBoundaries = np.vstack([
            peakAmplitudes - 0.001,
            peakAmplitudes + 0.001
        ]).T
        amplitudeBoundaries[0, :] = np.array([
            peakAmplitude - kwargs['maximumAmplitudeShift'],
            peakAmplitude + kwargs['maximumAmplitudeShift']
        ])
        latencyBoundaries = np.vstack([
            peakLatencies - 0.001,
            peakLatencies + 0.001
        ]).T
        latencyBoundaries[0, :] = np.array([
            peakLatency - kwargs['maximumLatencyShift'],
            peakLatency + kwargs['maximumLatencyShift']
        ])
        widthBoundaries = np.vstack([
            peakWidths - 0.001,
            peakWidths + 0.001
        ]).T
        widthBoundaries[0, :] = np.array([
            kwargs['minimumPeakWidth'],
            kwargs['maximumPeakWidth']
        ])

        # Initialize the parameter space and bounds
        p0 = np.concatenate([
            np.array([d]),
            peakAmplitudes,
            peakLatencies,
            peakWidths
        ])
        bounds = np.vstack([
            np.array([[
                d - 0.001,
                d + 0.001
            ]]),
            amplitudeBoundaries,
            latencyBoundaries,
            widthBoundaries
        ]).T

        # Fit the GMM and compute the residual sum of squares (rss)
        gmm = GaussianMixturesModel(k)
        gmm.fit(
            self.t,
            y,
            p0=p0,
            bounds=bounds
        )
        yFit = gmm.predict(self.t)

        # Extract the parameters of the fit GMM
        d, abc = gmm._popt[0], gmm._popt[1:]
        A2, B2, C2 = np.split(abc, 3)
        order = np.argsort(np.abs(A1))[::-1] # Sort by amplitude
        params = np.concatenate([
            A2[order],
            B2[order],
            C2[order],
        ])
        paramsPadded = np.full(self.params.shape[1], np.nan)
        paramsPadded[:params.size] = params
        paramsPadded[-1] = d

        return yFit, paramsPadded

    def generateNullSamples(
        self,
        nRuns=None
        ):
        """
        """

        #
        nUnits, nBins, nRuns_ = self.pethsResampled.shape
        if nRuns is None:
            nRuns = nRuns_
        nComponents = int(np.nanmax(self.k.flatten()))
        self.samples = np.full([nUnits, nRuns, nComponents], np.nan)
        for iUnit in range(nUnits):

            #
            self.ukey = self.ukeys[iUnit]
            end = '\r' if iUnit + 1 != nUnits else None
            print(f'Generating null samples for unit {iUnit + 1} out of {nUnits}', end=end)

            #
            sample = np.full([nRuns, nComponents], np.nan)
            if np.isnan(self.modulation[self.iUnit, :, -1]).all():
                continue
            for iRun in range(nRuns):
                peth = self.pethsResampled[iUnit, :, iRun]
                if np.isnan(peth).all():
                    continue
                dr, latencies, popt = super().refitSinglePeth(
                    self.ukey,
                    peth=peth
                )
                sample[iRun, :] = dr

            #
            self.samples[iUnit, :, :] = sample

    def computeProbabilityValues(self):
        """
        """

        nUnits, nBins, nRuns = self.pethsResampled.shape
        nComponents = int(np.nanmax(self.k.flatten()))
        nWindows = len(self.windows)
        self.pvalues = np.full([nUnits, nWindows, nComponents], np.nan)
        self.msign = np.full([nUnits, nWindows, nComponents], np.nan)

        for ukey in self.ukeys:

            #
            self.ukey = ukey
            end = '\n' if self.iUnit + 1 == nUnits else '\r'
            print(f'Computing p-values for unit {self.iUnit + 1} out of {nUnits}', end=end)
            for iWindow in range(nWindows):
                for iComp in range(nComponents):

                    #
                    sample = self.samples[self.iUnit, iComp, :]
                    tv = self.modulation[self.iUnit, iComp, iWindow]
                    sign = -1 if tv < 0 else +1
                    m = np.invert(np.isnan(sample))
                    p = np.sum(np.abs(sample[m]) > np.abs(tv)) / m.sum()
                    self.pvalues[self.iUnit, iWindow, iComp] = p
                    self.msign[self.iUnit, iWindow, iComp] = sign

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
        xrange=(-1, 1),
        iWindow=-1,
        ):
        """
        """

        #
        if minimumResponseAmplitude is None:
            minimumResponseAmplitude = 0

        #
        cmap = LinearSegmentedColormap.from_list('mycmap', colorspace, N=3)

        #
        fig, ax = plt.subplots()

        #
        samples = ([], [], [])
        for i, l in enumerate(labels):
            for polarity in [-1, 1]:
                m = np.vstack([
                    self.msign[:, iWindow, 0] == polarity,
                    np.ravel(self.labels) == l,
                    np.abs(self.params[:, 0]) >= minimumResponseAmplitude,
                ]).all(0)
                for dr, p, iUnit in zip(self.modulation[m, 0, iWindow], self.pvalues[m, iWindow, 0], np.arange(len(self.ukeys))[m]):
                    if l == -1:
                        dr *= -1
                    if normalize:
                        dr /= self.params[iUnit, 0]
                    if p < a:
                        if polarity == -1:
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
        ax.hist(
            [np.clip(sample, xmin, xmax) for sample in samples],
            range=(xmin, xmax),
            bins=nBins,
            histtype='barstacked',
            facecolor='none',
            edgecolor='k',
        )

        #
        ax.set_xlabel(r'$\Delta$R')
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
        xrange=(-1, 1),
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
        arange=np.arange(0, 3.1, 0.1)
        ):
        """
        """

        fig, axs = plt.subplots(nrows=3, sharex=True)
        nUnitsTotal = list()
        nUnitsSuppressed = list()
        nUnitsEnhanced = list()

        #
        for a in arange:
            nUnitsTotal.append(np.sum(self.params[:, 0] > a))
            m = np.logical_and(
                self.pvalues < 0.05,
                self.params[:, 0] >= a
            )
            nUnitsSuppressed.append(np.sum(
                self.modulation[m, 0, 5] < 0
            ))
            nUnitsEnhanced.append(np.sum(
                self.modulation[m, 0, 5] > 0
            ))

        #
        axs[0].plot(arange, nUnitsTotal)
        axs[1].plot(arange, nUnitsSuppressed)
        axs[1].plot(arange, nUnitsEnhanced)

        return fig, axs