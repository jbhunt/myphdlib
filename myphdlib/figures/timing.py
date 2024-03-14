import h5py
import numpy as np
from scipy.stats import sem
from matplotlib import pyplot as plt
from myphdlib.figures.analysis import AnalysisBase, GaussianMixturesModel, g, findOverlappingUnits
from myphdlib.figures.modulation import SimpleSaccadicModulationAnalysis
from myphdlib.general.toolkit import psth2

class SimpleSaccadicModulationTimingAnalysis(SimpleSaccadicModulationAnalysis):
    """
    """

    def __init__(
        self,
        **kwargs
        ):
        """
        """

        super().__init__(**kwargs)

        self.latencies = None
        self.templates = None
        self.peths = { # Standardized PSTHs
            'extrasaccadic': None,
            'perisaccadic': None
        }
        self.terms = { # Raw PSTHs
            'rProbe': {
                'extrasaccadic': None,
                'perisaccadic': None
            },
            'rMixed': None,
            'rSaccade': None,
        }
        self.windows = None

        return

    def loadNamespace(
        self,
        hdf,
        ):
        """
        """

        #
        super().loadNamespace(hdf)
        m = findOverlappingUnits(self.ukeys, hdf)

        #
        self.templates = {}
        with h5py.File(hdf, 'r') as stream:
            for direction in ('nasal', 'temporal'):
                path = f'rSaccade/dg/preferred/{direction}/fr'
                if path in stream:
                    ds = stream[path]
                    self.templates[direction] = np.array(ds)[m, :]
                    self.tSaccade = ds.attrs['t']

            if 'gmm/latencies' in stream:
                self.latencies_ = np.array(stream['gmm/latencies'])[m, :]
            if 'rProbe/dg/preferred/raw/fr' in stream:
                ds = stream['rProbe/dg/preferred/raw/fr']
                self.t = ds.attrs['t']
            if 'rProbe/dg/preferred/standardized/fr' in stream:
                self.peths['extrasaccadic'] = np.array(stream['rProbe/dg/preferred/standardized/fr'])[m, :]
            if 'gmm/modulation' in stream:
                self.modulation_ = np.array(stream['gmm/modulation'])[m, :]

        return

    def saveNamespace(self, hdf):
        """
        """
        return

    def computePeths(
        self,
        binsize=0.1,
        trange=(-0.5, 0.5),
        ):
        """
        """

        #
        leftEdges = np.arange(trange[0], trange[1], binsize)
        rightEdges = leftEdges + binsize
        self.windows = np.vstack([leftEdges, rightEdges]).T

        #
        nUnits = len(self.ukeys)
        nBins = self.t.size
        nWindows = self.windows.shape[0]
        self.terms['rMixed'] = np.full([nUnits, nBins, nWindows], np.nan)
        self.terms['rSaccade'] = np.full([nUnits, nBins, nWindows], np.nan)
        self.terms['rProbe']['perisaccadic'] = np.full([nUnits, nBins, nWindows], np.nan)
        self.peths['perisaccadic'] = np.full([nUnits, nBins, nWindows], np.nan)

        #
        for iUnit in range(nUnits):

            #
            end = '\r' if iUnit + 1 != nUnits else None
            print(f'Computing PETHs for unit {iUnit + 1} out of {nUnits}', end=end)

            #
            for iWin, perisaccadicWindow in enumerate(self.windows):

                #
                rMixed, rSaccade = super().computeTerms(
                    ukey=self.ukeys[iUnit],
                    perisaccadicWindow=perisaccadicWindow,
                )

                # Standardize the PETHs
                mu, sigma = self.ambc[iUnit, 2], self.ambc[iUnit, 3]
                yResidual = rMixed - rSaccade
                yStandard = (yResidual - mu) / sigma

                #
                self.terms['rMixed'][iUnit, :, iWin] = rMixed
                self.terms['rSaccade'][iUnit, :, iWin] = rSaccade
                self.peths['perisaccadic'][iUnit, :, iWin] = yStandard

        return
    
    def fitPeths(
        self,
        ):
        """
        """

        #
        nUnits, nBins, nWindows = self.peths['perisaccadic'].shape
        nComponents = int(np.max(self.k))
        self.modulation = np.full([nUnits, nComponents, nWindows], np.nan)
        self.latencies = np.full([nUnits, nComponents, nWindows], np.nan)

        #
        for iUnit in range(nUnits):

            #
            end = '\r' if iUnit + 1 != nUnits else None
            print(f'Re-fitting peri-saccadic PETHs for unit {iUnit + 1} out of {nUnits}', end=end)

            #
            for iWin in range(nWindows):
                peth = self.peths['perisaccadic'][iUnit, :, iWin]
                dr, latencies, params = super().refitPeth(
                    ukey=self.ukeys[iUnit],
                    sortby='amplitude',
                    peth=peth
                )
                self.modulation[iUnit, :, iWin] = dr
                self.latencies[iUnit, :, iWin] = latencies

        return

    def plotExampleRasterplot(
        self,
        ukey=('2023-07-21', 'mlati10', 262),
        responseWindow=(-0.5, 0.5),
        perisaccadicWindow=(-0.8, 1.2),
        figsize=(6, 6),
        **kwargs_
        ):

        """
        """

        kwargs = {
            'color': 'k',
            'marker': '.',
            'alpha': 0.5,
            's': 5
        }
        kwargs.update(kwargs_)

        iUnit = self.lookupUnitKey(ukey)
        if iUnit is None:
            raise Exception('Could not locate example unit')
        self.ukey = self.ukeys[iUnit]
        probeMotion = self.ambc[iUnit, 1]
        trialIndices = np.where(np.vstack([
            self.session.gratingMotionDuringProbes == probeMotion,
            self.session.probeLatencies >= perisaccadicWindow[0],
            self.session.probeLatencies <= perisaccadicWindow[1]
        ]).all(0))[0]
        latencySortedIndex = np.argsort(self.session.probeLatencies[trialIndices])
        t, M, spikeTimestamps = psth2(
            self.session.probeTimestamps[trialIndices],
            self.unit.timestamps,
            window=responseWindow,
            binsize=None,
            returnTimestamps=True
        )
        fig, axs = plt.subplots(ncols=2, sharey=True)
        x = list()
        y = list()
        for i, trialIndex in enumerate(latencySortedIndex):
            nSpikes = spikeTimestamps[trialIndex].size
            for t in np.atleast_1d(spikeTimestamps[trialIndex]):
                x.append(t)
            for r in np.full(nSpikes, i):
                y.append(r)
        axs[0].scatter(x, y, rasterized=True, **kwargs)
        axs[1].plot(self.session.probeLatencies[trialIndices][latencySortedIndex], np.arange(latencySortedIndex.size), color='k')
        axs[0].set_xlabel('Time from probe onset (sec)')
        axs[0].set_ylabel('Trial #')
        axs[1].set_xlabel('Probe latency (sec)')
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, axs

    def plotModulationByProbeLatency(
        self,
        figsize=(3, 4),
        **kwargs_
        ):
        """
        """

        kwargs = {
            'color': '0.7',
            'alpha': 0.5,
            'marker': '.',
            's': 5,
        }
        kwargs.update(kwargs_)

        #
        fig, ax = plt.subplots()

        #
        nUnits, nBins, nWindows = self.peths['perisaccadic'].shape

        #
        y = list()
        for iWin in range(nWindows):
            leftEdge, rightEdge = self.windows[iWin]
            windowCenter = rightEdge - ((rightEdge - leftEdge) / 2)
            x = np.full(nUnits, windowCenter) + np.random.normal(loc=0, scale=0.01, size=nUnits)
            ax.scatter(
                x,
                np.clip(self.modulation[:, 0, iWin], -5, 5),
                **kwargs
            )
            y.append(np.nanmean(self.modulation[:, 0, iWin]))
        binCenters = np.mean(self.windows, axis=1)
        ax.plot(binCenters, y, color='k')

        #
        ax.set_xlabel('Time from saccade initiation (sec)')
        ax.set_ylabel(r'Modulation ($\Delta R$)')
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, ax

    def plotModulationByComponentLatency(
        self,
        responseWindow=(-0.2, 0.5),
        binsize=0.05,
        figsize=(3, 4),
        **kwargs_
        ):
        """
        """

        # Keywords arguments for the scatter function
        kwargs = {
            'color': '0.7',
            'marker': '.',
            'alpha': 0.5,
            's': 5,
        }
        kwargs.update(kwargs_)

        #
        leftEdges = np.arange(responseWindow[0], responseWindow[1], binsize)
        rightEdges = leftEdges + binsize
        samples = [[] for i in range(leftEdges.shape[0])]

        #
        fig, ax = plt.subplots()
        for l, m in zip(self.latencies_.flatten(), self.modulation_.flatten()):
            mask = np.logical_and(
                l > leftEdges,
                l <= rightEdges
            )
            if mask.sum() != 1:
                continue
            iBin = np.where(mask)[0].item()
            samples[iBin].append(m)

        #
        binCenters = leftEdges + (binsize / 2)
        binMeans = list()
        for i, sample in enumerate(samples):
            if len(sample) == 0:
                binMeans.append(np.nan)
                continue
            binMeans.append(np.mean(sample))
            x = binCenters[i] + np.random.normal(loc=0, scale=0.007, size=len(sample))
            ax.scatter(x, np.clip(sample, -5, 5), **kwargs)
        ax.plot(binCenters, binMeans, color='k')

        #
        ax.set_xlabel('Time from probe onset (sec)')
        ax.set_ylabel(r'Modulation ($\Delta R$)')
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, ax
    
class IntegratedSaccadicModulationTimingAnalysis(SimpleSaccadicModulationAnalysis):
    """
    """

    def __init__(self):
        """
        """

        super().__init__()

        return

    def loadNamespace(
        self,
        hdf,
        ):
        """
        """

        #
        super().loadNamespace(hdf)
        m = findOverlappingUnits(self.ukeys, hdf)

        #
        self.templates = {}
        with h5py.File(hdf, 'r') as stream:
            for direction in ('nasal', 'temporal'):
                path = f'rSaccade/dg/preferred/{direction}/fr'
                if path in stream:
                    ds = stream[path]
                    self.templates[direction] = np.array(ds)[m, :]
                    self.tSaccade = ds.attrs['t']

            if 'gmm/latencies' in stream:
                self.latencies_ = np.array(stream['gmm/latencies'])[m, :]
            if 'rProbe/dg/preferred/raw/fr' in stream:
                ds = stream['rProbe/dg/preferred/raw/fr']
                self.t = ds.attrs['t']
            if 'rProbe/dg/preferred/standardized/fr' in stream:
                self.peths['extrasaccadic'] = np.array(stream['rProbe/dg/preferred/standardized/fr'])[m, :]
            if 'gmm/modulation' in stream:
                self.modulation_ = np.array(stream['gmm/modulation'])[m, :]

        return
    
    def saveNamespace(self, hdf):
        return
    
    def computePeths(
        self,
        binsize=0.1,
        trange=(-0.5, 0.5),
        ):
        """
        """

        #
        leftEdges = np.arange(trange[0], trange[1], binsize)
        rightEdges = leftEdges + binsize
        self.tWindowCenters = leftEdges + (binsize / 2)
        nUnits = len(self.ukeys)
        nWindows = len(leftEdges)
        nBins = self.peths['extrasaccadic'].shape[1]
        nComponents = int(self.k.max())
        self.windows = np.full([nUnits, nComponents, nWindows, 2], np.nan)

        #
        self.terms['rMixed'] = np.full([nUnits, nBins, nWindows, nComponents], np.nan)
        self.terms['rSaccade'] = np.full([nUnits, nBins, nWindows, nComponents], np.nan)
        self.terms['rProbe']['perisaccadic'] = np.full([nUnits, nBins, nWindows, nComponents], np.nan)
        self.peths['perisaccadic'] = np.full([nUnits, nBins, nWindows, nComponents], np.nan)

        #
        for iUnit in range(nUnits):

            #
            end = '\r' if iUnit + 1 != nUnits else None
            print(f'computing PETHs for unit {iUnit + 1} out of {nUnits}', end=end)

            #
            self.ukey = self.ukeys[iUnit]
            mu, sigma = self.ambc[iUnit, 2], self.ambc[iUnit, 3]

            for iComp in range(nComponents):

                #
                peakLatency = self.latencies_[iUnit, iComp]
                
                for iWin in range(nWindows):                

                    # Shift the time bins by the 
                    leftEdge = leftEdges[iWin] + peakLatency
                    rightEdge = rightEdges[iWin] + peakLatency
                    perisaccadicWindow = (leftEdge, rightEdge)
                    self.windows[iUnit, iComp, iWin] = np.array(perisaccadicWindow)

                    #
                    rMixed, rSaccade = super().computeTerms(
                        self.ukey,
                        perisaccadicWindow=perisaccadicWindow,
                    )
                    if np.isnan(rMixed).all():
                        continue

                    # Standardize the PETHs
                    yResidual = rMixed - rSaccade
                    yStandard = (yResidual - mu) / sigma

                    #
                    self.terms['rMixed'][iUnit, :, iWin, iComp] = rMixed
                    self.terms['rSaccade'][iUnit, :, iWin, iComp] = rSaccade
                    self.peths['perisaccadic'][iUnit, :, iWin, iComp] = yStandard

        return

    def fitPeths(
        self,
        ):
        """
        """

        nUnits = len(self.ukeys)
        nWindows = self.windows.shape[2]
        nComponents = int(self.k.max())
        self.modulation = np.full([nUnits, nWindows, nComponents], np.nan)
        
        #
        for iUnit in range(nUnits):

            #
            end = '\r' if iUnit + 1 != nUnits else None
            print(f'computing PETHs for unit {iUnit + 1} out of {nUnits}', end=end)

            #
            self.ukey = self.ukeys[iUnit]

            for iComp in range(nComponents):
                
                for iWin in range(nWindows): 

                    peth = self.peths['perisaccadic'][iUnit, :, iWin, iComp]
                    if np.isnan(peth).all():
                        continue
                    dr, latencies, params = super().refitPeth(
                        ukey=self.ukey,
                        sortby='amplitude',
                        peth=peth
                    )
                    self.modulation[iUnit, iWin, iComp] = dr[iComp]          

        return

    def plotModulationByResponseLatency(
        self,
        ymin=-5,
        ymax=5,
        figsize=(3, 4),
        ):
        """
        """

        fig, ax = plt.subplots()
        nUnits, nWindows, nComponents = self.modulation.shape

        curves = list()
        for iUnit in range(nUnits):
            for iComp in range(nComponents):
                curves.append(self.modulation[iUnit, :, iComp])
        curves = np.array(curves)
        ax.plot(self.tWindowCenters, np.nanmean(curves, axis=0), color='k')
        for i in range(curves.shape[1]):
            x = self.tWindowCenters[i] + np.random.normal(loc=0, scale=0.01, size=curves.shape[0])
            y = np.clip(curves[:, i], ymin, ymax)
            ax.scatter(x, y, color='0.7', alpha=0.5, marker='.', s=5, rasterized=True)

        #
        ax.set_xlabel('Time from saccade onset (sec)')
        ax.set_ylabel(r'Modulation ($\Delta R$)')
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, ax