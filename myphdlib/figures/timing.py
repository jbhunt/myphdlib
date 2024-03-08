import h5py
import numpy as np
from matplotlib import pyplot as plt
from myphdlib.figures.analysis import AnalysisBase, GaussianMixturesModel, g, findOverlappingUnits
from myphdlib.figures.modulation import SimpleSaccadicModulationAnalysis

class ExtendedSaccadicModulationAnalysis(SimpleSaccadicModulationAnalysis):
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
        self.pethsExpanded = {
            'extrasaccadic': None,
            'perisaccadic': None
        }
        self.termsExpanded = {
            'rProbe': {
                'extrasaccadic': None,
                'perisaccadic': None
            },
            'rMixed': None,
            'rSaccade': None,
        }
        self.perisaccadicWindows = None

        return

    def loadNamespace(
        self,
        hdf,
        ):
        """
        """

        #
        super().loadNamespace()

        #
        self.templates = {}
        with h5py.File(hdf, 'r') as stream:
            for direction in ('nasal', 'temporal'):
                path = f'rSaccade/dg/preferred/{direction}/fr'
                if path in stream:
                    ds = stream[path]
                    self.templates[direction] = np.array(ds)
                    self.tSaccade = ds.attrs['t']

            if 'gmm/latencies' in stream:
                self.latencies = np.array(stream['gmm/latencies'])
            if 'rProbe/dg/preferred/raw/fr' in stream:
                ds = stream['rProbe/dg/preferred/raw/fr']
                self.t = ds.attrs['t']
            if 'rProbe/dg/preferred/standardized/fr' in stream:
                self.pethsExpanded['extrasaccadic'] = np.array(stream['rProbe/dg/preferred/standardized/fr'])
            if 'gmm/modulation' in stream:
                self.modulation = np.array(stream['gmm/modulation'])

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
        self.perisaccadicWindows = np.vstack([leftEdges, rightEdges]).T

        #
        nUnits = len(self.ukeys)
        nBins = self.t.size
        nWindows = self.perisaccadicWindows.shape[0]
        self.termsExpanded['rMixed'] = np.full([nUnits, nBins, nWindows], np.nan)
        self.termsExpanded['rSaccade'] = np.full([nUnits, nBins, nWindows], np.nan)
        self.termsExpanded['rProbe']['perisaccadic'] = np.full([nUnits, nBins, nWindows], np.nan)
        self.pethsExpanded['perisaccadic'] = np.full([nUnits, nBins, nWindows], np.nan)

        #
        for iWin, perisaccadicWindow in enumerate(self.perisaccadicWindows):
            super().computePeths(
                perisaccadicWindow=perisaccadicWindow
            )
            self.termsExpanded['rMixed'][:, :, iWin] = self.terms['rMixed']
            self.termsExpanded['rSaccade'][:, :, iWin] = self.terms['rSaccade']
            self.termsExpanded['rProbe']['perisaccadic'][:, :, iWin] = self.terms['rProbe']['perisaccadic']
            self.pethsExpanded['perisaccadic'][:, :, iWin] = self.terms['rProbe']['perisaccadic']

        return
    
    def fitPeths(
        self,
        ):
        """
        """

        nUnits, nBins, nWindows = self.pethsExpanded['perisaccadic'].shape
        self.modulationExpanded = np.full([nUnits, nBins, nWindows], np.nan)
        self.latenciesExpanded = np.full([nUnits, nBins, nWindows])
        for iWin in range(nWindows):
            self.peths['perisaccadic'] = self.pethsExpanded['perisaccadic'][:, :, iWin]
            super().fitPeths(sortby='amplitude')
            self.modulationExpanded[:, :, iWin] = self.modulation
            self.latenciesExpanded[:, :, iWin] = self.latencies

        return

    def plotExampleRasterplot(
        self,
        ):
        """
        """

        return

    def plotExamplePeth(
        self,
        ):
        """
        """

        return

    def plotModulationByProbeLatency(
        self,
        ):
        """
        """

        fig, ax = plt.subplots()

        nUnits, nBins, nWindows = self.pethsExpanded['perisaccadic'].shape
        for iWin in range(nWindows):
            leftEdge, rightEdge = self.perisaccadicWindows[iWin]
            windowCenter = rightEdge - ((rightEdge - leftEdge) / 2)
            x = np.full(nUnits, windowCenter)
            ax.scatter(
                x,
                self.modulationExpanded[:, 0, iWin],
                color='k',
                alpha=0.3,
                marker='.',
                s=12,
            )

        return fig, ax

    def plotModulationByComponentLatency(
        self,
        **kwargs_
        ):
        """
        """

        # Keywords arguments for the scatter function
        kwargs = {
            'color': 'k',
            'marker': '.',
            'alpha': 0.3,
            's': 12,
        }
        kwargs.update(kwargs_)

        # Need to do this to reset the latency and modulation attributes
        self.loadNamespace() 

        # Plot
        fig, ax = plt.subplots()
        for l, m in zip(self.latencies.flatten(), self.modulation.flatten()):
            ax.scatter(l, m, **kwargs)

        return fig, ax
    
class ModulationTimingAnalysis(AnalysisBase):
    """
    """

    def __init__(self):
        """
        """

        super().__init__()

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
        self.perisaccadicWindows = np.vstack([leftEdges, rightEdges]).T

        #
        nUnits = len(self.ukeys)
        nWindows = len(self.perisaccadicWindows)

        #
        for iUnit in range(nUnits):
            self.ukey = self.ukeys[iUnit]
            for iWin in range(nWindows):
                leftEdge, rightEdge = self.perisaccadicWindows[iWin]
                for iComp in range(5):
                    responseLatency = self.session.probeLatency + self.latencies[iUnit, iComp]
                    trialIndices = np.where(np.vstack([
                        responseLatency > leftEdge,
                        responseLatency <= rightEdge,
                        self.session.gratingMotionDuringProbes == -1, # TODO: Use the preferred direction of motion

                    
                    ]).all(0))[0]

        return