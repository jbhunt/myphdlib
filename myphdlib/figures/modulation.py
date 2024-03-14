import h5py
import numpy as np
from scipy.signal import find_peaks as findPeaks
from matplotlib import pylab as plt
from myphdlib.general.toolkit import psth2
from myphdlib.figures.analysis import AnalysisBase, GaussianMixturesModel, g, findOverlappingUnits

class SimpleSaccadicModulationAnalysis(AnalysisBase):
    """
    """

    def __init__(
        self,
        **kwargs,
        ):
        """
        """

        super().__init__(**kwargs)

        self.peths = {
            'extrasaccadic': None,
            'perisaccadic': None
        }
        self.terms = {
            'rProbe': {
                'extrasaccadic': None,
                'perisaccadic': None
            },
            'rMixed': None,
            'rSaccade': None,
        }
        self.ambc = None # Amplitude, Preferred direction (motion), baseline FR mean, STD
        self.labels = None
        self.params = None
        self.k = None
        self.t = None
        self.templates = None
        self.tSaccade = None
        self.refits = None
        self.latencies = None
        self.examples = (
            ('2023-05-26', 'mlati7', 83),
            ('2023-06-30', 'mlati9', 17),
            ('2023-07-12', 'mlati9', 710)
        )

        return

    def loadNamespace(
        self,
        hdf,
        ):
        """
        """

        d = {
            'rProbe/dg/preferred/ambc': 'ambc',
            'gmm/params': 'params',
            'gmm/labels': 'labels',
            'gmm/k': 'k',
        }
        with h5py.File(hdf, 'r') as stream:

            #
            for k, v in d.items():
                if k in stream == False:
                    continue
                m = findOverlappingUnits(self.ukeys, hdf)
                ds = np.array(stream[k])
                if len(ds.shape) == 1:
                    ds = ds.flatten()
                self.__setattr__(v, ds[m])

            #
            path = f'/rProbe/dg/preferred/raw/fr'
            if path in stream:
                ds = stream[path]
                self.t = ds.attrs['t']
                self.terms['rProbe']['extrasaccadic'] = np.array(ds)[m]

            #
            path = f'/rProbe/dg/preferred/standardized/fr'
            if path in stream:
                ds = stream[path]
                self.peths['extrasaccadic']= np.array(ds)[m]

        return
    
    def saveNamespace(
        self,
        hdf
        ):
        """
        """

        d = {
            'gmm/latencies': self.latencies,
            'gmm/modulation': self.modulation,
            'rSaccade/dg/preferred/nasal/fr': self.templates['nasal'],
            'rSaccade/dg/preferred/temporal/fr': self.templates['temporal'],
            'rProbe/dg/preferred/perisaccadic/fr': self.peths['perisaccadic'],
        }
        m = findOverlappingUnits(self.ukeys, hdf)
        with h5py.File(hdf, 'a') as stream:
            for k, v in d.items():
                if v is None:
                    continue
                nCols = 1 if len(v.shape) == 1 else v.shape[1]
                data = np.full([m.size, nCols], np.nan)
                data[m, :] = v.reshape(-1, nCols)
                if k in stream:
                    del stream[k]
                ds = stream.create_dataset(
                    k,
                    data.shape,
                    data.dtype,
                    data=data
                )
                ds.attrs['t'] = self.tSaccade

        return

    def computeTemplates(
        self,
        responseWindow=(-0.2, 0.5),
        binsize=0.01,
        pad=3,
        ):
        """
        """

        t, nTrials, nBins = psth2(
            np.zeros(1),
            np.zeros(1),
            window=(responseWindow[0] - pad, responseWindow[1] + pad),
            binsize=binsize,
            returnShape=True
        )
        nUnits = len(self.ukeys)
        self.templates = {
            'nasal': np.full([nUnits, nBins], np.nan),
            'temporal': np.full([nUnits, nBins], np.nan),
        }

        #
        for iUnit in range(nUnits):

            #
            end = None if iUnit + 1 == nUnits else '\r'
            print(f'Copmuting saccade response templates for unit {iUnit + 1} out of {nUnits}', end=end)
            
            # Set the unit (and session)
            self.ukey = self.ukeys[iUnit]

            # Compute saccade response templates
            for saccadeLabel, saccadeDirection in zip([-1, 1], ['temporal', 'nasal']):
                trialIndices = np.where(np.vstack([
                    np.logical_or(
                        self.session.saccadeLatencies < -0.1,
                        self.session.saccadeLatencies > 0.05,
                    ),
                    self.session.saccadeLabels == saccadeLabel,
                    self.session.gratingMotionDuringSaccades == self.ambc[iUnit, 1] # TODO: Check if this is a thing
                ]).all(0))[0]
                if trialIndices.size == 0:
                    continue
                tSaccade, M = psth2(
                    self.session.saccadeTimestamps[trialIndices, 0],
                    self.unit.timestamps,
                    window=(responseWindow[0] - pad, responseWindow[1] + pad),
                    binsize=binsize,
                )
                fr = M.mean(0) / binsize
                self.templates[saccadeDirection][iUnit] = fr
                if self.tSaccade is None:
                    self.tSaccade = tSaccade

        return

    def computeTerms(
        self,
        ukey=None,
        responseWindow=(-0.2, 0.5),
        baselineWindow=(-3.0, -2.0),
        perisaccadicWindow=(-0.05, 0.1),
        binsize=0.01,
        smoothingKernelWidth=0.01,
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

        # Determine the unit index
        nUnits = len(self.ukeys)
        for iUnit in range(nUnits):
            date, animal, cluster = self.ukeys[iUnit]
            if date == self.ukey[0] and animal == self.ukey[1]:
                break

        # Compute peri-saccadic response
        trialIndices = np.where(self.session.parseEvents(
            eventName='probe',
            coincident=True,
            eventDirection=self.ambc[iUnit, 1],
            coincidenceWindow=perisaccadicWindow,
        ))[0]
        if trialIndices.size < 3:
            return (
                np.full(nBins, np.nan),
                np.full(nBins, np.nan),
            )
        t_, rMixed = self.unit.kde(
            self.session.probeTimestamps[trialIndices],
            responseWindow=responseWindow,
            binsize=binsize,
            sigma=smoothingKernelWidth
        )

        # Compute latency-shifted saccade response
        saccadeLabels = self.session.load('stimuli/dg/probe/dos')
        trialIndices = np.where(
            self.session.parseEvents('probe', True, self.ambc[iUnit, 1], perisaccadicWindow)
        )[0]
        iterable = zip(
            self.session.probeLatencies[trialIndices],
            saccadeLabels[trialIndices]
        )
        rSaccade = list()
        rBaseline = list()
        for probeLatency, saccadeLabel in iterable:
            saccadeDirection = 'temporal' if saccadeLabel == -1 else 'nasal'
            fp = self.templates[saccadeDirection][iUnit]
            x = t + probeLatency
            fr = np.interp(x, self.tSaccade, fp, left=np.nan, right=np.nan)
            rSaccade.append(fr)
            bl = self.templates[saccadeDirection][iUnit][np.logical_and(self.tSaccade >= baselineWindow[0], self.tSaccade <= baselineWindow[1])].mean()
            rBaseline.append(bl)

        #
        rSaccade = np.nanmean(np.array(rSaccade) - np.array(rBaseline).reshape(-1, 1), 0)

        return rMixed, rSaccade

    def measureModulation(
        self,
        ukey=None,
        sortby='amplitude',
        maximumAmplitudeShift=30,
        peth=None
        ):
        """
        """

        #
        if ukey is not None:
            self.ukey = ukey

        # Determine the unit index
        nUnits = len(self.ukeys)
        for iUnit in range(nUnits):
            date, animal, cluster = self.ukeys[iUnit]
            if date == self.ukey[0] and animal == self.ukey[1] and cluster == self.ukey[2]:
                break

        #
        params = self.params[iUnit]
        if np.isnan(params).all():
            return
        abcd = params[np.invert(np.isnan(params))]
        abc, d = abcd[:-1], abcd[-1]
        A1, B, C = np.split(abc, 3)
        k = A1.size
        if sortby == 'amplitude':
            order = np.argsort(A1)[::-1]
        elif sortby == 'latency':
            order = np.argsort(B)

        #
        kmax = int(np.max(self.k))
        dr = np.full(kmax, np.nan)
        latencies = np.full(kmax, np.nan)
        latencies[:k] = B[order]
        if peth is None:
            peth = self.peths['perisaccadic'][iUnit]

        for i, iComp in enumerate(order):

            # Refit
            amplitudeBoundaries = np.vstack([A1 - 0.001, A1 + 0.001]).T
            amplitudeBoundaries[iComp, 0] -= maximumAmplitudeShift
            amplitudeBoundaries[iComp, 1] += maximumAmplitudeShift
            bounds = np.vstack([
                [[d - 0.001, d + 0.001]],
                amplitudeBoundaries,
                np.vstack([B - 0.001, B + 0.001]).T,
                np.vstack([C - 0.001, C + 0.001]).T,
            ]).T
            p0 = np.concatenate([
                np.array([d]),
                A1,
                B,
                C
            ])
            gmm = GaussianMixturesModel(k)
            gmm.fit(self.t, peth, p0, bounds)

            #
            A2 = np.split(gmm._popt[1:], 3)[0]
            dr[i] = A2[iComp] - A1[iComp]

        return np.array(dr), latencies, gmm._popt

    def computePeths(
        self,
        responseWindow=(-0.2, 0.5),
        baselineWindow=(-3.0, -2.0),
        perisaccadicWindow=(-0.05, 0.1),
        binsize=0.01,
        smoothingKernelWidth=0.01,
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
        nUnits = len(self.ukeys)
        self.terms['rMixed'] = np.full([nUnits, nBins], np.nan)
        self.terms['rSaccade'] = np.full([nUnits, nBins], np.nan)
        self.terms['rProbe']['perisaccadic'] = np.full([nUnits, nBins], np.nan)
        self.peths['perisaccadic'] = np.full([nUnits, nBins], np.nan)

        #
        for iUnit in range(nUnits):

            #
            end = None if iUnit + 1 == nUnits else '\r'
            print(f'Copmuting peri-saccadic PSTHs for unit {iUnit + 1} out of {nUnits}', end=end)

            #
            rMixed, rSaccade = self.computeTerms(
                self.ukeys[iUnit],
                responseWindow,
                baselineWindow,
                perisaccadicWindow,
                binsize,
                smoothingKernelWidth,
            )

            # Standardize the PETHs
            mu, sigma = self.ambc[iUnit, 2], self.ambc[iUnit, 3]
            yResidual = rMixed - rSaccade
            yStandard = (yResidual - mu) / sigma

            #
            self.terms['rMixed'][iUnit] = rMixed
            self.terms['rSaccade'][iUnit] = rSaccade
            self.terms['rProbe']['perisaccadic'][iUnit] = yResidual
            self.peths['perisaccadic'][iUnit] = yStandard

        return

    def fitPeths(
        self,
        sortby='amplitude',
        maximumAmplitudeShift=30,
        ):
        """
        """

        #
        nUnits = len(self.ukeys)
        kmax = int(np.max(self.k))        
        self.modulation = np.full([nUnits, kmax], np.nan)
        self.latencies = np.full([nUnits, kmax], np.nan)

        #
        for iUnit in range(nUnits):

            end = None if iUnit + 1 == nUnits else '\r'
            print(f'Measuring modulation for unit {iUnit + 1} out of {nUnits}', end=end)

            #
            dr, latencies, params = self.refitPerisaccadicPeth(
                self.ukeys[iUnit],
                sortby=sortby,
                maximumAmplitudeShift=maximumAmplitudeShift
            )

            #
            self.modulation[iUnit, :] = dr
            self.latencies[iUnit, :] = latencies

        return

    def plotResponseTerms(
        self,
        figsize=(6, 2),
        ukey=('2023-07-21', 'mlati10', 262),
        ):
        """
        """

        #
        unitIndex = self.lookupUnitKey(ukey)
        rMixed = self.terms['rMixed'][unitIndex]
        rSaccade = self.terms['rSaccade'][unitIndex]
        rProbe = self.terms['rProbe']['extrasaccadic'][unitIndex]

        #
        fig, axs = plt.subplots(ncols=4, sharey=True)
        axs[0].plot(self.t, rMixed, color='k')
        # for y in rSaccade:
        #     axs[1].plot(self.t, y, color='0.8', alpha=0.5)
        axs[1].plot(self.t, rSaccade, color='k')
        axs[2].plot(self.t, rMixed - rSaccade, color='k')
        axs[3].plot(self.t, rProbe, color='k')
        axs[0].set_ylabel('FR (spikes/sec)')
        axs[0].set_xlabel('Time (sec)')
        axs[0].set_title(r'$R_{Probe, Saccade}$', fontsize=10)
        axs[1].set_title(r'$R_{Saccade}$', fontsize=10)
        axs[2].set_title(r'$R_{Probe (Peri)}$', fontsize=10)
        axs[3].set_title(r'$R_{Probe (Extra)}$', fontsize=10)
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, axs

    def plotExampleCurves(
        self,
        figsize=(6, 2),
        ):
        """
        """

        fig, axs = plt.subplots(ncols=3)
        for j, ukey in enumerate(self.examples):
            unitIndex = self.lookupUnitKey(ukey)
            if unitIndex is None:
                continue
            rMixed = self.terms['rMixed'][unitIndex]
            rSaccade = self.terms['rSaccade'][unitIndex]
            rProbe = self.terms['rProbe']['extrasaccadic'][unitIndex]
            axs[j].plot(
                self.t,
                rProbe,
                color='k'
            )
            axs[j].plot(
                self.t,
                rMixed - rSaccade,
                color='k',
                linestyle=':'
            )
            axs[j].plot(
                self.t,
                rMixed - rSaccade,
                color='k',
                alpha=0.3
            )
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, axs

    def plotModulationDistributionsWithHistogram(
        self,
        nBins=30,
        figsize=(3, 7),
        xlimits=(-10, 10),
        ):
        """
        """

        fig, axs = plt.subplots(nrows=len(np.unique(self.labels)), sharex=True)
        titles = (
            'Negative',
            'k=1',
            'k=2',
            'k=3',
            'k=4',
            'k=5'
        )
        for i, l in enumerate(np.unique(self.labels)):
            if l == -1:
                coef = -1
            else:
                coef = + 1
            axs[i].hist(
                coef * self.modulation[np.ravel(self.labels == l), 0],
                range=xlimits,
                bins=nBins,
                facecolor='0.7',
            )
            axs[i].set_title(titles[i], fontsize=10)
            ylim = axs[i].get_ylim()
            axs[i].vlines(0, *ylim, color='k')
            axs[i].set_ylim(ylim)
        axs[-1].set_xlabel('MI')
        axs[0].set_ylabel('# of units')
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, axs

    def plotModulationDistributionsWithScatterplot(
        self,
        figsize=(3, 4),
        scale=0.07,
        **kwargs_
        ):
        """
        """

        kwargs = {
            'marker': '.',
            's': 12,
            'alpha': 0.3,
            'color': '0.6'
        }
        kwargs.update(kwargs_)

        fig, ax = plt.subplots()
        for i, l in enumerate(np.unique(self.k)):
            m = np.ravel(self.k) == l
            jitter = np.random.normal(loc=0, scale=scale, size=m.sum())
            ax.scatter(
                self.modulation[m, 0],
                np.full(m.sum(), l) + jitter,
                **kwargs
            )
        ylim = ax.get_ylim()
        ax.vlines(0, *ylim, color='k')
        ax.set_ylim(ylim)
        xmax = np.max(np.abs(ax.get_xlim()))
        ax.set_xlim([-xmax, xmax])
        ax.set_ylabel('# of components (k)')
        ax.set_xlabel(r'$\Delta R$')
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, ax
