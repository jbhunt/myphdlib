import h5py
import numpy as np
from scipy.signal import find_peaks as findPeaks
from matplotlib import pylab as plt
from myphdlib.general.toolkit import psth2
from myphdlib.figures.analysis import AnalysisBase, GaussianMixturesModel, g, findOverlappingUnits

class BasicSaccadicModulationAnalysis(AnalysisBase):
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
            'extra': None,
            'peri': None
        }
        self.terms = {
            'rpe': None,
            'rpp': None,
            'rps': None,
            'rs': None
        }
        self.features = {
            'm': None,
            's': None,
            'd': None
        }
        self.model = {
            'k': None,
            'params': None,
            'labels': None,
            'refits': None
        }
        self.templates = {
            'nasal': None,
            'temporal': None
        }
        self.tProbe = None
        self.tSaccade = None
        self.windows = None
        self.mi = None # Modulation index
        self.filter = None

        #
        self.examples = (
            ('2023-07-05', 'mlati9', 271),
            ('2023-05-12', 'mlati7', 163),
            ('2023-05-12', 'mlati7', 104)
        )

        return

    def loadNamespace(
        self,
        ):
        """
        """

        #
        d = {
            'clustering/peths/standard': (self.peths, 'extra'),
            'clustering/model/params': (self.model, 'params'),
            'clustering/model/labels': (self.model, 'labels'),
            'clustering/model/k': (self.model, 'k'),
            'clustering/description/d': (self.features, 'd'),
            'clustering/description/m': (self.features, 'm'),
            'clustering/description/s': (self.features, 's'),
            'clustering/filter': ('filter', None),
        }

        with h5py.File(self.hdf, 'r') as stream:
            for path, (attr, key) in d.items():
                if path in stream:
                    ds = stream[path]
                    if 't' in ds.attrs.keys() and self.t is None:
                        self.t = ds.attrs['t']
                    value = np.array(ds)
                    if len(value.shape) == 2 and value.shape[-1] == 1:
                        value = value.flatten()
                    if key is None:
                        setattr(self, attr, value)
                    else:
                        attr[key] = value

        return
    
    def saveNamespace(
        self,
        hdf
        ):
        """
        """

        d = {
            'modulation/mi': self.mi,
            'modulation/model/params': self.model['refits'],
            'modulation/templates/nasal': self.templates['nasal'],
            'modulation/templates/temporal': self.templates['temporal'],
            'modulation/peths/peri': self.peths['peri'],
        }
        m = findOverlappingUnits(self.ukeys, hdf)
        with h5py.File(hdf, 'a') as stream:
            for k, v in d.items():
                if v is None:
                    continue
                if np.isnan(v).sum() == v.size:
                    continue
                nd = len(v.shape)
                if nd == 1:
                    data = np.full([m.size, 1], np.nan)
                    data[m, :] = v.reshape(-1, 1)
                else:
                    data = np.full([m.size, *v.shape[1:]], np.nan)
                    data[m] = v
                if k in stream:
                    del stream[k]
                ds = stream.create_dataset(
                    k,
                    data.shape,
                    data.dtype,
                    data=data
                )
                if k == 'terms/dg/extra/rProbe':
                    ds.attrs['t'] = self.t
                if 'temp' in k:
                    ds.attrs['t'] = self.tSaccade
                if 'peri' in k.split('/'):
                    ds.attrs['windows'] = self.windows

        return

    def _loadEventDataForSaccades(
        self,
        ):
        """
        """

        return (
            self.session.saccadeTimestamps[:, 0],
            self.session.saccadeLatencies,
            self.session.saccadeLabels,
            self.session.gratingMotionDuringSaccades
        )
    
    def _loadEventDataForProbes(
        self,
        perisaccadicWindow=(-0.05, 0.1),
        ):
        """
        """

        trialIndices = np.where(self.session.parseEvents(
            eventName='probe',
            coincident=True,
            eventDirection=self.ambc[self.iUnit, 1],
            coincidenceWindow=perisaccadicWindow,
        ))[0]
        saccadeLabels = self.session.load('stimuli/dg/probe/dos')

        return trialIndices, self.session.probeTimestamps, self.session.probeLatencies, saccadeLabels, self.session.gratingMotionDuringProbes

    def _computeSaccadeResponseTemplates(
        self,
        responseWindow=(-0.2, 0.5),
        perisaccadicWindow=(-0.2, 0.2),
        binsize=0.01,
        pad=3,
        gaussianKernelWidth=0.01,
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
                    gratingMotion == self.ambc[self.iUnit, 1]
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

                self.templates[saccadeDirection][self.iUnit] = fr
                self.tSaccade = tSaccade

        return

    def _computeResponseTerms(
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

        # Compute peri-saccadic response
        trialIndices, probeTimestamps, probeLatencies, saccadeLabels, gratingMotion = self._loadEventDataForProbes(perisaccadicWindow=perisaccadicWindow)

        # NOTE: This might fail with not enough spikes in the peri-saccadic window
        try:
            t_, rMixed = self.unit.kde(
                probeTimestamps[trialIndices],
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
            probeLatencies[trialIndices],
            saccadeLabels[trialIndices]
        )
        rSaccade = list()
        rBaseline = list()
        for probeLatency, saccadeLabel in iterable:
            saccadeDirection = 'temporal' if saccadeLabel == -1 else 'nasal'
            fp = self.templates[saccadeDirection][self.iUnit]
            x = t + probeLatency
            fr = np.interp(x, self.tSaccade, fp, left=np.nan, right=np.nan)
            rSaccade.append(fr)
            bl = self.templates[saccadeDirection][self.iUnit][np.logical_and(self.tSaccade >= baselineWindow[0], self.tSaccade <= baselineWindow[1])].mean()
            rBaseline.append(bl)

        rSaccade = np.nanmean(np.array(rSaccade) - np.array(rBaseline).reshape(-1, 1), 0)

        return rMixed, rSaccade

    def _refitSinglePeth(
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

        #
        if peth is None:
            peth = self.peths['peri'][self.iUnit]

        #
        params = self.params[self.iUnit]
        if np.isnan(params).all():
            return None, None, None
        abcd = params[np.invert(np.isnan(params))]
        abc, d = abcd[:-1], abcd[-1]
        A1, B1, C1 = np.split(abc, 3)
        k = A1.size
        if sortby == 'amplitude':
            order = np.argsort(A1)[::-1]
        elif sortby == 'latency':
            order = np.argsort(B1)

        #
        nComponents = int(np.nanmax(self.k))
        dr = np.full(nComponents, np.nan)
        latencies = np.full(nComponents, np.nan)
        latencies[:k] = B1[order]

        #
        nParams = self.params.shape[1]
        paramsRefit = np.full([nComponents, nParams], np.nan)
        for i, iComp in enumerate(order):

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
            gmm = GaussianMixturesModel(k)
            gmm.fit(self.t, peth, p0, bounds)
            if i == 0:
                d, abc = gmm._popt[0], gmm._popt[1:]
                paramsRefit[iComp, :abc.size] = abc
                paramsRefit[iComp, -1] = d

            #
            A2 = np.split(gmm._popt[1:], 3)[0]
            dr[i] = A2[iComp] - A1[iComp]

        return np.array(dr), latencies, paramsRefit

    def computePerisaccadicPeths(
        self,
        binsize=0.1,
        trange=(-0.5, 0.5),
        perisaccadicWindow=(-0.05, 0.1),
        baselineWindow=(-0.2, 0),
        zeroBaseline=True
        ):
        """
        """

        #
        if binsize is None:
            self.windows = np.array([perisaccadicWindow,])
        else:
            leftEdges = np.arange(trange[0], trange[1], binsize)
            rightEdges = leftEdges + binsize
            perisaccadicWindows = np.vstack([leftEdges, rightEdges]).T
            self.windows = np.vstack([
                perisaccadicWindows,
                np.array(perisaccadicWindow)
            ])

        #
        nUnits = len(self.ukeys)
        nBins = self.t.size
        nWindows = self.windows.shape[0]
        self.terms['rps'] = np.full([nUnits, nBins, nWindows], np.nan)
        self.terms['rs'] = np.full([nUnits, nBins, nWindows], np.nan)
        self.terms['rpp'] = np.full([nUnits, nBins, nWindows], np.nan)
        self.peths['peri'] = np.full([nUnits, nBins, nWindows], np.nan)

        #
        for ukey in self.ukeys:

            #
            self.ukey = ukey

            #
            end = '\r' if self.iUnit + 1 != nUnits else None
            print(f'Computing PETHs for unit {self.iUnit + 1} out of {nUnits}', end=end)

            #
            for iWin, perisaccadicWindow in enumerate(self.windows):

                #
                rMixed, rSaccade = self.computeResponseTerms(
                    ukey=self.ukeys[self.iUnit],
                    perisaccadicWindow=perisaccadicWindow,
                )

                # Standardize the PETHs
                mu, sigma = self.features['m'][self.iUnit], self.features['s'][self.iUnit]
                yResidual = np.clip(rMixed - rSaccade, 0, np.inf)
                yStandard = (yResidual - mu) / sigma

                # Correct for baseline shift
                binIndices = np.where(np.logical_and(
                    self.t >= baselineWindow[0],
                    self.t < baselineWindow[1]
                ))
                yCorrected = yStandard - yStandard[binIndices].mean()

                #
                self.terms['rps'][self.iUnit, :, iWin] = rMixed
                self.terms['rs'][self.iUnit, :, iWin] = rSaccade
                if zeroBaseline:
                    self.peths['peri'][self.iUnit, :, iWin] = yCorrected
                else:
                    self.peths['peri'][self.iUnit, :, iWin] = yStandard

        return
    
    def fitPerisaccadicPeths(
        self,
        ):
        """
        """

        #
        nUnits, nBins, nWindows = self.peths['peri'].shape
        nParams = self.params.shape[1]
        nComponents = int((self.params.shape[1] - 1) / 3)
        self.modulation = np.full([nUnits, nComponents, nWindows], np.nan)
        self.latencies = np.full([nUnits, nComponents, nWindows], np.nan)
        self.paramsRefit = np.full([nUnits, nComponents, nWindows, nParams], np.nan)

        #
        for ukey in self.ukeys:

            #
            self.ukey = ukey

            #
            end = '\r' if self.iUnit + 1 != nUnits else None
            print(f'Re-fitting peri-saccadic PETHs for unit {self.iUnit + 1} out of {nUnits}', end=end)

            #
            for iWin in range(nWindows):
                peth = self.peths['peri'][self.iUnit, :, iWin]
                if np.isnan(peth).all():
                    continue
                dr, latencies, params = self._refitSinglePeth(
                    ukey=self.ukeys[self.iUnit],
                    sortby='amplitude',
                    peth=peth
                )
                if all([dr is None, latencies is None, params is None]):
                    continue
                self.modulation[self.iUnit, :, iWin] = dr
                self.latencies[self.iUnit, :, iWin] = latencies
                self.paramsRefit[self.iUnit, :, iWin, :] = params

        return

    def _plotLatencySortedRasterplot(
        self,
        ukey,
        ax=None,
        responseWindow=(-0.5, 0.5),
        perisaccadicWindow=(-0.05, 0.1),
        **kwargs_
        ):

        """
        """

        #
        kwargs = {
            'color': 'k',
            'marker': '.',
            'alpha': 0.5,
            's': 5
        }
        kwargs.update(kwargs_)

        #
        if ax is None:
            fig, ax = plt.subplots()

        #
        if self.ukey is not None:
            self.ukey = ukey

        #
        probeMotion = self.ambc[self.iUnit, 1]
        trialIndices = np.where(np.vstack([
            self.session.gratingMotionDuringProbes == probeMotion,
            self.session.probeLatencies > perisaccadicWindow[0],
            self.session.probeLatencies <= perisaccadicWindow[1]
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

        #
        ax.scatter(x, y, rasterized=True, **kwargs)

        #
        x, y = list(), list()
        for i, t in enumerate(self.session.probeLatencies[trialIndices][latencySortedIndex]):
            x.append(-1 * t)
            y.append(i)
        kwargs['color'] = 'tab:green'
        ax.scatter(
            x,
            y,
            rasterized=True,
            **kwargs
        )

        return

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
        mu, sigma = self.ambc[self.iUnit, 2], self.ambc[self.iUnit, 3] 

        #
        rMixed = self.terms['rMixed'][self.iUnit, :, iWindow]
        rSaccade = self.terms['rSaccade'][self.iUnit, :, iWindow]
        rProbeExtra = self.peths['extra'][self.iUnit]
        rProbePeri = self.peths['peri'][self.iUnit, :, iWindow]

        #
        params = self.params[self.iUnit, :]
        params = params[np.invert(np.isnan(params))]
        abc, d = params[:-1], params[-1]
        A, B, C = np.split(abc, 3)
        a, b, c = A[0], B[0], C[0]
        rProbeExtraFit = g(self.t, a, b, c, d)

        #
        params = self.paramsRefit[self.iUnit, 0, iWindow, :]
        params = params[np.invert(np.isnan(params))]
        abc, d = params[:-1], params[-1]
        A, B, C = np.split(abc, 3)
        a, b, c = A[0], B[0], C[0]
        rProbePeriFit = g(self.t, a, b, c, d)

        #
        if axs is None:
            fig, axs = plt.subplots(ncols=4, sharey=True)
        axs[0].plot(self.t, (rMixed - mu) / sigma, color='k')
        axs[1].plot(self.t, (rSaccade) / sigma, color='k')
        # axs[2].plot(self.t, rProbeExtra, color=colors[0], alpha=0.7, linestyle=':')
        axs[2].plot(self.t, rProbePeri, color=colors[1])

        return

    def plotAnalysisDemo(
        self,
        ukey=('2023-05-12', 'mlati7', 163),
        responseWindow=(-0.5, 0.5),
        windowIndices=(1, 2, 3, 4, 5, 6, 7, 8),
        figsize=(5, 5),
        useTrueRatios=False,
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
            windowIndices = np.arange(10)
        elif type(windowIndices) in (list, tuple):
            windowIndices = np.array(windowIndices)

        # Set the unit key
        self.ukey = ukey
        if self.iUnit is None:
            raise Exception('Could not locate unit')

        # Figure out how many trials are in each peri-saccadic time window
        if useTrueRatios:
            heightRatios = list()
            for t1, t2 in self.windows[windowIndices, :]:
                trialIndices = np.where(self.session.parseEvents(
                    eventName='probe',
                    coincident=True,
                    eventDirection=self.ambc[self.iUnit, 1],
                    coincidenceWindow=(t1, t2),
                ))[0]
                heightRatios.append(trialIndices.size)
            heightRatios = np.array(heightRatios)
        else:
            heightRatios = np.ones(windowIndices.size)

        # Create the subplots
        nWindows = len(windowIndices)
        fig, grid = plt.subplots(
            nrows=nWindows,
            ncols=4,
            gridspec_kw={'height_ratios': heightRatios}
        )

        # For each row, plot the raster and the trial-averages PSTHs for each term
        for i, axs, w in zip(windowIndices, grid, self.windows[windowIndices, :]):
            self._plotLatencySortedRasterplot(
                ukey=ukey,
                ax=axs[0],
                perisaccadicWindow=w,
                responseWindow=responseWindow,
                **kwargs
            )
            self._plotResponseTerms(
                ukey=ukey,
                axs=axs[1:],
                iWindow=i,
            )

        # Set the y-axis limits for the tiral-averaged responses
        ylim = [np.inf, -np.inf]
        for ax in grid[:, 1:].flatten():
            y1, y2 = ax.get_ylim()
            if y1 < ylim[0]:
                ylim[0] = y1
            if y2 > ylim[1]:
                ylim[1] = y2
        for ax in grid[:, 1:].flatten():
            ax.set_ylim(ylim)
            for sp in ('top', 'right', 'bottom', 'left'):
                ax.spines[sp].set_visible(False)

        #
        for ax in grid[:-1, 1:].flatten():
            ax.set_xticklabels([])
            ax.set_yticklabels([])
        for ax in grid[:-1, 0].flatten():
            ax.set_yticks(grid[-1, 0].get_yticks())
            ax.set_yticklabels([])
            ax.set_xticks([])

        #
        ylim = [np.inf, -np.inf]
        for ax in grid[:, 0].flatten():
            y1, y2 = ax.get_ylim()
            if y1 < ylim[0]:
                ylim[0] = y1
            if y2 > ylim[1]:
                ylim[1] = y2

        for ax in grid[:, 0].flatten():
            for sp in ('top', 'right', 'bottom', 'left'):
                ax.spines[sp].set_visible(False)
            ax.set_xlim(responseWindow)
            ax.set_ylim(ylim)
            ax.set_yticks([0, 25])

        #
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])

        return fig, grid

    def plotExamplePeths(
        self,
        figsize=(3, 3.5),
        ):
        """
        """

        #
        fig, grid = plt.subplots(nrows=len(self.examples), ncols=2, sharex=True)
        if len(self.examples) == 1:
            grid = np.atleast_2d(grid)
        for i in range(len(self.examples)):
            self.ukey = self.examples[i]
            rProbePeri = self.peths['peri'][self.iUnit, :, -1]
            rProbeExtra = self.peths['extra'][self.iUnit, :]
            grid[i, 0].plot(self.t, rProbePeri, 'k')
            grid[i, 0].plot(self.t, rProbeExtra, color='k', linestyle=':')
            grid[i, 0].fill_between(
                self.t,
                rProbeExtra,
                rProbePeri,
                where=rProbeExtra > rProbePeri,
                color='b',
                alpha=0.2,
            )
            grid[i, 0].fill_between(
                self.t,
                rProbeExtra,
                rProbePeri,
                where=rProbeExtra < rProbePeri,
                color='r',
                alpha=0.2,
            )
            # paramsExtra = self.params[self.iUnit, :]
            # paramsPeri = self.paramsRefit[self.iUnit, 0, -1, :]
            # for j, params in enumerate([paramsPeri, paramsExtra]):
            #    params = params[np.invert(np.isnan(params))]
            #     if len(params) == 0:
            #         continue
            #     abc, d = params[:-1], params[-1]
            #     A, B, C = np.split(abc, 3)
            #     a, b, c = A[0], B[0], C[0]
            #     yFit = g(self.t, a, b, c, d)
            #     grid[i, j].plot(self.t, yFit, color='k' )

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
        for ax in grid[:, 1].flatten():
            ax.set_yticklabels([])

        #
        for ax in grid.flatten():
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)

        #
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, grid
