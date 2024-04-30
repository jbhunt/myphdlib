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
            'labels': None,
            'params1': None,
            'params2': None
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
            ('2023-07-12', 'mlati9', 710),
            ('2023-07-20', 'mlati9', 337),
            ('2023-05-26', 'mlati7', 336)
        )

        return

    def loadNamespace(
        self,
        ):
        """
        """

        #
        datasets = {
            'clustering/peths/standard': (self.peths, 'extra'),
            'clustering/model/params': (self.model, 'params1'),
            'clustering/model/labels': (self.model, 'labels'),
            'clustering/model/k': (self.model, 'k'),
            'clustering/features/d': (self.features, 'd'),
            'clustering/features/m': (self.features, 'm'),
            'clustering/features/s': (self.features, 's'),
            'clustering/filter': ('filter', None),
            'modulation/mi': ('mi', None),
            'modulation/windows': ('windows', None),
            'modulation/model/params': (self.model, 'params2'),
            'modulation/templates/nasal': (self.templates, 'nasal'),
            'modulation/templates/temporal': (self.templates, 'temporal'),
            'modulation/peths/peri': (self.peths, 'peri'),
            'modulation/terms/rps': (self.terms, 'rps'),
            'modulation/terms/rs': (self.terms, 'rs')
        }

        with h5py.File(self.hdf, 'r') as stream:
            for path, (attr, key) in datasets.items():
                parts = path.split('/')
                if path in stream:
                    ds = stream[path]
                    if path == 'modulation/peths/peri':
                        self.tProbe = ds.attrs['t']
                    if path == 'modulation/templates/nasal':
                        self.tSaccade = ds.attrs['t']
                    value = np.array(ds)
                    if 'filter' in parts:
                        value = value.astype(bool)
                    if len(value.shape) == 2 and value.shape[-1] == 1:
                        value = value.flatten()
                    if key is None:
                        setattr(self, attr, value)
                    else:
                        attr[key] = value

        return
    
    def saveNamespace(
        self,
        ):
        """
        """

        datasets = {
            'modulation/mi': (self.mi, True),
            'modulation/windows': (self.windows, False),
            'modulation/model/params': (self.model['params2'], True),
            'modulation/templates/nasal': (self.templates['nasal'], True),
            'modulation/templates/temporal': (self.templates['temporal'], True),
            'modulation/peths/peri': (self.peths['peri'], True),
            'modulation/terms/rps': (self.terms['rps'], True),
            'modulation/terms/rs': (self.terms['rs'], True)
        }
        mask = self._intersectUnitKeys(self.ukeys)
        with h5py.File(self.hdf, 'a') as stream:

            #
            for k, (v, f) in datasets.items():

                #
                if v is None:
                    continue
                if np.isnan(v).sum() == v.size:
                    continue

                #
                if f:
                    nd = len(v.shape)
                    if nd == 1:
                        data = np.full([mask.size, 1], np.nan)
                        data[mask, :] = v.reshape(-1, 1)
                    else:
                        data = np.full([mask.size, *v.shape[1:]], np.nan)
                        data[mask] = v
                else:
                    data = v

                #
                if k in stream:
                    del stream[k]
                ds = stream.create_dataset(
                    k,
                    data.shape,
                    data.dtype,
                    data=data
                )

                #
                parts = k.split('/')
                if 'peths' in parts:
                    ds.attrs['t'] = self.tProbe
                if 'templates' in parts:
                    ds.attrs['t'] = self.tSaccade

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
            eventDirection=self.features['d'][self.iUnit],
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
                    gratingMotion == self.features['d'][self.iUnit]
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

    def _fitPerisaccadicPeth(
        self,
        ukey=None,
        maximumAmplitudeShift=100,
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
        params1 = self.model['params1'][self.iUnit]
        if np.isnan(params1).all():
            return None, None, None
        abcd = params1[np.invert(np.isnan(params1))]
        abc, d = abcd[:-1], abcd[-1]
        A1, B1, C1 = np.split(abc, 3)
        order = np.arange(A1.size) # NOTE: This is unnecessary, but I left it
        k = A1.size

        #
        nComponents = int(np.nanmax(self.model['k']))
        dr = np.full(nComponents, np.nan)

        #
        nParams = params1.size
        params2 = np.full([nComponents, nParams], np.nan)
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
            gmm.fit(self.tProbe, peth, p0, bounds)
            if i == 0:
                d, abc = gmm._popt[0], gmm._popt[1:]
                params2[iComp, :abc.size] = abc
                params2[iComp, -1] = d

            #
            A2 = np.split(gmm._popt[1:], 3)[0]
            dr[i] = A2[iComp] - A1[iComp]

        return dr, params2

    def _fitPerisaccadicPeth2(
        self,
        ukey=None,
        maximumAmplitudeShift=100,
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
        params1 = self.model['params1'][self.iUnit]
        if np.isnan(params1).all():
            return None, None, None
        abcd = params1[np.invert(np.isnan(params1))]
        abc, d = abcd[:-1], abcd[-1]
        A1, B1, C1 = np.split(abc, 3)

        #
        gmm = GaussianMixturesModel(k=A1.size)
        # TODO: Finish coding this method

        return

    def computePerisaccadicPeths(
        self,
        trange=(-0.5, 0.5),
        tstep=0.1,
        responseWindow=(-0.2, 0.5),
        baselineWindow=(-0.2, 0),
        binsize=0.01,
        zeroBaseline=True
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

        #
        nUnits = len(self.ukeys)
        nBins = self.tProbe.size
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
                rMixed, rSaccade = self._computeResponseTerms(
                    ukey=self.ukeys[self.iUnit],
                    responseWindow=responseWindow,
                    perisaccadicWindow=perisaccadicWindow,
                )

                # Standardize the PETHs
                mu, sigma = self.features['m'][self.iUnit], self.features['s'][self.iUnit]
                yResidual = np.clip(rMixed - rSaccade, 0, np.inf)
                yStandard = (yResidual - mu) / sigma

                # Correct for baseline shift
                binIndices = np.where(np.logical_and(
                    self.tProbe >= baselineWindow[0],
                    self.tProbe < baselineWindow[1]
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
        maximumAmplitudeShift=200,
        ):
        """
        """

        #
        nUnits, nBins, nWindows = self.peths['peri'].shape
        nParams = self.model['params1'].shape[1]
        nComponents = int((nParams - 1) / 3)
        self.mi = np.full([nUnits, nWindows, nComponents], np.nan)
        self.model['params2'] = np.full([nUnits, nParams, nWindows, nComponents], np.nan)

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
                dr, params2 = self._fitPerisaccadicPeth(
                    ukey=self.ukeys[self.iUnit],
                    peth=peth,
                    maximumAmplitudeShift=maximumAmplitudeShift
                )
                if all([dr is None, params2 is None]):
                    continue
                self.mi[self.iUnit, iWin, :] = dr
                self.model['params2'][self.iUnit, :, iWin, :] = params2.T

        return

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
            's': 5
        }
        kwargs.update(kwargs_)

        #
        if self.ukey is not None:
            self.ukey = ukey

        #
        nWindows = len(self.windows)
        nTrialsPerWindow = list()
        for window in self.windows:
            gratingMotion = self.features['d'][self.iUnit]
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
            gratingMotion = self.features['d'][self.iUnit]
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

            #
            axs[iWin].scatter(x, y, rasterized=True, color='k', alpha=alpha, **kwargs)

            # Indicate the time of saccade onset
            x, y = list(), list()
            for i, t in enumerate(self.session.probeLatencies[trialIndices][latencySortedIndex]):
                x.append(-1 * t)
                y.append(i)
            axs[iWin].scatter(
                x,
                y,
                rasterized=True,
                color='m',
                **kwargs
            )

            # TODO: Indicate the time of probe onset
            y = np.arange(trialIndices.size)
            x = np.full(y.size, 0)
            axs[iWin].scatter(x, y, rasterized=True, color='c', **kwargs)

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
        mu, sigma = self.features['m'][self.iUnit], self.features['s'][self.iUnit]

        #
        rMixed = self.terms['rps'][self.iUnit, :, iWindow]
        rSaccade = self.terms['rs'][self.iUnit, :, iWindow]
        rProbeExtra = self.peths['extra'][self.iUnit]
        rProbePeri = self.peths['peri'][self.iUnit, :, iWindow]

        #
        params = self.model['params1'][self.iUnit, :]
        params = params[np.invert(np.isnan(params))]
        abc, d = params[:-1], params[-1]
        A, B, C = np.split(abc, 3)
        a, b, c = A[0], B[0], C[0]
        rProbeExtraFit = g(self.tProbe, a, b, c, d)

        #
        params = self.model['params2'][self.iUnit, :, iWindow, 0]
        params = params[np.invert(np.isnan(params))]
        abc, d = params[:-1], params[-1]
        A, B, C = np.split(abc, 3)
        a, b, c = A[0], B[0], C[0]
        rProbePeriFit = g(self.tProbe, a, b, c, d)

        #
        if axs is None:
            fig, axs = plt.subplots(ncols=4, sharey=True)
        axs[0].plot(self.tProbe, (rMixed - mu) / sigma, color='k')
        axs[1].plot(self.tProbe, (rSaccade) / sigma, color='k')
        # axs[2].plot(self.t, rProbeExtra, color=colors[0], alpha=0.7 linestyle=':')
        axs[2].plot(self.tProbe, rProbePeri, color=colors[1])

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
        figsize=(3, 3.5),
        windowIndex=5,
        yticks=(20,),
        xticks=(0, 0.3)
        ):
        """
        """

        #
        fig, grid = plt.subplots(nrows=len(self.examples), ncols=2, sharex=True)
        if len(self.examples) == 1:
            grid = np.atleast_2d(grid)
        cmap = plt.get_cmap('rainbow', np.nanmax(self.model['k']))

        #
        for i in range(len(self.examples)):

            #
            self.ukey = self.examples[i]

            # Plot the raw PETHs
            rProbePeri = self.peths['peri'][self.iUnit, :, windowIndex]
            rProbeExtra = self.peths['extra'][self.iUnit, :]
            grid[i, 1].plot(self.tProbe, rProbePeri, 'k')
            grid[i, 0].plot(self.tProbe, rProbeExtra, color='k')

            # Plot the fit for the largest component of the response
            paramsExtra = self.model['params1'][self.iUnit, :]
            paramsPeri = self.model['params2'][self.iUnit, :, windowIndex, :]
            for j, params in enumerate([paramsExtra, paramsPeri]):
                params = params[np.invert(np.isnan(params))]
                if len(params) == 0:
                    continue
                abc, d = params[:-1], params[-1]
                A, B, C = np.split(abc, 3)
                a, b, c = A[0], B[0], C[0]
                t = np.linspace(-15 * C[0], 15 * C[0], 100) + B[0]
                yFit = g(t, a, b, c, d)
                grid[i, j].plot(t, yFit, color=cmap(0))

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

        #
        for ax in grid.flatten():
            for sp in ('top', 'right', 'bottom', 'left'):
                ax.spines[sp].set_visible(False)

        #
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
