import h5py
import numpy as np
from scipy.signal import find_peaks as findPeaks
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
from myphdlib.figures.analysis import AnalysisBase, findOverlappingUnits, GaussianMixturesModel, g
from myphdlib.figures.modulation import BasicSaccadicModulationAnalysis
from myphdlib.figures.bootstrap import BoostrappedSaccadicModulationAnalysis
from myphdlib.general.toolkit import psth2

def convertGratingMotionToSaccadeDirection(
    gratingMotion=-1,
    referenceEye='left'
    ):
    """
    """

    saccadeDirection = None
    if referenceEye == 'left':
        if gratingMotion == -1:
            saccadeDirection = 'nasal'
        else:
            saccadeDirection = 'temporal'
    elif referenceEye == 'right':
        if gratingMotion == -1:
            saccadeDirection = 'temporal'
        else:
            saccadeDirection = 'nasal'

    return saccadeDirection

class FictiveSaccadesAnalysis(BoostrappedSaccadicModulationAnalysis, BasicSaccadicModulationAnalysis):
    """
    """

    def __init__(self):
        """
        """

        self.templates = {
            'nasal': None,
            'temporal': None
        }
        self.tSaccade = None
        self.t = None
        self.ambc = None
        self.peths = {
            'extra': None,
            'peri': None
        }
        self.terms = {
            'rMixed': None,
            'rProbe': {
                'peri': None,
                'extra': None,
            },
            'rSaccade': None
        }
        self.pethsRaw = {
            'extra': None,
            'peri': None,
        }
        self.params = None
        self.paramsRefit = None
        self.latencies = None
        self.modulation = None

        #
        self.paramsActual = None
        self.modualtionActual= None

        #
        self.pethsResampled = None
        self.pvalues = None
        self.samples = None
        self.msign = None

        #
        self.windows = np.array([
            [0, 0.1]
        ])

        super().__init__()

        return

    def saveNamespace(
        self,
        hdf,
        nUnitsPerChunk=100,
        ):
        """
        """

        #
        m = findOverlappingUnits(self.ukeys, hdf)

        #
        datasets = {

            #
            'temps/fs/nasal/fr': self.templates['nasal'],
            'temps/fs/temporal/fr': self.templates['temporal'],

            #
            'peths/fs/extra/fr': self.peths['extra'],
            'peths/fs/peri/fr': self.peths['peri'],

            #
            'terms/fs/extra/rProbe': self.terms['rProbe']['extra'],
            'terms/fs/peri/rProbe': self.terms['rProbe']['peri'],
            'terms/fs/peri/rSaccade': self.terms['rSaccade'],
            'terms/fs/peri/rMixed': self.terms['rMixed'],
            
            #
            'gmm/fs/extra/k': self.k.reshape(-1, 1),
            'gmm/fs/extra/params': self.params,
            'gmm/fs/peri/params': self.paramsRefit,
            'gmm/fs/extra/latencies': self.latencies,
            'gmm/fs/extra/modulation': self.modulation,

            #
            'peths/fs/extra/params': self.ambc,

            #
            'bootstrap/fs/p': self.pvalues,

        }

        with h5py.File(hdf, 'a') as stream:
            for path, dense in datasets.items():
                if dense is None:
                    continue
                if path in stream:
                    del stream[path]
                sparse = np.full([m.size, *dense.shape[1:]], np.nan)
                sparse[m] = dense
                ds = stream.create_dataset(
                    path,
                    shape=sparse.shape,
                    dtype=sparse.dtype,
                    data=sparse
                )
                if 'temps' in path.split('/'):
                    ds.attrs['t'] = self.tSaccade

        #
        self._saveLargeDataset(
            hdf,
            path='bootstrap/fs/peths',
            dataset=self.pethsResampled,
            nUnitsPerChunk=nUnitsPerChunk,
        )

        #
        self._saveLargeDataset(
            hdf,
            path='bootstrap/fs/samples',
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

        # Load saccade response templates
        paths = (
            'temps/fs/nasal/fr',
            'temps/fs/temporal/fr'
        )
        keys = (
            'nasal',
            'temporal'
        )
        self.templates = dict()

        #
        with h5py.File(hdf, 'r') as stream:
            for path, key in zip(paths, keys):
                if path in stream:
                    ds = stream[path]
                    if 't' in ds.attrs.keys() and self.tSaccade is None:
                        self.tSaccade = ds.attrs['t']
                    self.templates[key] = np.array(ds)[m]

        # Load extra and peri-saccadic PETHs
        self.peths = {
            'peri': None,
            'extra': None,
        }
        paths = (
            'peths/fs/peri/fr',
            'peths/fs/extra/fr',
        )
        with h5py.File(hdf, 'r') as stream:
            for tt, path in zip(self.peths.keys(), paths):
                if path in stream:
                    ds = stream[path]
                    if 't' in ds.attrs.keys() and self.t is None:
                        self.t = ds.attrs['t']
                    self.peths[tt] = np.array(ds[m])

        #
        self.terms = {
            'rProbe': {
                'peri': None,
                'extra': None,
            },
            'rMixed': None,
            'rSaccade': None
        }
        paths = (
            'terms/fs/peri/rProbe',
            'terms/fs/peri/rMixed',
            'terms/fs/peri/rSaccade',
            'terms/fs/extra/rProbe'
        )
        keySets = (
            ('rProbe', 'peri'),
            ('rMixed',),
            ('rSaccade',),
            ('rProbe', 'extra'),
        )
        with h5py.File(hdf, 'r') as stream:
            for path, keySet in zip(paths, keySets):
                if path in stream:
                    data = np.array(stream[path][m])
                    if len(keySet) == 1:
                        k1 = keySet[0]
                        self.terms[k1] = data
                    else:
                        k1, k2 = keySet
                        self.terms[k1][k2] = data

        #
        datasets = {
            'gmm/fs/extra/params': 'params',
            'gmm/dg/extra/params': 'paramsActual',
            'gmm/fs/extra/modulation': 'modulation',
            'gmm/dg/extra/modulation': 'modulationActual',
            'bootstrap/p': 'pvaluesActual',
            'peths/fs/extra/params': 'ambc',
            'gmm/dg/extra/k': 'k',
            'bootstrap/fs/samples': 'samples',
            'bootstrap/fs/p': 'pvalues',
            'bootstrap/fs/peths': 'pethsResampled',
        }
        with h5py.File(hdf, 'r') as stream:
            for path, name in datasets.items():
                if path in stream:
                    data = np.array(stream[path][m])
                    self.__setattr__(name, data)

        #
        with h5py.File(hdf, 'r') as stream:
            path = 'gmm/fs/extra/k'
            if path in stream:
                self.k = np.array(stream[path][m]).flatten()

        # TODO: Define "t" attribute

        return

    def _loadEventDataForSaccades(self):
        """
        """

        probeTimestamps = self.session.load('stimuli/fs/probe/timestamps')
        gratingMotion = self.session.load('stimuli/fs/saccade/motion')
        saccadeTimestamps = self.session.load('stimuli/fs/saccade/timestamps')

        #
        saccadeLatencies = np.full(saccadeTimestamps.size, np.nan)
        for iTrial in range(saccadeTimestamps.size):
            iProbe = np.argmin(np.abs(saccadeTimestamps[iTrial] - probeTimestamps))
            saccadeLatencies[iTrial] = saccadeTimestamps[iTrial] - probeTimestamps[iProbe]

        saccadeDirection = np.array([
            convertGratingMotionToSaccadeDirection(gm, self.session.eye)
                for gm in gratingMotion
        ])
        saccadeLabels = np.array([-1 if sd == 'temporal' else 1 for sd in saccadeDirection])
            
        return saccadeTimestamps, saccadeLatencies, saccadeLabels, gratingMotion

    def _loadEventDataForProbes(self, perisaccadicWindow=(-0.2, 0.2)):
        """
        """

        #
        probeTimestamps = self.session.load('stimuli/fs/probe/timestamps')
        gratingMotionDuringProbes = self.session.load('stimuli/fs/probe/motion')
        saccadeTimestamps = self.session.load('stimuli/fs/saccade/timestamps')
        gratingMotionDuringSaccades = self.session.load('stimuli/fs/saccade/motion')

        #
        probeLatencies = np.full(probeTimestamps.size, np.nan)
        saccadeLabels = np.full(probeTimestamps.size, np.nan)
        for iTrial in range(probeTimestamps.size):
            iSaccade = np.argmin(np.abs(probeTimestamps[iTrial] - saccadeTimestamps))
            probeLatencies[iTrial] = probeTimestamps[iTrial] - saccadeTimestamps[iSaccade]
            saccadeLabels[iTrial] = gratingMotionDuringSaccades[iSaccade] * -1

        #
        trialIndices = np.where(np.vstack([
            probeLatencies >= perisaccadicWindow[0],
            probeLatencies <= perisaccadicWindow[1],
            gratingMotionDuringProbes == self.ambc[self.iUnit, 1]
        ]).all(0))[0]

        return trialIndices, probeTimestamps, probeLatencies, saccadeLabels, gratingMotionDuringProbes

    def computeSaccadeResponseTemplates(
        self,
        responseWindow=(-0.2, 0.5),
        perisaccadicWindow=(-0.2, 0.2),
        binsize=0.01,
        pad=3,
        gaussianKernelWidth=0.01,
        ):
        """
        """

        super().computeSaccadeResponseTemplates(
            responseWindow=responseWindow,
            perisaccadicWindow=perisaccadicWindow,
            binsize=binsize,
            pad=pad,
            gaussianKernelWidth=gaussianKernelWidth,    
        )

        return

    def computePerisaccadicPeths(
        self,
        binsize=None,
        trange=(-0.5, 0.5),
        perisaccadicWindow=(-0.2, 0.2),
        baselineWindow=(-0.2, 0),
        zeroBaseline=True
        ):
        """
        """

        super().computePerisaccadicPeths(
            binsize,
            trange,
            perisaccadicWindow,
            baselineWindow,
            zeroBaseline
        )

        return

    def computeExtrasaccadicPeths(
        self,
        responseWindow=(-0.2, 0.5),
        perisaccadicWindow=(-0.5, 0.5),
        baselineWindow=(-0.7, -0.5),
        binsize=0.01,
        gaussianKernelWidth=0.01,
        ):
        """
        """

        self.t, nTrials, nBins = psth2(
            np.zeros(1),
            np.zeros(1),
            window=responseWindow,
            binsize=binsize,
            returnShape=True
        )

        #
        nUnits = len(self.ukeys)
        self.ambc = np.full([nUnits, 4], np.nan)
        for k in self.peths.keys():
            self.peths[k] = np.full([nUnits, nBins], np.nan)
            self.pethsRaw[k] = np.full([nUnits, nBins], np.nan)

        for session in self.sessions:

            #
            if session.hasDataset('stimuli/fs') == False:
                continue

            #
            # probeTimestamps = session.load('stimuli/fs/probe/timestamps')
            # gratingMotion = session.load('stimuli/fs/probe/motion')
            # saccadeTimestamps = session.load('stimuli/fs/saccade/timestamps')

            #
            # probeLatencies = np.full(probeTimestamps.size, np.nan)
            # for iTrial in range(probeTimestamps.size):
            #     iSaccade = np.argmin(np.abs(probeTimestamps[iTrial] - saccadeTimestamps))
            #    probeLatencies[iTrial] = probeTimestamps[iTrial] - saccadeTimestamps[iSaccade]

            #
            self._session = session # NOTE: This is ugly
            trialIndices_, probeTimestamps, probeLatencies, saccadeLabels, gratingMotion = self._loadEventDataForProbes()

            #
            for ukey in self.ukeys:
                if ukey[0] == str(session.date) and ukey[1] == session.animal:
                    self.ukey = ukey
                else:
                    continue

                #
                # if self.iUnit == 500:
                #     import pdb; pdb.set_trace()

                #
                end = None if self.iUnit + 1 == nUnits else '\r'
                print(f'Copmuting extra-saccadic PSTHs for unit {self.iUnit + 1} out of {nUnits}', end=end)

                # Initialize parameters
                x = np.full(nBins, np.nan)
                a = np.nan # Amplitude of preferred direction
                m = np.nan # Probe direction
                b = np.nan # Baseline
                c = np.nan # Scaling factor

                #
                for gm in np.unique(gratingMotion):
                    trialIndices = np.where(np.vstack([
                        np.logical_or(
                            probeLatencies > perisaccadicWindow[1],
                            probeLatencies < perisaccadicWindow[0],
                        ),
                        gratingMotion == gm
                    ]).all(0))[0]
                    if trialIndices.size == 0:
                        continue
                    try:
                        t, fr = self.unit.kde(
                            probeTimestamps[trialIndices],
                            responseWindow=responseWindow,
                            binsize=binsize,
                            sigma=gaussianKernelWidth
                        )
                    except:
                        continue

                    #
                    t, M = psth2(
                        probeTimestamps[trialIndices],
                        self.unit.timestamps,
                        window=baselineWindow,
                        binsize=None
                    )
                    bl = M.flatten() / np.diff(baselineWindow).item()
                    if np.isnan(a) or np.abs(fr - bl.mean()).max() > a:
                        x = fr
                        a = np.abs(fr - bl.mean()).max()
                        m = gm
                        b = bl.mean()
                        c = bl.std()

                #
                if np.isnan(c) or c == 0:
                    c = np.nan
                else:
                    self.peths['extra'][self.iUnit, :] = (x - b) / c
                self.ambc[self.iUnit] = np.array([a, m, b, c])
                self.pethsRaw['extra'][self.iUnit] = x

        return

    def fitExtrasaccadicPeths(
        self,
        kmax=5,
        **kwargs_
        ):
        """
        Algorithm
        ---------
        1. Find peaks using the normalized PSTH
        2. Discard all but the k largest peaks
        3. Use peak positions and amplitudes from the standardized PSTHs to initialize the GMM
        """

        kwargs = {
            'minimumPeakHeight': 0.15,
            'maximumPeakHeight': 1,
            'minimumPeakProminence': 0.05,
            'minimumPeakWidth': 0.001,
            'maximumPeakWidth': 0.02,
            'minimumPeakLatency': -0.2,
            'initialPeakWidth': 0.001,
            'maximumLatencyShift': 0.003,
            'maximumBaselineShift': 0.001,
            'maximumAmplitudeShift': 0.01 
        }
        kwargs.update(kwargs_)

        #
        nUnits, nBins = self.peths['extra'].shape
        self.k = np.full(nUnits, np.nan)
        self.rss = np.full(nUnits, np.nan)
        self.params = np.full([nUnits, int(3 * kmax + 1)], np.nan)
        self.fits = np.full([nUnits, nBins], np.nan)

        #
        for iUnit in range(nUnits):

            end = None if iUnit + 1 == nUnits else '\r'
            print(f'Fitting GMM for unit {iUnit + 1} out of {nUnits} units', end=end)

            #
            yRaw = self.pethsRaw['extra'][iUnit]
            bl = self.ambc[iUnit, 2]
            a = self.ambc[iUnit, 0]
            yNormed = (yRaw - bl) / a
            yStandard = self.peths['extra'][iUnit]

            #
            if np.isnan(yStandard).all():
                continue

            #
            peakIndices = list()
            peakProminences = list()
            for coef in (-1, 1):
                peakIndices_, peakProperties = findPeaks(
                    coef * yNormed,
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
                    peakProminences.append(peakProperties['prominences'][iPeak])

            # 
            peakIndices = np.array(peakIndices)
            if peakIndices.size == 0:
                continue
            peakProminences = np.array(peakProminences)
            peakAmplitudes = yStandard[peakIndices]
            peakLatencies = self.t[peakIndices]

            # Use only the k largest peaks
            if peakIndices.size > kmax:
                index = np.argsort(np.abs(peakAmplitudes))[::-1]
                peakIndices = peakIndices[index][:kmax]
                peakProminences = peakProminences[index][:kmax]
                peakAmplitudes = peakAmplitudes[index][:kmax]
                peakLatencies = peakLatencies[index][:kmax]
            
            #
            k = peakIndices.size
            self.k[iUnit] = k

            # Initialize the parameter space
            p0 = np.concatenate([
                np.array([0]),
                peakAmplitudes,
                peakLatencies,
                np.full(k, kwargs['initialPeakWidth'])
            ])
            bounds = np.vstack([
                np.array([[
                    -1 * kwargs['maximumBaselineShift'],
                    kwargs['maximumBaselineShift']
                ]]),
                np.vstack([
                    peakAmplitudes - kwargs['maximumAmplitudeShift'],
                    peakAmplitudes + kwargs['maximumAmplitudeShift']
                ]).T,
                np.vstack([
                    peakLatencies - kwargs['maximumLatencyShift'],
                    peakLatencies + kwargs['maximumLatencyShift']
                ]).T,
                np.repeat([[
                    kwargs['minimumPeakWidth'],
                    kwargs['maximumPeakWidth']
                ]], k, axis=0)
            ]).T

            # Fit the GMM and compute the residual sum of squares (rss)
            gmm = GaussianMixturesModel(k)
            gmm.fit(
                self.t,
                yStandard,
                p0=p0,
                bounds=bounds
            )
            yFit = gmm.predict(self.t)
            self.fits[iUnit, :] = yFit
            self.rss[iUnit] = np.sum(np.power(yFit - yStandard, 2)) / np.sum(np.power(yStandard, 2))

            # Extract the parameters of the fit GMM
            d, abc = gmm._popt[0], gmm._popt[1:]
            A, B, C = np.split(abc, 3)
            order = np.argsort(np.abs(A))[::-1] # Sort by amplitude
            params = np.concatenate([
                A[order],
                B[order],
                C[order],
            ])
            self.params[iUnit, :params.size] = params
            self.params[iUnit, -1] = d

        return

    def measureResponseCorrelation(
        self,
        ):
        """
        Measure the correlation between responses to the probe presented during
        the fictive saccades protocol and the drifting grating protocol
        """

        # Load datasets
        m = self.findOverlappingUnits(self.ukeys, self.hdf)
        with h5py.File(self.hdf, 'r') as stream:
            # probeMotion = np.array(stream['peths/dg/extra/params'][m, 1])
            pethsFromDriftingGrating = np.array(stream['peths/dg/extra/fr'][m, :])

        nUnits = len(self.ukeys)
        self.similarity = np.full(nUnits, np.nan)
        for i, ukey in enumerate(self.ukeys):
            self.ukey = ukey
            peth = self.peths['extra'][self.iUnit, :]
            r, p = pearsonr(
                peth,
                pethsFromDriftingGrating[self.iUnit, :]
            )
            self.similarity[i] = r

        return

    def downsampleExtrasaccadicPeths(
        self,
        perisaccadicWindow=(-0.5, 0.5),
        minimumTrialCount=10,
        rate=0.05,
        **kwargs
        ):
        """
        """
        kwargs.update({
            'perisaccadicWindow': perisaccadicWindow,
            'minimumTrialCount': minimumTrialCount,
            'rate': rate,
        })
        super().downsampleExtrasaccadicPeths(**kwargs)
        return

    def plotAnalysisDemo(
        self,
        examples=(
            ('2023-07-20', 'mlati9', 73),
            ('2023-07-11', 'mlati10', 448),
            ('2023-07-11', 'mlati10', 434)
        ),
        responseWindow=(-0.2, 0.5),
        perisaccadicWindow=(-0.5, 0.5),
        figsize=(7, 4),
        **kwargs_
        ):
        """
        """

        kwargs = {
            'marker': '.',
            's': 3,
            'alpha': 0.3
        }
        kwargs.update(kwargs_)

        fig, axs = plt.subplots(ncols=5, nrows=len(examples), sharex=True)

        #
        for i in range(len(examples)):
            self.ukey = examples[i]
            trialIndices_, probeTimestamps, probeLatencies, saccadeLabels, gratingMotionDuringProbes = self._loadEventDataForProbes(
                perisaccadicWindow=perisaccadicWindow,
            )
            t, M, spikeTimestamps = psth2(
                probeTimestamps,
                self.unit.timestamps,
                window=responseWindow,
                binsize=None,
                returnTimestamps=True
            )
            
            #
            if trialIndices_.size == 0:
                plt.close(fig)
                print(f"Warning: Could not determine unit's preferred direction of motion")
                return None, None

            trialIndices = {
                'peri': trialIndices_,
                'extra': np.where(np.vstack([
                    np.logical_or(
                        probeLatencies < perisaccadicWindow[0],
                        probeLatencies > perisaccadicWindow[1],
                    ),
                    gratingMotionDuringProbes == self.ambc[self.iUnit, 1]
                ]).all(0))[0]
            }
            xy = {
                'peri': list(),
                'extra': list()
            }
            y = 0
            for k in ['extra', 'peri']:
                for iTrial in trialIndices[k]:
                    for x in spikeTimestamps[iTrial]:
                        xy[k].append([x, y])
                    y += 1
            for k in xy.keys():
                xy[k] = np.array(xy[k])

            if len(xy['extra']) != 0:
                axs[i, 0].scatter(xy['extra'][:, 0], xy['extra'][:, 1], color='k', **kwargs)
            if len(xy['peri']) != 0:
                axs[i, 0].scatter(xy['peri'][:, 0], xy['peri'][:, 1], color='r', **kwargs)

            #
            saccadeTimestamps, saccadeLatencies, saccadeLabels, gratingMotionDuringSaccades = self._loadEventDataForSaccades()
            trialIndices = np.where(np.logical_and(
                np.logical_or(
                    saccadeLatencies < perisaccadicWindow[1] * -1,
                    saccadeLatencies > perisaccadicWindow[0] * -1
                ),
                gratingMotionDuringSaccades == self.ambc[self.iUnit, 1]
            ))[0]
            averageLatency = np.mean(saccadeLatencies[
                np.logical_and(
                    saccadeLatencies >= perisaccadicWindow[1] * -1,
                    saccadeLatencies <= perisaccadicWindow[0] * -1
                )
            ])
            t, M, spikeTimestamps = psth2(
                saccadeTimestamps[trialIndices],
                self.unit.timestamps,
                window=responseWindow,
                binsize=None,
                returnTimestamps=True
            )
            xy = list()
            for iTrial in range(len(trialIndices)):
                # l = saccadeLatencies[trialIndices][iTrial]
                for x in spikeTimestamps[iTrial]:
                    xy.append([x + averageLatency, iTrial + y])
            xy = np.array(xy)
            if len(xy) != 0:
                axs[i, 0].scatter(xy[:, 0], xy[:, 1], color='b', **kwargs)

            #
            mu, sigma = self.ambc[self.iUnit, 2], self.ambc[self.iUnit, 3]
            axs[i, 1].plot(
                self.t,
                (self.terms['rMixed'][self.iUnit, :, 0] - mu) / sigma,
                color='k'
            )
            axs[i, 2].plot(
                self.t,
                self.terms['rSaccade'][self.iUnit, :, 0] / sigma,
                color='k'
            )
            #
            for k, j in zip(['peri', 'extra'], [3, 4]):
                if k == 'peri':
                    params = self.paramsRefit[self.iUnit, :]
                    yRaw = self.peths[k][self.iUnit, :, 0]
                else:
                    params = self.params[self.iUnit, :]
                    yRaw = self.peths[k][self.iUnit, :]
                abcd = params[np.invert(np.isnan(params))]
                abc, d = abcd[:-1], abcd[-1]
                A, B, C = np.split(abc, 3)
                yFit = g(self.t, A[0], B[0], C[0], d)
                axs[i, j].plot(
                    self.t,
                    yRaw,
                    color='0.5'
                ) 
                axs[i, j].plot(
                    self.t,
                    yFit,
                    color='k'
                )

            #
            ylim = [np.inf, -np.inf]
            for ax in axs[i, 1:].flatten():
                y1, y2 = ax.get_ylim()
                if y1 < ylim[0]:
                    ylim[0] = y1
                if y2 > ylim[1]:
                    ylim[1] = y2
            if ylim[1] < 5:
                ylim[1] = 5
            if abs(ylim[0]) < 5:
                ylim[0] = -5
            for ax in axs[i, 1:].flatten():
                ax.set_ylim(ylim)
                ax.set_yticks([-5, 0, 5])
            for ax in axs[i, 2:].flatten():
                ax.set_yticklabels([])

        #
        labels = (
            f'Raster',
            r'$R_{P}, R_{S}$',
            r'$R_{S}$',
            r'$R_{P (Peri)}$',
            r'$R_{P (Extra)}$'
        )
        for j, l in enumerate(labels):
            axs[0, j].set_title(l, fontsize=10)
        axs[-1, 0].set_xlabel('Time from probe (s)')
        
        #
        for ax in axs[:, 0].flatten():
            ax.set_ylabel('Trial #')
        for ax in axs[:, 1].flatten():
            ax.set_ylabel('FR (z-scored)')
        for ax in axs.flatten():
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
        for ax in axs[:, 0].flatten():
            for sp in ('bottom', 'left'):
                ax.spines[sp].set_visible(False)
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, axs

    def plotModulationBySaccadeType(
        self,
        minimumResponseAmplitude=1,
        alphaLevel=0.05,
        bounds=(-1.7, 1.7),
        colors=('tab:red', 'tab:purple', 'tab:blue'),
        windowIndex=5,
        figsize=(3, 3),
        ):
        """
        """

        fig, ax = plt.subplots()

        m = np.vstack([
            self.params[:, 0] >= minimumResponseAmplitude,
            self.paramsActual[:, 0] >= minimumResponseAmplitude,
            self.pvaluesActual[:, windowIndex, 0] < alphaLevel,
            # np.logical_or(
            #     self.pvaluesActual[:, windowIndex, 0] < alphaLevel,
            #     self.pvalues[:, 0, 0] < alphaLevel
            # )
        ]).all(0)
        x = np.clip(self.modulation[m, 0, 0] / self.params[m, 0], *bounds)
        y = np.clip(self.modulationActual[m, 0, windowIndex] / self.paramsActual[m, 0], *bounds)
        c = np.full(x.size, 'k')
        
        ax.scatter(
            x,
            y,
            marker='.',
            s=10,
            c=c,
            alpha=0.7
        )
        ax.vlines(0, *bounds, color='k', alpha=0.5)
        ax.hlines(0, *bounds, color='k', alpha=0.5)
        ax.set_ylim(bounds)
        ax.set_xlim(bounds)
        ax.set_xlabel('Modulation (Fictive)')
        ax.set_ylabel('Modulation (Actual)')
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, ax
