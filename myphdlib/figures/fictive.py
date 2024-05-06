import h5py
import numpy as np
from scipy.signal import find_peaks as findPeaks
from scipy.stats import pearsonr
from matplotlib import pyplot as plt
from myphdlib.figures.analysis import AnalysisBase, GaussianMixturesModel, g
from myphdlib.figures.modulation import BasicSaccadicModulationAnalysis
from myphdlib.figures.bootstrap import BoostrappedSaccadicModulationAnalysis
from myphdlib.figures.clustering import GaussianMixturesFittingAnalysis
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

class FictiveSaccadesAnalysis(
    GaussianMixturesFittingAnalysis,
    BoostrappedSaccadicModulationAnalysis,
    BasicSaccadicModulationAnalysis,
    ):
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
            'peri': None,
            'resampled': None,
            'normal': None,
            'standard': None,
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
            'fits': None,
            'peaks': None,
            'labels': None,
            'params1': None, # Extra-saccadic, fictive
            'params2': None, # Peri-saccadic, fictive
            'params3': None, # Extra-saccadic, real
        }
        self.templates = {
            'nasal': None,
            'temporal': None
        }
        self.tProbe = None
        self.tSaccade = None
        self.windows = None
        self.mi ={
            'real': None,
            'fictive': None,
        }
        self.p = {
            'real': None,
            'fictive': None,
        }
        self.samples = {
            'real': None,
            'fictive': None,
        }
        self.filter = None

        #
        self.windows = np.array([
            [0, 0.1]
        ])

        #
        self.examples = (

        )

        return

    def loadNamespace(
        self,
        ):
        """
        """

        #
        datasets = {
            'clustering/filter': ('filter', None),
            'clustering/model/params': (self.model, 'params3'),
            'clustering/features/s': (self.features, 's'),
            'modulation/mi': (self.mi, 'real'),
            'bootstrap/p': (self.p, 'real'),
            'fictive/peths/extra': (self.peths, 'extra'),
            'fictive/peths/extra': (self.peths, 'standard'),
            'fictive/peths/peri': (self.peths, 'peri'),
            'fictive/peths/resampled': (self.peths, 'resampled'),
            'fictive/peths/normal': (self.peths, 'normal'),
            'fictive/terms/rpe': (self.terms, 'rpe'),
            'fictive/terms/rps': (self.terms, 'rps'),
            'fictive/terms/rs': (self.terms, 'rs'),
            'fictive/templates/nasal': (self.templates, 'nasal'),
            'fictive/templates/temporal': (self.templates, 'temporal'),
            'fictive/model/params1': (self.model, 'params1'),
            'fictive/model/params2': (self.model, 'params2'),
            'fictive/model/k': (self.model, 'k'),
            'fictive/model/fits': (self.model, 'fits'),
            'fictive/model/peaks': (self.model, 'peaks'),
            'fictive/features/a': (self.features, 'a'),
            'fictive/features/m': (self.features, 'm'),
            'fictive/features/d': (self.features, 'd'),
            'fictive/samples': (self.samples, 'fictive'),
            'fictive/p': (self.p, 'fictive'),
            'fictive/mi': (self.mi, 'fictive'),
        }
        with h5py.File(self.hdf, 'r') as stream:
            for path, (attr, key) in datasets.items():
                parts = path.split('/')
                if path in stream:
                    ds = stream[path]
                    if path == 'fictive/peths/extra':
                        self.tProbe = ds.attrs['t']
                    if path == 'fictive/templates/nasal':
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
        nUnitsPerChunk=100,
        ):
        """
        """

        datasets = {
            'fictive/peths/extra': (self.peths['extra'], True),
            'fictive/peths/peri': (self.peths['peri'], True),
            'fictive/peths/resampled': (self.peths['resampled'], True),
            'fictive/peths/normal': (self.peths['normal'], True),
            'fictive/terms/rpe': (self.terms['rpe'], True),
            'fictive/terms/rps': (self.terms['rps'], True),
            'fictive/terms/rs': (self.terms['rs'], True),
            'fictive/templates/nasal': (self.templates['nasal'], True),
            'fictive/templates/temporal': (self.templates['temporal'], True),
            'fictive/model/params1': (self.model['params1'], True),
            'fictive/model/params2': (self.model['params2'], True),
            'fictive/model/k': (self.model['k'], True),
            'fictive/model/fits': (self.model['fits'], True),
            'fictive/model/peaks': (self.model['peaks'], True),
            'fictive/features/a': (self.features['a'], True),
            'fictive/features/m': (self.features['m'], True),
            'fictive/features/d': (self.features['d'], True),
            'fictive/p': (self.p['fictive'], True),
            'fictive/mi': (self.mi['fictive'], True)
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

        #
        if self.peths['resampled'] is not None:
            self._saveLargeDataset(
                self.hdf,
                path='fictive/peths/resampled',
                dataset=self.peths['resampled'],
                nUnitsPerChunk=nUnitsPerChunk,
            )

        #
        if self.samples['fictive'] is not None:
            self._saveLargeDataset(
                self.hdf,
                path='fictive/samples',
                dataset=self.samples['fictive'],
                nUnitsPerChunk=nUnitsPerChunk
            )

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
        if self.ukey is None:
            gratingMotionMask = np.full(gratingMotionDuringProbes.size, True)
        else:
            gratingMotionMask = gratingMotionDuringProbes == self.features['d'][self.iUnit]
        trialIndices = np.where(np.vstack([
            probeLatencies >= perisaccadicWindow[0],
            probeLatencies <= perisaccadicWindow[1],
            gratingMotionMask
        ]).all(0))[0]

        return trialIndices, probeTimestamps, probeLatencies, saccadeLabels, gratingMotionDuringProbes

    def computeExtrasaccadicPeths(
        self,
        responseWindow=(-0.2, 0.5),
        perisaccadicWindow=(-0.5, 0.5),
        baselineWindow=(-0.7, -0.5),
        binsize=0.01,
        smoothingKernelWidth=0.01,
        ):
        """
        """

        self.tProbe, nTrials, nBins = psth2(
            np.zeros(1),
            np.zeros(1),
            window=responseWindow,
            binsize=binsize,
            returnShape=True
        )

        #
        nUnits = len(self.ukeys)
        for key in ('a', 'm', 'd'):
            self.features[key] = np.full(nUnits, np.nan)
        self.peths['extra'] = np.full([nUnits, nBins], np.nan)
        self.peths['normal'] = np.full([nUnits, nBins], np.nan)

        for session in self.sessions:

            #
            if session.hasDataset('stimuli/fs') == False:
                continue

            #
            self._session = session # NOTE: This is ugly
            trialIndices_, probeTimestamps, probeLatencies, saccadeLabels, gratingMotionDuringProbes = self._loadEventDataForProbes()

            #
            for ukey in self.ukeys:

                # Look for units in the target session
                if ukey[0] == str(session.date) and ukey[1] == session.animal:
                    self.ukey = ukey
                else:
                    continue

                #
                end = None if self.iUnit + 1 == nUnits else '\r'
                print(f'Copmuting extra-saccadic PSTHs for unit {self.iUnit + 1} out of {nUnits}', end=end)

                # Initialize feature set
                y = np.full(nBins, np.nan)
                a = None # Amplitude
                d = None # Probe direction
                m = None # Mean FR
                s = self.features['s'][self.iUnit] # Standard deviation (from DG protocol)

                #
                for gratingMotion in (-1, 1):

                    # Select just the extra-saccadic trials
                    trialIndices = np.where(np.vstack([
                        gratingMotionDuringProbes == gratingMotion,
                        np.logical_or(
                            probeLatencies > perisaccadicWindow[1],
                            probeLatencies < perisaccadicWindow[0]
                        )
                    ]).all(0))[0]
                    if trialIndices.size == 0:
                        continue

                    # 
                    try:

                        #Compute firing rate
                        t, y_ = self.unit.kde(
                            probeTimestamps[trialIndices],
                            responseWindow=responseWindow,
                            binsize=binsize,
                            sigma=smoothingKernelWidth,
                        )

                        # Estimate baseline firing rate
                        t, bl1 = self.unit.kde(
                            probeTimestamps[trialIndices],
                            responseWindow=baselineWindow,
                            binsize=binsize,
                            sigma=smoothingKernelWidth
                        )
                        m_ = bl1.mean()

                        # Estimate standard deviation of firing rate
                        # t, bl2 = self.unit.kde(
                        #     probeTimestamps[trialIndices],
                        #     responseWindow=standardizationWindow,
                        #     binsize=binsize,
                        #     sigma=smoothingKernelWidth
                        # )
                        # s_ = bl2.std()

                    #
                    except:
                        continue

                    # Compute new features
                    a_ = np.abs(y_[self.tProbe > 0] - m_).max()
                    d_ = gratingMotion

                    # Override current feature set if amplitude is greater
                    if a is None or a_ > a:
                        y = y_
                        a = a_
                        d = d_
                        m = m_

                #
                if a is None:
                    continue

                #
                self.features['a'][self.iUnit] = a
                self.features['d'][self.iUnit] = d
                self.features['m'][self.iUnit] = m

                # Store the raw PSTH
                if np.isnan(s) == False:
                    self.peths['extra'][self.iUnit] = (y - m) / s

                # Normalize
                self.peths['normal'][self.iUnit] = (y - m) / a

        # Copy the PETHs (for fitting)
        self.peths['standard'] = self.peths['extra']

        return

    def fitExtrasaccadicPeths(
        self,
        kmax=5,
        key='params1',
        **kwargs_
        ):
        """
        """
        super().fitExtrasaccadicPeths(kmax, key, **kwargs_)
        return

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

        super()._computeSaccadeResponseTemplates(
            responseWindow=responseWindow,
            perisaccadicWindow=perisaccadicWindow,
            binsize=binsize,
            pad=pad,
            gaussianKernelWidth=gaussianKernelWidth,    
        )

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

        super().computePerisaccadicPeths(
            trange=trange,
            tstep=tstep,
            responseWindow=responseWindow,
            baselineWindow=baselineWindow,
            binsize=binsize,
            zeroBaseline=zeroBaseline
        )

        return

    def fitPerisaccadicPeths(
        self,
        maximumAmplitudeShift=200,
        key='fictive'
        ):
        """
        """
        super().fitPerisaccadicPeths(maximumAmplitudeShift, key)
        return

    def downsampleExtrasaccadicPeths(
        self,
        minimumTrialCount=10,
        rate=0.05,
        **kwargs_
        ):
        """
        """
        kwargs = {
            'minimumTrialCount': minimumTrialCount,
            'rate': rate,
        }
        kwargs.update(kwargs_)
        self.iw = 0
        super().downsampleExtrasaccadicPeths(**kwargs)
        return

    def generateNullSamples(
        self,
        nRuns=None,
        useFilter=True,
        parallelize=True,
        key='fictive',
        ):
        """
        """
        super().generateNullSamples(nRuns, useFilter, parallelize, key)
        return

    def computeProbabilityValues(
        self,
        key='fictive'
        ):
        """
        """
        super().computeProbabilityValues(key)
        return

    def run(
        self,
        ):
        """
        """

        # Get extra-saccadic responses
        self.computeExtrasaccadicPeths()
        self.fitExtrasaccadicPeths()

        # Get peri-saccadic responses and measure modulation
        self._computeSaccadeResponseTemplates()
        self.computePerisaccadicPeths()
        self.fitPerisaccadicPeths()

        # Compute p-values
        self.downsampleExtrasaccadicPeths()
        self.generateNullSamples()
        self.computeProbabilityValues()

        return

    # TODO: Refactor this method
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

    def plotExamples(
        self,
        windowIndex=5,
        ):
        """
        """

        with h5py.File(self.hdf, 'r') as stream:
            pethsRealExtra = np.array(stream['clustering/peths/standard'])
            pethsRealPeri = np.array(stream['modulation/peths/peri'])

        fig, grid = plt.subplots(nrows=len(self.examples), ncols=2, sharey=True)
        if len(self.examples) == 1:
            grid = np.atleast_2d(grid)

        for i, ukey in enumerate(self.examples):
            iUnit = self._indexUnitKey(ukey)
            grid[i, 0].plot(pethsRealExtra[iUnit])
            grid[i, 0].plot(pethsRealPeri[iUnit, :, windowIndex])
            grid[i, 1].plot(self.peths['standard'][iUnit])
            grid[i, 1].plot(self.peths['peri'][iUnit, :, windowIndex])

        return fig, grid

    def plotModulationBySaccadeType(
        self,
        bounds=(-2, 2),
        windowIndex=5,
        figsize=(3, 3),
        ):
        """
        """

        fig, ax = plt.subplots()

        mask = np.vstack([
            self.filter,
            np.logical_or(
                self.p['real'][:, 0] < 0.05,
                self.p['fictive'][:, 0] < 0.05
            ),
        ]).all(0)
        x = np.clip(self.mi['real'][mask, windowIndex, 0] / self.model['params3'][mask, 0], *bounds)
        y = np.clip(self.mi['fictive'][mask, windowIndex, 0] / self.model['params1'][mask, 0], *bounds)
        c = np.full(x.size, 'k')
        
        ax.scatter(
            x,
            y,
            marker='.',
            s=10,
            c=c,
            alpha=0.7,
            clip_on=False,
        )
        ax.vlines(0, *bounds, color='k', alpha=0.5)
        ax.hlines(0, *bounds, color='k', alpha=0.5)
        ax.set_ylim(bounds)
        ax.set_xlim(bounds)
        ax.set_aspect('equal')
        ax.set_xlabel('Modulation Index (Real)')
        ax.set_ylabel('Modulation Index (Fictive)')
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, ax
