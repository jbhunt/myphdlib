import h5py
import numpy as np
import pathlib as pl
from scipy.signal import find_peaks as findPeaks
from matplotlib import pyplot as plt
from myphdlib.general.toolkit import psth2
from myphdlib.figures.analysis import AnalysisBase, GaussianMixturesModel
from myphdlib.figures.clustering import GaussianMixturesFittingAnalysis

class DirectionSectivityAnalysis(
    GaussianMixturesFittingAnalysis,
    ):
    """
    """

    def __init__(self, **kwargs):
        """
        """

        super().__init__(**kwargs)

        self.dsi = {
            'bar': None,
            'probe': None,
            'saccade': None,
        }
        self.pd = {
            'bar': None,
            'probe': None,
            'saccade': None,
        }
        self.peths = {
            'standard': None,
            'normal': None,
            'raw': None,
            'preferred': None,
            'null': None,
        }
        self.tProbe = None
        self.tSaccade = None
        self.features = {
            'a': None,
            'd': None,
            'm': None,
            's': None,
        }
        self.model = {
            'params1': None, # Preferred
            'params2': None, # Null
        }
        self.templates = {
            'nasal': None,
            'temporal': None,
        }
        self.mi = {
            'real': None,
        }
        self.p = {
            'real': None,
        }

        return

    def loadNamespace(
        self,
        ):
        """
        """

        datasets = {
            'clustering/features/d': (self.features, 'd'),
            'clustering/peths/standard': (self.peths, 'preferred'),
            'clustering/model/params': (self.model, 'params1'),
            'preference/peths/null': (self.peths, 'null'),
            'preference/peths/normal': (self.peths, 'normal'),
            'preference/peths/standard': (self.peths, 'standard'),
            'preference/model/params2': (self.model, 'params2'),
            'preference/dsi/bar': (self.dsi, 'bar'),
            'preference/dsi/probe': (self.dsi, 'probe'),
            'preference/dsi/saccade': (self.dsi, 'saccade'),
            'preference/pd/bar': (self.pd, 'bar'),
            'preference/pd/probe': (self.pd, 'probe'),
            'preference/pd/saccade': (self.pd, 'saccade'),
            'modulation/templates/nasal': (self.templates, 'nasal'),
            'modulation/templates/temporal': (self.templates, 'temporal'),
            'modulation/mi': (self.mi, 'real'),
            'bootstrap/p': (self.p, 'real'),
        }
        with h5py.File(self.hdf, 'r') as stream:
            for path, (attr, key) in datasets.items():
                parts = path.split('/')
                if path in stream:
                    ds = stream[path]
                    if path == 'clustering/peths/standard':
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
            'preference/peths/null': (self.peths['null'], True),
            'preference/peths/normal': (self.peths['normal'], True),
            'preference/peths/standard': (self.peths['standard'], True),
            'preference/model/params2': (self.model['params2'], True),
            'preference/dsi/bar': (self.dsi['bar'], True),
            'preference/dsi/probe': (self.dsi['probe'], True),
            'preference/dsi/saccade': (self.dsi['saccade'], True),
            'preference/pd/bar': (self.pd['bar'], True),
            'preference/pd/probe': (self.pd['probe'], True),
            'preference/pd/saccade': (self.pd['saccade'], True),
        }

        #
        mask = self._intersectUnitKeys(self.ukeys)

        #
        with h5py.File(self.hdf, 'a') as stream:
            for k, (v, f) in datasets.items():
                if v is None:
                    continue
                nCols = 1 if len(v.shape) == 1 else v.shape[1]
                data = np.full([mask.size, nCols], np.nan)
                data[mask, :] = v.reshape(-1, nCols)
                if k in stream:
                    del stream[k]
                ds = stream.create_dataset(
                    k,
                    data.shape,
                    data.dtype,
                    data=data
                )

                # Save the bin centers for all PETH datasets
                if 'peths' in pl.Path(k).parts:
                    ds.attrs['t'] = self.tProbe

        return

    def computeNullPeths(
        self,
        **kwargs
        ):
        """
        """

        super().computeExtrasaccadicPeths(
            preferred=False,
            **kwargs
        )
        self.peths['null'] = np.copy(self.peths['standard'])

        return

    def fitNullPeths(self, **kwargs):
        """
        """

        super().fitExtrasaccadicPeths(
            key='params2',
            **kwargs
        )

        return

    def measureDirectionSelectivityForProbes(
        self,
        responseWindow=(0, 0.5),
        ):
        """
        """

        binIndices = np.where(np.logical_and(
            self.tProbe >= responseWindow[0],
            self.tProbe <= responseWindow[1]
        ))[0]
        self.dsi['probe'] = np.full(len(self.ukeys), np.nan)
        self.pd['probe'] = np.full(len(self.ukeys), np.nan)
        for iUnit in range(len(self.ukeys)):

            #
            a1 = np.abs(self.peths['preferred'][iUnit, binIndices]).max()
            a2 = np.abs(self.peths['null'][iUnit, binIndices]).max()
            pd = self.features['d'][iUnit]

            #
            vectors = np.full([2, 2], np.nan)
            vectors[:, 0] = np.array([a1, a2]).T
            vectors[:, 1] = np.array([
                np.pi if pd == -1 else 0,
                0 if pd == -1 else np.pi
            ]).T

            # Compute the coordinates of the polar plot vertices
            vertices = np.vstack([
                vectors[:, 0] * np.cos(vectors[:, 1]),
                vectors[:, 0] * np.sin(vectors[:, 1])
            ]).T

            # Compute direction selectivity index
            a, b = vertices.sum(0) / vectors[:, 0].sum()
            dsi = np.sqrt(np.power(a, 2) + np.power(b, 2))
            self.dsi['probe'][iUnit] = dsi
            self.pd['probe'][iUnit] = vectors[0, 1]

        return

    def measureDirectionSelectivityForSaccades(
        self,
        responseWindow=(-0.2, 0.5),
        ):
        """
        """

        binIndices = np.where(np.logical_and(
            self.tSaccade >= responseWindow[0],
            self.tSaccade <= responseWindow[1]
        ))[0]
        self.dsi['saccade'] = np.full(len(self.ukeys), np.nan)
        self.pd['saccade'] = np.full(len(self.ukeys), np.nan)
        for iUnit in range(len(self.ukeys)):

            #
            self.ukey = self.ukeys[iUnit]

            #
            a1 = np.max(np.abs(self.templates['nasal'][iUnit, binIndices]))
            a2 = np.max(np.abs(self.templates['temporal'][iUnit, binIndices]))
            pd = 'nasal' if a1 > a2 else 'temporal'

            #
            vectors = np.full([2, 2], np.nan)
            if self.session.eye == 'left':
                if pd == 'nasal':
                    vectors[:, 0] = np.array([a1, a2]).T
                    vectors[:, 1] = np.array([np.deg2rad(180), np.deg2rad(0)]).T
                    self.pd['saccade'][iUnit] = np.deg2rad(180)
                elif pd == 'temporal':
                    vectors[:, 0] = np.array([a2, a1]).T
                    vectors[:, 1] = np.array([np.deg2rad(0), np.deg2rad(180)]).T
                    self.pd['saccade'][iUnit] = np.deg2rad(0)

            # Compute the coordinates of the polar plot vertices
            vertices = np.vstack([
                vectors[:, 0] * np.cos(vectors[:, 1]),
                vectors[:, 0] * np.sin(vectors[:, 1])
            ]).T

            # Compute direction selectivity index
            a, b = vertices.sum(0) / vectors[:, 0].sum()
            dsi = np.sqrt(np.power(a, 2) + np.power(b, 2))
            self.dsi['saccade'][iUnit] = dsi
            
        return

    def measureDirectionSelectivityForMovingBars(
        self,
        ):
        """
        Compute DSI for the moving bars stimulus
        """

        self.dsi['bar'] = np.full(len(self.ukeys), np.nan)
        self.pd['bar'] = np.full(len(self.ukeys), np.nan)
        date = None
        for session in self.sessions:

            #
            self._session = session

            # 
            movingBarOrientations = self.session.load('stimuli/mb/orientation')
            barOnsetTimestamps = self.session.load('stimuli/mb/onset/timestamps')
            barOffsetTimestamps = self.session.load('stimuli/mb/offset/timestamps')
            movingBarTimestamps = np.hstack([
                barOnsetTimestamps.reshape(-1, 1),
                barOffsetTimestamps.reshape(-1, 1)
            ])
            uniqueOrientations = np.unique(movingBarOrientations)
            uniqueOrientations.sort()

            #
            for ukey in self.ukeys:

                #
                if ukey[0] != str(session.date):
                    continue
                self.ukey = ukey

                #
                vectors = np.full([uniqueOrientations.size, 2], np.nan)
                for rowIndex, orientation in enumerate(uniqueOrientations):

                    #
                    trialIndices = np.where(movingBarOrientations == orientation)[0]
                    amplitudes = list()
                    for trialIndex in trialIndices:
                        t1, t2 = movingBarTimestamps[trialIndex, :]
                        dt = t2 - t1
                        t, M = psth2(
                            np.array([t1]),
                            self.unit.timestamps,
                            window=(0, dt),
                            binsize=None
                        )
                        fr = M.item() / dt
                        amplitudes.append(fr)

                    #
                    vectors[rowIndex, 0] = np.mean(amplitudes)
                    vectors[rowIndex, 1] = np.deg2rad(orientation)

                # Compute the coordinates of the polar plot vertices
                vertices = np.vstack([
                    vectors[:, 0] * np.cos(vectors[:, 1]),
                    vectors[:, 0] * np.sin(vectors[:, 1])
                ]).T

                # Compute direction selectivity index
                a, b = vertices.sum(0) / vectors[:, 0].sum()
                self.dsi['bar'][self.iUnit] = np.sqrt(np.power(a, 2) + np.power(b, 2))
                self.pd['bar'][self.iUnit] = np.arctan2(b, a) % (2 * np.pi)


        return

    def run(
        self,
        ):
        """
        """

        self.loadNamespace()
        self.computeNullPeths()
        self.fitNullPeths()
        self.measureDirectionSelectivityForMovingBars()
        self.measureDirectionSelectivityForProbes()
        self.measureDirectionSelectivityForSaccades()
        self.saveNamespace()

        return

    def plotModulationByDirectionSelectivity(
        self,
        threshold=0.3,
        nBins=50,
        figsize=(4, 5),
        ):
        """
        """

        #
        fig, grid = plt.subplots(nrows=3, sharex=True)
        mi = self.mi['real'][:, 5, 0] / self.model['params1'][:, 0]
        modulated = self.p['real'][:, 0] < 0.05
        for i, ev in enumerate(['bar', 'probe', 'saccade']):
            samples = (
                mi[np.vstack([self.filter, modulated, self.dsi[ev] >= threshold]).all(0)],
                mi[np.vstack([self.filter, modulated, self.dsi[ev] <  threshold]).all(0)],
            )
            counts, edges, patches = grid[i].hist(
                samples,
                range=(-3, 3),
                bins=nBins,
                histtype='barstacked'
            )
            for patch in patches[0]:
                patch.set_facecolor('k')
                patch.set_edgecolor('k')
            for patch in patches[1]:
                patch.set_facecolor('w')
                patch.set_edgecolor('k')

        #
        titles = (
            'Moving bars',
            'Probe',
            'Saccade'
        )
        for i, ax in enumerate(grid):
            ax.set_ylabel('N units')
            ax.set_title(titles[i], fontsize=10)
        grid[-1].set_xlabel('Modualtion index (MI)')
        grid[0].legend([r'$DSI\geq 0.3$', r'$DSI<0.3$'])
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, grid

    def scatterDirectionSelectivityIndices(
        self,
        events=('probe', 'saccade'),
        figsize=(4, 4),
        ):
        """
        """

        ev1, ev2 = events
        fig, ax = plt.subplots()
        data = {
            events[0]: list(),
            events[1]: list()
        }
        for iUnit in range(len(self.ukeys)):
            if self.filter[iUnit] == False:
                continue
            for ev in events:
                xRad = self.pd[ev][iUnit]
                xSigned = -1 if xRad > 1.5 else +1
                xNorm = xSigned * self.dsi[ev][iUnit]
                data[ev].append(xNorm)

        ax.scatter(
            *data.values(),
            marker='.',
            s=7,
            color='k',
            alpha=0.5,
            clip_on=False
        )
        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_aspect('equal')
        ax.set_xlabel(r'$DSI_{Probe}\cdot PD_{Probe}$')
        ax.set_ylabel(r'$DSI_{Saccade}\cdot PD_{Saccade}$')
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, ax