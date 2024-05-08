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
            'modulation/templates/nasal': (self.templates, 'nasal'),
            'modulation/templates/temporal': (self.templates, 'temporal'),
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
        ):
        """
        """

        self.dsi['probe'] = np.full(len(self.ukeys), np.nan)
        self.pd['probe'] = np.full(len(self.ukeys), np.nan)
        for ukey in self.ukeys:

            #
            self.ukey = ukey

            #
            a1 = self.model['params1'][self.iUnit, 0]
            a2 = self.model['params2'][self.iUnit, 0]
            t1 = np.deg2rad(180 if self.features['d'][self.iUnit] == -1 else 0)
            t2 = np.deg2rad(0 if t1 == 180 else 180)
            vectors = np.array([
                [a1, t1],
                [a2, t2]
            ])

            # Compute the coordinates of the polar plot vertices
            vertices = np.vstack([
                vectors[:, 0] * np.cos(vectors[:, 1]),
                vectors[:, 0] * np.sin(vectors[:, 1])
            ]).T
            return vertices, vectors

            import pdb; pdb.set_trace()

            # Compute direction selectivity index
            a, b = vertices.sum(0) / vectors[:, 0].sum()
            dsi = np.sqrt(np.power(a, 2) + np.power(b, 2))
            self.dsi['probe'][self.iUnit] = dsi
            self.pd['probe'][self.iUnit] = t1

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
        for ukey in self.ukeys:

            #
            self.ukey = ukey

            #
            a1 = np.max(np.abs(self.templates['nasal'][self.iUnit, binIndices]))
            a2 = np.max(np.abs(self.templates['temporal'][self.iUnit, binIndices]))
            pd = 'nasal' if a1 > a2 else 'temporal'

            #
            vectors = np.full([2, 2], np.nan)
            if self.session.eye == 'left':
                if pd == 'nasal':
                    vectors[:, 0] = np.array([a1, a2]).T
                    vectors[:, 1] = np.array([np.deg2rad(180), np.deg2rad(0)]).T
                    self.pd['saccade'][self.iUnit] = np.deg2rad(180)
                elif pd == 'temporal':
                    vectors[:, 0] = np.array([a2, a1]).T
                    vectors[:, 1] = np.array([np.deg2rad(0), np.deg2rad(180)]).T
                    self.pd['saccade'][self.iUnit] = np.deg2rad(0)
            elif self.session.eye == 'right':
                raise Exception('Right eye sessions not implemented yet')

            # Compute the coordinates of the polar plot vertices
            vertices = np.vstack([
                vectors[:, 0] * np.cos(vectors[:, 1]),
                vectors[:, 0] * np.sin(vectors[:, 1])
            ]).T

            # Compute direction selectivity index
            a, b = vertices.sum(0) / vectors[:, 0].sum()
            dsi = np.sqrt(np.power(a, 2) + np.power(b, 2))
            self.dsi['saccade'][self.iUnit] = dsi
            
        return

    def measureDirectionSelectivityForMovingBars(
        self,
        ):
        """
        Extract DSI for the moving bars stimulus
        """

        self.dsi['bar'] = np.full(len(self.ukeys), np.nan)
        self.pd['bar'] = np.full(len(self.ukeys), np.nan)
        date = None
        for ukey in self.ukeys:
            self.ukey = ukey
            if date is None or ukey[0] != date:
                dsi = self.session.load('metrics/dsi')
                date = self.session.date
            iUnit = self._indexUnitKey(ukey)
            self.dsi['bar'][iUnit] = dsi[self.unit.index]

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

    def plotDirectionSelectivityIndices(
        self,
        ):
        """
        """

        fig, ax = plt.subplots()
        ax.scatter(
            self.dsi['probe'],
            self.dsi['saccade'],
            marker='.',
            s=10,
            color='k',
            alpha=0.7
        )

        return fig, ax