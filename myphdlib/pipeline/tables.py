import h5py
import numpy as np
import pathlib as pl
from types import FunctionType
from myphdlib.general.toolkit import psth2

class TableBase():
    """
    """

    def make(
        self,
        filename,
        sessions,
        mapping
        ):
        """
        """

        #
        if type(filename) != pl.Path:
            filename = pl.Path(filename)

        #
        with h5py.File(str(filename), 'w') as stream:
            for key, (value, kwargs) in mapping.items():
                dataset = list()
                attrs = {}
                for index, session in enumerate(sessions):
                    if index + 1 == len(sessions):
                        end = '\n'
                    else:
                        end = '\r'
                    print(f'Working on session {index + 1} out of {len(sessions)} (dataset={key})', end=end)
                    if session.probeTimestamps is None:
                        continue
                    if type(value) == FunctionType:
                        f = value
                        data, attrs_ = f(session, **kwargs)
                        for k in attrs_.keys():
                            if k not in attrs.keys():
                                attrs[k] = attrs_[k]
                    elif type(value) == str:
                        data = session.load(value)
                    if data is None:
                        import pdb; pdb.set_trace()
                    for x in data:
                        dataset.append(x)
                dataset = np.array(dataset)
                ds = stream.create_dataset(key, dataset.shape, dataset.dtype, data=dataset)
                del dataset
                if len(attrs.keys()) != 0:
                    for k in attrs.keys():
                        ds.attrs[k] = attrs[k]

        return

def _loadPeths(
    session,
    path='peths/rProbe/left',
    ):
    """
    """


    peths, metadata = session.load(path, returnMetadata=True)
    if 'rProbe' in path and len(peths.shape) == 3:
        pethsFlattened = peths[:, :, 0]
        del peths
        return pethsFlattened, metadata

    return peths, metadata

def _getZetaTestProbabilities(
    session,
    probeMotion=-1
    ):
    """
    """

    probeDirection = 'left' if probeMotion == -1 else 'right'

    if session.probeTimestamps is None:
        return np.full(session.population.count(), np.nan).reshape(-1, 1)

    pvalues = session.load(f'population/zeta/probe/{probeDirection}/p')
    return pvalues.reshape(-1, 1), {}

unitsTableMapping = {
    'date': (
        lambda session:
            (np.array([str(unit.session.date) for unit in session.population], dtype='S').reshape(-1, 1), {}),
        {}
    ),
    'animal': (
        lambda session:
            (np.array([unit.session.animal for unit in session.population], dtype='S').reshape(-1, 1), {}),
        {}
    ),
    'unitNumber': (
        lambda session:
            (np.array([unit.cluster for unit in session.population]).reshape(-1, 1), {}),
        {}
    ),
    'unitLabel': (
        lambda session:
            (np.array([unit.label for unit in session.population]).reshape(-1, 1), {}),
        {},
    ),
    'nSpikes': (
        lambda session:
            (np.array([unit.timestamps.size for unit in session.population]).reshape(-1, 1), {}),
        {},
    ),
    'directionSelectivityIndex':
        ('population/metrics/dsi', {}),
    'luminanceSelectivityIndex':
        ('population/metrics/lpi', {}),
    'refractoryPeriodViolationRate':
        ('population/metrics/rpvr', {}),
    'amplitudeCutoff':
        ('population/metrics/ac', {}),
    'presenceRatio':
        ('population/metrics/pr', {}),
    'kilosortLabel':
        ('population/metrics/ksl', {}),
    'spikeWaveforms':
        ('population/metrics/bsw', {}),
    'rMixed/dg/left/fr':
        (_loadPeths, {'path': 'peths/rMixed/dg/left/fr'}),
    'rMixed/dg/left/sd':
        (_loadPeths, {'path': 'peths/rMixed/dg/left/sd'}),
    'rMixed/dg/right/fr':
        (_loadPeths, {'path': 'peths/rMixed/dg/right/fr'}),
    'rMixed/dg/right/sd':
        (_loadPeths, {'path': 'peths/rMixed/dg/right/sd'}),
    'rProbe/dg/left/fr':
        (_loadPeths, {'path': 'peths/rProbe/dg/left/fr'}),
    'rProbe/dg/left/sd':
        (_loadPeths, {'path': 'peths/rProbe/dg/left/sd'}),
    'rProbe/dg/right/fr':
        (_loadPeths, {'path': 'peths/rProbe/dg/right/fr'}),
    'rProbe/dg/right/sd':
        (_loadPeths, {'path': 'peths/rProbe/dg/right/sd'}),
    'rSaccade/dg/left/fr':
       (_loadPeths, {'path': 'peths/rSaccade/dg/left/fr'}),
    'rSaccade/dg/left/sd':
       (_loadPeths, {'path': 'peths/rSaccade/dg/left/sd'}),
    'rSaccade/dg/right/fr':
        (_loadPeths, {'path': 'peths/rSaccade/dg/right/fr'}),
    'rSaccade/dg/right/sd':
        (_loadPeths, {'path': 'peths/rSaccade/dg/right/sd'}),
    'rSaccade/dg/nasal/fr':
        (_loadPeths, {'path': 'peths/rSaccade/dg/nasal/fr'}),
    'rSaccade/dg/nasal/sd':
        (_loadPeths, {'path': 'peths/rSaccade/dg/nasal/sd'}),
    'rSaccade/dg/temporal/fr':
        (_loadPeths, {'path': 'peths/rSaccade/dg/temporal/fr'}),
    'rSaccade/dg/temporal/sd':
        (_loadPeths, {'path': 'peths/rSaccade/dg/temporal/sd'}),
    'rMixed/fs/left/fr':
        (_loadPeths, {'path': 'peths/rMixed/fs/left/fr'}),
    'rMixed/fs/left/sd':
        (_loadPeths, {'path': 'peths/rMixed/fs/left/sd'}),
    'rMixed/fs/right/fr':
        (_loadPeths, {'path': 'peths/rMixed/fs/right/fr'}),
    'rMixed/fs/right/sd':
        (_loadPeths, {'path': 'peths/rMixed/fs/right/sd'}),
    'rProbe/fs/left/fr':
        (_loadPeths, {'path': 'peths/rProbe/fs/left/fr'}),
    'rProbe/fs/left/sd':
        (_loadPeths, {'path': 'peths/rProbe/fs/left/sd'}),
    'rProbe/fs/right/fr':
        (_loadPeths, {'path': 'peths/rProbe/fs/right/fr'}),
    'rProbe/fs/right/sd':
        (_loadPeths, {'path': 'peths/rProbe/fs/right/sd'}),
    'rSaccade/fs/left/fr':
        (_loadPeths, {'path': 'peths/rSaccade/fs/left/fr'}),
    'rSaccade/fs/left/sd':
        (_loadPeths, {'path': 'peths/rSaccade/fs/left/sd'}),
    'rSaccade/fs/right/fr':
        (_loadPeths, {'path': 'peths/rSaccade/fs/right/fr'}),
    'rSaccade/fs/right/sd':
        (_loadPeths, {'path': 'peths/rSaccade/fs/right/sd'}),
    'rSaccade/fs/nasal/fr':
        (_loadPeths, {'path': 'peths/rSaccade/fs/nasal/fr'}),
    'rSaccade/fs/nasal/sd':
        (_loadPeths, {'path': 'peths/rSaccade/fs/nasal/sd'}),
    'rSaccade/fs/temporal/fr':
        (_loadPeths, {'path': 'peths/rSaccade/fs/temporal/fr'}),
    'rSaccade/fs/temporal/sd':
        (_loadPeths, {'path': 'peths/rSaccade/fs/temporal/sd'}),
    'xProbe':
        (_loadPeths, {'path': 'peths/rProbe/dg/preferred'}),
    'xSaccade':
       (_loadPeths, {'path': 'peths/rSaccade/dg/preferred'}),
    'pZeta/left':
        (_getZetaTestProbabilities, {'probeMotion': -1}),
    'pZeta/right':
        (_getZetaTestProbabilities, {'probeMotion': +1}),
}

class UnitsTable(TableBase):
    """
    """

    def make(
        self,
        filename,
        sessions,
        minimumFiringRate=None,
        minimumResponseAmplitude=None,
        filterUnits=False
        ):
        """
        """

        #
        if filterUnits:
            for index, session in enumerate(sessions):
                if index + 1 == len(sessions):
                    end = None
                else:
                    end = '\r'
                print(f'Filtering units for session {index + 1} out of {len(sessions)}', end=end)
                session.population.filter2(
                    minimumFiringRate=minimumFiringRate,
                    minimumResponseAmplitude=minimumResponseAmplitude,
                )

        #
        super().make(
            filename,
            sessions,
            unitsTableMapping
        )

        return

def _getSaccadeLabels(session):
    """
    """
    saccadeDirections = list()
    for saccadeLabel in session.saccadeLabels:
        if saccadeLabel == -1:
            saccadeDirections.append('temporal')
        elif saccadeLabel == +1:
            saccadeDirections.append('nasal')
        else:
            saccadeDirections.append('unclassified')
    return np.array(saccadeDirections, dtype='S').reshape(-1, 1)

def _getSaccadeAmplitudes(session):
    return

def _getSaccadeWaveforms(sessions):
    return

saccadesTableMapping = {
    'date': (
        lambda session: np.array([str(session.date) for i in range(session.saccadeTimestamps.shape[0])], dtype='S'),
        {}
    ),
    'animal': (
        lambda session: np.array([session.animal for i in range(session.saccadeTimestamps.shape[0])], dtype='S'),
        {}
    ),
    'treatment': (
        lambda session: np.array([session.treatment for i in range(session.saccadeTimestamps.shape[0])], dtype='S'),
        {}
    ),
    'saccadeLabels': (
        _getSaccadeLabels, {}
    ),
    'saccadeAmplitudes': (
        _getSaccadeAmplitudes, {}
    ),
    'saccadeWaveforms': (
        _getSaccadeWaveforms, {}
    )
}

class SaccadesTable(TableBase):
    """
    """

    def make(
        self,
        filename,
        sessions,
        ):
        """
        """

        super().make(
            filename,
            sessions,
            saccadesTableMapping
        )

        return
