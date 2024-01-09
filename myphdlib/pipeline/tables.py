import h5py
import numpy as np
import pathlib as pl
from types import FunctionType
from myphdlib.general.toolkit import psth2

def _getVisualResponseSign(session):
    return np.array([unit.visualResponseSign for unit in session.population]).reshape(-1, 1), {}

def _getVisualResponseLatency(session):
    """
    """

    latencies = list()
    for unit in session.population:
        lPos, lNeg = unit.visualResponseLatency[unit.preferredProbeDirection]
        l = lNeg if unit.visualResponseSign == -1 else lPos
        latencies.append(l)

    return np.array(latencies).reshape(-1, 1), {}

def _getVisualResponseAmplitude(session):
    """
    """

    amplitudes = list()
    for unit in session.population:
        a = unit.visualResponseAmplitude[unit.preferredProbeDirection]
        amplitudes.append(a)

    return np.array(amplitudes).reshape(-1, 1), {}

def _getComponentResponse(
    session,
    component='probe',
    eventDirection=-1,
    shifted=False
    ):
    """
    """


    probeDirection = 'left' if eventDirection == -1 else 'right'
    saccadeDirection = 'temporal' if eventDirection == -1 else 'nasal'

    if component == 'probe':
        peths, metadata = session.load(f'curves/rProbe/{probeDirection}', returnMetadata=True)

    elif component == 'saccade':
        if shifted:
            peths, metadata = session.load(f'curves/rSaccade/{probeDirection}', returnMetadata=True)
        else:
            peths, metadata = session.load(f'curves/rSaccadeUnshifted/{saccadeDirection}', returnMetadata=True)

    elif component == 'mixed':
        peths, metadata = session.load(f'curves/rMixed/{probeDirection}', returnMetadata=True)

    return peths, metadata

# TODO: Code these functions
def _getSaccadicModulationIndex(session):
    return np.full(session.population.count(), np.nan).reshape(-1, 1), {}

def _getSaccadicModulationProbabilities(session):
    return np.full(session.population.count(), np.nan).reshape(-1, 1), {}

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

def _getSigmaValues(
    session,
    probeMotion=-1,
    ):
    """
    """
    probeDirection = 'left' if probeMotion == -1 else 'right'
    sigmas = session.load(f'population/metrics/sigma/{probeDirection}')

    return sigmas, {}

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
    'xPreferred':
        ('peths/probe/preferred', {}),
    'xNonpreferred':
        ('peths/probe/nonpreferred', {}),
    'rMixed/left':
        (_getComponentResponse, {'eventDirection': -1, 'component': 'mixed'}),
    'rMixed/right':
        (_getComponentResponse, {'eventDirection': +1, 'component': 'mixed'}),
    'rProbe/left':
        (_getComponentResponse, {'eventDirection': -1, 'component': 'probe'}),
    'rProbe/right':
        (_getComponentResponse, {'eventDirection': +1, 'component': 'probe'}),
    'rSaccade/left':
        (_getComponentResponse, {'eventDirection': -1, 'component': 'saccade', 'shifted': True}),
    'rSaccade/right':
        (_getComponentResponse, {'eventDirection': +1, 'component': 'saccade', 'shifted': True}),
    'rSaccadeUnshifted/nasal':
        (_getComponentResponse, {'eventDirection': -1, 'component': 'saccade', 'shifted': False}),
    'rSaccadeUnshifted/temporal':
        (_getComponentResponse, {'eventDirection': +1, 'component': 'saccade', 'shifted': False}),
    'pZeta/left':
        (_getZetaTestProbabilities, {'probeMotion': -1}),
    'pZeta/right':
        (_getZetaTestProbabilities, {'probeMotion': +1}),
    'sigma/left':
        (_getSigmaValues, {'probeMotion': -1}),
    'sigma/right':
        (_getSigmaValues, {'probeMotion': +1}),
}

class UnitsTable():
    """
    """

    def make(
        self,
        filename,
        sessions,
        minimumFiringRate=0.5,
        minimumResponseAmplitude=None,
        filterUnits=True
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
        if type(filename) != pl.Path:
            filename = pl.Path(filename)

        #
        with h5py.File(str(filename), 'w') as stream:
            for key, (value, kwargs) in unitsTableMapping.items():
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
                    if len(data.shape) in (1, 2):
                        for iUnit in range(data.shape[0]):
                            x = data[iUnit]
                            dataset.append(x)
                    elif len(data.shape) == 3:
                        for iUnit in range(data.shape[1]):
                            x = data[:, iUnit, :]
                            dataset.append(x)
                dataset = np.array(dataset)
                ds = stream.create_dataset(key, dataset.shape, dataset.dtype, data=dataset)
                if len(attrs.keys()) != 0:
                    for k in attrs.keys():
                        ds.attrs[k] = attrs[k]

        return
