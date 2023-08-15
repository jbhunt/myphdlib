import numpy as np
from zetapy import getZeta
from joblib import Parallel, delayed
from myphdlib.general.toolkit import psth2, smooth
from simple_spykes.util.ecephys import run_quality_metrics
samplingRateNeuropixels = 30000.0

def extractSingleUnitData(session):
    """
    """

    print(f'INFO[{session.animal}, {session.date}]: Extracting spike clusters and timestamps')

    spikeTimestamps = np.array([])
    result = list(session.folders.ephys.joinpath('sorting').glob('spike_times.npy'))
    if len(result) != 1:
        raise Exception('Could not locate the spike times data')
    else:
        spikeTimestamps = np.around(
            np.load(str(result.pop())).flatten() / samplingRateNeuropixels,
            3
        )
    
    #
    spikeClusters = np.array([])
    result = list(session.folders.ephys.joinpath('sorting').glob('spike_clusters.npy'))
    if len(result) != 1:
        raise Exception('Could not locate the cluster ID data')
    else:
        spikeClusters = np.around(
            np.load(str(result.pop())).flatten(),
            3
        )

    #
    session.save('spikes/timestamps', spikeTimestamps)
    session.save('spikes/clusters', spikeClusters)

    #
    uids = np.unique(spikeClusters)
    session.save('population/uids', uids)
    nUnits = uids.size
    print(f'INFO[{session.animal}, {session.date}]: {nUnits} single-units detected')

    return

def _runZetaTest(spikes, event, index, offset, window):
    """
    Helper function that runs the ZETA test

    keywords
    --------
    offset: float
        This value is added to the spike timestamps to identify event related activity that
        happens prior to the target event
    """

    try:
        pvalue, out = getZeta(spikes + offset, event, intLatencyPeaks=3, tplRestrictRange=window)
        latency = out[-1]
    except:
        pvalue, latency = np.nan, np.nan

    return pvalue, latency - offset, index

def identifyUnitsWithEventRelatedActivity(
    session,
    threshold=0.01,
    parallel=True,
    overwrite=False,
    offsetForMotorResponse=-0.5,
    windowForMotorResponse=(0, 1.5),
    offsetForVisualResponse=0,
    windowForVisualResponse=(0, 0.5),
    stimulus='spots',
    skipVisualUnitIdentification=False
    ):
    """
    """

    # Identify visual units
    if skipVisualUnitIdentification == False:
        if stimulus == 'spots':
            paths = (
                'stimuli/sn/pre/timestamps',
                'stimuli/sn/post/timestamps'
            )
            timestamps = list()
            for path in paths:
                if session.hasGroup(path):
                    for ts in session.load(path):
                        timestamps.append(ts)
            timestamps = np.array(timestamps)
        else:
            raise Exception(f'{stimulus} is not a valid target stimulus')

        # Distributed/parallel computation
        if parallel:
            result = Parallel(n_jobs=-1)(delayed(_runZetaTest)(unit.timestamps, timestamps, index, offsetForVisualResponse, windowForVisualResponse)
                for index, unit in enumerate(session.population)
            )
            indices = np.argsort(np.array([element[2] for element in result]))
            pvalues = np.array([el[0] for el in result])[indices]
            latency = np.array([el[1] for el in result])[indices]

        #
        isVisualUnit = pvalues <= threshold
        nVisualUnits = isVisualUnit.sum()
        nTotalUnits = isVisualUnit.size
        print(f'INFO[{session.animal}, {session.date}]: {nVisualUnits} out of {nTotalUnits} single-units classified as visual')
        session.save('population/filters/visual', isVisualUnit)
        session.save('population/metrics/zeta/visual/pvalues', pvalues)
        session.save('population/metrics/zeta/visual/latency', latency)

    # Identify motor units
    M = list()
    L = list()
    P = list()
    for eye in ('left', 'right'):
        for direction in ('nasal', 'temporal'):

            # Load the saccade onset timestamps
            timestamps = session.saccadeOnsetTimestamps[eye][direction]
            if timestamps.size == 0:
                continue

            # Distributed/parallel computation
            if parallel:
                result = Parallel(n_jobs=-1)(delayed(_runZetaTest)(unit.timestamps, timestamps, index, offsetForMotorResponse, windowForMotorResponse)
                    for index, unit in enumerate(session.population)
                )
                indices = np.argsort(np.array([element[2] for element in result]))
                pvalues = np.array([el[0] for el in result])[indices]
                latency = np.array([el[1] for el in result])[indices]

            #
            mask = pvalues <= threshold
            M.append(mask)
            L.append(latency)
            P.append(pvalues)
            session.save(f'population/metrics/zeta/motor/{eye}/{direction}/pvalues', pvalues)
            session.save(f'population/metrics/zeta/motor/{eye}/{direction}/latency', latency)
    
    #
    M, L, P = np.array(M), np.array(L), np.array(P)
    isMotorUnit = M.any(0)
    nMotorUnits = isMotorUnit.sum()
    nTotalUnits = isMotorUnit.size
    print(f'INFO[{session.animal}, {session.date}]: {nMotorUnits} out of {nTotalUnits} single-units classified as motor')
    session.save('population/filters/motor', isMotorUnit)

    #
    # latency = np.array([np.min(column) for column in L.T])
    # pvalues = np.array([np.min(column) for column in P.T])

    return

def _measureStabilityForMotorUnits(
    session,
    ):
    """
    """

    timestamps = 

    return

def measureSpikeSortingQuality(
    session,
    presenceRatioThreshold=0.9,
    isiViolationRateThreshold=1000,
    **kwargs
    ):
    """
    """

    print(f'INFO[{session.animal}, {session.date}]: Measuring spike-sorting quality')

    params_ = {
        "isi_threshold": 0.0015,
        "min_isi": 0.000166,
        "num_channels_to_compare": 7,
        "max_spikes_for_unit": 500,
        "max_spikes_for_nn": 10000,
        "n_neighbors": 4,
        'n_silhouette': 10000,
        "drift_metrics_interval_s": 51,
        "drift_metrics_min_spikes_per_interval": 10,
        "include_pc_metrics": False
    }
    params_.update(kwargs)

    #
    sortingResultsFolder = session.folders.ephys.joinpath('sorting')
    metrics = run_quality_metrics(
        str(sortingResultsFolder),
        30000.0,
        params_,
        save_to_file=str(sortingResultsFolder.joinpath('quality_metrics.json'))
    )
    presenceRatios = np.array(list(metrics['presence_ratio'].values())).astype(float)
    isiViolationRates = np.array(list(metrics['isi_viol'].values())).astype(float)

    #
    session.save('population/metrics/stability/ecephys', presenceRatios)
    session.save('population/metrics/contamination/ecephys', isiViolationRates)

    #
    for unit in session.population:
        pass
    #
    quality = np.logical_and(
        presenceRatios >= presenceRatioThreshold,
        isiViolationRates >= isiViolationRateThreshold
    )
    session.save('population/filters/quality', quality)

    return