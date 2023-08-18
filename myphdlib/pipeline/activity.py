import numpy as np
from zetapy import getZeta
from joblib import Parallel, delayed
from myphdlib.general.toolkit import psth2, smooth
from simple_spykes.util.ecephys import run_quality_metrics
from scipy.stats import ttest_rel, ttest_ind
from scipy.signal import find_peaks
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

def _runZetaTest(
    spikes,
    event,
    index,
    offset,
    window,
    timing='peak'
    ):
    """
    Helper function that runs the ZETA test

    keywords
    --------
    offset: float
        This value is added to the spike timestamps to identify event related activity that
        happens prior to the target event
    timing: str
        Latency definition. "onset" return the time from event to response onset. "peak" return
        the time from the event to the response peak.
    """

    try:
        pvalue, (tZenith, tInverse, tPeak), dZeta = getZeta(
            spikes + offset,
            event,
            intLatencyPeaks=3,
            dblUseMaxDur=np.max(window),
            tplRestrictRange=window,
            boolReturnZETA=True
        )
        if timing == 'zenith':
            latency = tZenith
        elif timing == 'peak':
            latency = tPeak
        elif timing == 'inverse':
            latency = tInverse
        else:
            latency = np.nan
    except Exception as error:
        pvalue, latency = np.nan, np.nan

    return pvalue, latency - offset, index

def identifyUnitsWithEventRelatedActivity(
    session,
    parallel=True,
    overwrite=False,
    offsetForMotorResponse=0.3,
    windowForMotorResponse=(0, 0.6),
    offsetForVisualResponse=0,
    windowForVisualResponse=(0, 0.3),
    stimulus='probe',
    ):
    """
    """

    # Decide whether to run the ZETA test for visual responses
    runZetaTest = not np.all([
        session.hasGroup('population/metrics/zeta/visual/pvalues'),
        session.hasGroup('population/metrics/zeta/visual/latency/peak'),
    ]).item()
    if overwrite:
        runZetaTest = True

    # Get the timestamps for the target stimulus
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
    elif stimulus == 'probe':
        timestamps = session.load('stimuli/dg/timestamps')
        if timestamps.size == 0:
            print(f'WARNING[{session.date}, {session.animal}]:No timestamps found for the drifting grating stimulus')
            runZetaTest = False
    else:
        raise Exception(f'{stimulus} is not a valid target stimulus')
    
    # Identify units with visual responses
    if runZetaTest:

        #
        print(f'INFO[{session.animal}, {session.date}]: Running ZETA test for probe stimulus-related activity')

        # Distributed/parallel computation
        if parallel:
            result = Parallel(n_jobs=-1)(delayed(_runZetaTest)(unit.timestamps, timestamps, unit.index, offsetForVisualResponse, windowForVisualResponse)
                for unit in session.population
            )
            indices = np.array([el[2] for el in result])
            order = np.argsort(indices)
            pvalues = np.array([el[0] for el in result])[order]
            latency = np.array([el[1] for el in result])[order]

        #
        session.save('population/metrics/zeta/visual/pvalues', pvalues)
        session.save('population/metrics/zeta/visual/latency/peak', latency)

    # Decide whether to run ZETA test for motor responses
    runZetaTest = False
    for eye in ('left', 'right'):
        for direction in ('nasal', 'temporal'):
            if session.hasGroup(f'population/metrics/zeta/motor/{eye}/{direction}/pvalues') == False:
                runZetaTest = True
    if overwrite:
        runZetaTest = True

    # Identify units with saccade-related activity
    for eye in ('left', 'right'):
        for direction in ('nasal', 'temporal'):

            #
            if runZetaTest:

                # Load the saccade onset timestamps
                timestamps = session.saccadeOnsetTimestamps[eye][direction]
                if timestamps.size == 0:
                    continue

                #
                print(f'INFO[{session.animal}, {session.date}]: Running ZETA test for saccade-related activity (eye={eye}, direction={direction})')

                # Distributed/parallel computation
                if parallel:
                    result = Parallel(n_jobs=-1)(delayed(_runZetaTest)(unit.timestamps, timestamps, unit.index, offsetForMotorResponse, windowForMotorResponse)
                        for unit in session.population
                    )
                    indices = np.array([el[2] for el in result])
                    order = np.argsort(indices)
                    pvalues = np.array([el[0] for el in result])[order]
                    latency = np.array([el[1] for el in result])[order]

                #
                session.save(f'population/metrics/zeta/motor/{eye}/{direction}/pvalues', pvalues)
                session.save(f'population/metrics/zeta/motor/{eye}/{direction}/latency/peak', latency)

    return

def estimateResponseLatency(
    session,
    baselineResponseWindow=(-0.4, -0.3),
    targetResponseWindow=(-0.3, 0.3),
    binsize=0.01,
    logProbThresholdForPeakDetection=15,
    logProbThresholdForOnsetDetection=5,
    smoothingWindowSize=7,
    ):
    """
    """

    # Load the timestamps for the probe stimuli
    probeOnsetTimestamps = session.load('stimuli/dg/timestamps')

    #
    P, L = list(), list()
    for unit in session.population:
        
        # Compute baseline response
        t, M = psth2(
            probeOnsetTimestamps,
            unit.timestamps,
            window=baselineResponseWindow,
            binsize=None
        )
        rBaseline = M.flatten() / np.diff(baselineResponseWindow).item()

        # Compute T-statistic for each time bin
        pVec = list()
        binCenters, M = psth2(
            probeOnsetTimestamps,
            unit.timestamps,
            window=targetResponseWindow,
            binsize=binsize
        )
        for column in M.T:
            rTarget = column / binsize
            t, p = ttest_ind(rBaseline, rTarget)
            pVec.append(p)
        pVec = np.array(pVec)
        P.append(pVec)

        # Log the p-values
        onsetLatency = np.nan
        if smoothingWindowSize == 0 or smooth is None:
            signal = -1 * np.log(pVec)
        else:
            signal = -1 * smooth(np.log(pVec), smoothingWindowSize)

        # Find the peak of the p-values vector
        peaks, props = find_peaks(
            signal,
            height=logProbThresholdForPeakDetection
        )
        if peaks.size == 0:
            L.append(onsetLatency)
            continue

        # Look backwards for the first threshold crossing
        binIndex = peaks[np.argmax(signal[peaks])]
        while True:
            p2 = signal[binIndex]
            p1 = signal[binIndex - 1]
            if p1 < logProbThresholdForOnsetDetection < p2:
                onsetLatency = binCenters[binIndex]
                break
            binIndex -= 1
            if binIndex < 0:
                break
        L.append(onsetLatency)

    #
    P = np.array(P)
    L = np.array(L)
    session.save(f'population/metrics/zeta/visual/latency/onset', L)
    return P, L, binCenters

def predictUnitClassification(
    session,
    threshold=0.001,
    maximumPeakLatencyForMotorUnits=0.035,
    ):
    """
    """

    #
    nTotalUnits = len(session.population)

    # Visually responsive units
    pvalues = session.load('population/metrics/zeta/visual/pvalues')
    isVisualUnit = pvalues <= threshold
    nVisualUnits = isVisualUnit.sum()
    print(f'INFO[{session.animal}, {session.date}]: {nVisualUnits} out of {nTotalUnits} single-units classified as visual')
    session.save('population/filters/visual', isVisualUnit)

    # Unit with saccade-related activity
    M, P, L = list(), list(), list()
    for eye in ('left', 'right'):
        for direction in ('nasal', 'temporal'):
            pvalues = session.load(f'population/metrics/zeta/motor/{eye}/{direction}/pvalues')
            latency = session.load(f'population/metrics/zeta/motor/{eye}/{direction}/latency/peak')
            mask = pvalues <= threshold
            M.append(mask)
            P.append(pvalues)
            L.append(latency)

    # TODO: ignore saccade-related units if the response latency is too delayed (putatively visual units)
    M, P, L = map(np.array, [M, P, L])
    hasMotorResponse = M.any(axis=0)
    maximumPeakLatency = np.max(L, axis=0)
    isMotorUnit = np.logical(
        hasMotorResponse,
        maximumPeakLatency <= maximumPeakLatencyForMotorUnits,
    )
    nMotorUnits = isMotorUnit.sum()
    print(f'INFO[{session.animal}, {session.date}]: {nMotorUnits} out of {nTotalUnits} single-units classified as motor')
    session.save('population/filters/motor', isMotorUnit)

    return

def _measureUnitStability(
    session,
    ):
    """
    """

    stability = list()
    for unit in session.population:
        pass

    return

def _measureUnitContamination(
    session,
    ):
    """
    """

    return

def measureSpikeSortingQuality(
    session,
    presenceRatioThreshold=0.9,
    isiViolationRateThreshold=1000,
    backend='ecephys',
    **kwargs
    ):
    """
    """

    print(f'INFO[{session.animal}, {session.date}]: Measuring spike-sorting quality')

    # ecephys spike sorting backend
    if backend == 'ecephys':
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
        session.save('population/metrics/stability', presenceRatios)
        session.save('population/metrics/contamination', isiViolationRates)

    # Custom implementation of quality metrics
    elif backend == 'custom':
        for unit in session.population:
            pass

    #
    quality = np.logical_and(
        presenceRatios >= presenceRatioThreshold,
        isiViolationRates >= isiViolationRateThreshold
    )
    session.save('population/filters/quality', quality)

    return