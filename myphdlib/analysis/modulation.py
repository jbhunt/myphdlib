import numpy as np
from joblib import Parallel, delayed
from myphdlib.general.toolkit import psth2, smooth

def _estimateVisualOnlyComponent(
    unit,
    session,
    window=(0, 0.3),
    binsize=0.01,
    baselineWindowSize=1,
    baselineWindowOffset=-1,
    zscore=True,
    ):
    """
    """

    #
    perisaccadic = session.load('stimuli/dg/perisaccadic')
    driftingGratingMotion = session.load('stimuli/dg/motion')
    probeOnsetTimestamps = session.load('stimuli/dg/timestamps')

    #
    energy = {
        'left': None,
        'right': None,
    }
    curves = {
        'left': None,
        'right': None,
    }

    #
    for motion in np.unique(driftingGratingMotion):
        km = 'left' if motion == - 1 else 'right'
        trialIndices = np.where(np.logical_and(
            driftingGratingMotion == motion,
            np.invert(perisaccadic)
        ))[0] 
        t, M = psth2(
            probeOnsetTimestamps[trialIndices],
            unit.timestamps,
            window=window,
            binsize=binsize
        )
        mu, sigma = unit.describe(
            probeOnsetTimestamps[trialIndices], 
            window=(
                0 + baselineWindowOffset - baselineWindowSize,
                0 + baselineWindowOffset
            )
        )
        if sigma == 0:
            curves[km] = np.full(t.size, np.nan)
            energy[km] = np.full(t.size, np.nan)
            continue

        fr = M.mean(0) / binsize
        z = (fr - mu) / sigma
        if zscore:
            curves[km] = z
        else:
            curves[km] = fr
        e = np.power(z, 2)
        energy[km] = e

    #
    response = {
        'left': np.nansum(energy['left']),
        'right': np.nansum(energy['right'])
    }

    return response, energy, curves


def _estimateVisuomotorComponent(
    unit,
    session,
    window=(0, 0.3),
    binsize=0.01,
    baselineWindowSize=1,
    baselineWindowOffset=-1,
    zscore=True,
    ):
    """
    """

    #
    perisaccadic = session.load('stimuli/dg/perisaccadic')
    driftingGratingMotion = session.load('stimuli/dg/motion')
    probeOnsetTimestamps = session.load('stimuli/dg/timestamps')

    #
    energy = {
        'left': None,
        'right': None,
    }
    curves = {
        'left': None,
        'right': None,
    }

    #
    for motion in np.unique(driftingGratingMotion):
        km = 'left' if motion == - 1 else 'right'
        trialIndicesPerisaccadic = np.where(np.logical_and(
            driftingGratingMotion == motion,
            perisaccadic
        ))[0]
        trialIndicesControl = np.where(np.logical_and(
            driftingGratingMotion == motion,
            np.invert(perisaccadic)
        ))
        t, M = psth2(
            probeOnsetTimestamps[trialIndicesPerisaccadic],
            unit.timestamps,
            window=window,
            binsize=binsize
        )
        mu, sigma = unit.describe(
            probeOnsetTimestamps[trialIndicesControl], 
            window=(
                0 + baselineWindowOffset - baselineWindowSize,
                0 + baselineWindowOffset
            )
        )
        if sigma == 0:
            curves[km] = np.full(t.size, np.nan)
            energy[km] = np.full(t.size, np.nan)
            continue
        fr = M.mean(0) / binsize
        z = (fr - mu) / sigma
        if zscore:
            curves[km] = z
        else:
            curves[km] = fr
        e = np.power(z, 2)
        energy[km] = e

    #
    response = {
        'left': np.nansum(energy['left']),
        'right': np.nansum(energy['right'])
    }

    return response, energy, curves

def _estimateMotorOnlyComponent(
    unit,
    session,
    window=(0, 0.3),
    binsize=0.01,
    eye='right',
    baselineWindowSize=1,
    baselineWindowOffset=-0.5,
    zscore=True,
    includeExtrasaccadicTrials=False,
    ):
    """
    """

    #
    perisaccadic = session.load('stimuli/dg/perisaccadic')
    probeLatencies = session.load('stimuli/dg/latency')
    saccadeDirections = session.load('stimuli/dg/direction')
    driftingGratingMotion = session.load('stimuli/dg/motion')
    saccadeOnsetTimestamps = {
        'nasal': session.load(f'saccades/predicted/{eye}/nasal/timestamps'),
        'temporal': session.load(f'saccades/predicted/{eye}/temporal/timestamps')

    }

    #
    energy = {
        'left': list(),
        'right': list(),
    }
    curves = {
        'left': list(),
        'right': list(),
    }

    #
    for motion in np.unique(driftingGratingMotion):
        km = 'left' if motion == - 1 else 'right'

        #
        if includeExtrasaccadicTrials:
            trialIndices = np.where(driftingGratingMotion == motion)[0]
        else:
            trialIndices = np.where(np.logical_and(
                driftingGratingMotion == motion,
                perisaccadic
            ))[0]

        #
        for probeLatency, direction_ in zip(probeLatencies[trialIndices], saccadeDirections[trialIndices]):
            saccadeLatency = -1 * probeLatency #
            direction = 'nasal' if direction_ == -1 else 'temporal'
            t, M = psth2(
                saccadeOnsetTimestamps[direction] + probeLatency,
                unit.timestamps,
                window=window,
                binsize=binsize
            )
            fr = M.mean(0) / binsize
            mu, sigma = unit.describe(
                saccadeOnsetTimestamps[direction] + probeLatency,
                window=(
                    0 + baselineWindowOffset - baselineWindowSize,
                    0 + baselineWindowOffset
                )
            )
            if sigma == 0:
                continue
            z = (fr - mu) / sigma
            if zscore:
                curves[km].append(z)
            else:
                curves[km].append(fr)
            e = np.power(z, 2)
            energy[km].append(e)

    #
    response = {
        'left': None,
        'right': None,
    }
    for km in ('left', 'right'):
        curves[km] = np.array(curves[km])
        energy[km] = np.array(energy[km])
        response[km] = np.nanmean(energy[km], axis=0).sum()

    return response, energy, curves

def _estimateExpectedPerisaccadicVisualResponse(
    unit,
    session,
    window=(0, 0.3),
    binsize=0.01,
    eye='right',
    baselineWindowSize=1,
    baselineWindowOffset=0,
    zscore=True,
    ):
    """
    """
    #
    perisaccadic = session.load('stimuli/dg/perisaccadic')
    driftingGratingMotion = session.load('stimuli/dg/motion')
    probeOnsetTimestamps = session.load('stimuli/dg/timestamps')
    probeLatencies = session.load('stimuli/dg/latency')
    saccadeDirections = session.load('stimuli/dg/direction')
    saccadeOnsetTimestamps = {
        'nasal': session.load(f'saccades/predicted/{eye}/nasal/timestamps'),
        'temporal': session.load(f'saccades/predicted/{eye}/temporal/timestamps')

    }

    #
    energy = {
        'left': list(),
        'right': list(),
    }
    curves = {
        'left': list(),
        'right': list(),
    }

    #
    for motion in np.unique(driftingGratingMotion):

        #
        km = 'left' if motion == - 1 else 'right'
        trialIndicesPerisaccadic = np.where(np.logical_and(
            driftingGratingMotion == motion,
            perisaccadic
        ))[0]
        trialIndicesControl = np.where(np.logical_and(
            driftingGratingMotion == motion,
            np.invert(perisaccadic)
        ))[0]

        # Estimate the visual-only response
        t, M = psth2(
            probeOnsetTimestamps[trialIndicesControl],
            unit.timestamps,
            window=window,
            binsize=binsize
        )
        fr = M.mean(0) / binsize
        mu, sigma = unit.describe(
            probeOnsetTimestamps[trialIndicesControl],
            window=(
                0 + baselineWindowOffset - baselineWindowSize,
                0 + baselineWindowOffset
            )
        )
        if sigma == 0:
            return None, None, None
        z = (fr - mu) / sigma
        e = np.power(z, 2)
        visualOnlyParameters = {
            'fr': fr,
            'mu': mu,
            'sigma': sigma,
            'z': z,
            'e': e
        }
        
        #
        for iTrial in trialIndicesPerisaccadic:
            probeLatency = probeLatencies[iTrial]
            saccadeDirection = saccadeDirections[iTrial]
            kd = 'nasal' if saccadeDirection == -1 else 'temporal'
            t, M = psth2(
                saccadeOnsetTimestamps[kd] + probeLatency,
                unit.timestamps,
                window=window,
                binsize=binsize
            )
            fr = M.mean(0) / binsize
            mu, sigma = unit.describe(
                saccadeOnsetTimestamps[kd] + probeLatency,
                window=(
                    0 + baselineWindowOffset - baselineWindowSize,
                    0 + baselineWindowOffset
                )
            )
            if sigma == 0:
                continue
            z = (fr - visualOnlyParameters['mu']) / visualOnlyParameters['sigma']
            e = np.power(z, 2)
            if zscore:
                curves[km].append(visualOnlyParameters['z'] + z)
            else:
                curves[km].append(visualOnlyParameters['fr'] + fr)
            energy[km].append(visualOnlyParameters['e'] + e)

    #
    response = {
        'left': None,
        'right': None,
    }
    for km in ('left', 'right'):
        curves[km] = np.array(curves[km])
        energy[km] = np.array(energy[km])
        response[km] = np.nanmean(energy[km], axis=0).sum()

    return response, energy, curves

def _computeModulationIndexForSingleUnit(
    session,
    window,
    binsize,
    version,
    unit=None,
    ):
    """
    """

    #
    MIn = {
        'left': np.nan,
        'right': np.nan,
    }
    if unit is None:
        return MIn
    
    #
    try:
        if version == 1:
            Rp, visualOnlyEnergy, visualOnlyCurves = _estimateVisualOnlyComponent(
                unit,
                session,
                window,
                binsize,
            )
            Rs, motorOnlyEnergy, motorOnlyCurves = _estimateMotorOnlyComponent(
                unit,
                session,
                window,
                binsize,
            )
            Rsp, visuomotorEnergy, visuomotorCurves = _estimateVisuomotorComponent(
                unit,
                session,
                window,
                binsize
            )

            for km in ('left', 'right'):
                if np.any(np.isnan([Rp[km], Rs[km], Rsp[km]])):
                    MIn[km] = np.nan
                expected = Rp[km] + Rs[km]
                observed = Rsp[km]
                MIn[km] = round((observed - expected) / (observed + expected), 3)
        
        #
        elif version == 2:
            Re, energy, curves = _estimateExpectedPerisaccadicVisualResponse(
                unit,
                session,
                window,
                binsize
            )
            Rsp, energy, curves = _estimateVisuomotorComponent(
                unit,
                session,
                window,
                binsize
            )
            for km in ('left', 'right'):
                if np.any(np.isnan([Re[km], Rsp[km]])):
                    MIn[km] = np.nan
                expected = Re[km]
                observed = Rsp[km]
                MIn[km] = round((observed - expected) / (observed + expected), 3)

    except:
        pass
        
    return MIn

def _computeModulationCurveForSingleUnit(
    session,
    window,
    binsize,
    unit=None,
    ):
    """
    """

    C = {
        'left': None,
        'right': None,
    }
    if unit is None:
        return C

    Re, eExpected, zExpected = _estimateExpectedPerisaccadicVisualResponse(
        unit,
        session,
        window,
        binsize
    )
    Rsp, eObserved, zObserved = _estimateVisuomotorComponent(
        unit,
        session,
        window,
        binsize
    )
    for km in ('left', 'right'):
        if np.any(np.isnan([Re[km], Rsp[km]])):
            C[km] = np.full(zExpected[km].size, np.nan)
        expected = zExpected[km].mean(0)
        observed = zObserved[km]
        C[km] = np.around((observed - expected) / (observed + expected), 3)

    return C

def computeModulationIndexAcrossPopulation(
    session,
    window=(0, 0.3),
    binsize=0.01,
    parallel=True,
    version=1,
    ):
    """
    """

    #
    MI = {
        'left': list(),
        'right': list()
    }
    args = (
        session,
        window,
        binsize,
        version
    )

    #
    visual = session.load('analysis/typing/visual')
    clusters = [unit.cluster for unit in session.population]
    population = list()
    for unitIndex in range(len(clusters)):
        if visual[unitIndex]:
            unit = session.population.index(clusters[unitIndex])
            population.append(unit)

    #
    if parallel:
        result = Parallel(n_jobs=-1)(delayed(_computeModulationIndexForSingleUnit)(*args, unit=unit)
            for unit in population
        )
        MI['left'] = np.array([element['left'] for element in result])
        MI['right'] = np.array([element['right'] for element in result])
    else:
        for unit in population:
            MIn = _computeModulationIndexForSingleUnit(*args, unit=unit)
            MI['left'].append(MIn['left'])
            MI['right'].append(MIn['right'])
        for km in ('left', 'right'):
            MI[km] = np.array(MI[km])
    
    return MI