import numpy as np
from scipy.signal import find_peaks
from joblib import Parallel, delayed
from myphdlib.general.toolkit import psth2, smooth
from decimal import Decimal

# TODO
# [X] If standardizing the response, use a common estimate of mu and sigma
# [X] Base the estimate of latency-shifted sacade-related activity on the direction of motion of the grating for each saccade
# [-] Write a funciton that determines the response window
# [X] Figure out how to deal with negative spike rates in the MI formula
#     - Wait until computing the expected response to subtract off the baseline activity
#     - Add an offset to the psths such that the actual and expected responses are non-zero
# [X] Make the functions that estimate/measure responses do one direction of motion at a time
# [-] Get rid of the "perisaccadicWindow" argument (it's redundant)
# [X] Write an alternative MI computation the measures modulation on a trial-by-trial basis
# [ ] Have the option to compute MI as the difference in expected and observed response amplitude
#     - Use the peak latency determined with the ZETA test to identify which bin to look in
# [ ] Come up with a different way to estimate the baseline saccade-related activity to add to the observed response

def measureVisualOnlyResponse(
    unit,
    probeMotion=-1,
    responseWindow=(0, 0.3),
    baselineWindow=(-3, -1),
    perisaccadicWindow=(-0.05, 0.1),
    perisaccadicTrialIndices=None,
    excludePerisaccadicTrials=True,
    binsize=0.01,
    fictiveSaccadesProtocol=False
    ):
    """
    Measure the visual-only response
    """

    # Determine the extra-saccadic trial indices
    if perisaccadicTrialIndices is None:
        trialIndices = np.where(unit.session.filterProbes(
            trialType='es',
            perisaccadicWindow=perisaccadicWindow,
            probeDirections=(probeMotion,),
        ))[0]

    # Determine the extra-saccadic trial indices (based on the peri-saccadic trial indices)
    else:

        # Gather all trial indices, unsorted
        trialIndicesUnsorted = np.where(unit.session.filterProbes(
            trialType=None,
            probeDirections=(probeMotion,),
            perisaccadicWindow=perisaccadicWindow
        ))[0]

        # Filter out the peri-saccadic trials
        trialIndices = list()
        for iTrial in trialIndicesUnsorted:

            # User-defined peri-saccadic trial indices
            if iTrial in perisaccadicTrialIndices:
                continue

            # Ground truth peri-saccadic trial indices (optional)
            if excludePerisaccadicTrials:
                probeLatency = unit.session.probeLatencies[iTrial]
                if probeLatency >= perisaccadicWindow[0] and probeLatency <= perisaccadicWindow[1]:
                    continue

            #
            trialIndices.append(iTrial)
        #
        trialIndices = np.array(trialIndices)

    #
    if binsize is None:
        dt = np.diff(responseWindow).item()
    else:
        dt = binsize
    t, M = psth2(
        unit.session.probeTimestamps[trialIndices],
        unit.timestamps,
        window=responseWindow,
        binsize=binsize
    )  
    fr = M / dt

    #
    mu, sigma = unit.describe3(
        unit.session.probeTimestamps[trialIndices],
        baselineWindow=baselineWindow,
        binsize=binsize
    )

    return t, fr, mu, sigma

def measurePerisaccadicVisualResponse(
    unit,
    probeMotion=-1,
    responseWindow=(0, 0.3),
    baselineWindow=(-3, -1),
    perisaccadicWindow=(-0.05, 0.1),
    perisaccadicTrialIndices=None,
    binsize=0.01,
    ):
    """
    Measure the observed peri-saccadic response
    """

    #
    if perisaccadicTrialIndices is None:
        trialIndices = np.where(unit.session.filterProbes(
            trialType='ps',
            probeDirections=(probeMotion,),
            perisaccadicWindow=perisaccadicWindow
        ))[0]
    else:
        trialIndices = perisaccadicTrialIndices

    #
    if binsize is None:
        dt = np.diff(responseWindow).item()
    else:
        dt = binsize

    #
    t, M = psth2(
        unit.session.probeTimestamps[trialIndices],
        unit.timestamps,
        window=responseWindow,
        binsize=binsize
    )
    fr = M / dt

    #
    mu, sigma = unit.describe3(
        unit.session.probeTimestamps[trialIndices],
        baselineWindow=baselineWindow,
        binsize=binsize
    )

    return t, fr, mu, sigma

def estimateSaccadeRelatedActivity(
    unit,
    probeMotion=-1,
    responseWindow=(0, 0.3),
    baselineWindow=(-3, -1),
    perisaccadicWindow=(-0.05, 0.1),
    perisaccadicTrialIndices=None,
    binsize=0.01,
    ):
    """
    Estimate the activity attributable to saccades in the peri-saccadic window
    """

    #
    if perisaccadicTrialIndices is None:
        trialIndices = np.where(unit.session.filterProbes(
            trialType='ps',
            probeDirections=(probeMotion,),
        ))[0]
    else:
        trialIndices = perisaccadicTrialIndices

    # Estimate the baseline level of activity prior to saccades
    saccadeIndices = np.where(unit.session.gratingMotionDuringSaccades == probeMotion)[0]

    #
    fr = list()
    if binsize is None:
        dt = np.diff(responseWindow).item()
    else:
        dt = binsize
    for probeLatency in unit.session.probeLatencies[trialIndices]:

        # Shift the saccade psth by the latency from the saccade to the probe
        t, M = psth2(
            unit.session.saccadeTimestamps[saccadeIndices] + probeLatency,
            unit.timestamps,
            window=responseWindow,
            binsize=binsize
        )
        fr_ = M.mean(0) / dt
        fr.append(fr_)

    #
    fr = np.array(fr)

    #
    mu, sigma = unit.describe3(
        unit.session.saccadeTimestamps[saccadeIndices],
        baselineWindow=baselineWindow,
        binsize=binsize
    )

    return t, fr, mu, sigma

def computeFictiveSaccadeResponses(
    unit,
    probeMotion=-1,
    responseWindow=(0, 0.3),
    responseWindowSize=0.3,
    baselineWindow=(-3, -1),
    perisaccadicWindow=(-0.05, 0.1),
    binsize=0.01,
    priorParameterEstimates=None,
    sigma=1
    ):
    """
    """

    # TODO: Implement a property that loads and stores these data
    probeTimestamps = unit.session.load('stimuli/fs/probe/timestamps')
    gratingMotionDuringProbes = unit.session.load('stimuli/fs/probe/motion')
    saccadeTimestamps = unit.session.load('stimuli/fs/saccade/timestamps')
    gratingMotionDuringSaccades = unit.session.load('stimuli/fs/saccade/motion')

    #
    if binsize is None:
        dt = np.diff(responseWindow).item()
    else:
        dt = binsize

    # Parse probes
    probeIndices, probeLatencies = dict(ps=list(), es=list()), list()
    for trialIndex, probeTimestamp in enumerate(probeTimestamps):
        if gratingMotionDuringProbes[trialIndex] != probeMotion:
            continue
        saccadeTimestampsRelative = probeTimestamp - saccadeTimestamps
        closestSaccadeIndex = np.argmin(np.abs(saccadeTimestampsRelative))
        probeLatency = probeTimestamp - saccadeTimestamps[closestSaccadeIndex]
        probeLatencies.append(probeLatency)
        if probeLatency >= perisaccadicWindow[0] and probeLatency <= perisaccadicWindow[1]:
            probeIndices['ps'].append(trialIndex)
        else:
            probeIndices['es'].append(trialIndex)

    # Parse saccades
    saccadeIndices, saccadeLatencies = dict(ps=list(), es=list()), list()
    for trialIndex, saccadeTimestamp in enumerate(saccadeTimestamps):
        if gratingMotionDuringSaccades[trialIndex] != probeMotion:
            continue
        probeTimestampsRelative = saccadeTimestamp - probeTimestamps
        closestProbeIndex = np.argmin(np.abs(probeTimestampsRelative))
        saccadeLatency = saccadeTimestamp - probeTimestamps[closestProbeIndex]
        saccadeLatencies.append(saccadeLatency)
        if saccadeLatency >= perisaccadicWindow[0] and saccadeLatency <= perisaccadicWindow[1]:
            saccadeIndices['ps'].append(trialIndex)
        else:
            saccadeIndices['es'].append(trialIndex)

    # === Measure visual-only response ===

    mu, sigma_ = unit.describe3(
        probeTimestamps[probeIndices['es']],
        baselineWindow=baselineWindow,
        binsize=binsize
    )

    t, M = psth2(
        probeTimestamps[probeIndices['es']],
        unit.timestamps,
        window=responseWindow,
        binsize=binsize
    )  
    fr = M / dt
    rProbe = np.mean((fr - mu) / sigma, axis=0)

    # === Measure peri-saccadic response ===

    mu, sigma_ = unit.describe3(
        probeTimestamps[probeIndices['ps']],
        baselineWindow=baselineWindow,
        binsize=binsize
    )

    t_, M = psth2(
        probeTimestamps[probeIndices['ps']],
        unit.timestamps,
        window=responseWindow,
        binsize=binsize
    )
    fr = M / dt
    rMixed = np.mean((fr - mu) / sigma, axis=0)

    # === Compute saccade-related activity ===
    mu, sigma_ = unit.describe3(
        saccadeTimestamps[saccadeIndices['es']],
        baselineWindow=baselineWindow,
        binsize=binsize
    )

    rSaccade = list()
    for saccadeLatency in saccadeLatencies:
        t_, M = psth2(
            saccadeTimestamps[saccadeIndices['es']],
            unit.timestamps,
            window=np.array(responseWindow) + saccadeLatency,
            binsize=binsize
        )
        fr = M.mean(0) / dt
        z = (fr - mu) / sigma
        rSaccade.append(z)
    rSaccade = np.mean(np.array(rSaccade), axis=0)

    # === Compute baseline activity prior to saccades ===

    rControl = 0

    return t, (rProbe, rMixed, rSaccade, rControl), sigma

def computeModulationIndices(
    unit,
    probeMotion=-1,
    responseWindowSize=0.05,
    baselineWindowEdge=-5,
    baselineWindowSize=3,
    responseLatencyOffset=0.0,
    standardize=True,
    perisaccadicWindow=(-0.05, 0.1),
    perisaccadicTrialIndices=None,
    binsize=None,
    priorParameterEstimates=None,
    analyzeFictiveSaccades=False
    ):
    """
    """

    # Define the center of the visual response window
    if probeMotion == -1:
        peakResponseLatency = unit.visualResponseLatency.left + responseLatencyOffset
    else:
        peakResponseLatency = unit.visualResponseLatency.right + responseLatencyOffset
    
    # Define the edges of the visual response window
    windowHalfWidth = round(responseWindowSize / 2, 2)
    responseWindow = np.around(np.array([
        peakResponseLatency - windowHalfWidth,
        peakResponseLatency + windowHalfWidth
    ]), 2)

    # Define the baseline response window
    baselineWindow = np.around(np.array([
        baselineWindowEdge,
        baselineWindowEdge + baselineWindowSize,
    ]), 2)

    # Keywords arguments for the response functions
    kwargs = {
        'probeMotion': probeMotion,
        'responseWindow': responseWindow,
        'baselineWindow': baselineWindow,
        'perisaccadicWindow': perisaccadicWindow,
        'perisaccadicTrialIndices': perisaccadicTrialIndices,
        'binsize': binsize
    }

    # Estimate variability of FR prior to the stimulus
    trialIndices = np.where(unit.session.filterProbes(
        trialType='es',
        probeDirections=(probeMotion,)
    ))[0]

    # Use the response window size as the binsize for estimating baseline variability
    if binsize is None:
        binsize_ = responseWindowSize
    else:
        binsize_ = binsize

    #
    t = None

    #
    if priorParameterEstimates is not None and 'sigma' in priorParameterEstimates.keys():
        sigma = priorParameterEstimates['sigma']
    else:
        mu, sigma = unit.describe2(
            unit.session.probeTimestamps[trialIndices],
            baselineWindowBoundaries=baselineWindow,
            binsize=binsize_
        )
    if sigma == 0:
        raise Exception(f'Standard deviation could not be estimated')

    # Run analysis on fictive saccades
    if analyzeFictiveSaccades:
        t, rs, sigma = computeFictiveSaccadeResponses(
            unit=unit,
            probeMotion=probeMotion,
            responseWindow=responseWindow,
            responseWindowSize=responseWindowSize,
            baselineWindow=baselineWindow,
            perisaccadicWindow=perisaccadicWindow,
            binsize=binsize,
            priorParameterEstimates=priorParameterEstimates,
            sigma=sigma
        )
        rProbe, rMixed, rSaccade, rControl = rs

    # Run analysis on actual saccades
    else:

        # Observed extra-saccadic responses
        if priorParameterEstimates is not None and 'rProbe' in priorParameterEstimates.keys():
            rProbe = np.atleast_1d(priorParameterEstimates['rProbe'])
        else:
            t_, mProbe, mu, sigma_ = measureVisualOnlyResponse(
                unit,
                **kwargs
            )
            rProbe = mProbe.mean(0)
            if standardize:
                rProbe = np.around((rProbe - mu) / sigma, 3)
            t = np.copy(t_)

        # Observed peri-saccadic responses
        if priorParameterEstimates is not None and 'rMixed' in priorParameterEstimates.keys():
            rMixed = np.atleast_1d(priorParameterEstimates['rMixed'])
        else:
            t_, mMixed, mu, sigma_ = measurePerisaccadicVisualResponse(
                unit,
                **kwargs
            )
            rMixed = mMixed.mean(0)
            if standardize:
                rMixed = np.around((rMixed - mu) / sigma, 3)

        # Predicted saccade-related activity
        if priorParameterEstimates is not None and 'rSaccade' in priorParameterEstimates.keys():
            rSaccade = np.atleast_1d(priorParameterEstimates['rSaccade'])
        else:
            t_, mSaccade, mu, sigma_ = estimateSaccadeRelatedActivity(
                unit,
                **kwargs
            )
            rSaccade = mSaccade.mean(0)
            if standardize:
                rSaccade = np.around((rSaccade - mu) / sigma, 3)

        #
        if priorParameterEstimates is not None and 'rControl' in priorParameterEstimates.keys():
            rControl = np.atleast_1d(priorParameterEstimates['rControl'])
        else:
            t_, mControl, mu, sigma_ = estimateSaccadeRelatedActivity(
                unit,
                probeMotion=probeMotion,
                responseWindow=np.around([
                    baselineWindowEdge - responseWindowSize,
                    baselineWindowEdge
                ], 2),
                baselineWindow=baselineWindow,
                binsize=binsize
            )
            rControl = round(mControl.mean(0).mean(), 3)
            if standardize:
                rControl = round((rControl - mu) / sigma, 3)

    #
    rExpected = np.around(rProbe + rSaccade, 2)
    rObserved = np.around(rMixed + rControl, 2)

    #
    if rObserved.sum() < 0 and rExpected.sum() < 0:
        rOffset = 0
        rSign = -1
    elif rObserved.sum() < 0 and rExpected.sum() > 0: 
        rOffset = abs(rObserved.sum())
        rSign = +1
    elif rObserved.sum() > 0 and rExpected.sum() < 0:
        rSign = +1
        rOffset = abs(rExpected.sum())
    else:
        rSign = +1
        rOffset = 0
    
    #
    rDifference = (rObserved.sum() + rOffset) - (rExpected.sum() + rOffset)
    rSum = (rObserved.sum() + rOffset) + (rExpected.sum() + rOffset) 
    mi = np.around(rDifference / rSum * rSign, 3)
    
    #
    dr = np.around(rObserved - rExpected, 3)

    # Flatten all of the metrics and responses
    if binsize == None:
        if t is not None: t = round(t.item(), 2)
        dr = dr.item()
        rExpected = rExpected.item()
        rObserved = rObserved.item()
        rProbe = rProbe.item()
        rMixed = rMixed.item()
        rSaccade = rSaccade.item()

    return t, (rObserved, rExpected, rProbe, rMixed, rSaccade, rControl), mi, dr, sigma

def estimateNullDistributions(
    unit,
    probeMotion=-1,
    nRuns=100,
    priorParameterEstimates=None,
    **kwargs
    ):
    """
    Estimate the spread of the modulation indices by re-sampling extra-saccadic trials,
    recomputing the visual-only response, and plugging that back into the MI formula
    """

    # Get the indices for extra-saccadic trials
    trialIndicesExtrasaccadic = np.where(unit.session.filterProbes(
        trialType='es',
        probeDirections=(probeMotion,)
    ))[0]

    # Get the indices for peri-saccadic trials
    trialIndicesPerisaccadic = np.where(unit.session.filterProbes(
        trialType='ps',
        probeDirections=(probeMotion,)
    ))[0]

    # All trial indices
    trialIndicesUnsorted = np.where(unit.session.filterProbes(
        trialType=None,
        probeDirections=(probeMotion,)
    ))[0]

    # Number of trials to resample
    nTrialsPerisaccadic = trialIndicesPerisaccadic.size

    # Remove the visual-only response estimate so that it will be re-computed
    if 'rProbe' in priorParameterEstimates.keys():
        rProbe = priorParameterEstimates.pop('rProbe')

    # For each run select a set of random extra-saccadic trials and compute the MI
    samples = list()
    for iRun in range(nRuns):

        # Resample extra-saccadic trials
        trialIndicesExtrasaccadicResampled = np.random.choice(
            trialIndicesExtrasaccadic,
            size=nTrialsPerisaccadic,
            replace=False
        )

        # Classify all other trials as peri-saccadic (for computing resampled visual-only response)
        trialIndicesPerisaccadicResampled = list()
        for iTrial in trialIndicesUnsorted:
            if iTrial in trialIndicesExtrasaccadicResampled:
                continue
            trialIndicesPerisaccadicResampled.append(iTrial)
        trialIndicesPerisaccadicResampled = np.array(trialIndicesPerisaccadicResampled)
    
        # Estimate saccadic modulation
        t, rs, mi, dv, sd = computeModulationIndices(
            unit,
            probeMotion,
            perisaccadicTrialIndices=trialIndicesPerisaccadicResampled,
            priorParameterEstimates=priorParameterEstimates,
            **kwargs
        )

        #
        samples.append([mi, dv])

    return np.array(samples)

def measureSaccadicModulationForSingleUnit(
    unit,
    nRuns=100,
    testNullHypothesis=True,
    **kwargs,
    ):
    """
    """

    result = {
        ('left', 'mi', 'x'): None,
        ('left', 'mi', 'p'): None,
        ('left', 'dr', 'x'): None,
        ('left', 'dr', 'p'): None,
        ('right', 'mi', 'x'): None,
        ('right', 'mi', 'p'): None,
        ('right', 'dr', 'x'): None,
        ('right', 'dr', 'p'): None,
    }
    for probeMotion in (-1, 1):

        #
        probeDirection = 'left' if probeMotion == -1 else 'right'

        #
        t, rs, mi, dr, sd = computeModulationIndices(
            unit,
            probeMotion,
            **kwargs
        )

        #
        priorParameterEstimates = {
            'rProbe': rs[2],
            'rMixed': rs[3],
            'rSaccade': rs[4],
            'rControl': rs[5],
            'sigma': sd,
        }

        #
        result[(probeDirection, 'mi', 'x')] = mi
        result[(probeDirection, 'dr', 'x')] = dr

        #
        if testNullHypothesis:

            #
            samples = estimateNullDistributions(
                unit,
                probeMotion,
                nRuns,
                priorParameterEstimates,
                **kwargs,
            )

            #
            for iColumn, (metricKey, x) in enumerate(zip(['mi', 'dr'], [mi, dr])):
                sample = samples[:, iColumn]
                if np.isnan(x):
                    p = np.nan
                elif x <= 0:
                    p = np.sum(sample >= 0) / sample.size
                elif x > 0:
                    p = np.sum(sample <= 0) / sample.size
                result[(probeDirection, metricKey, 'p')] = p

    return result

def measureSaccadicModulation(
    session,
    responseWindowSize=0.05,
    baselineWindowEdge=-5, # This fucked me up so bad ):
    baselineWindowSize=3,
    perisaccadicWindow=(-0.05, 0.1),
    binsize=None,
    nRunsForBootstrap=1000,
    testNullHypothesis=True,
    parallel=False,
    returnData=False
    ):
    """
    """

    #
    kwargs = {
        'responseWindowSize': responseWindowSize,
        'baselineWindowEdge': baselineWindowEdge,
        'baselineWindowSize': baselineWindowSize,
        'perisaccadicWindow': perisaccadicWindow,
        'binsize': binsize
    }

    #
    session.population.filter(probeMotion=None)
    nUnitsFiltered = session.population.count(filtered=True)
    nUnitsUnfiltered = session.population.count(filtered=False)

    #
    data = {
        ('left', 'mi', 'x'): np.full(nUnitsUnfiltered, np.nan),
        ('left', 'mi', 'p'): np.full(nUnitsUnfiltered, np.nan),
        ('left', 'dr', 'x'): np.full(nUnitsUnfiltered, np.nan),
        ('left', 'dr', 'p'): np.full(nUnitsUnfiltered, np.nan),
        ('right', 'mi', 'x'): np.full(nUnitsUnfiltered, np.nan),
        ('right', 'mi', 'p'): np.full(nUnitsUnfiltered, np.nan),
        ('right', 'dr', 'x'): np.full(nUnitsUnfiltered, np.nan),
        ('right', 'dr', 'p'): np.full(nUnitsUnfiltered, np.nan),
    }

    # Run MI estimation in parallel
    if parallel:
        raise Exception('Parallel processing not implemented yet')

    # Run MI estimation in serial
    else:
        for iUnit, unit in enumerate(session.population):
            message = f'Working on unit {iUnit + 1} out of {nUnitsFiltered}'
            session.log(message, 'info')
            result = measureSaccadicModulationForSingleUnit(
                unit,
                nRuns=nRunsForBootstrap,
                testNullHypothesis=testNullHypothesis,
                **kwargs
            )
            for k in data.keys():
                data[k][unit.index] = result[k]

    # Save the estimates
    for (probeDirection, metricName, featureName) in data.keys():
        session.save(f'population/metrics/{metricName}/{probeDirection}/{featureName}', np.array(data[(probeDirection, metricName, featureName)]))
    
    #
    if returnData:
        return data

from matplotlib import pylab as plt

class SaccadicModulationAnalysis():
    """
    """

    def analyzeSaccadicModulationOverTimeForSingleUnit(
        self,
        unit,
        plot=True,
        figsize=(4, 7),
        **kwargs_
        ):
        """
        """

        #
        kwargs = {
            'responseWindowSizes': (0.05, 1),
            'centerBinLeftEdge': -0.05,
            'binWidths': np.full(7, 0.15),
            'probeMotion': -1,
        }
        kwargs.update(kwargs_)

        #
        probeDirection = 'left' if kwargs['probeMotion'] == -1 else 'right'

        #
        self.data = {
            't': None,
            'mi': list(),
            'dr': list(),
            'ro': list(),
            're': list(),
            'ep': None
        }

        # Determine the bin edges
        nBins = len(kwargs['binWidths'])
        leftEdges = np.cumsum(kwargs['binWidths']) - kwargs['binWidths'][0]
        centerBinIndex = int((nBins - 1) / 2)
        leftEdges -= (leftEdges[centerBinIndex] - kwargs['centerBinLeftEdge'])
        rightEdges = leftEdges + kwargs['binWidths']
        epochs = np.transpose(np.vstack([leftEdges, rightEdges]))[::-1]
        self.data['ep'] = epochs
        self.data['t'] = np.mean(epochs, 1)

        #
        if plot:
            fig, axs = plt.subplots(nrows=nBins + 1, sharex=True, sharey=True)
            fig.set_figwidth(figsize[0])
            fig.set_figheight(figsize[1])

        #
        for iBin, epoch in enumerate(epochs):

            #
            trialIndices = np.where(np.vstack([
                unit.session.probeLatencies > epoch[0],
                unit.session.probeLatencies <= epoch[1],
                unit.session.gratingMotionDuringProbes == kwargs['probeMotion']
            ]).all(0))[0]

            #
            t, rs, mi, dr, sd = computeModulationIndices(
                unit,
                kwargs['probeMotion'],
                kwargs['responseWindowSizes'][1],
                perisaccadicTrialIndices=trialIndices,
                binsize=0.02,
            )
            self.data['ro'].append(rs[0])
            self.data['re'].append(rs[1])

            #
            if plot:
                axs[iBin].plot(t, rs[0], color='k', alpha=0.5)
                axs[iBin].plot(t, rs[1], color='r', alpha=0.5)

            #
            t, rs, mi, dr, sd = computeModulationIndices(
                unit,
                kwargs['probeMotion'],
                kwargs['responseWindowSizes'][0],
                perisaccadicTrialIndices=trialIndices,
                binsize=None
            )
            self.data['mi'].append(mi)
            self.data['dr'].append(dr)

            #
            if plot:
                xl = axs[iBin].get_xlim()
                y1, y2 = axs[iBin].get_ylim()
                axs[iBin].text(np.mean(xl), y2 * 0.8, f'dr={dr:.2f}')

        #
        trialIndices = np.where(np.vstack([
            np.logical_or(
                unit.session.probeLatencies > epochs.max(),
                unit.session.probeLatencies <= epochs.min(),
            ),
            unit.session.gratingMotionDuringProbes == kwargs['probeMotion']
        ]).all(0))[0]

        #
        t, rs, mi, dr, sd = computeModulationIndices(
            unit,
            kwargs['probeMotion'],
            kwargs['responseWindowSizes'][1],
            perisaccadicTrialIndices=trialIndices,
            binsize=0.02,
        )
        self.data['ro'].append(rs[0])
        self.data['re'].append(rs[1])

        #
        if plot:
            axs[-1].plot(t, rs[0], color='k', alpha=0.5)
            axs[-1].plot(t, rs[1], color='b', alpha=0.5)

        #
        if plot:

            #
            axs[-1].set_xlabel('Time from probe (sec)')
            axs[-1].set_ylabel('FR (standardized)')
            fig.tight_layout()

            #
            yl = axs[-1].get_ylim()
            x1 = unit.visualResponseLatency[probeDirection] - (kwargs['responseWindowSizes'][0] / 2)
            x2 = unit.visualResponseLatency[probeDirection] + (kwargs['responseWindowSizes'][0] / 2)
            for iBin in np.arange(nBins):

                #
                axs[iBin].fill_between([x1, x2], *yl, color='y', alpha=0.1)
                x3 = -1 * epochs[iBin][0]
                x4 = -1 * epochs[iBin][1]
                axs[iBin].fill_between([x3, x4], *yl, color='k', alpha=0.1)
                axs[iBin].vlines(0, *yl, color='k')

            #
            axs[-1].vlines(0, *yl, color='k')

        #
        self.data['ro'] = np.array(self.data['ro'])
        self.data['re'] = np.array(self.data['re'])

        if plot:
            fig.tight_layout()
            return fig

    def analyzeSaccadicModulationOverTimeAcrossUnits(
        self,
        session,
        ):
        """
        """

        curves = {
            'observed': list(),
            'expected': list()
        }
        session.population.filter()
        for unit in session.population:
            self.analyzeSaccadicModulationOverTimeForSingleUnit(
                unit,
                plot=False
            )
            curves['observed'].append(self.data['ro'])
            curves['expected'].append(self.data['re'])

        #
        curves['observed'] = np.array(curves['observed'])
        curves['expected'] = np.array(curves['expected'])

        return curves
