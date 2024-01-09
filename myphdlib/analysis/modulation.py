import numpy as np
from scipy.signal import find_peaks
from joblib import Parallel, delayed
from myphdlib.general.toolkit import psth2, smooth
from decimal import Decimal
import h5py

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



class ModulationAnalysisMixin():
    """
    """

    def _measureVisualOnlyResponse(
        self,
        unit,
        probeMotion=-1,
        responseWindow=(0, 0.3),
        baselineWindow=(-0.25, 0),
        perisaccadicWindow=(-0.05, 0.1),
        binsize=0.01,
        trialIndicesPerisaccadic=None,
        trialIndicesExtrasaccadic=None,
        ):
        """
        Measure the visual-only response
        """

        # Determine the extra-saccadic trial indices
        if trialIndicesExtrasaccadic is None:
            trialIndices = np.where(self.parseEvents(
                eventName='probe',
                coincident=False,
                eventDirection=probeMotion,
                coincidenceWindow=perisaccadicWindow,
            ))[0]
        else:
            trialIndices = trialIndicesExtrasaccadic

        #
        if binsize is None:
            dt = np.diff(responseWindow).item()
        else:
            dt = binsize
        t, M = psth2(
            self.probeTimestamps[trialIndices],
            unit.timestamps,
            window=responseWindow,
            binsize=binsize
        )  
        fr = M / dt

        t_, M = psth2(
            self.probeTimestamps[trialIndices],
            unit.timestamps,
            window=baselineWindow,
            binsize=None
        )
        bl = np.mean(M.flatten()) / np.diff(baselineWindow).item()

        return t, fr, bl

    def _measurePerisaccadicVisualResponse(
        self,
        unit,
        probeMotion=-1,
        responseWindow=(0, 0.3),
        baselineWindow=(-0.025, 0),
        perisaccadicWindow=(-0.05, 0.1),
        binsize=0.01,
        trialIndicesPerisaccadic=None,
        trialIndicesExtrasaccadic=None,
        ):
        """
        Measure the observed peri-saccadic response
        """

        tBins, nTrials, nBins = psth2(
            np.array([0]),
            np.array([0]),
            window=responseWindow,
            binsize=binsize,
            returnShape=True
        )

        #
        if trialIndicesPerisaccadic is None:
            trialIndices = np.where(self.parseEvents(
                eventName='probe',
                coincident=True,
                eventDirection=probeMotion,
                coincidenceWindow=perisaccadicWindow,
            ))[0]
        else:
            trialIndices = trialIndicesPerisaccadic

        # Return NaN if trial count is 0
        if trialIndices.size == 0:
            return tBins, np.full([1, nBins], np.nan), np.nan

        #
        if binsize is None:
            dt = np.diff(responseWindow).item()
        else:
            dt = binsize

        #
        t, M = psth2(
            self.probeTimestamps[trialIndices],
            unit.timestamps,
            window=responseWindow,
            binsize=binsize
        )
        fr = M / dt

        #
        t_, M = psth2(
            self.probeTimestamps[trialIndices],
            unit.timestamps,
            window=baselineWindow,
            binsize=None
        )
        bl = np.mean(M.flatten()) / np.diff(baselineWindow).item()

        return t, fr, bl

    def _estimateSaccadeRelatedActivityWithInterpolation(
        self,
        unit,
        probeMotion=-1,
        responseWindow=(0, 0.3),
        baselineWindow=(-1, -0.8),
        perisaccadicWindow=(-0.05, 0.1),
        binsize=0.01,
        trialIndicesPerisaccadic=None,
        trialIndicesExtrasaccadic=None,
        ):
        """
        """

        #
        if trialIndicesPerisaccadic is None:
            trialIndices = np.where(self.parseEvents(
                eventName='probe',
                coincident=True,
                eventDirection=probeMotion,
                coincidenceWindow=perisaccadicWindow,
            ))[0]
        else:
            trialIndices = trialIndicesPerisaccadic

        #
        fr = list()
        bl = list()
        if binsize is None:
            dt = np.diff(responseWindow).item()
        else:
            dt = binsize

        # Mask that filers out saccades with a coincident probe
        extrastimulusMask = self.parseEvents(
            eventName='saccade',
            coincident=False,
            eventDirection=None,
            coincidenceWindow=-1 * np.array(perisaccadicWindow),
        )

        # Compute PETH templates
        templates = {
            (-1, -1): None,
            (-1, +1): None,
            (+1, -1): None,
            (+1, +1): None,
        }
        for saccadeLabel, probeMotion_ in templates.keys():
            saccadeIndices = np.where(
                np.vstack([
                    extrastimulusMask,
                    self.saccadeLabels == saccadeLabel,
                    self.gratingMotionDuringSaccades == probeMotion_
                ]).all(0)
            )[0]
            t, M = psth2(
                self.saccadeTimestamps[saccadeIndices, 0],
                unit.timestamps,
                window=responseWindow,
                binsize=binsize
            )
            fr_ = M.mean(0) / dt
            templates[(saccadeLabel, probeMotion_)] = fr_

        #
        tBins, nTrials, nBins = psth2(
            np.array([0]),
            np.array([0]),
            window=responseWindow,
            binsize=binsize,
            returnShape=True
        )

       # Iterate over the peri-saccadic trials
        saccadeLabelsProximate = self.load('stimuli/dg/probe/dos')
        for probeLatency, saccadeLabel in zip(self.probeLatencies[trialIndices], saccadeLabelsProximate[trialIndices]):

            # PETH must satisfy all criteria
            # - No coincident probe detected
            # - Same label as the closest (proximate) saccade
            # - Same motion of the grating during the closest (proximate) saccade

            if binsize is not None:
                fp = templates[(saccadeLabel, probeMotion)]
                xp = tBins
                x = tBins + probeLatency
                y = np.interp(x, xp, fp)
                fr.append(y)
                bl.append(np.nan)

        #
        fr = np.array(fr)
        bl = np.array(bl).reshape(-1, 1)

        return t, fr, bl

    def _estimateSaccadeRelatedActivity(
        self,
        unit,
        probeMotion=-1,
        responseWindow=(0, 0.3),
        baselineWindow=(-1, -0.8),
        perisaccadicWindow=(-0.05, 0.1),
        binsize=0.01,
        trialIndicesPerisaccadic=None,
        trialIndicesExtrasaccadic=None,
        ):

        """
        Estimate the activity attributable to saccades in the peri-saccadic window
        """

        #
        if trialIndicesPerisaccadic is None:
            trialIndices = np.where(self.parseEvents(
                eventName='probe',
                coincident=True,
                eventDirection=probeMotion,
                coincidenceWindow=perisaccadicWindow,
            ))[0]
        else:
            trialIndices = trialIndicesPerisaccadic

        #
        fr = list()
        bl = list()
        if binsize is None:
            dt = np.diff(responseWindow).item()
        else:
            dt = binsize

        # Mask that filers out saccades with a coincident probe
        extrastimulusMask = self.parseEvents(
            eventName='saccade',
            coincident=False,
            eventDirection=None,
            coincidenceWindow=-1 * np.array(perisaccadicWindow),
        )

        # Iterate over the peri-saccadic trials
        saccadeLabelsProximate = self.load('stimuli/dg/probe/dos')
        for probeLatency, saccadeLabel in zip(self.probeLatencies[trialIndices], saccadeLabelsProximate[trialIndices]):

            # PETH must satisfy all criteria
            # - No coincident probe detected
            # - Same label as the closest (proximate) saccade
            # - Same motion of the grating during the closest (proximate) saccade

            saccadeIndices = np.where(
                np.vstack([
                    extrastimulusMask,
                    self.saccadeLabels == saccadeLabel,
                    self.gratingMotionDuringSaccades == probeMotion
                ]).all(0)
            )[0]

            t, M = psth2(
                self.saccadeTimestamps[saccadeIndices, 0] + probeLatency,
                unit.timestamps,
                window=responseWindow,
                binsize=binsize
            )
            fr_ = M.mean(0) / dt
            fr.append(fr_)

            #
            t_, M = psth2(
                self.saccadeTimestamps[saccadeIndices, 0] + probeLatency,
                unit.timestamps,
                window=baselineWindow,
                binsize=None
            )
            bl_ = np.mean(M.flatten()) / np.diff(baselineWindow).item()
            bl.append(bl_)

        #
        fr = np.array(fr)
        bl = np.array(bl).reshape(-1, 1)

        return t, fr, bl

    def _computeResponseSetForActualSaccades(
        self,
        unit,
        probeMotion=-1,
        responseWindow=(0, 0.3),
        baselineWindow=(-0.02, 0),
        perisaccadicWindow=(-0.05, 0.1),
        binsize=0.01,
        priorEstimates=None,
        trialIndicesPerisaccadic=None,
        trialIndicesExtrasaccadic=None
        ):
        """
        """

        #
        if priorEstimates is None:
            priorEstimates = dict()
        currentEstimates = dict()
        currentEstimates.update(priorEstimates)

        #
        kwargs = {
            'unit': unit,
            'probeMotion': probeMotion,
            'responseWindow': responseWindow,
            'baselineWindow': baselineWindow,
            'perisaccadicWindow': perisaccadicWindow,
            'binsize': binsize,
            'trialIndicesPerisaccadic': trialIndicesPerisaccadic,
            'trialIndicesExtrasaccadic': trialIndicesExtrasaccadic
        }

        #
        if 'rProbe' not in currentEstimates.keys():
            t, fr, bl = self._measureVisualOnlyResponse(**kwargs)
            currentEstimates['rProbe'] = fr.mean(0) - bl

        #
        if 'rMixed' not in currentEstimates.keys():
            t, fr, bl = self._measurePerisaccadicVisualResponse(**kwargs)
            currentEstimates['rMixed'] = fr.mean(0)

        #
        if 'rSaccade' not in currentEstimates.keys():
            t, fr, bl = self._estimateSaccadeRelatedActivity(**kwargs)
            currentEstimates['rSaccade'] = fr.mean(0)

        #
        if 'rControl' not in currentEstimates.keys():
            currentEstimates['rControl'] = 0

        #
        if 'sigma' not in currentEstimates.keys():
            trialIndices = np.where(self.parseEvents(
                eventName='probe',
                coincident=False,
                eventDirection=probeMotion,
                coincidenceWindow=perisaccadicWindow,
            ))[0]
            mu, sigma = unit.describeWithBootstrap(
                self.probeTimestamps[trialIndices],
                baselineWindowBoundaries=baselineWindow,
                windowSize=np.diff(responseWindow).item()
            )
            currentEstimates['sigma'] = sigma

        return (
            currentEstimates['rProbe'],
            currentEstimates['rMixed'],
            currentEstimates['rSaccade'],
            currentEstimates['rControl'],
            currentEstimates['sigma']
        )

    def _computeResponseSetForFictiveSaccades(
        self,
        unit,
        probeMotion=-1,
        responseWindow=(0, 0.3),
        baselineWindow=(-0.02, 0),
        perisaccadicWindow=(-0.05, 0.1),
        binsize=0.01,
        priorEstimates=None,
        trialIndicesPerisaccadic=None,
        trialIndicesExtrasaccadic=None,
        ):
        """
        """

        # TODO: Implement a property that loads and stores these data
        probeTimestamps = self.load('stimuli/fs/probe/timestamps')
        gratingMotionDuringProbes = self.load('stimuli/fs/probe/motion')
        saccadeTimestamps = self.load('stimuli/fs/saccade/timestamps')
        gratingMotionDuringSaccades = self.load('stimuli/fs/saccade/motion')

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

        #
        if priorEstimates is None:
            priorEstimates = dict()
        currentEstimates = dict()
        currentEstimates.update(priorEstimates)

        # Measure visual-only response
        if 'rProbe' not in currentEstimates.keys():
            t, M = psth2(
                probeTimestamps[probeIndices['es']],
                unit.timestamps,
                window=responseWindow,
                binsize=binsize
            )  
            fr = M.mean(0) / dt
            currentEstimates['rProbe'] = fr

        # Measure peri-saccadic response
        if 'rMixed' not in currentEstimates.keys():
            t_, M = psth2(
                probeTimestamps[probeIndices['ps']],
                unit.timestamps,
                window=responseWindow,
                binsize=binsize
            )
            fr = M.mean(0) / dt
            currentEstimates['rMixed'] = fr

        # Compute saccade-related activity
        if 'rSaccade' not in currentEstimates.keys():
            M = list()
            for saccadeLatency in saccadeLatencies:
                t_, mi = psth2(
                    saccadeTimestamps[saccadeIndices['es']],
                    unit.timestamps,
                    window=np.array(responseWindow) + saccadeLatency,
                    binsize=binsize
                )
                fr = mi.mean(0) / dt
                M.append(fr)
            M = np.array(M)
            currentEstimates['rSaccade'] = M.mean(0)

        # Compute baseline activity prior to saccades
        if 'rControl' not in currentEstimates.keys():
            currentEstimates['rControl'] = 0

        # TODO: Estimate sigma here and add it to the dict
        # TODO: Unpack the dictionary so that the terms are in the expected order

        return tuple(currentEstimates.values())

    def _measureSaccadicModulationValue(
        self,
        unit,
        probeMotion=-1,
        responseWindowSize=0.05,
        baselineWindow=(-0.2, 0),
        perisaccadicWindow=(-0.05, 0.1),
        analyzeFictiveSaccades=False,
        trialIndicesPerisaccadic=None,
        trialIndicesExtrasaccadic=None,
        priorEstimates=None,
        ):
        """
        """

        # Define the response window
        responseWindow = unit.determineResponseWindow(
            probeMotion=probeMotion,
            windowSize=responseWindowSize
        )
        if np.isnan(responseWindow).all():
            return np.nan, dict()

        # Collect the response sets
        if analyzeFictiveSaccades:
            rProbe, rMixed, rSaccade, rControl = self._computeResponseSetForFictiveSaccades(
                unit,
                priorEstimates=priorEstimates
            )
        else:
            rProbe, rMixed, rSaccade, rControl, sigma = self._computeResponseSetForActualSaccades(
                unit,
                probeMotion=probeMotion,
                responseWindow=responseWindow,
                baselineWindow=baselineWindow,
                perisaccadicWindow=perisaccadicWindow,
                binsize=None,
                priorEstimates=priorEstimates,
                trialIndicesPerisaccadic=trialIndicesPerisaccadic,
                trialIndicesExtrasaccadic=trialIndicesExtrasaccadic
            )

        # Compute the difference between the observed and expected responses and scale
        rObserved = rMixed
        rExpected = rProbe + rSaccade
        rDiff = (rObserved - rExpected) / sigma

        #
        currentEstimates = {
            'rProbe': rProbe,
            'rMixed': rMixed,
            'rSaccade': rSaccade,
            'rControl': rControl,
            'sigma': sigma
        }

        return round(rDiff.item(), 3), currentEstimates

    def _measureSaccadicModulationLikelihood(
        self,
        unit,
        probeMotion=-1,
        perisaccadicWindow=(-0.05, 0.1),
        nRuns=500,
        priorEstimates=None,
        ):
        """
        """

        #
        if priorEstimates is None:
            _, priorEstimates = self._measureSaccadicModulationValue(
                unit,
                probeMotion=probeMotion,
            )
        if 'rProbe' in priorEstimates.keys():
            rProbe = priorEstimates.pop('rProbe') # Remove this entry so that it will get re-computed

        #
        trialIndicesPerisaccadic = np.where(self.parseEvents(
            eventName='probe',
            coincident=True,
            eventDirection=probeMotion,
            coincidenceWindow=perisaccadicWindow
        ))[0]
        trialIndicesExtrasaccadic = np.where(self.parseEvents(
            eventName='probe',
            coincident=False,
            eventDirection=probeMotion,
            coincidenceWindow=perisaccadicWindow
        ))[0]
        nTrialsForResampling = trialIndicesPerisaccadic.size

        #
        sample = np.full(nRuns, np.nan)
        for iRun in range(nRuns):
            trialIndicesExtrasaccadic_ = np.random.choice(
                trialIndicesExtrasaccadic,
                replace=False,
                size=nTrialsForResampling
            )
            tv, currentEstimates = self._measureSaccadicModulationValue(
                unit,
                probeMotion,
                trialIndicesExtrasaccadic=trialIndicesExtrasaccadic_,
                priorEstimates=priorEstimates
            )
            sample[iRun] = tv

        #
        if sample.mean() < 0:
            mask = sample > 0
        elif sample.mean() >= 0:
            mask = sample < 0
        p = mask.sum() / mask.size * 2

        return p, sample


    def _measureSaccadicModulation(
        self,
        bootstrap=False,
        nRuns=500,
        minimumFiringRate=0.5,
        minimumResponseAmplitude=1,
        ):
        """
        """

        #
        nUnits = self.population.count()
        if self.probeTimestamps is None:
            for probeDirection in ('left', 'right'):
                for datasetName in ('smv', 'smp'):
                    self.save(f'population/metrics/{datasetName}/{probeDirection}', np.full(nUnits, np.nan))
            return

        #
        for probeMotion, probeDirection in zip([-1, 1], ['left', 'right']):

            #
            mvalues = np.full(nUnits, np.nan)
            pvalues = np.full(nUnits, np.nan)

            #
            for iUnit in range(nUnits):

                if iUnit == nUnits - 1:
                    end = None
                else:
                    end = '\r'
                self.log(f'Measuring saccadic modulation for unit {iUnit + 1} out of {nUnits} (motion={probeMotion})', end=end)

                unit = self.population[iUnit]

                #
                if unit.timestamps.size / self.tRange[-1] < minimumFiringRate:
                    continue

                #
                if unit.visualResponseAmplitude[probeDirection] < minimumResponseAmplitude:
                    continue

                #
                m, currentEstimates = self._measureSaccadicModulationValue(
                    unit,
                    probeMotion=probeMotion
                )
                if np.isnan(m):
                    continue
                mvalues[iUnit] = m

                #
                if bootstrap:
                    p, sample = self._measureSaccadicModulationLikelihood(
                        unit,
                        probeMotion=probeMotion,
                        nRuns=nRuns,
                        priorEstimates=currentEstimates
                    )
                    pvalues[iUnit] = p

            #
            self.save(f'population/metrics/smv/{probeDirection}', mvalues)
            self.save(f'population/metrics/smp/{probeDirection}', pvalues)

        return

class SaccadicModulationAnalysis():
    """
    """

    def measureSaccadicModulation(
        self,
        table,
        probeDirection='left',
        responseWindow=(0, 0.3),
        baselineWindow=(-0.2, 0),
        standardize=False,
        pMax=None,
        rMin=None,
        ):
        """
        """

        # Read the table
        file = h5py.File(table, 'r')

        # Load datasets
        rProbe = np.array(file[f'rProbe/{probeDirection}'])
        rMixed = np.array(file[f'rMixed/{probeDirection}'])
        rSaccade = np.array(file[f'rSaccade/{probeDirection}'])
        sigmas = np.array(file[f'sigma/{probeDirection}'])
        pvalues = np.array(file[f'pZeta/{probeDirection}'])
        # dates = np.array(file['date'])
        # clusters = np.array(file['unitNumber'])

        #
        tPoints = file[f'rProbe/{probeDirection}'].attrs['t']
        binEdges = file[f'rMixed/{probeDirection}'].attrs['edges']
        binCenters = binEdges.mean(1)

        #
        nBins = rMixed.shape[1]
        nUnits = rProbe.shape[0]
        data = np.full([nUnits, nBins], np.nan)

        #
        for iUnit in range(nUnits):

            #
            if pMax is not None and pvalues[iUnit] > pMax:
                continue

            # Find the maximum of the PETH
            rProbe_ = rProbe[iUnit]
            tMask = np.logical_and(
                tPoints >= responseWindow[0],
                tPoints <= responseWindow[1]
            )
            tOffset = np.sum(tPoints < responseWindow[0])
            tMax = tPoints[np.argmax(rProbe_[tMask]) + tOffset]

            # Get the baseline FR for the visual-only PETH
            tMask = np.logical_and(
                tPoints >= baselineWindow[0],
                tPoints <= baselineWindow[1]
            )
            rBaseline = rProbe_[tMask].mean()

            #
            if rMin is not None and np.interp(tMax, tPoints, rProbe_ - rBaseline) < rMin:
                continue

            #
            for iBin in range(nBins):

                # Compute the difference between the expected and observed responses
                rObserved = np.interp(tMax, tPoints, rMixed[iUnit][iBin])
                rExpected = np.interp(
                    tMax,
                    tPoints,
                    rSaccade[iUnit][iBin] + (rProbe_ - rBaseline)
                )
                if standardize:
                    factor = sigmas[iUnit]
                    if factor == 0:
                        continue
                    dR = (rObserved - rExpected) / factor
                else:
                    dR = (rObserved - rExpected)
                data[iUnit, iBin] = dR

        # Close the opened file
        file.close()

        #
        data = np.delete(
            data,
            np.isnan(data).all(axis=1),
            axis=0
        )

        return binCenters, data