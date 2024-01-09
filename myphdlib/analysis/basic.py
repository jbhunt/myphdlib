import numpy as np
from zetapy import ifr
from myphdlib.general.toolkit import psth2, smooth
from scipy.ndimage import gaussian_filter1d as smooth2

class BasicResponseCharacterizationAnalysisMixin():
    """
    """

    def _measureVisualResponseLatencies(
        self,
        responseWindow=(0, 0.5),
        binsize=0.001,
        sigma=0.005,
        minimumSpikeRate=0.5,
        ):
        """
        Measure latency to the largest positive and negative peaks in the PSTH
        """

        #
        for probeDirection, probeMotion in zip(['left', 'right'], [-1, 1]):

            #
            responseLatencies = np.full([self.population.count(), 2], np.nan)
            if self.probeTimestamps is None:
                self.save(f'population/metrics/rl/probe/{probeDirection}', responseLatencies)
                continue

            #
            nUnits = self.population.count()
            for iRun, unit in enumerate(self.population):
                
                #
                # Log progress
                if iRun == nUnits - 1:
                    end = None
                else:
                    end = '\r'
                self.log(f'Measuring visual response latencies for {probeDirection}-ward probes ({unit.index + 1} / {nUnits} units)', end=end)

                # ZETA test will fail with 3 or fewer spikes
                if minimumSpikeRate is not None and unit.timestamps.size / self.tRange[-1] < minimumSpikeRate:
                    l1, l2 = np.nan, np.nan

                #
                else:
                    t1, fr1, res = ifr(
                        unit.timestamps,
                        self.probeTimestamps[self.gratingMotionDuringProbes == probeMotion],
                        dblUseMaxDur=responseWindow[1],
                    )
                    if fr1.size == 0:
                        l1, l2 = np.nan, np.nan
                    else:
                        t2 = np.arange(0, responseWindow[1] + binsize, binsize)
                        fr2 = np.interp(t2, t1, fr1)
                        fr3 = smooth2(fr2, sigma / binsize)
                        l1 = round(t2[np.argmax(fr3)], 3)
                        l2 = round(t2[np.argmin(fr3)], 3)

                #
                responseLatencies[unit.index, 0] = l1
                responseLatencies[unit.index, 1] = l2

            self.save(f'population/metrics/vrl/{probeDirection}', responseLatencies)

        return

    def _measureVisualResponseAmplitude(
        self,
        responseWindow=(-0.025, 0.025),
        baselineWindow=(-0.2, 0),
        ):
        """
        Measure the response amplitude for the largest positive and negative peaks in the PSTH
        """

        nUnits = self.population.count()
        for probeDirection, probeMotion in zip(['left', 'right'], [-1, 1]):

            #
            responseAmplitudes = np.full([nUnits, 2], np.nan)
            if self.probeTimestamps is None:
                self.save(f'population/metrics/vra/{probeDirection}', responseAmplitudes)
                continue

            #
            responseLatencies = self.load(f'population/metrics/vrl/{probeDirection}')
            probeTimestamps = self.probeTimestamps[self.gratingMotionDuringProbes == probeMotion]

            for iRun, unit in enumerate(self.population):

                # Log progress
                if iRun == nUnits - 1:
                    end = None
                else:
                    end = '\r'
                self.log(f'Measuring visual response amplitudes for {probeDirection}-ward probes ({unit.index + 1} / {nUnits} units)', end=end)

                # Estimate baseline level of activity
                bl, sigma = unit.describeAcrossTrials(
                    probeTimestamps,
                    baselineWindow,
                )

                # Measure the response amplitude for the positive and negative components of the PSTH
                for columnIndex, responseLatency in zip([0, 1], responseLatencies[unit.index]):
                    responseWindowCentered = np.around(np.array(responseWindow) + responseLatency, 3)
                    fr, sigma = unit.describeAcrossTrials(
                        probeTimestamps,
                        responseWindowCentered
                    )
                    responseAmplitudes[unit.index, columnIndex] = abs(fr - bl)
                
            #
            self.save(f'population/metrics/vra/{probeDirection}', responseAmplitudes)

        return

    def _measureVisualResponseSign(
        self,
        ):
        """
        Determine whether the PSTH is dominated by a positive or negative component
        """

        self.log(f'Measuring dominant sign of the response to the probe stimulus')

        nUnits = self.population.count()
        for probeDirection, probeMotion in zip(['left', 'right'], [-1, 1]):
            responseSignIndices = np.full(nUnits, np.nan)
            if self.probeTimestamps is None:
                self.save(f'population/metrics/rsi/{probeDirection}', responseSignIndices)
                continue
            responseAmplitudes = self.load(f'population/metrics/vra/{probeDirection}')
            for unitIndex, unit in enumerate(self.population):
                rPositive, rNegative = responseAmplitudes[unitIndex]
                if np.isnan([rPositive, rNegative]).all():
                    continue
                responseSignIndex = (rPositive - rNegative) / (rPositive + rNegative)
                responseSignIndices[unitIndex] = responseSignIndex
            self.save(f'population/metrics/rsi/{probeDirection}', responseSignIndices)

        return

    def _measureVisualResponsePreference(
        self,
        ):
        """
        Determine which direction of motion is preferred by each unit
        """

        self.log(f'Measuring preference for direction of motion of the probe stimulus')

        nUnits = self.population.count()
        responsePreferenceIndices = np.full(nUnits, np.nan)
        if self.probeTimestamps is None:
            self.save(f'population/metrics/rpi', responsePreferenceIndices)
            return

        responseSignIndices = np.hstack([
            self.load(f'population/metrics/rsi/left').reshape(-1, 1),
            self.load(f'population/metrics/rsi/right').reshape(-1, 1)
        ])
        columnIndices = np.around((np.nanmean(responseSignIndices, axis=1) + 1) / 2, 0).astype(int)
        responseAmplitudesLeft = self.load(f'population/metrics/vra/left')
        responseAmplitudesRight = self.load(f'population/metrics/vra/right')
        responseAmplitudesPreferred = list()
        for rowIndex, columnIndex in enumerate(columnIndices):
            if np.isnan(columnIndex):
                responseAmplitudesPreferred.append(np.nan)
            elif columnIndex == 0:
                responseAmplitudesPreferred.append(responseAmplitudesLeft[rowIndex])
            else:
                responseAmplitudesPreferred.append(responseAmplitudesRight[rowIndex])
        responseAmplitudesPreferred = np.array(responseAmplitudesPreferred)
        for iUnit in range(nUnits):
            rLeft, rRight = responseAmplitudesPreferred[iUnit]
            if np.isnan([rLeft, rRight]).any():
                continue
            if rRight - rLeft == 0:
                responsePreferenceIndex = 0
            else:
                responsePreferenceIndex = (rRight - rLeft) / (rRight + rLeft)
            responsePreferenceIndices[iUnit] = responsePreferenceIndex
        self.save(f'population/metrics/rpi', np.around(responsePreferenceIndices, 2))

        return

    def _measureVisualResponseProperties(
        self,
        minimumSpikeRate=0.5,
        ):
        """
        """

        self._measureVisualResponseLatencies(
            minimumSpikeRate=minimumSpikeRate
        )
        self._measureVisualResponseAmplitude()
        self._measureVisualResponseSign()
        self._measureVisualResponsePreference()

        return

    def _measureLuminancePolarity(
        self,
        responseWindow=(0, 0.3),
        baselineWindow=(-0.2, 0),
        minimumFiringRate=0.5,
        ):
        """
        """

        #
        spotTimestamps = list()
        spotPolarities = list()
        for phase in ('pre', 'post'):
            spotTimestamps_ = self.load(f'stimuli/sn/{phase}/timestamps')
            spotPolarities_ = self.load(f'stimuli/sn/{phase}/signs')
            if spotTimestamps_ is None:
                continue
            for t in spotTimestamps_:
                spotTimestamps.append(t)
            for p in spotPolarities_:
                spotPolarities.append(p)
        spotTimestamps = np.array(spotTimestamps)
        spotPolarities = np.array(spotPolarities)

        #
        nUnits = self.population.count()
        polarityIndices = np.full(nUnits, np.nan)
        for iUnit, unit in enumerate(self.population):

            #
            if unit.timestamps.size / self.tRange[-1] < minimumFiringRate:
                continue

            # Measure ON response
            trialIndices = np.where(spotPolarities == True)
            t, M = psth2(
                spotTimestamps[trialIndices],
                unit.timestamps,
                window=responseWindow,
                binsize=None
            )
            fr = M.mean(0).item() / np.diff(responseWindow).item()
            t, M = psth2(
                spotTimestamps[trialIndices],
                unit.timestamps,
                window=baselineWindow,
                binsize=None
            )
            bl = M.mean(0).item() / np.diff(baselineWindow).item()
            rOn = np.clip(fr - bl, 0, np.inf)

            # Measure OFF response
            trialIndices = np.where(spotPolarities == False)
            t, M = psth2(
                spotTimestamps[trialIndices],
                unit.timestamps,
                window=responseWindow,
                binsize=None
            )
            fr = M.mean(0).item() / np.diff(responseWindow).item()
            t, M = psth2(
                spotTimestamps[trialIndices],
                unit.timestamps,
                window=baselineWindow,
                binsize=None
            )
            bl = M.mean(0).item() / np.diff(baselineWindow).item()
            rOff = np.clip(fr - bl, 0, np.inf)

            # Compute polarity index
            if rOn + rOff == 0:
                continue
            else:
                polarityIndex = (rOn - rOff) / (rOn + rOff)
                polarityIndices[iUnit] = round(polarityIndex, 2)

        #
        self.save(f'population/metrics/lpi', polarityIndices)

        return

    def _measureDirectionSelectivity(
        self,
        ):
        """
        """

        # Load stimulus metadata
        movingBarOrientations = self.load('stimuli/mb/orientation')
        barOnsetTimestamps = self.load('stimuli/mb/onset/timestamps')
        barOffsetTimestamps = self.load('stimuli/mb/offset/timestamps')
        movingBarTimestamps = np.hstack([
            barOnsetTimestamps.reshape(-1, 1),
            barOffsetTimestamps.reshape(-1, 1)
        ])

        #
        uniqueOrientations = np.unique(movingBarOrientations)
        uniqueOrientations.sort()

        #
        nUnits = self.population.count()
        directionSelectivityIndices = np.full(nUnits, np.nan).astype(float)

        #
        for unitIndex, unit in enumerate(self.population):

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
                        unit.timestamps,
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
            dsi = np.sqrt(np.power(a, 2) + np.power(b, 2))
            directionSelectivityIndices[unitIndex] = dsi

        #
        self.save('population/metrics/dsi', directionSelectivityIndices)

        return

    # TODO: Update this funciton
    def estimateReceptiveField(
        self,
        phase='on',
        responseWindow=(0, 0.5)
        ):
        """
        """

        rfs = list()
        fields = self.load(f'stimuli/sn/pre/fields')
        if fields is None:
            fields = self.load(f'stimuli/sn/post/fields')
        nDots, nRows, nCols = fields.shape

        #
        for unit in self.population:
            stack = np.full([nRows, nCols], 0.0)
            for block in ('pre', 'post'):
                fields = self.load(f'stimuli/sn/{block}/fields')
                if fields is None:
                    continue
                if phase == 'on':
                    timestamps = self.load(f'stimuli/sn/{block}/timestamps')[0::2]
                elif phase == 'off':
                    timestamps = self.load(f'stimuli/sn/{block}/timestamps')[1::2]
                    
                for field, timestamp in zip(fields, timestamps):
                    i, j = np.where(field != -1)
                    t, M = psth2(
                        np.array([timestamp]),
                        unit.timestamps,
                        window=responseWindow,
                        binsize=None,
                    )
                    nSpikes = M.sum()
                    stack[i, j] += nSpikes
            rf = stack / 6
            rfs.append(rf)

        #
        self.save(f'population/rfs', np.array(rfs))

        return

    def _measureVisualOnlyBaselineVariability(
        self,
        baselineWindow=(-0.2, 0),
        perisaccadicWindow=(-0.05, 0.1),
        binsize=0.02,
        nRuns=5,
        ):
        """
        """


        nUnits = self.population.count()

        for probeDirection, probeMotion in zip(['left', 'right'], [-1, 1]):
            sigmas = np.full(nUnits, np.nan)
            if self.probeTimestamps is None:
                self.save(f'population/metrics/sigma/{probeDirection}', sigmas, metadata={'binsize': binsize})
                continue
            trialIndices = np.where(self.parseEvents(
                eventName='probe',
                coincident=False,
                eventDirection=probeMotion,
                coincidenceWindow=perisaccadicWindow
            ))[0]
            for iUnit, unit in enumerate(self.population):
                if iUnit + 1 == nUnits:
                    end = None
                else:
                    end = '\r'
                self.log(f'Measuring visual-only baseline variability for unit {iUnit + 1} out of {nUnits} (motion={probeMotion})', end=end)
                mu, sigma = unit.describeWithBootstrap(
                    self.probeTimestamps[trialIndices],
                    baselineWindowBoundaries=baselineWindow,
                    windowSize=binsize,
                    nRuns=nRuns
                )
                sigmas[unit.index] = round(sigma, 2)
            self.save(f'population/metrics/sigma/{probeDirection}', sigmas, metadata={'binsize': binsize})

        return

    def _runBasicAnalysisModule(self):
        """
        """

        self._measureVisualResponseLatencies()
        self._measureVisualResponseAmplitude()

        return