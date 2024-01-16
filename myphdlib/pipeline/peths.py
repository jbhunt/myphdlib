import numpy as np
from myphdlib.general.toolkit import psth2

def _getPerisaccadicEpochs(
    leftEdge=-0.5,
    rightEdge=0.5,
    binsize=0.1,
    ):
    """
    """

    leftEdges = np.arange(leftEdge, rightEdge, binsize)
    rightEdges = leftEdges + binsize
    binEdges = np.vstack([leftEdges, rightEdges]).T
    return binEdges

def _loadEventData(
    session,
    protocol='dg'
    ):
    """
    """

    #
    if protocol == 'dg':
        probeData = (
            session.probeTimestamps,
            session.probeLatencies,
            session.gratingMotionDuringProbes,
            session.load('stimuli/dg/probe/dos')
        )
        saccadeData = (
            session.saccadeTimestamps[:, 0],
            session.saccadeLatencies,
            session.saccadeLabels,
            session.gratingMotionDuringSaccades
        )

    #
    elif protocol == 'fs':

        # Load datasets
        probeTimestamps = session.load('stimuli/fs/probe/timestamps')
        gratingMotionDuringProbes = session.load('stimuli/fs/probe/motion')
        saccadeTimestamps = session.load('stimuli/fs/saccade/timestamps')
        gratingMotionDuringSaccades = session.load('stimuli/fs/saccade/motion')

        #
        if session.eye == 'left':
            saccadeLabels = gratingMotionDuringSaccades * -1
        else:
            saccadeLabels = gratingMotionDuringSaccades

        # Parse probes
        probeLatencies = list()
        saccadeLabelsProximate = list()
        for trialIndex, probeTimestamp in enumerate(probeTimestamps):
            saccadeTimestampsRelative = probeTimestamp - saccadeTimestamps
            closestSaccadeIndex = np.argmin(np.abs(saccadeTimestampsRelative))
            saccadeLabelsProximate.append(saccadeLabels[closestSaccadeIndex])
            probeLatency = probeTimestamp - saccadeTimestamps[closestSaccadeIndex]
            probeLatencies.append(probeLatency)

        # Parse saccades
        saccadeLatencies = list()
        for trialIndex, saccadeTimestamp in enumerate(saccadeTimestamps):
            probeTimestampsRelative = saccadeTimestamp - probeTimestamps
            closestProbeIndex = np.argmin(np.abs(probeTimestampsRelative))
            saccadeLatency = saccadeTimestamp - probeTimestamps[closestProbeIndex]
            saccadeLatencies.append(saccadeLatency)

        #
        probeData = (
            probeTimestamps,
            np.array(probeLatencies),
            gratingMotionDuringProbes,
            np.array(saccadeLabelsProximate),
        )
        saccadeData = (
            saccadeTimestamps,
            np.array(saccadeLatencies),
            saccadeLabels,
            gratingMotionDuringSaccades
        )

    return probeData, saccadeData

def _getResponseTemplatesForSaccades(
    unit,
    probeData=None,
    saccadeData=None,
    responseWindow=(-0.2, 0.5),
    perisaccadicWindow=(-0.05, 0.1),
    binsize=0.02,
    protocol='dg'
    ):
    """
    """

    # Load event data (if necessary)
    if any([probeData is None, saccadeData is None]):
        probeData, saccadeData = _loadEventData(unit.session, protocol=protocol)
    
    # Unpack event data
    probeTimestamps, probeLatencies, gratingMotionDuringProbes, saccadeLabelsProximate = probeData
    saccadeTimestamps, saccadeLatencies, saccadeLabels, gratingMotionDuringSaccades = saccadeData

    #
    tBins, nTrials, nBins = psth2(
        np.array([0]),
        np.array([0]),
        window=responseWindow,
        binsize=binsize,
        returnShape=True,
    )

    #
    templates = {
        ('left', 'nasal'): np.full(nBins, np.nan),
        ('left', 'temporal'): np.full(nBins, np.nan),
        ('right', 'nasal'): np.full(nBins, np.nan),
        ('right', 'temporal'): np.full(nBins, np.nan),
    }

    #
    peristimulusWindow = (
        perisaccadicWindow[1] * -1,
        perisaccadicWindow[0] * -1
    )
    extrastimulusMask = np.logical_or(
        saccadeLatencies < peristimulusWindow[0],
        saccadeLatencies > perisaccadicWindow[1]
    )

    #
    for gratingDirection, saccadeDirection in templates.keys():
        gratingMotion = -1 if gratingDirection == 'left' else +1
        saccadeLabel = -1 if saccadeDirection == 'temporal' else +1
        trialIndices = np.where(
            np.vstack([
                extrastimulusMask,
                gratingMotionDuringSaccades == gratingMotion,
                saccadeLabels == saccadeLabel
            ]).all(0)
        )[0]
        if trialIndices.size == 0:
            continue
        t, fr = unit.peth(
            saccadeTimestamps[trialIndices],
            responseWindow=responseWindow,
            binsize=binsize
        )
        templates[(gratingDirection, saccadeDirection)] = fr

    return templates

class TimeHistogramProcessingMixin():
    """
    """

    def _extractVisualOnlyPeths(
        self,
        responseWindow=(-0.2, 0.5),
        perisaccadicWindow=(-0.05, 0.1),
        binsize=0.02,
        protocol='dg',
        overwrite=True,
        ):
        """
        """

        #
        tBins, nTrials, nBins = psth2(
            np.array([0]),
            np.array([0]),
            window=responseWindow,
            binsize=binsize,
            returnShape=True,
        )
        nUnits = self.population.count()

        #
        metadata = {
            't': tBins,
            'binsize': binsize,
        }

        # Select event data
        probeData, saccadeData = _loadEventData(
            self,
            protocol=protocol
        )
        probeTimestamps, probeLatencies, gratingMotionDuringProbes, saccadelabelsProximate = probeData
        saccadeTimestamps, saccadeLatencies, saccadelabels, gratingMotionDuringSacccades = saccadeData

        #
        for probeMotion, probeDirection in zip([-1, 1], ['left', 'right']):

            #
            datasetPath = f'peths/rProbe/{protocol}/{probeDirection}'
            if self.hasDataset(datasetPath) and overwrite == False:
                continue
            peths = np.full([nUnits, nBins], np.nan)

            #
            if self.probeTimestamps is None:
                self.save(datasetPath, peths, metadata=metadata)
                continue

            # Identify extra-saccadic trials
            trialIndices = np.where(np.logical_and(
                np.logical_or(
                    probeLatencies < perisaccadicWindow[0],
                    probeLatencies > perisaccadicWindow[1]
                ),
                gratingMotionDuringProbes == probeMotion
            ))[0]

            #
            for iUnit, unit in enumerate(self.population):

                #
                if iUnit + 1 == nUnits:
                    end = None
                else:
                    end = '\r'
                self.log(f'Extracting visual-only PSTHs for unit {iUnit + 1} out of {nUnits} (motion={probeMotion}, protocol={protocol})', end=end)    

                t, fr = unit.peth(
                    probeTimestamps[trialIndices],
                    responseWindow=responseWindow,
                    binsize=binsize
                )
                peths[iUnit] = fr
            self.save(datasetPath, peths, metadata=metadata)

        return

    def _extractSaccadeOnlyPeths(
        self,
        responseWindow=(-0.2, 0.5),
        perisaccadicWindow=(-0.05, 0.1),
        binsize=0.02,
        protocol='dg',
        overwrite=True,
        ):
        """
        """

        #
        tBins, nTrials, nBins = psth2(
            np.array([0]),
            np.array([0]),
            window=responseWindow,
            binsize=binsize,
            returnShape=True,
        )
        nUnits = self.population.count()

        #
        metadata = {
            't': tBins,
            'binsize': binsize,
        }

        # Select event data
        probeData, saccadeData = _loadEventData(
            self,
            protocol=protocol
        )
        probeTimestamps, probeLatencies, gratingMotionDuringProbes, saccadelabelsProximate = probeData
        saccadeTimestamps, saccadeLatencies, saccadelabels, gratingMotionDuringSacccades = saccadeData

        #
        peristimulusWindow = np.array([
            perisaccadicWindow[1] * -1,
            perisaccadicWindow[0] * -1
        ])

        #
        for saccadeLabel, saccadeDirection in zip([-1, 1], ['temporal', 'nasal']):

            #
            datasetPath = f'peths/rSaccade/{protocol}/{saccadeDirection}'
            if self.hasDataset(datasetPath) and overwrite == False:
                continue
            peths = np.full([nUnits, nBins], np.nan)

            #
            if self.probeTimestamps is None:
                self.save(datasetPath, peths, metadata=metadata)
                continue

            # Identify extra-saccadic trials
            trialIndices = np.where(np.logical_and(
                np.logical_or(
                    saccadeLatencies < peristimulusWindow[0],
                    saccadeLatencies > peristimulusWindow[1]
                ),
                saccadelabels == saccadeLabel
            ))[0]

            #
            for iUnit, unit in enumerate(self.population):

                #
                if iUnit + 1 == nUnits:
                    end = None
                else:
                    end = '\r'
                self.log(f'Extracting saccade-only PSTHs for unit {iUnit + 1} out of {nUnits} (direction={saccadeDirection}, protocol={protocol})', end=end)   

                t, fr = unit.peth(
                    saccadeTimestamps[trialIndices],
                    responseWindow=responseWindow,
                    binsize=binsize
                )
                peths[iUnit] = fr
            
            #
            self.save(datasetPath, peths, metadata=metadata)

        return

    def _extractPerisaccadicPeths(
        self,
        responseWindow=(-0.2, 0.5),
        perisaccadicWindow=(-0.05, 0.1),
        binsize=0.02,
        binEdges=None,
        protocol='dg',
        overwrite=True
        ):
        """
        """

        #
        if binEdges is None:
            binEdges = _getPerisaccadicEpochs()

        #
        tBins, nTrials, nBins = psth2(
            np.array([0]),
            np.array([0]),
            window=responseWindow,
            binsize=binsize,
            returnShape=True,
        )
        nUnits = self.population.count()

        #
        metadata = {
            't': tBins,
            'binsize': binsize,
            'edges': binEdges,
        }

        # Select event data
        probeData, saccadeData = _loadEventData(
            self,
            protocol=protocol
        )
        probeTimestamps, probeLatencies, gratingMotionDuringProbes, saccadelabelsProximate = probeData
        saccadeTimestamps, saccadeLatencies, saccadelabels, gratingMotionDuringSacccades = saccadeData

        #
        for probeMotion, probeDirection in zip([-1, 1], ['left', 'right']):

            #
            peths = np.full([nUnits, nBins, binEdges.shape[0]], np.nan)
            datasetPath = f'peths/rMixed/{protocol}/{probeDirection}'
            if self.hasDataset(datasetPath) and overwrite == False:
                continue

            #
            if self.probeTimestamps is None:
                self.save(datasetPath, peths, metadata=metadata)
                continue

            #
            for iUnit, unit in enumerate(self.population):

                #
                if iUnit + 1 == nUnits:
                    end = None
                else:
                    end = '\r'
                self.log(f'Extracting peri-saccadic PSTHs for unit {iUnit + 1} out of {nUnits} (motion={probeMotion}, protocol={protocol})', end=end)   
                
                for iBin, (leftEdge, rightEdge) in enumerate(binEdges):

                    #
                    perisaccadicWindow = (leftEdge, rightEdge)
                    trialIndices = np.where(np.logical_and(
                        np.logical_and(
                            probeLatencies >= perisaccadicWindow[0],
                            probeLatencies <= perisaccadicWindow[1]
                        ),
                        gratingMotionDuringProbes == probeMotion
                    ))[0]

                    #
                    t, fr = unit.peth(
                        probeTimestamps[trialIndices],
                        responseWindow=responseWindow,
                        binsize=binsize
                    )
                    peths[iUnit, :, iBin] = fr
            
            #
            self.save(datasetPath, peths, metadata=metadata)

        return

    def _extractLatencyShiftedSaccadePeths(
        self,
        binEdges=None,
        responseWindow=(-0.2, 0.5),
        perisaccadicWindow=(-0.05, 0.1),
        binsize=0.02,
        protocol='dg',
        overwrite=True
        ):
        """
        """

        #
        if binEdges is None:
            binEdges = _getPerisaccadicEpochs()

        #
        tBins, nTrials, nBins = psth2(
            np.array([0]),
            np.array([0]),
            window=responseWindow,
            binsize=binsize,
            returnShape=True,
        )
        nUnits = self.population.count()

        #
        metadata = {
            't': tBins,
            'binsize': binsize,
            'edges': binEdges,
        }

        # Select event data
        probeData, saccadeData = _loadEventData(
            self,
            protocol=protocol
        )
        probeTimestamps, probeLatencies, gratingMotionDuringProbes, saccadelabelsProximate = probeData
        saccadeTimestamps, saccadeLatencies, saccadelabels, gratingMotionDuringSacccades = saccadeData

        #
        for probeMotion, probeDirection in zip([-1, 1], ['left', 'right']):

            #
            peths = np.full([nUnits, nBins, binEdges.shape[0]], np.nan)
            datasetPath = f'peths/rSaccade/{protocol}/{probeDirection}'
            if self.hasDataset(datasetPath) and overwrite == False:
                continue

            #
            if self.probeTimestamps is None:
                self.save(datasetPath, peths, metadata=metadata)
                continue

            #
            for iUnit, unit in enumerate(self.population):

                #
                if iUnit + 1 == nUnits:
                    end = None
                else:
                    end = '\r'
                self.log(f'Extracting latency-shifted saccade PSTHs for unit {iUnit + 1} out of {nUnits} (motion={probeMotion}, protocol={protocol})', end=end)   

                #
                templates = _getResponseTemplatesForSaccades(
                    unit,
                    probeData,
                    saccadeData,
                    responseWindow=responseWindow,
                    perisaccadicWindow=perisaccadicWindow,
                    binsize=binsize
                )
                
                for iBin, (leftEdge, rightEdge) in enumerate(binEdges):

                    #
                    trialIndices = np.where(np.logical_and(
                        np.logical_and(
                            probeLatencies >= leftEdge,
                            probeLatencies <= rightEdge
                        ),
                        gratingMotionDuringProbes == probeMotion
                    ))[0]
                    if trialIndices.size == 0:
                        peths[iUnit, :, iBin] = np.full(nBins, np.nan)
                        continue
                
                    #
                    curves = list()
                    iterable = zip(
                        gratingMotionDuringProbes[trialIndices],
                        probeLatencies[trialIndices],
                        saccadelabelsProximate[trialIndices]
                    )
                    for probeMotion_, probeLatency, saccadeLabel in iterable:
                        probeDirection = 'left' if probeMotion_ == -1 else 'right'
                        saccadeDirection = 'temporal' if saccadeLabel == -1 else 'nasal'
                        fp = templates[(probeDirection, saccadeDirection)]
                        xp = tBins
                        x = tBins + probeLatency
                        curve = np.interp(x, xp, fp)
                        curves.append(curve)
                    curves = np.array(curves)
                    peths[iUnit, :, iBin] = np.nanmean(curves, axis=0)
            
            #
            self.save(datasetPath, peths, metadata=metadata)

        return

    def _extractVisualOnlyPethsWithResampling(
        self,
        nRuns=5000,
        rTrials=0.05,
        minimumTrialCount=30,
        responseWindow=(-0.2, 0.5),
        perisaccadicWindow=(-0.05, 0.1),
        binsize=0.02,
        protocol='dg',
        overwrite=True,
        ):
        """
        """

        #
        tBins, nTrials_, nBins = psth2(
            np.array([0]),
            np.array([0]),
            window=responseWindow,
            binsize=binsize,
            returnShape=True,
        )
        nUnits = self.population.count()

        #
        metadata = {
            't': tBins,
            'binsize': binsize,
        }

        # Select event data
        probeData, saccadeData = _loadEventData(
            self,
            protocol=protocol
        )
        probeTimestamps, probeLatencies, gratingMotionDuringProbes, saccadelabelsProximate = probeData
        saccadeTimestamps, saccadeLatencies, saccadelabels, gratingMotionDuringSacccades = saccadeData

        #
        for probeMotion, probeDirection in zip([-1, 1], ['left', 'right']):

            #
            datasetPath = f'peths/rProbe/{protocol}/{probeDirection}'
            if self.hasDataset(datasetPath) and overwrite == False:
                continue
            peths = np.full([nUnits, nBins, nRuns + 1], np.nan)

            #
            if self.probeTimestamps is None:
                self.save(datasetPath, peths, metadata=metadata)
                continue

            # Identify extra-saccadic trials
            trialIndices = np.where(np.logical_and(
                np.logical_or(
                    probeLatencies < perisaccadicWindow[0],
                    probeLatencies > perisaccadicWindow[1]
                ),
                gratingMotionDuringProbes == probeMotion
            ))[0]

            # Determine the size of the samples
            nTrials = int(round(trialIndices.size * rTrials))
            if nTrials < minimumTrialCount:
                nTrials = minimumTrialCount

            #
            for iUnit, unit in enumerate(self.population):

                #
                t, fr = unit.peth(
                    probeTimestamps[trialIndices],
                    responseWindow=responseWindow,
                    binsize=binsize
                )
                peths[iUnit, :, 0] = fr

                #
                if iUnit + 1 == nUnits:
                    end = None
                else:
                    end = '\r'
                self.log(f'Extracting visual-only PSTHs for unit {iUnit + 1} out of {nUnits} (motion={probeMotion}, protocol={protocol})', end=end)

                #
                for iRun in range(nRuns):

                    #
                    trialIndicesResampled = np.random.choice(
                        trialIndices,
                        size=nTrials,
                        replace=False
                    )

                    t, fr = unit.peth(
                        probeTimestamps[trialIndicesResampled],
                        responseWindow=responseWindow,
                        binsize=binsize
                    )
                    peths[iUnit, :, iRun + 1] = fr

            self.save(datasetPath, peths, metadata=metadata)

        return

    def _runPethsModule(
        self,
        overwrite=True
        ):
        """
        """

        for protocol in ('dg', 'fs'):
            self._extractVisualOnlyPeths(
                protocol=protocol,
                overwrite=overwrite
            )
            self._extractPerisaccadicPeths(
                protocol=protocol,
                overwrite=overwrite
            )
            self._extractSaccadeOnlyPeths(
                protocol=protocol,
                overwrite=overwrite
            )
            self._extractLatencyShiftedSaccadePeths(
                protocol=protocol,
                overwrite=overwrite
            )

        return