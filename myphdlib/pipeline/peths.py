import numpy as np
from myphdlib.general.toolkit import psth2

def _getTimeBins(
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

def _loadFictiveSaccadesEventData(
    session,
    probeMotion=-1,
    ):
    """
    """

    # Load datasets
    probeTimestamps = session.load('stimuli/fs/probe/timestamps')
    gratingMotionDuringProbes = session.load('stimuli/fs/probe/motion')
    saccadeTimestamps = session.load('stimuli/fs/saccade/timestamps')
    gratingMotionDuringSaccades = session.load('stimuli/fs/saccade/motion')

    # Parse probes
    probeLatencies = list()
    for trialIndex, probeTimestamp in enumerate(probeTimestamps):
        if gratingMotionDuringProbes[trialIndex] != probeMotion:
            continue
        saccadeTimestampsRelative = probeTimestamp - saccadeTimestamps
        closestSaccadeIndex = np.argmin(np.abs(saccadeTimestampsRelative))
        probeLatency = probeTimestamp - saccadeTimestamps[closestSaccadeIndex]
        probeLatencies.append(probeLatency)

    # Parse saccades
    saccadeLatencies = list()
    for trialIndex, saccadeTimestamp in enumerate(saccadeTimestamps):
        if gratingMotionDuringSaccades[trialIndex] != probeMotion:
            continue
        probeTimestampsRelative = saccadeTimestamp - probeTimestamps
        closestProbeIndex = np.argmin(np.abs(probeTimestampsRelative))
        saccadeLatency = saccadeTimestamp - probeTimestamps[closestProbeIndex]
        saccadeLatencies.append(saccadeLatency)

    #
    if session.eye == 'left':
        saccadeDirections = gratingMotionDuringSaccades * -1
    else:
        saccadeDirections = gratingMotionDuringSaccades

    return probeTimestamps, np.array(probeLatencies), gratingMotionDuringProbes, saccadeTimestamps, np.array(saccadeLatencies), saccadeDirections

class TimeHistogramProcessingMixin():
    """
    """

    def _extractVisualOnlyPethsDuringDriftingGrating(
        self,
        responseWindow=(-0.2, 0.5),
        perisaccadicWindow=(-0.05, 0.1),
        binsize=0.02,
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

        #
        for probeMotion, probeDirection in zip([-1, 1], ['left', 'right']):

            #
            datasetPath = f'curves/rProbe/actual/{probeDirection}'
            peths = np.full([nUnits, nBins], np.nan)

            #
            if self.probeTimestamps is None:
                self.save(datasetPath, peths, metadata=metadata)
                continue

            # Identify extra-saccadic trials
            trialIndices = np.where(self.parseEvents(
                eventName='probe',
                coincident=False,
                eventDirection=probeMotion,
                coincidenceWindow=perisaccadicWindow,
            ))[0]

            #
            for iUnit, unit in enumerate(self.population):

                #
                if iUnit + 1 == nUnits:
                    end = None
                else:
                    end = '\r'
                self.log(f'Extracting visual-only PSTHs for unit {iUnit + 1} out of {nUnits} (motion={probeMotion})', end=end)    

                t, fr, bl = self._measureVisualOnlyResponse(
                    unit,
                    probeMotion=probeMotion,
                    responseWindow=responseWindow,
                    trialIndicesExtrasaccadic=trialIndices,
                    binsize=binsize,
                )
                peths[iUnit] = fr.mean(0)
            self.save(datasetPath, peths, metadata=metadata)

        return

    def _extractSaccadeOnlyPethsDuringDriftingGrating(
        self,
        responseWindow=(-0.2, 0.5),
        perisaccadicWindow=(-0.05, 0.1),
        binsize=0.02,
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

        for saccadeDirectionCoded, saccadeDirection in zip([-1, 1], ['temporal', 'nasal']):

            #
            datasetPath = f'curves/rSaccadeUnshifted/actual/{saccadeDirection}'
            peths = np.full([nUnits, nBins], np.nan)

            #
            if self.probeTimestamps is None:
                self.save(datasetPath, peths, metadata=metadata)
                continue

            # Identify extra-saccadic trials
            trialIndices = np.where(self.parseEvents(
                eventName='saccade',
                coincident=False,
                eventDirection=saccadeDirectionCoded,
                coincidenceWindow=-1 * np.array(perisaccadicWindow),
            ))[0]

            #
            for iUnit, unit in enumerate(self.population):

                #
                if iUnit + 1 == nUnits:
                    end = None
                else:
                    end = '\r'
                self.log(f'Extracting saccade-only PSTHs for unit {iUnit + 1} out of {nUnits} (direction={saccadeDirection})', end=end)   

                t, fr = unit.peth(
                    self.saccadeTimestamps[trialIndices, 0],
                    responseWindow=responseWindow,
                    binsize=binsize
                )
                peths[iUnit] = fr
            
            #
            self.save(datasetPath, peths, metadata=metadata)

        return

    def _extractPerisaccadicPethsDuringDriftingGrating(
        self,
        responseWindow=(-0.2, 0.5),
        perisaccadicWindow=(-0.05, 0.1),
        binsize=0.02,
        binEdges=None,
        ):
        """
        """

        #
        if binEdges is None:
            binEdges = _getTimeBins()

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

        #
        for probeMotion, probeDirection in zip([-1, 1], ['left', 'right']):

            #
            peths = np.full([binEdges.shape[0], nUnits, nBins], np.nan)
            datasetPath = f'curves/rMixed/actual/{probeDirection}'

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
                self.log(f'Extracting peri-saccadic PSTHs for unit {iUnit + 1} out of {nUnits} (motion={probeMotion})', end=end)   
                
                for iBin, (leftEdge, rightEdge) in enumerate(binEdges):

                    #
                    perisaccadicWindow = (leftEdge, rightEdge)
                
                    t, fr, bl = self._measurePerisaccadicVisualResponse(
                        unit,
                        probeMotion=probeMotion,
                        responseWindow=responseWindow,
                        binsize=binsize,
                        perisaccadicWindow=perisaccadicWindow
                    )
                    peths[iBin, iUnit, :] = fr.mean(0)
            
            #
            self.save(datasetPath, peths, metadata=metadata)

        return

    def _extractLatencyShiftedSaccadePethsDuringDriftingGrating(
        self,
        binEdges=None,
        responseWindow=(-0.2, 0.5),
        perisaccadicWindow=(-0.05, 0.1),
        binsize=0.02,):
        """
        """

        #
        if binEdges is None:
            binEdges = _getTimeBins()

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

        #
        for probeMotion, probeDirection in zip([-1, 1], ['left', 'right']):

            #
            peths = np.full([binEdges.shape[0], nUnits, nBins], np.nan)
            datasetPath = f'curves/rSaccade/actual/{probeDirection}'

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
                self.log(f'Extracting latency-shifted saccade PSTHs for unit {iUnit + 1} out of {nUnits} (motion={probeMotion})', end=end)   
                
                for iBin, (leftEdge, rightEdge) in enumerate(binEdges):

                    #
                    perisaccadicWindow = (leftEdge, rightEdge)
                
                    t, fr, bl = self._estimateSaccadeRelatedActivityWithInterpolation(
                        unit,
                        probeMotion=probeMotion,
                        responseWindow=responseWindow,
                        binsize=binsize,
                        perisaccadicWindow=perisaccadicWindow
                    )
                    peths[iBin, iUnit, :] = fr.mean(0)
            
            #
            self.save(datasetPath, peths, metadata=metadata)

        return
    
    def _extractVisualOnlyPethsDuringFictiveSaccades(
        self,
        responseWindow=(-0.2, 0.5),
        perisaccadicWindow=(-0.05, 0.1),
        binsize=0.02,
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

        #
        probeTimestamps, probeLatencies, gratingMotionDuringProbes, saccadeTimestamps, saccadeLatencies, saccadeDirections = _loadFictiveSaccadesEventData(
            self,
        )

        #
        for probeMotion, probeDirection in zip([-1, 1], ['left', 'right']):

            #
            datasetPath = f'curves/rProbe/fictive/{probeDirection}'
            peths = np.full([nUnits, nBins], np.nan)

            #
            if self.probeTimestamps is None:
                self.save(datasetPath, peths, metadata=metadata)
                continue

            #
            probeIndices = np.where(np.logical_and(
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
                self.log(f'Extracting visual-only PSTHs for unit {iUnit + 1} out of {nUnits} (motion={probeMotion})', end=end)    

                t, fr = unit.peth(
                    probeTimestamps[probeIndices],
                    responseWindow=responseWindow,
                    binsize=binsize
                )
                peths[iUnit] = fr
            self.save(datasetPath, peths, metadata=metadata)

        return
    
    def _extractSaccadeOnlyPethsDuringFictiveSaccades(
        self,
        responseWindow=(-0.2, 0.5),
        perisaccadicWindow=(-0.05, 0.1),
        binsize=0.02,
        ):
        """
        """

        #
        (probeTimestamps,
            probeLatencies,
            gratingMotionDuringProbes,
            saccadeTimestamps,
            saccadeLatencies,
            saccadeDirections
        ) = _loadFictiveSaccadesEventData(self)

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

        for saccadeDirectionCoded, saccadeDirection in zip([-1, 1], ['temporal', 'nasal']):

            #
            datasetPath = f'curves/rSaccadeUnshifted/fictive/{saccadeDirection}'
            peths = np.full([nUnits, nBins], np.nan)

            #
            if self.probeTimestamps is None:
                self.save(datasetPath, peths, metadata=metadata)
                continue

            #
            saccadeIndices = np.where(np.logical_and(
                np.logical_or(
                    probeLatencies < -1 * perisaccadicWindow[0],
                    probeLatencies > -1 * perisaccadicWindow[1]
                ),
                saccadeDirections == saccadeDirectionCoded
            ))[0]

            #
            for iUnit, unit in enumerate(self.population):

                #
                if iUnit + 1 == nUnits:
                    end = None
                else:
                    end = '\r'
                self.log(f'Extracting saccade-only PSTHs for unit {iUnit + 1} out of {nUnits} (direction={saccadeDirection})', end=end)   

                t, fr = unit.peth(
                    self.saccadeTimestamps[trialIndices, 0],
                    responseWindow=responseWindow,
                    binsize=binsize
                )
                peths[iUnit] = fr
            
            #
            self.save(datasetPath, peths, metadata=metadata)

        return

        return
    
    def _extractPerisaccadicPethsDuringFictiveSaccades(
        self,
        responseWindow=(-0.2, 0.5),
        perisaccadicWindow=(-0.05, 0.1),
        binsize=0.02,
        binEdges=None,
        ):
        """
        """

        return
    
    def _extractLatencyShiftedSaccadePethsDuringFictiveSaccades(
        self,
        binEdges=None,
        responseWindow=(-0.2, 0.5),
        perisaccadicWindow=(-0.05, 0.1),
        binsize=0.02,):
        """
        """

        return

    def _runPethsModule(
        self,
        ):
        """
        """

        self._extractVisualOnlyPethsDuringDriftingGrating()
        self._extractSaccadeOnlyPethsDuringDriftingGrating()
        self._extractPerisaccadicPethsDuringDriftingGrating()
        self._extractLatencyShiftedSaccadePethsDuringDriftingGrating()

        return