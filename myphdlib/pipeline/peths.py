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

class TimeHistogramProcessingMixin():
    """
    """

    def _extractVisualOnlyPeths(
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
            datasetPath = f'curves/rProbe/{probeDirection}'
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

    def _extractSaccadeOnlyPeths(
        self,
        responseWindow=(-0.2, 0.5),
        perisaccadicWindow=(-0.05, 0.1),
        binsize=0.02,):
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
            datasetPath = f'curves/rSaccadeUnshifted/{saccadeDirection}'
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

    def _extractPerisaccadicPeths(
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
            datasetPath = f'curves/rMixed/{probeDirection}'

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

    def _extractLatencyShiftedSaccadePeths(
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
            datasetPath = f'curves/rSaccade/{probeDirection}'

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

    def _runPethsModule(
        self,
        ):
        """
        """

        self._extractVisualOnlyPeths()
        self._extractSaccadeOnlyPeths()
        self._extractPerisaccadicPeths()
        self._extractLatencyShiftedSaccadePeths()

        return