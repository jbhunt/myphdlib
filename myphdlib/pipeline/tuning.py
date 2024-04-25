import numpy as np
from myphdlib.general.toolkit import psth2

class TuningProcessingMixin(object):
    """
    """

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
        self.save(f'metrics/lpi', polarityIndices)

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
        self.save('metrics/dsi', directionSelectivityIndices)

        return

    def _extractReceptiveFields(
        self,
        responseWindow=(0, 0.2),
        baselineWindow=(-0.1, 0),
        ):
        """
        """

        #
        self.log(f'Extracting sparse noise responses')

        # Load the event timestamps
        spotTimestamps = {
            'on': list(),
            'off': list(),
        }
        stimulusFields = {
            'on': list(),
            'off': list()
        }
        for block in ('pre', 'post'):
            if self.hasDataset(f'stimuli/sn/{block}'):
                stimulusFields_ = self.load(f'stimuli/sn/{block}/fields')
                spotTimestamps_ = self.load(f'stimuli/sn/{block}/timestamps')
                for f, t in zip(stimulusFields_, spotTimestamps_[0::2]):
                    if np.isnan(t):
                        continue
                    spotTimestamps['on'].append(t)
                    stimulusFields['on'].append(f)
                for f, t in zip(stimulusFields_, spotTimestamps_[1::2]):
                    if np.isnan(t):
                        continue
                    spotTimestamps['off'].append(t)
                    stimulusFields['off'].append(f)
        for block in spotTimestamps.keys():
            spotTimestamps[block] = np.array(spotTimestamps[block])
            stimulusFields[block] = np.array(stimulusFields[block])

        #
        if spotTimestamps['on'].size == 0:
            return

        #
        nRows, nCols = stimulusFields['on'][0].shape
        values, counts = np.unique(stimulusFields['on'], axis=0, return_counts=True)
        nTrials = counts[0]
        heatmaps = {
            'on': list(),
            'off': list()
        }
        for unit in self.population:
            hm = {
                'on': np.full([nRows, nCols], 0.0),
                'off': np.full([nRows, nCols], 0.0)
            }
            for block in heatmaps.keys():
                t, M = psth2(
                    spotTimestamps[block],
                    unit.timestamps,
                    window=baselineWindow,
                    binsize=None
                )
                bl = M.flatten() / np.diff(baselineWindow).item()
                mu, sigma = bl.mean(), bl.std()
                if sigma == 0:
                    hm[block] = np.full([nRows, nCols], np.nan)
                    continue
                for t, f in zip(spotTimestamps[block], stimulusFields[block]):
                    t, M = psth2(
                        np.array([t]),
                        unit.timestamps,
                        window=responseWindow,
                        binsize=None
                    )
                    fr = M.flatten().item() / np.diff(responseWindow).item()
                    z = round((fr - mu) / sigma, 2)
                    i, j = np.where(f != -1)
                    hm[block][i, j] += z
                hm[block] /= nTrials
            for block in heatmaps.keys():
                heatmaps[block].append(hm[block])
        for block in heatmaps.keys():
            heatmaps[block] = np.array(heatmaps[block])

        #
        for block in heatmaps.keys():
            self.save(f'rf/{block}', heatmaps[block])

        return

    def _runTuningModule(
        self,
        ):
        """
        """

        self._measureDirectionSelectivity()
        self._measureLuminancePolarity()
        # self._extractReceptiveFields()

        return