import numpy as np
from myphdlib.general.toolkit import psth2

class TuningProcessingMixin(object):
    """
    """

    def _extractReceptiveFields(
        self,
        responseWindow=(0, 0.2),
        baselineWindow=(-0.1, 0),
        ):
        """
        """

        #
        self.log(f'Extracting sparse noise responses')

        #
        if self.hasDataset('stimuli/sn/pre') == False:
            return

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
            self.save(f'population/rf/{block}', heatmaps[block])

        return

    def _runTuningModule(
        self,
        ):
        """
        """

        return