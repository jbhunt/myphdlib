import numpy as np
from myphdlib.general.toolkit import psth2
from myphdlib.figures.analysis import AnalysisBase

class SaccadeResponseAnalysis(AnalysisBase):
    """
    """

    def __init__(self, **kwargs):
        """
        """

        super().__init__(event='saccade', **kwargs)

        return
    
    def computeSaccadePeths(
        self,
        responseWindow=(-0.5, 0.5),
        baselineWindow=(-1, -0.5),
        binsize=0.01,
        saccadeType='real',
        ):
        """
        """

        peths = {
            'nasal': list(),
            'temporal': list()
        }
        for ukey in self.ukeys:
            self.ukey = ukey
            for saccadeLabel, saccadeDirection in zip([1, -1], ['nasal', 'temporal']):
                saccadeIndices = np.where(self.session.saccadeLabels == saccadeLabel)[0]
                self.tSaccade, fr = self.unit.kde(
                    self.session.saccadeTimestamps[saccadeIndices, 0],
                    responseWindow=responseWindow,
                    binsize=binsize
                )
                t, M = psth2(
                    self.unit.timestamps,
                    window=baselineWindow,
                    binsize=None
                )
                bl = M.mean(0) / np.diff(baselineWindow)
                fr -= bl
                peths[saccadeDirection].append(fr)
        
        #
        for k in peths.keys():
            peths[k] = np.array(peths[k])

        # Identify the preferred saccade direction
        pethsByPreference = {
            'pref': list(),
            'null': list()
        }
        nUnits = len(peths['nasal'])
        ssi = np.full(nUnits, np.nan)
        for iUnit in range(nUnits):
            rSaccadeNasal = np.max(np.abs(peths['nasal'][iUnit]))
            rSaccadeTemporal = np.max(np.abs(peths['temporal'][iUnit]))
            if rSaccadeNasal > rSaccadeTemporal:
                pethsByPreference['pref'].append(peths['nasal'][iUnit])
                pethsByPreference['null'].append(peths['temporal'][iUnit])
            else:
                pethsByPreference['pref'].append(peths['temporal'][iUnit])
                pethsByPreference['null'].append(peths['nasal'][iUnit])
            ssi[iUnit] = (rSaccadeNasal - rSaccadeTemporal) / (rSaccadeNasal + rSaccadeTemporal)

        #
        for k in ('pref', 'null'):
            self.ns[f'psths/{k}/{saccadeType}'] = pethsByPreference[k]
        self.ns[f'globals/ssi'] = ssi

        return