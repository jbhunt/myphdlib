import numpy as np
from myphdlib.general.toolkit import psth2
from myphdlib.figures.analysis import AnalysisBase
from matplotlib import pyplot as plt

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
        nUnits = len(self.ukeys)
        for iUnit, ukey in enumerate(self.ukeys):
            print(f'Computing PSTH for unit {iUnit} out of {nUnits}', end='\r')
            self.ukey = ukey
            for saccadeLabel, saccadeDirection in zip([1, -1], ['nasal', 'temporal']):
                saccadeIndices = np.where(self.session.saccadeLabels == saccadeLabel)[0]
                self.tSaccade, fr = self.unit.kde(
                    self.session.saccadeTimestamps[saccadeIndices, 0],
                    responseWindow=responseWindow,
                    binsize=binsize
                )
                t, M = psth2(
                    self.session.saccadeTimestamps[saccadeIndices, 0],
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

    def plotPeths(
        self,
        minimumResponseAmplitude=5,
        normalize=True,
        vrange=(-1, 1),
        sortby='latency',
        cmap='viridis',
        figsize=(4, 4),
        ):
        """
        """

        fig, ax = plt.subplots()
        pethsNormed = list()
        sortingData = list()
        for fr in self.ns['psths/pref/real']:
            i = np.argmax(np.abs(fr))
            a = fr[i]
            if abs(a) < minimumResponseAmplitude:
                continue
            if sortby == 'latency':
                sortingData.append(i)
            elif sortby == 'amplitude':
                # a2 = fr[np.logical_and(self.tSaccade >= 0, self.tSaccade < 0.3)].mean()
                sortingData.append(a)
            if normalize:
                pethsNormed.append(fr / a)
            else:
                pethsNormed.append(fr)
        sortedIndex = np.argsort(sortingData)
        pethsNormed = np.array(pethsNormed)

        ax.pcolor(self.tSaccade, np.arange(sortedIndex.size), pethsNormed[sortedIndex], vmin=vrange[0], vmax=vrange[1], cmap=cmap, rasterized=True)
        ax.vlines(0, 0, sortedIndex.size, color='k')
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, [ax,]