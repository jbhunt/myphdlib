import numpy as np
from matplotlib import pyplot as plt
from myphdlib.figures.analysis import AnalysisBase

class UnitFilteringAnalysis(AnalysisBase):
    """
    """

    def __init__(self, **kwargs):
        """
        """

        super().__init__(**kwargs)
        self.y = None

        return

    def measureUnitSurvival(
        self,
        ):
        """
        """

        y = list()
        
        #
        nUnits = 0
        for session in self.sessions:
            nUnits += len(session.population)
        y.append(nUnits)

        #
        nUnits = 0
        for session in self.sessions:
            firingRates = session.load('metrics/fr')
            nUnits += np.sum(firingRates >= 0.2)
        y.append(nUnits)

        #
        nUnits = 0
        for session in self.sessions:
            firingRates = session.load('metrics/fr')
            pZeta = np.nanmin(np.vstack([
                session.load('zeta/probe/left/p'),
                session.load('zeta/probe/right/p')
            ]), axis=0)
            nUnits += np.sum(np.logical_and(
                firingRates >= 0.2,
                pZeta < 0.01,
            ))
        y.append(nUnits)

        #
        nUnits = 0
        for session in self.sessions:
            firingRates = session.load('metrics/fr')
            qualityLabels = session.load('metrics/ql')
            pZeta = np.nanmin(np.vstack([
                session.load('zeta/probe/left/p'),
                session.load('zeta/probe/right/p')
            ]), axis=0)
            nUnits += np.sum(np.vstack([
                firingRates >= 0.2,
                pZeta < 0.01,
                qualityLabels == 1
            ]).all(0))
        y.append(nUnits)

        #
        nUnits = 0
        for session in self.sessions:
            firingRates = session.load('metrics/fr')
            qualityLabels = session.load('metrics/ql')
            passingMetricThresholds = np.vstack([
                session.load('metrics/pr') >= 0.9,
                session.load('metrics/ac') <= 0.1,
                session.load('metrics/rpvr') <= 0.5
            ]).all(0)
            for iUnit in range(len(qualityLabels)):
                if qualityLabels[iUnit] != 1 and passingMetricThresholds[iUnit] == True:
                    qualityLabels[iUnit] = 1
            pZeta = np.nanmin(np.vstack([
                session.load('zeta/probe/left/p'),
                session.load('zeta/probe/right/p')
            ]), axis=0)
            nUnits += np.sum(np.vstack([
                firingRates >= 0.2,
                qualityLabels == 1,
                pZeta < 0.01
            ]).all(0))
        y.append(nUnits)

        #
        self.y = np.array(y)

        return

    def plotFilteringCurve(
        self,
        figsize=(5, 4)
        ):
        """
        """

        fig, ax = plt.subplots()
        ax.plot(np.arange(len(self.y)) + 1, self.y, color='k')
        ax.set_xticks(np.arange(len(self.y)) + 1)
        ax.set_xticklabels([
            'No filter',
            'FR filter',
            'ZETA test',
            'Manual spike-sorting',
            'Recovery'
        ], rotation=45)
        ax.set_ylabel('N units')
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, ax