import h5py
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
        self.peths = {
            'recovered': None
        }

        return

    def _computePeths(
        self,
        minimumPeakIndex=23,
        ):
        """
        """

        self.peths['recovered'] = list()
        with h5py.File(self.hdf, 'r') as stream:
            peths = np.array(stream['clustering/peths/normal'])
        for iUnit in np.where(self._intersectUnitKeys(self.ukeys))[0]:
            peth = peths[iUnit, :]
            if np.argmax(np.abs(peth)) < minimumPeakIndex:
                continue
            self.peths['recovered'].append(peth)
        self.peths['recovered'] = np.array(self.peths['recovered'])


        return

    def _identifyUnitsOfInterest(
        self,
        operation='d'
        ):
        """
        """

        self._ukeys = list()
        for session in self.sessions:

            #
            firingRates = session.load('metrics/fr')
            qualityLabels = session.load('metrics/ql')
            passingMetricThresholds = np.vstack([
                session.load('metrics/pr') >= 0.9,
                session.load('metrics/ac') <= 0.1,
                session.load('metrics/rpvr') <= 0.5
            ]).all(0)
            p1 = session.load('zeta/probe/left/p')
            p1[np.isnan(p1)] = 1.0
            p2 = session.load('zeta/probe/right/p')
            p2[np.isnan(p2)] = 1.0
            pZeta = np.min(np.vstack([p1, p2]), axis=0)

            #
            f1 = np.vstack([
                firingRates >= 0.2,
                qualityLabels == 1,
                pZeta < 0.01
            ]).all(0)
            s1 = list()
            for unit in session.population[f1]:
                s1.append(unit.cluster)
            s1 = set(s1)

            # Recovery
            for iUnit in range(len(qualityLabels)):
                if qualityLabels[iUnit] != 1 and passingMetricThresholds[iUnit] == True:
                    qualityLabels[iUnit] = 1

            #
            f2 = np.vstack([
                firingRates >= 0.2,
                qualityLabels == 1,
                pZeta < 0.01
            ]).all(0)
            s2 = list()
            for unit in session.population[f2]:
                s2.append(unit.cluster)
            s2 = set(s2)

            #
            if operation == 'd':
                for cluster in s2.difference(s1):
                    ukey = (
                        str(session.date),
                        session.animal,
                        cluster
                    )
                    self._ukeys.append(ukey)

            #
            elif operation == 'i':
                for cluster in s2.intersection(s1):
                    ukey = (
                        str(session.date),
                        session.animal,
                        cluster
                    )
                    self._ukeys.append(ukey)

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

    def plotPeths(
        self,
        figsize=(4, 5)
        ):
        """
        """

        fig, axs = plt.subplots(ncols=2, sharey=True)
        for j, op in enumerate(('i', 'd')):
            self._identifyUnitsOfInterest(op)
            self._computePeths()
            index = np.argsort([np.argmax(np.abs(y)) for y in self.peths['recovered']])
            axs[j].pcolor(self.peths['recovered'][index], vmin=-0.8, vmax=0.8)

        #
        axs[0].set_ylabel('N units')
        axs[0].set_title('Pass')
        axs[1].set_title('Pass*')
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, axs

    def plotResponseComplexity(
        self,
        figsize=(2, 4)
        ):
        """
        """

        complexity = {
            'i': None,
            'd': None
        }
        for op in ('i', 'd'):
            self._identifyUnitsOfInterest(op)
            self._computePeths()
            complexity[op] = np.abs(self.peths['recovered']).sum(1) / (self.peths['recovered'].shape[1])

        #
        fig, ax = plt.subplots()
        ax.boxplot(
            complexity.values(),
            labels=complexity.keys(),
            widths=0.4,
            medianprops={'color': 'k'},
        )
        ax.set_ylim([0, 1])
        ax.set_ylabel('Complexity index')
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['Pass', 'Pass*'])
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, ax, complexity