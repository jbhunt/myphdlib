import h5py
import numpy as np
from matplotlib import pyplot as plt
from myphdlib.figures.analysis import AnalysisBase, convertGratingMotionToSaccadeDirection
from myphdlib.general.toolkit import psth2, smooth
from matplotlib.gridspec import GridSpec

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

class SupplementaryFigure(AnalysisBase):
    """
    """

    def __init__(self, example=('2023-05-12', 'mlati7', 163), *args, **kwargs):
        """
        """

        self.examples = (example,)

        super().__init__(*args, **kwargs)

        return

    def makeHeatmaps(
        self,
        responseWindow=(-0.2, 0.5),
        perisaccadicWindow=(-0.5, 0.5),
        baselineWindowForSaccades=(-3, -2),
        baselineWindowForProbes=(-0.2, 0),
        binsize=0.02
        ):
        """
        """

        #
        self.ukey = self.examples[0]
        iUnit = self._indexUnitKey(self.ukey)
        gratingMotion = self.preference[self.iUnit]
        binIndicesForSaccadeBaseline = np.logical_and(
            self.tSaccade >= baselineWindowForSaccades[0],
            self.tSaccade <  baselineWindowForSaccades[1]
        )

        # Compute the mixed response
        trialIndicesPeri = np.where(np.vstack([
            self.session.gratingMotionDuringProbes == gratingMotion,
            self.session.probeLatencies > perisaccadicWindow[0],
            self.session.probeLatencies <= perisaccadicWindow[1]
        ]).all(0))[0]
        latencySortedIndex = np.argsort(self.session.probeLatencies[trialIndicesPeri])[::-1]

        #
        tProbe, M = psth2(
            self.session.probeTimestamps[trialIndicesPeri],
            self.unit.timestamps,
            window=responseWindow,
            binsize=binsize,
        )
        M1 = M[latencySortedIndex, :] / binsize

        #
        trialIndicesExtra = np.where(np.vstack([
            self.session.gratingMotionDuringProbes == gratingMotion,
            np.logical_or(
                self.session.probeLatencies <= perisaccadicWindow[0],
                self.session.probeLatencies >  perisaccadicWindow[1]
            ),
        ]).all(0))[0]
        tBaseline, M = psth2(
            self.session.probeTimestamps[trialIndicesExtra],
            self.unit.timestamps,
            window=baselineWindowForProbes,
            binsize=None
        )
        bl = np.mean(M / np.diff(baselineWindowForProbes), axis=0)

        # Compute the latency-shifted saccade response
        M2a, M2b = list(), list()
        saccadeLabels = self.session.load('stimuli/dg/probe/dos')
        for trialIndex in trialIndicesPeri[latencySortedIndex]:
            probeLatency = self.session.probeLatencies[trialIndex]
            saccadeLabel = saccadeLabels[trialIndex]
            saccadeDirection = 'temporal' if saccadeLabel == -1 else 'nasal'
            fp = self.ns[f'psths/{saccadeDirection}/real'][iUnit, :]
            bl = fp[binIndicesForSaccadeBaseline].mean()
            fr = np.interp(
                tProbe + probeLatency,
                self.tSaccade,
                fp,
                left=np.nan,
                right=np.nan
            )
            M2a.append(fr)
            M2b.append(fr - bl)
        M2a = np.array(M2a)
        M2b = np.array(M2b)

        #
        M3 = list()
        for m1, m2 in zip(M1, M2b):
            M3.append(np.clip(m1 - m2, 0, np.inf))
        M3 = np.array(M3)

        return M1, M2a, M3, tProbe

    def makeHeatmaps2(
        self,
        ):
        """
        """

        self.ukey = self.examples[0]
        iUnit = self._indexUnitKey(self.ukey)
        mu, sigma = self.ns['stats/pref/real/extra'][iUnit]
        M1 = self.ns['terms/pref/real/mixed'][iUnit].T - mu
        M2 = self.ns['terms/pref/real/saccade'][iUnit].T
        M3 = self.ns['terms/pref/real/peri'][iUnit].T * sigma

        return M1, M2, M3

    def plot(
        self,
        vmin=-10,
        vmax=120,
        figsize=(6, 3),
        ):
        """
        """

        m1, m2, m3, tProbe = self.makeHeatmaps()
        y = np.arange(m1.shape[0])
        fig, axs = plt.subplots(ncols=3, sharex=True, sharey=True)
        for m, ax in zip([m1, m2, m3], axs):
            ax.pcolor(tProbe, y, m, vmin=vmin, vmax=vmax, cmap='viridis')
        fig.supxlabel('Time from probe (s)', fontsize=10)
        fig.supylabel('Trials (sorted by latency)', fontsize=10)
        titles = (
            r'$R_{Probe, Saccade}$',
            r'$R_{Saccaade}$',
            r'$R_{Probe (Peri)}$'
        )
        for ax, title in zip(axs, titles):
            ax.set_title(title, fontsize=10)
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, axs

class SupplementaryFigure2(AnalysisBase):
    """
    """

    def __init__(self, example=('2023-05-12', 'mlati7', 163), *args, **kwargs):
        """
        """

        self.examples = (example,)

        super().__init__(*args, **kwargs)

        return

    def makeHeatmaps(
        self,
        responseWindow=(-0.2, 0.5),
        perisaccadicWindow=(-0.3, -0.2),
        baselineWindowForSaccades=(-3, -2),
        baselineWindowForProbes=(-0.2, 0),
        binsize=0.02
        ):
        """
        """

        #
        self.ukey = self.examples[0]
        iUnit = self._indexUnitKey(self.ukey)
        gratingMotion = self.preference[self.iUnit]
        binIndicesForSaccadeBaseline = np.logical_and(
            self.tSaccade >= baselineWindowForSaccades[0],
            self.tSaccade <  baselineWindowForSaccades[1]
        )

        # Compute the mixed response
        trialIndicesPeri = np.where(np.vstack([
            self.session.gratingMotionDuringProbes == gratingMotion,
            self.session.probeLatencies > perisaccadicWindow[0],
            self.session.probeLatencies <= perisaccadicWindow[1]
        ]).all(0))[0]
        latencySortedIndex = np.argsort(self.session.probeLatencies[trialIndicesPeri])[::-1]
        sortedLatencies = self.session.probeLatencies[trialIndicesPeri][latencySortedIndex]

        #
        tProbe, M = psth2(
            self.session.probeTimestamps[trialIndicesPeri],
            self.unit.timestamps,
            window=responseWindow,
            binsize=binsize,
        )
        M1a = M[latencySortedIndex, :] / binsize

        #
        trialIndicesExtra = np.where(np.vstack([
            self.session.gratingMotionDuringProbes == gratingMotion,
            np.logical_or(
                self.session.probeLatencies <= perisaccadicWindow[0],
                self.session.probeLatencies >  perisaccadicWindow[1]
            ),
        ]).all(0))[0]
        tBaseline, M = psth2(
            self.session.probeTimestamps[trialIndicesExtra],
            self.unit.timestamps,
            window=baselineWindowForProbes,
            binsize=None
        )
        bl = np.mean(M / np.diff(baselineWindowForProbes), axis=0)
        M1b = np.copy(M1a)
        M1b -= bl

        # Compute the latency-shifted saccade response
        M2a, M2b = list(), list()
        saccadeLabels = self.session.load('stimuli/dg/probe/dos')
        for trialIndex in trialIndicesPeri[latencySortedIndex]:
            probeLatency = self.session.probeLatencies[trialIndex]
            saccadeLabel = saccadeLabels[trialIndex]
            saccadeDirection = 'temporal' if saccadeLabel == -1 else 'nasal'
            fp = self.ns[f'psths/{saccadeDirection}/real'][iUnit, :]
            bl = fp[binIndicesForSaccadeBaseline].mean()
            fr = np.interp(
                tProbe + probeLatency,
                self.tSaccade,
                fp,
                left=np.nan,
                right=np.nan
            )
            M2a.append(fr)
            M2b.append(fr - bl)
        M2a = np.array(M2a)
        M2b = np.array(M2b)

        #
        M3 = list()
        for m1, m2 in zip(M1a, M2b):
            M3.append(np.clip(m1 - m2, 0, np.inf))
        M3 = np.array(M3)

        return M1a, M1b, M2a, M2b, M3, tProbe, sortedLatencies

    def plotExampleShift(
        self,
        responseWindow=(-0.2, 0.5),
        trialIndex=30,
        perisaccadicWindow=(-0.3, -0.2),
        saccadeDirection='nasal',
        binsize=0.02,
        ):
        """
        """

        self.ukey = self.examples[0]
        iUnit = self._indexUnitKey(self.ukey)
        t = np.arange(responseWindow[0], responseWindow[1] + binsize, binsize) + (binsize / 2)
        M1a, M1b, M2a, M2b, M3, tProbe, sortedLatencies = self.makeHeatmaps(
            perisaccadicWindow=perisaccadicWindow
        )
        fig, ax = plt.subplots()
        fp = self.ns[f'psths/{saccadeDirection}/real'][iUnit, :]
        if trialIndex is None:
            trialIndex = np.random.choice(np.arange(sortedLatencies.size), size=1).item()
        y1 = np.interp(t, self.tSaccade, fp)
        y2 = np.interp(t + sortedLatencies[trialIndex], self.tSaccade, fp)

        ax.plot(t, y1, color='k')
        ax.plot(t, y2, color='k')

        return fig, ax

    def plotHeatmaps(
        self,
        perisaccadicWindow=(0, 0.1),
        figsize=(3, 6),
        ):
        """
        """

        M1a, M1b, M2a, M2b, M3, tProbe, sortedLatencies = self.makeHeatmaps(perisaccadicWindow=perisaccadicWindow)
        fig, axs = plt.subplots(nrows=4, sharex=True)
        axs[0].pcolor(tProbe, np.arange(M1b.shape[0]), M1b, vmin=-30, vmax=100, cmap='viridis', rasterized=True)
        for fr in M1b:
            axs[1].plot(tProbe, fr, color='0.7', lw=0.5, alpha=0.5)
        axs[1].plot(tProbe, M1b.mean(0), color='k')
        axs[2].pcolor(tProbe, np.arange(M2b.shape[0]), M2b, vmin=-30, vmax=100, cmap='viridis', rasterized=True)
        for fr in M2b:
            axs[3].plot(tProbe, fr, color='0.7', lw=0.5, alpha=0.5)
        axs[3].plot(tProbe, M2b.mean(0), color='k')
        ylim = [np.inf, -np.inf]
        for ax in (axs[1], axs[3]):
            y1, y2 = ax.get_ylim()
            if y1 < ylim[0]:
                ylim[0] = y1
            if y2 > ylim[1]:
                ylim[1] = y2
        for ax in (axs[1], axs[3]):
            ax.set_ylim(ylim)
        ylabels = ('Trial #', 'FR', 'Trial #', 'FR')
        for ax, yl in zip(axs, ylabels):
            ax.set_ylabel(yl)
        axs[-1].set_xlabel('Time from probe (s)')
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, axs

    def plot(
        self,
        perisaccadicWindow=(-0.3, -0.2),
        baselineWindowForSaccades=(-3, -2),
        nSaccades=30,
        figsize=(5.75, 1.5),
        ):
        """
        """

        #
        # fig, axs = plt.subplots(ncols=3)
        fig = plt.figure()
        gs = GridSpec(2, 3)
        axs = list()
        axs.append(fig.add_subplot(gs[0, 0]))
        axs.append(fig.add_subplot(gs[1, 0]))
        axs.append(fig.add_subplot(gs[:, 1]))
        axs.append(fig.add_subplot(gs[:, 2]))

        # Plot raster and trial-average saccade response
        self.ukey = self.examples[0]
        iUnit = self._indexUnitKey(self.ukey)
        gratingMotion = self.preference[self.iUnit]
        saccadeDirection = convertGratingMotionToSaccadeDirection(gratingMotion, referenceEye=self.session.eye)
        saccadeLabel = -1 if saccadeDirection == 'temporal' else 1
        trialIndices = np.where(self.session.saccadeLabels == saccadeLabel)[0]
        saccadeTimestamps = self.session.saccadeTimestamps[trialIndices, 0]
        t, M, spikeTimestamps = psth2(
            saccadeTimestamps,
            self.unit.timestamps,
            window=(-0.2, 0.5),
            binsize=0.01,
            returnTimestamps=True
        )
        saccadeIndices = np.random.choice(np.arange(len(spikeTimestamps)), size=nSaccades, replace=False)
        for i, j in enumerate(saccadeIndices):
            x = spikeTimestamps[j]
            y = np.full(x.size, i)
            # axs[0].vlines(x, y - 0.4, y + 0.4, color='k', alpha=0.7, linewidth=1)
            # axs[0].vlines(0, i - 0.4, i + 0.4, color='m', alpha=0.7, linewidth=1)
            axs[0].scatter(x, y, color='k', marker='o', s=3, edgecolor='none', alpha=0.3)
            axs[0].scatter(0, i, color='m', marker='o', s=3, edgecolor='none')
        fp = self.ns[f'psths/{saccadeDirection}/real'][iUnit, :]
        y = np.interp(self.tProbe, self.tSaccade, fp)
        binIndicesForSaccadeBaseline = np.logical_and(
            self.tSaccade >= baselineWindowForSaccades[0],
            self.tSaccade <  baselineWindowForSaccades[1]
        )
        bl = fp[binIndicesForSaccadeBaseline].mean()
        y -= bl
        axs[1].plot(self.tProbe, y, color='k')
        axs[0].set_xticks(axs[1].get_xticks())
        axs[0].set_xticklabels([])
        axs[0].set_xlim(axs[1].get_xlim())

        # Plot timing of probe and saccades for target perisaccadic window
        frameTimestamps = self.session.load(f'frames/{self.session.eye}/timestamps')
        eyePosition = self.session.load('pose/filtered')
        saccadeIndices = np.where(np.vstack([
            self.session.saccadeLatencies > perisaccadicWindow[1] * -1,
            self.session.saccadeLatencies <= perisaccadicWindow[0] * -1,
            self.session.saccadeLabels == saccadeLabel
        ]).all(0))[0]
        saccadeIndices = np.random.choice(saccadeIndices, size=nSaccades, replace=False)
        exampleSaccadeIndex = np.random.choice(saccadeIndices, size=1).item()
        saccadeTimestamps = self.session.saccadeTimestamps[saccadeIndices, 0]
        saccadeLatencies = self.session.saccadeLatencies[saccadeIndices]
        x = np.linspace(-0.25, 0.25, 100) + np.mean(perisaccadicWindow)
        for i, t, l in zip(saccadeIndices, saccadeTimestamps, saccadeLatencies):
            w = np.interp(
                x + t + l,
                frameTimestamps,
                eyePosition[:, 0]
            )
            bl = w[np.logical_and(x > (-1 * l - 0.1), x <= (-1 * l))].mean()
            if i == exampleSaccadeIndex:
                color='r'
                alpha=1
            else:
                color='k'
                alpha=0.3
            axs[2].plot(x, w - bl, color=color, alpha=alpha, lw=0.8)
        axs[2].fill_between(
            perisaccadicWindow,
            *axs[2].get_ylim(),
            color='k',
            edgecolor='none',
            alpha=0.1
        )

        # Plot latency-shifted templates and average
        binIndicesForSaccadeBaseline = np.logical_and(
            self.tSaccade >= baselineWindowForSaccades[0],
            self.tSaccade <  baselineWindowForSaccades[1]
        )
        trialIndicesPeri = list()
        for l in saccadeLatencies:
            dt = np.abs(l - (self.session.probeLatencies * -1))
            iTrial = np.nanargmin(dt)
            trialIndicesPeri.append(iTrial)
        trialIndicesPeri = np.array(trialIndicesPeri)
        saccadeLabels = self.session.load('stimuli/dg/probe/dos')
        M = list()
        for saccadeIndex, trialIndex in enumerate(trialIndicesPeri):
            probeLatency = self.session.probeLatencies[trialIndex]
            saccadeLabel = saccadeLabels[trialIndex]
            saccadeDirection = 'temporal' if saccadeLabel == -1 else 'nasal'
            fp = self.ns[f'psths/{saccadeDirection}/real'][iUnit, :]
            bl = fp[binIndicesForSaccadeBaseline].mean()
            fr = np.interp(
                self.tProbe - probeLatency,
                self.tSaccade,
                fp,
                left=np.nan,
                right=np.nan
            )
            if saccadeIndices[saccadeIndex] == exampleSaccadeIndex:
                color='r'
                alpha=1
            else:
                color='0.8'
                alpha=0.3
            axs[3].plot(self.tProbe, fr - bl, color=color, alpha=alpha, lw=0.75)
            M.append(fr - bl)
        M = np.array(M)
        axs[3].plot(self.tProbe, M.mean(0), color='k')

        #
        for ax in (axs[0], axs[1], axs[3]):
            ax.set_xlim([-0.2, 0.5])
            ax.set_xticks([-0.2, 0, 0.5])
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()
        axs[0].set_xticks([])
        fig.subplots_adjust(hspace=0.1)

        return fig, axs