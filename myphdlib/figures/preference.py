import h5py
import numpy as np
import pathlib as pl
from matplotlib import pyplot as plt
from myphdlib.general.toolkit import psth2
from myphdlib.figures.analysis import AnalysisBase
from myphdlib.figures.fictive import convertSaccadeDirectionToGratingMotion, convertGratingMotionToSaccadeDirection

class DirectionSectivityAnalysis(AnalysisBase):
    """
    """

    def __init__(self, **kwargs):
        """
        """

        super().__init__(**kwargs)

        return

    def measureDirectionSelectivityForProbes(
        self,
        method='ratio'
        ):
        """
        """

        #
        self.ns['dsi/probe'] = np.full(len(self.ukeys), np.nan)

        #
        for iUnit in range(len(self.ukeys)):

            #
            aPref = self.ns[f'params/pref/real/extra'][iUnit][:, 0]
            aNull = self.ns[f'params/null/real/extra'][iUnit][:, 0]

            # Clip null direction if sign reverses
            if aPref > 0:
                if aNull < 0:
                    aNull = 0
            else:
                if aNull > 0:
                    aNull = 0

            #
            if self.preference[iUnit] == -1:
                aLeft = aPref
                aRight = aNull
            else:
                aLeft = aNull
                aRight = aPref

            # Ratio of difference and sum
            if method == 'ratio':
                dsi = (aRight - aLeft) / (aRight + aLeft)


            # Normalized vector sum
            elif method == 'vector-sum':
                vectors = np.full([2, 2], np.nan)
                vectors[:, 0] = np.array([aPref, aNull]).T
                vectors[:, 1] = np.array([
                    np.pi if self.preference[iUnit] == -1 else 0,
                    0 if self.preference[iUnit] == -1 else np.pi
                ]).T

                # Compute the coordinates of the polar plot vertices
                vertices = np.vstack([
                    vectors[:, 0] * np.cos(vectors[:, 1]),
                    vectors[:, 0] * np.sin(vectors[:, 1])
                ]).T

                # Compute direction selectivity index
                a, b = vertices.sum(0) / vectors[:, 0].sum()
                dsi = np.sqrt(np.power(a, 2) + np.power(b, 2))
            
            #
            self.ns['dsi/probe'][iUnit] = dsi

        return

    def measureDirectionSelectivityForSaccades(
        self,
        method='ratio',
        responseWindow=(-0.2, 0.5),
        ):
        """
        """

        #
        binIndices = np.where(np.logical_and(
            self.tSaccade >= responseWindow[0],
            self.tSaccade <= responseWindow[1]
        ))[0]
        self.ns['dsi/saccade'] = np.full(len(self.ukeys), np.nan)

        #
        for iUnit in range(len(self.ukeys)):

            #
            self.ukey = self.ukeys[iUnit]

            # Determine which saccade is preferred
            aNasal = np.max(np.abs(self.templates['nasal'][iUnit, binIndices]))
            aTemporal = np.max(np.abs(self.templates['temporal'][iUnit, binIndices]))
            if aNasal > aTemporal:
                aPref = aNasal
                aNull = aTemporal
                saccadeDirection = 'nasal'
            else:
                aPref = aTemporal
                aNull = aNasal
                saccadeDirection = 'tempoarl'

            # Clip null direction if sign reverses
            if aPref > 0:
                if aNull < 0:
                    aNull = 0
            else:
                if aNull > 0:
                    aNull = 0

            # Convert saccade direction to probe direction
            probeDirection = convertSaccadeDirectionToGratingMotion(
                saccadeDirection,
                self.session.eye,
            )
            if probeDirection == -1:
                aLeft = aPref
                aRight = aNull
            else:
                aLeft = aNull
                aRight = aPref

            #
            if method == 'ratio':
                dsi = (aRight - aLeft) / (aRight + aLeft)
            
            #
            self.ns['dsi/saccade'][iUnit] = dsi
            
        return

    # TODO: Project vector sum onto the horizontal axis (so that I can compare
    #       DSI from probes/saccades to DSI for the moving bars)
    def measureDirectionSelectivityForMovingBars(
        self,
        method='ratio',
        ):
        """
        Compute DSI for the moving bars stimulus
        """

        self.ns['dsi/bar'] = np.full(len(self.ukeys), np.nan)
        for session in self.sessions:

            #
            self._session = session

            # 
            movingBarOrientations = self.session.load('stimuli/mb/orientation')
            barOnsetTimestamps = self.session.load('stimuli/mb/onset/timestamps')
            barOffsetTimestamps = self.session.load('stimuli/mb/offset/timestamps')
            movingBarTimestamps = np.hstack([
                barOnsetTimestamps.reshape(-1, 1),
                barOffsetTimestamps.reshape(-1, 1)
            ])
            uniqueOrientations = np.unique(movingBarOrientations)
            uniqueOrientations.sort()

            #
            for ukey in self.ukeys:

                #
                if ukey[0] != str(session.date):
                    continue
                self.ukey = ukey

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
                            self.unit.timestamps,
                            window=(0, dt),
                            binsize=None
                        )
                        fr = M.item() / dt
                        amplitudes.append(fr)

                    #
                    vectors[rowIndex, 0] = np.mean(amplitudes)
                    vectors[rowIndex, 1] = np.deg2rad(orientation)

                #
                if method == 'vector-sum':

                    # Compute the coordinates of the polar plot vertices
                    vertices = np.vstack([
                        vectors[:, 0] * np.cos(vectors[:, 1]),
                        vectors[:, 0] * np.sin(vectors[:, 1])
                    ]).T

                    # Compute direction selectivity index
                    a, b = vertices.sum(0) / vectors[:, 0].sum()
                    dsi = np.sqrt(np.power(a, 2) + np.power(b, 2))
                    # preferredDirection = np.arctan2(b, a) % (2 * np.pi)
                
                #
                elif method == 'ratio':
                    
                    # TODO: Project vectors onto the horizontal axis
                    # I think the formula is cos(theta) * (a.b / |b|)
                    dsi = np.nan

                #
                self.nm['dsi/bar'][self.iUnit] = dsi

        return

    def histModulationByDirectionSelectivity(
        self,
        threshold=0.3,
        nBins=50,
        windowIndex=5,
        figsize=(4, 5),
        ):
        """
        """

        #
        fig, ax = plt.subplots()
        mi = self.ns[f'mi/pref/real'][:, windowIndex, 0]
        ds = np.abs(self.ns[f'dsi/probe']) >= threshold
        samples = (
            mi[np.invert(ds)],
            mi[ds]
        )
        counts, edges, patches = ax.hist(
            samples,
            range=(-3, 3),
            bins=nBins,
            histtype='barstacked'
        )
        for patch in patches[0]:
            patch.set_facecolor('k')
            patch.set_edgecolor('k')
        for patch in patches[1]:
            patch.set_facecolor('w')
            patch.set_edgecolor('k')

        #
        ax.set_ylabel('N units')
        ax.set_xlabel('Modualtion index (MI)')
        ax.legend([fr'$DSI\geq{threshold}$', fr'$DSI<{threshold}$'])
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, ax

    def scatterModulationByPreference(
        self,
        alpha=0.5,
        windowIndex=5,
        figsize=(4, 4),
        ):
        """
        """

        fig, ax = plt.subplots()

        #
        significant = np.logical_or(
            self.ns[f'p/pref/real'][:, windowIndex, 0] < alpha,
            self.ns[f'p/null/real'][:, windowIndex, 0] < alpha
        )
        x = self.ns[f'mi/pref/real'][significant, windowIndex, 0]
        y = self.ns[f'mi/null/real'][significant, windowIndex, 0]

        #
        colors = list()
        for iUnit in range(len(self.ukeys)):
            fPref = self.ns[f'p/pref/real'][iUnit, windowIndex, 0] < alpha
            fNull = self.ns[f'p/pref/null'][iUnit, windowIndex, 0] < alpha
            if fNull and fPref:
                color = (1, 1, 0, 1)
            elif fNull:
                color = (0, 1, 0, 1)
            else:
                color = (1, 0, 0, 1)
            colors.append(color)

        #
        ax.scatter(
            x,
            y,
            marker='.',
            s=10,
            color=colors,
            alpha=0.5,
        )

        #
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return

    def histResponseAmplitudeByEvent(
        self,
        responseWindow=(0, 0.5),
        baselineWindow=(-1, -0.5),
        figsize=(4, 4),
        ):
        """
        Create histogram which shows the distribution of response amplitude for
        preferred probes and the corresponding saccade direction
        """

        #
        nUnits = len(self.ukeys)
        x = np.full(nUnits, np.nan)
        y = np.full(nUnits, np.nan)

        #
        binIndicesForResponse = np.logical_and(
            self.tSaccade >= responseWindow[0],
            self.tSaccade <= responseWindow[1]
        )
        binIndicesForBaseline = np.logical_and(
            self.tSaccade >= baselineWindow[0],
            self.tSaccade <= baselineWindow[1]
        )

        #
        for iUnit in range(nUnits):

            # Visual response amplitude
            ppth = self.ns[f'ppths/pref/real/extra'][iUnit]
            x[iUnit] = np.max(np.abs(ppth[binIndicesForResponse]))

            # Saccade response amplitude
            self.ukey = self.ukeys[iUnit]
            saccadeDirection = convertGratingMotionToSaccadeDirection(
                self.preference[self.iUnit],
                self.session.eye,
            )
            psth = self.ns[f'psths/{saccadeDirection}/real'][iUnit]
            bl = psth[binIndicesForBaseline].mean()
            fr = (psth[binIndicesForResponse] - bl) / self.factor[iUnit]
            y[iUnit] = np.max(np.abs(fr))

        #
        fig, ax = plt.subplots()
        ax.hist(
            x=(x, y),
            color=('r', 'b'),
            edgecolor='k',
            histtype='barstacked'
        )

        #
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, ax