import h5py
import numpy as np
import pathlib as pl
from matplotlib import pyplot as plt
from myphdlib.general.toolkit import psth2
from myphdlib.figures.analysis import AnalysisBase
from myphdlib.figures.modulation import BasicSaccadicModulationAnalysis
from myphdlib.figures.fictive import convertSaccadeDirectionToGratingMotion, convertGratingMotionToSaccadeDirection

class DirectionSectivityAnalysis(BasicSaccadicModulationAnalysis, AnalysisBase):
    """
    """

    def __init__(self, **kwargs):
        """
        """

        super().__init__(**kwargs)

        self.examples = (
            ('2023-04-11', 'mlati6', 736),
            ('2023-04-11', 'mlati6', 585)
        )

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

            # Get amplitude of largest component of preferred PPTH
            paramsPref = self.ns['params/pref/real/extra'][iUnit]
            paramsPref = np.delete(paramsPref, np.isnan(paramsPref))
            A, B, C = np.split(paramsPref[:-1], 3)
            aPref = A[0]
            lPref = B[0]

            # Get amplitude of corresponding component from the null PPTH
            paramsNull = self.ns['params/null/real/extra'][iUnit]
            paramsNull = np.delete(paramsNull, np.isnan(paramsNull))
            A, B, C = np.split(paramsNull[:-1], 3)
            iComp = np.argmin(np.abs(B - lPref))
            aNull = A[iComp]

            # Clip null direction if sign reverses
            if aPref > 0:
                if aNull < 0:
                    aNull = 0
            else:
                if aNull > 0:
                    aNull = 0

            #
            if self.preference[iUnit] == -1:
                aLeft = abs(aPref)
                aRight = abs(aNull)
            else:
                aLeft = abs(aNull)
                aRight = abs(aPref)

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
            # self.ukey = self.ukeys[iUnit]

            # Determine which saccade is preferred
            aNasal = np.max(np.abs(self.ns['psths/nasal/real'][iUnit, binIndices]))
            aTemporal = np.max(np.abs(self.ns['psths/temporal/real'][iUnit, binIndices]))
            if aNasal > aTemporal:
                aPref = aNasal
                aNull = aTemporal
                saccadeDirection = 'nasal'
            else:
                aPref = aTemporal
                aNull = aNasal
                saccadeDirection = 'temporal'

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

    def plotModulationFrequencyByDirectionSelectivity(
        self,
        nq=10,
        windowIndex=5,
        componentIndex=0,
        minimumResponseAmplitude=0,
        figsize=(4, 2),
        ):
        """
        """

        fig, grid = plt.subplots(ncols=nq, sharex=True)

        #
        DSI = np.abs(self.ns['dsi/probe'])
        MI = self.ns['mi/pref/real'][:, windowIndex, componentIndex]
        P = self.ns['p/pref/real'][:, windowIndex, componentIndex]

        #
        exclude = np.vstack([
            np.isnan(DSI),
            np.isnan(MI),
            np.isnan(P),
            np.abs(self.ns['params/pref/real/extra'][:, 0]) < minimumResponseAmplitude
        ]).any(0)
        DSI = np.delete(DSI, exclude)
        MI = np.delete(MI, exclude)
        P = np.delete(P, exclude)

        #
        index = np.argsort(DSI)
        DSI = DSI[index]
        MI = MI[index]
        P = P[index]

        #
        stack = np.vstack([
            DSI,
            MI,
            P
        ])

        #
        ylims = list()
        for i, quantile in enumerate(np.array_split(stack, nq, axis=1)):

            #
            dsi = quantile[0, :]
            mi = quantile[1, :]
            p = quantile[2, :]

            #
            n1 = np.sum(np.logical_and(mi < 0, p < 0.05))
            grid[i].bar(0, n1, bottom=0, color='b', width=1)
            n2 = np.sum(p >= 0.05)
            grid[i].bar(0, n2, bottom=n1, color='w', width=1)
            n3 = np.sum(np.logical_and(mi > 0, p < 0.05))
            grid[i].bar(0, n3, bottom=n1 + n2, color='r', width=1)

            #
            ylims.append([0, p.size])

        #
        for iq, ax in enumerate(grid):
            ax.set_xlim([-0.5, 0.5])
            ax.set_ylim(ylims[iq])
            ax.set_xticks([0,])
            ax.set_xticklabels([])
        for ax in grid[1:]:
            ax.set_yticks([])
        fig.supxlabel('DSI', fontsize=10)
        fig.supylabel('N units', fontsize=10)
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.2)

        return fig, grid

    def plotModulationByPreference(
        self,
        figsize=(2, 3.5),
        windowIndex=5,
        transform=True,
        yrange=(-3, 3),
        ):
        """
        """

        fig, ax = plt.subplots()

        #
        miRaw = np.copy(self.ns[f'mi/null/real'])
        # self._correctModulationIndexForNullProbes()

        #
        mask = list()
        for iUnit in range(len(self.ukeys)):
            include = np.all([
                self.ns[f'mi/pref/real'][iUnit, windowIndex, 0] < 0,
                self.ns[f'p/pref/real'][iUnit, windowIndex, 0] < 0.05
            ])
            if include:
                mask.append(True)
            else:
                mask.append(False)
        mask = np.array(mask)

        #
        x = self.ns[f'mi/pref/real'][:, windowIndex, 0]
        y = self.ns[f'mi/null/real'][:, windowIndex, 0]
        if transform:
            x = np.tanh(x)
            y = np.tanh(y)
        
        #
        ax.scatter(
            np.full(mask.sum(), 0),
            np.clip(x[mask], *yrange),
            marker='.',
            color='0.5',
            s=15,
            alpha=0.1
        )
        s = np.full(x.size, 0.0)
        s[self.ns[f'p/null/real'][:, windowIndex, 0] < 0.05] = 15.0
        ax.scatter(
            np.full(mask.sum(), 1),
            np.clip(y[mask], *yrange),
            marker='.',
            color='0.5',
            s=s[mask],
            alpha=0.1
        )
        for xy in zip(x[mask], y[mask]):
            ax.plot([0, 1], np.clip(np.array(xy), *yrange), color='0.5', lw=0.5, alpha=0.1)

        for i, a in zip([0, 1], [x, y]):
            q1 = np.quantile(a[mask], 0.25)
            q2 = np.quantile(a[mask], 0.5)
            q3 = np.quantile(a[mask], 0.75)
            ax.bar(
                i,
                q3 - q1,
                0.35,
                q1,
                facecolor='none',
                edgecolor='k',
                zorder=2
            )
            ax.scatter(i, q2, color='k', s=30, zorder=2)
        ax.plot([0, 1], [np.median(x[mask]), np.median(y[mask])], color='k')

        #
        ax.set_ylabel('Modulation index')
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Pref', 'Null'], rotation=45)
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        #
        self.ns[f'mi/null/real'] = miRaw

        return fig, ax

    def scatterModulationByPreference(
        self,
        alpha=0.05,
        windowIndex=5,
        modulationSign=-1,
        minimumResponseAmplitude=10,
        minimumSelectivity=0.2,
        transform=False,
        xyrange=(-3, 3),
        figsize=(4, 4),
        ):
        """
        """

        #
        miRaw = np.copy(self.ns[f'mi/null/real'])
        # self._correctModulationIndexForNullProbes()

        #
        fig, ax = plt.subplots()

        #
        include = np.vstack([
            self.ns['params/pref/real/extra'][:, 0] >= minimumResponseAmplitude,
            np.logical_or(
                self.ns['p/pref/real'][:, windowIndex, 0] < alpha,
                self.ns['p/null/real'][:, windowIndex, 0] < alpha
            ),
            np.abs(self.ns['dsi/probe']) >= minimumSelectivity,
            self.ns['mi/pref/real'][:, windowIndex, 0] < 0 if modulationSign == -1 else self.ns['mi/pref/real'][:, windowIndex, 0] > 0
        ]).all(0)
        x = self.ns[f'mi/pref/real'][include, windowIndex, 0]
        y = self.ns[f'mi/null/real'][include, windowIndex, 0]
        if transform:
            x = np.tanh(x)
            y = np.tanh(y)
            xyrange = (-1, 1)
        else:
            x = np.clip(x, *xyrange)
            y = np.clip(y, *xyrange)

        #
        colors = list()
        markers = list()
        for iUnit in range(len(self.ukeys)):
            if include[iUnit]:
                fPref = self.ns[f'p/pref/real'][iUnit, windowIndex, 0] < alpha
                fNull = self.ns[f'p/null/real'][iUnit, windowIndex, 0] < alpha
                if fNull and fPref:
                    color = 'xkcd:purple'
                elif fNull:
                    color = 'xkcd:orange'
                elif fPref:
                    color = 'xkcd:green'
                # else:
                #     color = 'xkcd:gray'
                markers.append('.')
                colors.append(color)

        #
        for i in range(x.size):
            ax.scatter(
                x[i],
                y[i],
                s=30,
                color=colors[i],
                marker=markers[i],
                alpha=0.5,
                clip_on=False
            )
        ax.vlines(0, *xyrange, color='k', linestyle=':')
        ax.hlines(0, *xyrange, color='k', linestyle=':')

        #
        ax.set_xlabel(r'$MI_{Pref}$')
        ax.set_ylabel(r'$MI_{Null}$')
        ax.set_xlim(xyrange)
        ax.set_ylim(xyrange)
        ax.set_aspect('equal')
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        #
        self.ns[f'mi/null/real'] = miRaw

        return fig, ax

    def histSaccadeResponseAmplitudeBySelectivity(
        self,
        responseWindow=(0, 0.5),
        baselineWindow=(-1, -0.5),
        minimumResponseAmplitude=5,
        minimumSelectivity=0.3,
        saccadeType='real',
        nBins=50,
        xRange=(-100, 100),
        colors=('k', 'r'),
        labels=('Non-DS', 'DS'),
        figsize=(3.5, 2),
        ):
        """
        Create histogram which shows the distribution of response amplitude for
        preferred probes and the corresponding saccade direction
        """

        #
        binIndicesForResponse = np.logical_and(
            self.tSaccade >= responseWindow[0],
            self.tSaccade <= responseWindow[1]
        )
        #
        binIndicesForBaseline = np.logical_and(
            self.tSaccade >= baselineWindow[0],
            self.tSaccade <= baselineWindow[1]
        )

        #
        include = np.abs(self.ns['params/pref/real/extra'][:, 0]) >= minimumResponseAmplitude
        samples = {
            'x': np.full(include.size, np.nan),
            'y': np.full(include.size, np.nan)
        }
        amplitude = np.full(len(self.ukeys), np.nan)

        #
        for iUnit in range(len(self.ukeys)):

            #
            if include[iUnit] == False:
                continue

            # Saccade response amplitude
            self.ukey = self.ukeys[iUnit]
            saccadeDirection = convertGratingMotionToSaccadeDirection(
                self.preference[self.iUnit],
                self.session.eye,
            )
            psth = self.ns[f'psths/{saccadeDirection}/{saccadeType}'][iUnit]
            bl = psth[binIndicesForBaseline].mean()
            fr = (psth[binIndicesForResponse] - bl) / self.factor[iUnit]

            #
            dsi = self.ns['dsi/probe'][iUnit]
            if dsi < minimumSelectivity:
                samples['x'][iUnit] = fr[np.argmax(np.abs(fr))]
            else:
                samples['y'][iUnit] = fr[np.argmax(np.abs(fr))]
            amplitude[iUnit] = fr[np.argmax(np.abs(fr))]

        #
        fig, ax = plt.subplots()
        for i, key in enumerate(('x', 'y')):
            binCounts, binEdges = np.histogram(
                np.clip(samples[key], *xRange),
                range=xRange,
                bins=nBins,
            )
            binCenters = binEdges[:-1] + ((binEdges[1] - binEdges[0]) / 2)
            ax.plot(
                binCenters,
                binCounts / binCounts.sum(),
                color=colors[i],
                label=labels[i],
                alpha=0.5,
            )

        #
        ax.set_xlabel('Saccade response amplitude (SD)')
        ax.set_ylabel('Probability')
        ax.legend()
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, ax, amplitude

    def plotExampleDirectionSelectiveUnit(
        self,
        ukeys=None,
        figsize=(5, 2)
        ):
        """
        """

        #
        if ukeys is None:
            ukeys = self.examples
        fig, axs = plt.subplots(ncols=len(ukeys))
        axs = np.atleast_1d(axs)

        #
        for j, ukey in enumerate(ukeys):

            #
            self.ukey = ukey

            #
            yPref = self.ns['ppths/pref/real/extra'][self.iUnit]
            yNull = self.ns['ppths/null/real/extra'][self.iUnit]
            axs[j].plot(self.tProbe, yNull, color='0.7', label='Null')
            axs[j].plot(self.tProbe, yPref, color='k', label='Pref')

            #
            dsi = self.ns['dsi/probe'][self.iUnit]
            axs[j].set_title(f'DSI={dsi:.3f}', fontsize=10)
            axs[j].set_xlabel('Time from probe (sec)')

        #
        axs[0].set_ylabel('FR (z-scored)')
        axs[0].legend()

        #
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, axs

    def histDirectionSelectivity(
        self,
        threshold=0.5,
        minmumResponseAmplitude=10,
        nbins=30,
        figsize=(3, 2),
        ):
        """
        """

        fig, ax = plt.subplots()
        mask = np.abs(self.ns['params/pref/real/extra'][:, 0]) > minmumResponseAmplitude
        dsi = self.ns['dsi/probe'][mask]
        ax.hist(
            list(map(np.abs, (dsi[np.abs(dsi) < threshold], dsi[np.abs(dsi) >= threshold]))),
            color=('w', 'k'),
            edgecolor=None,
            bins=nbins,
            histtype='barstacked',
        )
        ax.hist(np.abs(dsi), bins=nbins, edgecolor='k', color=None, histtype='step')

        #
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, ax, dsi