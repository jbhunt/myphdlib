import h5py
import numpy as np
import pathlib as pl
from matplotlib import pyplot as plt
from myphdlib.general.toolkit import psth2
from myphdlib.figures.analysis import AnalysisBase, g
from myphdlib.figures.modulation import BasicSaccadicModulationAnalysis
from myphdlib.figures.analysis import convertSaccadeDirectionToGratingMotion, convertGratingMotionToSaccadeDirection
from matplotlib.colors import LinearSegmentedColormap
from itertools import combinations

def computeApparentMotion(
    session,
    ):
    """
    """

    gratingMotionByBlock = session.load('stimuli/dg/grating/motion')
    gratingTimestampsByBlock = np.vstack([
        session.load('stimuli/dg/grating/timestamps'),
        session.load('stimuli/dg/iti/timestamps')
    ]).T
    apparentMotion = list()
    for saccadeTimestamp in session.saccadeTimestamps[:, 0]:
        am = np.nan
        for gratingMotion, (t1, t2) in zip(gratingMotionByBlock, gratingTimestampsByBlock):
            if saccadeTimestamp >= t1 and saccadeTimestamp < t2:
                am = gratingMotion
                break
        apparentMotion.append(am)
    apparentMotion = np.array(apparentMotion)

    return apparentMotion

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
    
    def measureDirectionSelectivityForPerisaccadicProbes(
        self,
        method='ratio',
        minimumResponseAmplitude=5,
        perisaccadicWindow=(-0.5, 0.5),
        denovo=True,
        responseWindow=(0, 0.5),
        baselineWindow=(-0.2, 0),
        ):
        """
        """

#
        window = (
            np.min([responseWindow, baselineWindow]),
            np.max([responseWindow, baselineWindow])
        )
        tProbe, _ = psth2(
            np.array([0]),
            np.array([0]),
            window=window,
            binsize=0.01,
        )
        binIndicesForResponse = np.where(np.logical_and(
            tProbe >= responseWindow[0],
            tProbe <  responseWindow[1]
        ))[0]
        binIndicesForBaseline = np.where(np.logical_and(
            tProbe >= baselineWindow[0],
            tProbe <  baselineWindow[1]
        ))[0]

        #
        dsi = np.full([len(self.ukeys), len(self.windows)], np.nan)
        nUnits = len(self.ukeys)

        #
        for iUnit in range(len(self.ukeys)):

            #
            end = '\r' if iUnit + 1 != nUnits else '\n'
            print(f'Working on unit {iUnit + 1} out of {nUnits} ...', end=end)

            #
            session = self._getSessionFromUnitKey(self.ukeys[iUnit])
            date, animal, cluster = self.ukeys[iUnit]
            unit = session.population.indexByCluster(cluster)

            #
            for windowIndex in range(len(self.windows)):

                #
                perisaccadicWindow = self.windows[windowIndex]

                #
                if denovo:

                    #
                    trialIndices = np.where(np.logical_and(
                        session.gratingMotionDuringProbes == -1,
                        np.logical_and(
                            session.probeLatencies >= perisaccadicWindow[0],
                            session.probeLatencies <= perisaccadicWindow[1]
                        )
                    ))[0]

                    #
                    t, fr = unit.kde(
                        session.probeTimestamps[trialIndices],
                        responseWindow=window,
                        binsize=0.01,
                        sigma=0.01,
                    )
                    fr = (fr[binIndicesForResponse] - fr[binIndicesForBaseline].mean()) / self.factor[iUnit]
                    aLeft = np.max(np.abs(fr))

                    #
                    trialIndices = np.where(np.logical_and(
                        session.gratingMotionDuringProbes == 1,
                        np.logical_or(
                            session.probeLatencies >= perisaccadicWindow[0],
                            session.probeLatencies <= perisaccadicWindow[1]
                        )
                    ))[0]

                    #
                    t, fr = unit.kde(
                        session.probeTimestamps[trialIndices],
                        responseWindow=window,
                        binsize=0.01,
                        sigma=0.01,
                    )
                    fr = (fr[binIndicesForResponse] - fr[binIndicesForBaseline].mean()) / self.factor[iUnit]
                    aRight = np.max(np.abs(fr))

                    #
                    if aLeft > aRight:
                        aPref = aLeft
                        aNull = aRight
                    else:
                        aPref = aRight
                        aNull = aLeft

                else:
                    # Get amplitude of largest component of preferred PPTH
                    paramsPref = self.ns['params/pref/real/peri'][iUnit, windowIndex]
                    paramsPref = np.delete(paramsPref, np.isnan(paramsPref))
                    A, B, C = np.split(paramsPref[:-1], 3)
                    aPref = A[0]
                    lPref = B[0]

                    # Get amplitude of corresponding component from the null PPTH
                    paramsNull = self.ns['params/null/real/peri'][iUnit, windowIndex]
                    paramsNull = np.delete(paramsNull, np.isnan(paramsNull))
                    A, B, C = np.split(paramsNull[:-1], 3)
                    iComp = np.argmin(np.abs(B - lPref))
                    aNull = A[iComp]

                    #
                    if self.preference[iUnit] == -1:
                        aLeft = aPref
                        aRight = aNull
                    else:
                        aLeft = aNull
                        aRight = aPref

                #
                aNull = np.clip(aNull, 0, np.inf)

                #
                if abs(aPref) < minimumResponseAmplitude:
                    continue

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
                    dsi_ = np.sqrt(np.power(a, 2) + np.power(b, 2))

                    #
                    dsi[iUnit, windowIndex] = dsi_

        return dsi

    def measureDirectionSelectivityForExtrasaccadicProbes(
        self,
        method='ratio',
        minimumResponseAmplitude=5,
        perisaccadicWindow=(-0.5, 0.5),
        denovo=True,
        responseWindow=(0, 0.5),
        baselineWindow=(-0.2, 0),
        ):
        """
        Notes
        -----
        Extrasaccadic responses can be selected by setting the windowIndex to None 
        """

        #
        window = (
            np.min([responseWindow, baselineWindow]),
            np.max([responseWindow, baselineWindow])
        )
        tProbe, _ = psth2(
            np.array([0]),
            np.array([0]),
            window=window,
            binsize=0.01,
        )
        binIndicesForResponse = np.where(np.logical_and(
            tProbe >= responseWindow[0],
            tProbe <  responseWindow[1]
        ))[0]
        binIndicesForBaseline = np.where(np.logical_and(
            tProbe >= baselineWindow[0],
            tProbe <  baselineWindow[1]
        ))[0]

        #
        self.ns['dsi/probe'] = np.full(len(self.ukeys), np.nan)
        nUnits = len(self.ukeys)

        #
        for iUnit in range(len(self.ukeys)):

            #
            end = '\r' if iUnit + 1 != nUnits else '\n'
            print(f'Working on unit {iUnit + 1} out of {nUnits} ...', end=end)

            #
            session = self._getSessionFromUnitKey(self.ukeys[iUnit])
            date, animal, cluster = self.ukeys[iUnit]
            unit = session.population.indexByCluster(cluster)

            #
            if denovo:

                #
                trialIndices = np.where(np.logical_and(
                    session.gratingMotionDuringProbes == -1,
                    np.logical_or(
                        session.probeLatencies < perisaccadicWindow[0],
                        session.probeLatencies > perisaccadicWindow[1]
                    )
                ))[0]

                #
                t, fr = unit.kde(
                    session.probeTimestamps[trialIndices],
                    responseWindow=window,
                    binsize=0.01,
                    sigma=0.01,
                )
                fr = (fr[binIndicesForResponse] - fr[binIndicesForBaseline].mean()) / self.factor[iUnit]
                aLeft = np.max(np.abs(fr))

                #
                trialIndices = np.where(np.logical_and(
                    session.gratingMotionDuringProbes == 1,
                    np.logical_or(
                        session.probeLatencies < perisaccadicWindow[0],
                        session.probeLatencies > perisaccadicWindow[1]
                    )
                ))[0]

                #
                t, fr = unit.kde(
                    session.probeTimestamps[trialIndices],
                    responseWindow=window,
                    binsize=0.01,
                    sigma=0.01,
                )
                fr = (fr[binIndicesForResponse] - fr[binIndicesForBaseline].mean()) / self.factor[iUnit]
                aRight = np.max(np.abs(fr))

                #
                if aLeft > aRight:
                    aPref = aLeft
                    aNull = aRight
                else:
                    aPref = aRight
                    aNull = aLeft

            else:
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

                #
                if self.preference[iUnit] == -1:
                    aLeft = aPref
                    aRight = aNull
                else:
                    aLeft = aNull
                    aRight = aPref

            #
            aNull = np.clip(aNull, 0, np.inf)

            #
            if abs(aPref) < minimumResponseAmplitude:
                continue

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

    def measureDirectionSelectivityForRealSaccades1(
        self,
        method='ratio',
        responseWindow=(-0.2, 0.5),
        baselineWindow=(-1, -0.5),
        minimumResponseAmplitude=5,
        denovo=True,
        ):
        """
        """

        #
        window = (
            np.min([responseWindow, baselineWindow]),
            np.max([responseWindow, baselineWindow])
        )
        tSaccade, _ = psth2(
            np.array([0]),
            np.array([0]),
            window=window,
            binsize=0.01,
        )
        binIndicesForResponse = np.where(np.logical_and(
            tSaccade >= responseWindow[0],
            tSaccade <  responseWindow[1]
        ))[0]
        binIndicesForBaseline = np.where(np.logical_and(
            tSaccade >= baselineWindow[0],
            tSaccade <  baselineWindow[1]
        ))[0]
        self.ns[f'dsi/saccade/real1'] = np.full(len(self.ukeys), np.nan)

        #
        nUnits = len(self.ukeys)
        for iUnit in range(nUnits):

            #
            end = '\r' if iUnit + 1 != nUnits else '\n'
            print(f'Working on unit {iUnit + 1} out of {nUnits} ...', end=end)

            #
            session = self._getSessionFromUnitKey(self.ukeys[iUnit])
            date, animal, cluster = self.ukeys[iUnit]
            unit = session.population.indexByCluster(cluster)

            # Determine which saccade is preferred

            # Nasal saccade response
            if denovo:
                t, fr = unit.kde(
                    session.saccadeTimestamps[session.saccadeLabels == 1, 0],
                    responseWindow=window,
                    binsize=0.01,
                    sigma=0.01,
                )
                fr = (fr[binIndicesForResponse] - fr[binIndicesForBaseline].mean()) / self.factor[iUnit]
            else:
                fr = self.ns[f'psths/nasal/real'][iUnit, binIndicesForResponse]
                fr -= self.ns[f'psths/nasal/real'][iUnit, binIndicesForBaseline].mean()
                fr /= self.factor[iUnit]
            aNasal = np.max(np.abs(fr))

            # Temporal saccade response
            if denovo:
                t, fr = unit.kde(
                    session.saccadeTimestamps[session.saccadeLabels == -1, 0],
                    responseWindow=window,
                    binsize=0.01,
                    sigma=0.01,
                )
                fr = (fr[binIndicesForResponse] - fr[binIndicesForBaseline].mean()) / self.factor[iUnit]
            else:
                fr = self.ns[f'psths/temporal/real'][iUnit, binIndicesForResponse]
                fr -= self.ns[f'psths/temporal/real'][iUnit, binIndicesForBaseline].mean()
                fr /= self.factor[iUnit]
            aTemporal = np.max(np.abs(fr))

            #
            if aNasal > aTemporal:
                aPref = aNasal
                aNull = aTemporal
                saccadeDirection = 'nasal'
            else:
                aPref = aTemporal
                aNull = aNasal
                saccadeDirection = 'temporal'

            #
            if abs(aPref) < minimumResponseAmplitude:
                continue

            # Clip null direction if sign reverses
            aNull = np.clip(aNull, 0, np.inf)

            # Convert saccade direction to probe direction
            apparentMotion = convertSaccadeDirectionToGratingMotion(
                saccadeDirection,
                session.eye,
            )
            if apparentMotion == -1:
                aLeft = aPref
                aRight = aNull
            else:
                aLeft = aNull
                aRight = aPref

            #
            if method == 'ratio':
                dsi = (aRight - aLeft) / (aRight + aLeft)
            
            #
            self.ns[f'dsi/saccade/real1'][iUnit] = dsi
            
        return

    def measureDirectionSelectivityForRealSaccades2(
        self,
        responseWindow=(-0.2, 0.5),
        baselineWindow=(-1, -0.5),
        minimumResponseAmplitude=5, 
        ):
        """
        """

        window = (
            np.min([baselineWindow[0], responseWindow[0]]),
            np.max([baselineWindow[0], responseWindow[1]])
        )
        t, _ = psth2(
            np.array([0]),
            np.array([0]),
            window=window,
            binsize=0.01,
        )
        binIndicesForResponse = np.where(np.logical_and(
            t >= responseWindow[0],
            t <= responseWindow[1]
        ))[0]
        binIndicesForBaseline = np.where(np.logical_and(
            t >= baselineWindow[0],
            t <= baselineWindow[1]
        ))[0]
        self.ns['dsi/saccade/real'] = np.full(len(self.ukeys), np.nan)
        nUnits = len(self.ukeys)
        target = None
        for iUnit in range(nUnits):

            #
            end = '\r' if iUnit + 1 != nUnits else '\n'
            print(f'Working on {iUnit + 1} out of {nUnits} ... ', end=end)

            #
            date, animal, cluster = self.ukeys[iUnit]
            session = self._getSessionFromUnitKey(self.ukeys[iUnit])
            unit = session.population.indexByCluster(cluster)

            #
            if session != target:
                apparentMotion = computeApparentMotion(session)
                target = session

            # Response to leftward motion
            t, fr = unit.kde(
                target.saccadeTimestamps[apparentMotion == -1, 0],
                responseWindow=window,
                binsize=0.01,
                sigma=0.01
            )
            fr = (fr[binIndicesForResponse] - fr[binIndicesForBaseline].mean()) / self.factor[iUnit]
            aLeft = np.max(np.abs(fr))

            # Response to rightward motion
            t, fr = unit.kde(
                target.saccadeTimestamps[apparentMotion == 1, 0],
                responseWindow=window,
                binsize=0.01,
                sigma=0.01
            )
            fr = (fr[binIndicesForResponse] - fr[binIndicesForBaseline].mean()) / self.factor[iUnit]
            aRight = np.max(np.abs(fr))

            # Clip at 0 if the null response is negative
            if aLeft > aRight:
                aPref = aLeft
                aRight = np.clip(aRight, 0, np.inf)
            else:
                aPref = aRight
                aLeft = np.clip(aLeft, 0, np.inf)

            #
            if abs(aPref) < minimumResponseAmplitude:
                continue

            #
            dsi = (aRight - aLeft) / (aRight + aLeft)
            self.ns['dsi/saccade/real'][iUnit] = dsi

        return

    def measureDirectionSelectivityForFictiveSaccades(
        self,
        responseWindow=(-0.2, 0.5),
        baselineWindow=(-1, -0.5),
        minimumResposneAmplitude=3, 
        ):
        """
        """

        window = (
            np.min([baselineWindow[0], responseWindow[0]]),
            np.max([baselineWindow[0], responseWindow[1]])
        )
        t, _ = psth2(
            np.array([0]),
            np.array([0]),
            window=window,
            binsize=0.01,
        )
        binIndicesForResponse = np.where(np.logical_and(
            t >= responseWindow[0],
            t <= responseWindow[1]
        ))[0]
        binIndicesForBaseline = np.where(np.logical_and(
            t >= baselineWindow[0],
            t <= baselineWindow[1]
        ))[0]
        self.ns['dsi/saccade/fictive'] = np.full(len(self.ukeys), np.nan)

        #
        nUnits = len(self.ukeys)
        target = None
        for iUnit in range(nUnits):

            #
            end = '\r' if iUnit + 1 != nUnits else '\n'
            print(f'Working on {iUnit + 1} out of {nUnits} ... ', end=end)

            #
            date, animal, cluster = self.ukeys[iUnit]
            session = self._getSessionFromUnitKey(self.ukeys[iUnit])
            unit = session.population.indexByCluster(cluster)

            #
            if target != session:
                gratingMotion = session.load('stimuli/fs/saccade/motion')
                saccadeTimestamps = session.load('stimuli/fs/saccade/timestamps')
                target = session

            #
            if saccadeTimestamps is None or len(saccadeTimestamps) == 0:
                continue

            #
            try:

                # Response to leftward motion
                t, fr = unit.kde(
                    saccadeTimestamps[gratingMotion == -1],
                    responseWindow=window,
                    binsize=0.01,
                    sigma=0.01
                )
                fr = (fr[binIndicesForResponse] - fr[binIndicesForBaseline].mean()) / self.factor[iUnit]
                aLeft = np.max(np.abs(fr))

                # Response to rightward motion
                t, fr = unit.kde(
                    saccadeTimestamps[gratingMotion == 1],
                    responseWindow=window,
                    binsize=0.01,
                    sigma=0.01
                )
                fr = (fr[binIndicesForResponse] - fr[binIndicesForBaseline].mean()) / self.factor[iUnit]
                aRight = np.max(np.abs(fr))

            #
            except:
                continue

            #
            if np.any([aLeft < minimumResposneAmplitude, aRight < minimumResposneAmplitude]):
                continue

            # Clip at 0 if the null response is negative
            if aLeft > aRight:
                aRight = np.clip(aRight, 0, np.inf)
            else:
                aLeft = np.clip(aLeft, 0, np.inf)

            #
            dsi = (aRight - aLeft) / (aRight + aLeft)
            self.ns['dsi/saccade/fictive'][iUnit] = dsi

        return

    # TODO: Project vector sum onto the horizontal axis (so that I can compare
    #       DSI from probes/saccades to DSI for the moving bars)
    def measureDirectionSelectivityForMovingBars(
        self,
        method='ratio',
        minimumResponseAmplitude=5,
        baselineWindow=(-0.5, 0),
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


                #
                if ukey[0] != str(session.date):
                    continue
                self.ukey = ukey
                # unit = session.population.indexByCluster(ukey[-1])

                #
                vectors = np.full([uniqueOrientations.size, 2], np.nan)
                for rowIndex, orientation in enumerate(uniqueOrientations):

                    #
                    trialIndices = np.where(movingBarOrientations == orientation)[0]
                    amplitudes = list()
                    baselines = list()
                    for trialIndex in trialIndices:

                        # Response
                        t1, t2 = movingBarTimestamps[trialIndex, :]
                        dt = t2 - t1
                        t, M = psth2(
                            np.array([t1]),
                            self.unit.timestamps,
                            window=(0, dt),
                            binsize=None
                        )
                        fr = M.item() / dt

                        # Baseline
                        t, M = psth2(
                            np.array([t1]),
                            self.unit.timestamps,
                            window=baselineWindow,
                            binsize=None,
                        )
                        bl = M.item() / np.diff(baselineWindow)

                        #
                        amplitudes.append(fr - bl)

                    # t, M = psth2(
                    #     movingBarTimestamps[trialIndices, 0],
                    #     unit.timestamps,
                    #     window=window,
                    #     binsize=0.01,
                    # )
                    # fr = M.mean(0) / 0.01
                    # fr = (fr[binIndicesForResponse] - fr[binIndicesForBaseline].mean()) / self.factor[self.iUnit]

                    #
                    vectors[rowIndex, 0] = np.mean(amplitudes) / self.factor[self.iUnit]
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
                    self.ns['dsi/bar'][self.iUnit] = np.sqrt(np.power(a, 2) + np.power(b, 2))
                
                #
                elif method == 'ratio':
                    
                    #
                    aLeft = vectors[uniqueOrientations == 180, 0]
                    aRight = vectors[uniqueOrientations == 0, 0]

                    #
                    if aLeft < 0 and aRight > 0 or aLeft > 0 and aRight < 0:
                        continue

                    #
                    if aLeft > aRight:
                        aPref = aLeft
                    else:
                        aPref = aRight
                    if aPref < minimumResponseAmplitude:
                        continue

                    #
                    self.ns['dsi/bar'][self.iUnit] = (aRight - aLeft) / (aRight + aLeft)

        return

    def scatterDirectionSelectivityByEvent(
        self,
        figsize=(6, 6),
        ):
        """
        """

        fig, axs = plt.subplots(ncols=4, nrows=4, sharex=True, sharey=True)
        paths = (
            'dsi/probe',
            'dsi/saccade/real',
            'dsi/saccade/fictive',
            'dsi/bar'
        )
        
        for i, p1 in enumerate(paths):
            for j, p2 in enumerate(paths):
                if i == j:
                    continue
                if i < j:
                    continue
                x = self.ns[p1]
                y = self.ns[p2]
                axs[i, j].scatter(
                    x,
                    y,
                    color='k',
                    s=3,
                    edgecolor='none',
                    alpha=0.3
                )
                m = np.logical_and(np.logical_not(np.isnan(x)), np.logical_not(np.isnan(y)))
                f = np.poly1d(np.polyfit(x[m], y[m], deg=1))
                x1, x2 = np.nanmin(x), np.nanmax(x)
                axs[i, j].plot(
                    [x1, x2],
                    f([x1, x2]),
                    color='r'
                )
        xlabels = (
            r'$DSI_{Probe}$',
            r'$DSI_{Saccade (Real)}$',
            r'$DSI_{Saccade (Fictive)}$',
            r'$DSI_{Bars}$'
        )
        ylabels = (
            r'$DSI_{Probe}$',
            r'$DSI_{Saccade (Real)}$',
            r'$DSI_{Saccade (Fictive)}$',
            r'$DSI_{Bars}$'
        )
        for j, ax in enumerate(axs[-1, :]):
            ax.set_xlabel(xlabels[j])
        for i, ax in enumerate(axs[:, 0]):
            ax.set_ylabel(ylabels[i])
        for ax in axs.flatten():
            ax.set_xlim([-1, 1])
            ax.set_ylim([-1, 1])
            ax.set_aspect('equal')
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, axs

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

    def scatterModulationByPreference(
        self,
        alpha=0.05,
        windowIndex=5,
        modulationSign=-1,
        minimumResponseAmplitude=10,
        minimumSelectivity=0.25,
        transform=False,
        xyrange=(-3, 3),
        figsize=(2.5, 5),
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
        ratios = list()
        markers = list()
        uniqueColors = (
            '0.75',
            '0.75',
            '0.75', # both
            'xkcd:gray',
        )
        labels = list()
        for iUnit in range(len(self.ukeys)):
            if include[iUnit]:
                fPref = self.ns[f'p/pref/real'][iUnit, windowIndex, 0] < alpha
                fNull = self.ns[f'p/null/real'][iUnit, windowIndex, 0] < alpha
                if fNull and fPref:
                    color = uniqueColors[2]
                    label = 3
                elif fNull:
                    color = uniqueColors[0]
                    label = 1
                elif fPref:
                    color = uniqueColors[1]
                    label = 2
                else:
                    color = uniqueColors[3]
                    label = 0
                markers.append('o')
                colors.append(color)
                labels.append(label)
                ratio = self.ns['p/pref/real'][iUnit, windowIndex, 0] - self.ns['p/null/real'][iUnit, windowIndex, 0]
                ratios.append(ratio)

        #
        for label, color in zip([1, 2, 3, 0], uniqueColors):
            mask = np.array(labels) == label
            ax.scatter(
                x[mask],
                y[mask],
                s=10,
                color=color,
                alpha=1,
                marker='o',
                clip_on=False,
                edgecolor='none',
            )

        # ax.vlines(0, *xyrange, color='k', linestyle=':')
        ax.hlines(0, *xyrange, color='k', linestyle='-')

        #
        f = np.poly1d(np.polyfit(x, y, deg=1))
        x2 = np.linspace(x.min(), x.max(), 100)
        ax.plot(x2, f(x2), color='k', linestyle=':')

        #
        ax.set_ylim(xyrange)
        if modulationSign == -1:
            ax.set_xlim([xyrange[0], 0])
        else:
            ax.set_xlim([0, xyrange[1]])
        ax.set_xlabel(r'$MI_{Pref}$')
        ax.set_ylabel(r'$MI_{Null}$')
        ax.set_aspect('equal')
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        #
        self.ns[f'mi/null/real'] = miRaw

        return fig, ax, x, y

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

    def plotExamplePeths(
        self,
        componentIndex=0,
        figsize=(5, 2)
        ):
        """
        """

        #
        fig, axs = plt.subplots(ncols=int(len(self.examples) * 2))
        axs = np.atleast_1d(axs)

        #
        for i, ukey in enumerate(self.examples):

            #
            j = int(i * 2)

            #
            iUnit = self._indexUnitKey(ukey)

            #
            yPref = self.ns['ppths/pref/real/extra'][iUnit]
            params = self.ns['params/pref/real/extra'][iUnit]
            abcd = np.delete(params, np.isnan(params))
            abc, d = abcd[:-1], abcd[-1]
            A, B, C = np.split(abc, 3)
            a, b, c = A[componentIndex], B[componentIndex], C[componentIndex]
            t2 = np.linspace(-15 * c, 15 * c, 100) + b
            y2 = g(t2, a, b, c, d)
            axs[j].plot(self.tProbe, yPref, color='0.7')
            axs[j].plot(t2, y2, color='k')

            #
            yNull = self.ns['ppths/null/real/extra'][iUnit]
            params = self.ns['params/null/real/extra'][iUnit]
            abcd = np.delete(params, np.isnan(params))
            abc, d = abcd[:-1], abcd[-1]
            A, B, C = np.split(abc, 3)
            a, b, c = A[componentIndex], B[componentIndex], C[componentIndex]
            t2 = np.linspace(-15 * c, 15 * c, 100) + b
            y2 = g(t2, a, b, c, d)
            axs[j + 1].plot(self.tProbe, yNull, color='0.7')
            axs[j + 1].plot(t2, y2, color='k')

            #
            dsi = self.ns['dsi/probe'][iUnit]
            axs[j].set_title(f'DSI={dsi:.3f}', fontsize=10)
            axs[j].set_xlabel('Time from probe (sec)')

            #
            ylim = [np.inf, -np.inf]
            for ax in axs[j:j + 2]:
                y1, y2 = ax.get_ylim()
                if y1 < ylim[0]:
                    ylim[0] = y1
                if y2 > ylim[1]:
                    ylim[1] = y2
            for ax in axs[j:j + 2]:
                ax.set_ylim(ylim)
            axs[j + 1].set_yticklabels([])
            

        #
        axs[0].set_ylabel('FR (SD)')

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
    
    def plotSelectivityByUnitType(
        self,
        threshold=0.3,
        figsize=(1.5, 3.5)
        ):
        """
        """

        fig, ax = plt.subplots()
        dsi = self.ns['dsi/probe']
        ds = np.full(dsi.size, False)
        ds[dsi >= threshold] = True
        ds = ds.astype(int)
        combos = np.vstack([
            ds,
            self.labels
        ]).T
        uniqueCombos, counts = np.unique(combos, axis=0, return_counts=True)
        data = np.full([np.unique(self.labels).size, np.unique(ds).size], np.nan)
        for i, l1 in enumerate(np.unique(self.labels)):
            for j, l2 in enumerate(np.unique(ds)):
                k = np.where(np.logical_and(
                    uniqueCombos[:, 0] == l2,
                    uniqueCombos[:, 1] == l1
                ))[0]
                data[i, j] = counts[k]
        data[:, 0] /= np.nansum(data[:, 0])
        data[:, 1] /= np.nansum(data[:, 1])
        
        #
        for (i, j), f in np.ndenumerate(data):
            ax.scatter(np.unique(ds)[j], np.unique(self.labels)[i], s=f * 300, color='k', marker='o')

        #
        ax.set_xlim([-0.5, 1.5])
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, (ax,), data