import h5py
import numpy as np
from scipy.stats import sem
from matplotlib import pylab as plt
from myphdlib.general.toolkit import smooth

def getResponseParams(
    hdf,
    protocol='dg',
    responseWindow=(0, 0.3),
    baselineWindow=(-0.2, 0)
    ):
    """
    """

    #
    with h5py.File(hdf, 'r') as stream:
        rProbeLeft = np.array(stream[f'rProbe/{protocol}/left'])
        rProbeRight = np.array(stream[f'rProbe/{protocol}/right'])
        tPoints = np.array(stream[f'rProbe/{protocol}/left'].attrs['t'])

    #
    nUnits = rProbeLeft.shape[0]
    directionPreference = np.full(nUnits, np.nan)
    responseSign = np.full(nUnits, np.nan)
    baselineWindowMask = np.logical_and(
        tPoints >= baselineWindow[0],
        tPoints <  baselineWindow[1]
    )
    responseWindowMask = np.logical_and(
        tPoints >= responseWindow[0],
        tPoints <  responseWindow[1]
    )
    for iUnit in range(nUnits):

        #
        yLeft = rProbeLeft[iUnit, responseWindowMask] - rProbeLeft[iUnit, baselineWindowMask].mean()
        yRight = rProbeRight[iUnit, responseWindowMask] - rProbeRight[iUnit, baselineWindowMask].mean()
        yPeaks = list()
        for y in [yLeft, yRight]:
            i = np.argmax(np.abs(y))
            yPeaks.append(y[i])

        #
        if np.mean(yPeaks) < 0:
            responseSign[iUnit] = -1
        elif np.mean(yPeaks) > 0:
            responseSign[iUnit] = +1
        else:
            responseSign[iUnit] = np.nan

        #
        if yPeaks[0] > yPeaks[1]:
            directionPreference[iUnit] = -1
        elif yPeaks[0] < yPeaks[1]:
            directionPreference[iUnit] = +1
        else:
            directionPreference[iUnit] = np.nan

    return directionPreference, responseSign

def loadPeths(
    hdf,
    responseWindow=(0, 0.3),
    baselineWindow=(-0.2, 0),
    protocol='dg',
    maximumProbability=0.01,
    minimumResponseAmplitude=1,
    ):
    """
    """

    #
    directionPreference, responseSign = getResponseParams(
        hdf,
        responseWindow=responseWindow,
        baselineWindow=baselineWindow
    )

    # Read the table
    file = h5py.File(hdf, 'r')

    # Load datasets
    unitLabels = np.array(file[f'unitLabel'])
    tPoints = file[f'rProbe/{protocol}/left'].attrs['t']
    binEdges = file[f'rMixed/{protocol}/left'].attrs['edges']
    nSpikes = np.array(file['nSpikes'])
    binCenters = binEdges.mean(1)

    #
    nUnits, nBinsInHistogram, nBinsInWindow = file[f'rMixed/{protocol}/left'].shape
    data = np.full([nUnits, nBinsInHistogram, nBinsInWindow, 4, 2], np.nan)
    iPeak = np.full(nUnits, np.nan)

    #
    for probeDirection in ('left', 'right'):

        #
        rProbe = np.array(file[f'rProbe/{protocol}/{probeDirection}'])
        rMixed = np.array(file[f'rMixed/{protocol}/{probeDirection}'])
        rSaccade = np.array(file[f'rSaccade/{protocol}/{probeDirection}'])
        pZeta = np.array(file[f'pZeta/{probeDirection}'])

        #
        for iUnit in range(nUnits):

            #
            if directionPreference[iUnit] == -1 and probeDirection == 'left':
                index = 0
            elif directionPreference[iUnit] == -1 and probeDirection == 'right':
                index = 1
            elif directionPreference[iUnit] == +1 and probeDirection == 'left':
                index = 1
            elif directionPreference[iUnit == +1] and probeDirection == 'right':
                index = 0
            else:
                continue

            #
            if maximumProbability is not None and pZeta[iUnit] > maximumProbability:
                continue

            # Get the baseline FR for the visual-only PETH
            tMask = np.logical_and(
                tPoints >= baselineWindow[0],
                tPoints <= baselineWindow[1]
            )
            mu = round(rProbe[iUnit][tMask].mean(), 3)
            sigma = round(rProbe[iUnit][tMask].std(), 3)

            # Find the maximum of the PETH
            tMask = np.logical_and(
                tPoints >= responseWindow[0],
                tPoints <= responseWindow[1]
            )
            iPeak[iUnit] = np.argmax(np.abs(rProbe[iUnit][tMask] - mu))
            iPeak[iUnit] += np.sum(tPoints < responseWindow[0])

            #
            if sigma == 0:
                continue

            #
            rPeak = np.abs(rProbe[iUnit] - mu)[int(iPeak[iUnit])]
            if rPeak < minimumResponseAmplitude:
                continue

            for iBin in range(binCenters.size):

                #
                if responseSign[iUnit] == -1:
                    y1 = -1 * (rProbe[iUnit] - mu) + mu
                    y2 = -1 * (rSaccade[iUnit, :, iBin] - mu) + mu
                    y3 = -1 * (rMixed[iUnit, :, iBin] - mu) + mu
                
                #
                else:
                    y1 = rProbe[iUnit]
                    y2 = rSaccade[iUnit, :, iBin]
                    y3 = rMixed[iUnit, :, iBin]

                #
                rObserved = y3 - (y2 - mu)
                rExpected = y1
                y4 = rObserved - rExpected

                #
                data[iUnit, :, iBin, 0, index] = y1
                data[iUnit, :, iBin, 1, index] = y2
                data[iUnit, :, iBin, 2, index] = y3
                data[iUnit, :, iBin, 3, index] = y4

                #
                if nSpikes[iUnit] == 20346:
                    pass

    return data

class SaccadicModulationAcrossTimeByClusterFigure():
    """
    """

    def __init__(
        self,
        ):
        """
        """

        self.data = None
        self.binCenters = None
        self.unitLabels = None
        self.tPeak = None
        self.tPoints = None

        return

    def processPeths(
        self,
        table,
        probeDirection='left',
        responseWindow=(0, 0.3),
        baselineWindow=(-0.2, 0),
        pMax=None,
        rMin=None,
        protocol='dg',
        version=1,
        ):
        """
        """

        # Read the table
        file = h5py.File(table, 'r')

        # Load datasets
        rProbe = np.array(file[f'rProbe/{protocol}/{probeDirection}'])
        rMixed = np.array(file[f'rMixed/{protocol}/{probeDirection}'])
        rSaccade = np.array(file[f'rSaccade/{protocol}/{probeDirection}'])
        pvalues = np.array(file[f'pZeta/{probeDirection}'])
        unitLabels = np.array(file[f'unitLabel'])
        nSpikes = np.array(file[f'nSpikes'])

        #
        tPoints = file[f'rProbe/{protocol}/{probeDirection}'].attrs['t']
        binEdges = file[f'rMixed/{protocol}/{probeDirection}'].attrs['edges']
        binCenters = binEdges.mean(1)

        #
        nUnits, nBinsInHistogram, nBinsInWindow = rMixed.shape
        data = np.full([nUnits, nBinsInHistogram, nBinsInWindow, 4], np.nan)
        tPeak = np.full(nUnits, np.nan)

        #
        for iUnit in range(nUnits):

            #
            if pMax is not None and pvalues[iUnit] > pMax:
                continue

            # Find the maximum of the PETH
            tMask = np.logical_and(
                tPoints >= responseWindow[0],
                tPoints <= responseWindow[1]
            )
            tOffset = np.sum(tPoints < responseWindow[0])
            tPeak[iUnit] = np.argmax(rProbe[iUnit][tMask])

            # Get the baseline FR for the visual-only PETH
            tMask = np.logical_and(
                tPoints >= baselineWindow[0],
                tPoints <= baselineWindow[1]
            )
            rBaseline = rProbe[iUnit][tMask].mean()
            sigma = round(rProbe[iUnit][tMask].std(), 3)

            #
            if rMin is not None and np.interp(tPeak[iUnit], tPoints, rProbe[iUnit] - rBaseline) < rMin:
                continue

            #
            for iBin in range(nBinsInWindow):

                # Compute the difference between the expected and observed responses

                # Version #1
                if version == 1:
                    rObserved = np.clip(np.interp(
                        tPoints,
                        tPoints,
                        rMixed[iUnit, :, iBin] - (rSaccade[iUnit, :, iBin] - rBaseline)
                    ), 0, np.inf)
                    rExpected = np.interp(
                        tPoints,
                        tPoints,
                        rProbe[iUnit]
                    )

                # Version #2
                elif version == 2:
                    rObserved = np.interp(
                        tPoints,
                        tPoints,
                        rMixed[iUnit, :, iBin]
                    )
                    rExpected = np.interp(
                        tPoints,
                        tPoints,
                        rSaccade[iUnit, :, iBin] + (rProbe[iUnit] - rBaseline)
                    )

                #
                rDiff = smooth(rObserved - rExpected, 3)
                rOffset = rDiff[tMask].mean()
                rDiff -= rOffset # Correct for difference in baselines
                rDiff /= sigma # Scale by standard deviation

                #
                data[iUnit, :, iBin, 0] = rProbe[iUnit]
                data[iUnit, :, iBin, 1] = rSaccade[iUnit, :, iBin] - rBaseline
                data[iUnit, :, iBin, 2] = rMixed[iUnit, :, iBin]
                data[iUnit, :, iBin, 3] = rDiff

            #
            # if nSpikes[iUnit] == 26759:
            #    return data[iUnit]

        # Close the opened file
        file.close()

        #
        indices = np.where(np.isnan(unitLabels))[0]
        # indices = np.array([]).astype(int)
        data = np.delete(
            data,
            indices,
            axis=0
        )
        unitLabels = np.delete(unitLabels, indices)
        tPeak = np.delete(tPeak, indices)

        #
        self.data = data
        self.binCenters = binCenters
        self.unitLabels = unitLabels
        self.tPeak = tPeak.astype(int)
        self.tPoints = tPoints

        return data, binCenters, unitLabels

    def plotModulationCurves(
        self,
        figsize=(3, 8)
        ):
        """
        """

        uniqueLabels = np.unique(self.unitLabels)
        fig, axs = plt.subplots(nrows=len(uniqueLabels), sharex=True, sharey=True)

        #
        for labelIndex, unitLabel in enumerate(uniqueLabels):
            color = f'C{labelIndex}'
            labelMask = self.unitLabels == unitLabel
            y = list()
            for iBin in range(self.data.shape[-1]):
                sample = self.data[labelMask, self.tPeak[labelMask], iBin, -1]
                y.append([
                    np.mean(sample) - sem(sample),
                    np.mean(sample),
                    np.mean(sample) + sem(sample)
                ])
            y = np.array(y)
            axs[labelIndex].plot(self.binCenters, y[:, 1], color=color)
            axs[labelIndex].scatter(self.binCenters, y[:, 1], color=color)
            axs[labelIndex].vlines(self.binCenters, y[:, 0], y[:, 2], color=color)

        #
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, axs

    def plotPeths(
        self,
        ):
        """
        """

        uniqueLabels = np.unique(self.unitLabels)
        fig, axs = plt.subplots(nrows=len(uniqueLabels), sharex=True)

        #
        for labelIndex, unitLabel in enumerate(uniqueLabels):
            color = f'C{labelIndex}'
            labelMask = self.unitLabels == unitLabel
            peths = self.data[labelMask, :, 0, 0]
            axs[labelIndex].plot(peths.mean(0), color=color)
            axs[labelIndex].set_ylim([0, axs[labelIndex].get_ylim()[-1]])

        return fig, axs