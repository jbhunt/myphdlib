import h5py
import numpy as np
from scipy.stats import sem
from matplotlib import pylab as plt
from myphdlib.general.toolkit import smooth

def getTimePoints(
    hdf,
    ):
    """
    """

    return

def getUnitLabels(
    hdf,
    ):
    """
    """

    with h5py.File(hdf, 'r') as stream:
        unitLabels = np.array(stream['unitLabels'])

    return unitLabels

def determineResponseFeatures(
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
    maximumProbability=0.05,
    minimumResponseAmplitude=3,
    minimumBaselineActivity=1,
    ):
    """
    """

    #
    directionPreference, responseSign = determineResponseFeatures(
        hdf,
        responseWindow=responseWindow,
        baselineWindow=baselineWindow
    )

    # Read the table
    file = h5py.File(hdf, 'r')

    # Load datasets
    tPoints = file[f'rProbe/{protocol}/left'].attrs['t']
    binEdges = file[f'rMixed/{protocol}/left'].attrs['edges']
    nSpikes = np.array(file['nSpikes'])
    binCenters = binEdges.mean(1)

    #
    nUnits, nBinsInHistogram, nBinsInWindow = file[f'rMixed/{protocol}/left'].shape
    data = np.full([nUnits, nBinsInHistogram, nBinsInWindow, 4, 2], np.nan)
    iPeak = np.full(nUnits, np.nan)

    #
    rProbe = {
        'left': np.array(file[f'rProbe/{protocol}/left']),
        'right': np.array(file[f'rProbe/{protocol}/right'])
    }
    rSaccade = {
        'left': np.array(file[f'rSaccade/{protocol}/left']),
        'right': np.array(file[f'rSaccade/{protocol}/right'])
    }
    rMixed = {
        'left': np.array(file[f'rMixed/{protocol}/left']),
        'right': np.array(file[f'rMixed/{protocol}/right'])
    }
    pZeta = {
        'left': np.array(file[f'pZeta/left']),
        'right': np.array(file[f'pZeta/right'])
    }

    for iUnit in range(nUnits):

        #
        if directionPreference[iUnit] == -1:
            probeDirection = 'left'
        else:
            probeDirection = 'right'
        if responseSign[iUnit] < 0:
            coeff = -1
        else:
            coeff = +1

        #
        if maximumProbability is not None and pZeta[probeDirection][iUnit] > maximumProbability:
            continue

        # Get the baseline FR for the visual-only PETH
        binIndices = np.where(np.logical_and(
            tPoints >= baselineWindow[0],
            tPoints <= baselineWindow[1]
        ))[0]
        mu = round(rProbe[probeDirection][iUnit][binIndices].mean(), 3)
        sigma = round(rProbe[probeDirection][iUnit][binIndices].std(), 3)

        # Filter out units with low baselines
        if mu < minimumBaselineActivity:
            continue

        # Find the maximum of the PETH
        binIndices = np.where(np.logical_and(
            tPoints >= responseWindow[0],
            tPoints <= responseWindow[1]
        ))[0]
        iPeak[iUnit] = np.argmax(coeff * (rProbe[probeDirection][iUnit][binIndices] - mu)) + np.sum(tPoints < responseWindow[0])
        responseAmplitude = (coeff * (rProbe[probeDirection][iUnit] - mu))[int(iPeak[iUnit])]

        # Filter out units with low amplitude resopnses
        if responseAmplitude < minimumResponseAmplitude:
            continue

        #
        for probeDirection in ('left', 'right'):

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

            for iBin in range(binCenters.size):

                #
                if responseSign[iUnit] == -1:
                    y1 = -1 * (rProbe[probeDirection][iUnit] - mu) + mu
                    y2 = -1 * (rSaccade[probeDirection][iUnit, :, iBin] - mu) + mu
                    y3 = -1 * (rMixed[probeDirection][iUnit, :, iBin] - mu) + mu
                
                #
                else:
                    y1 = rProbe[probeDirection][iUnit]
                    y2 = rSaccade[probeDirection][iUnit, :, iBin]
                    y3 = rMixed[probeDirection][iUnit, :, iBin]

                #
                rObserved = y3 - (y2 - mu)
                rExpected = y1
                y4 = rObserved - rExpected

                #
                data[iUnit, :, iBin, 0, index] = y1 - mu
                data[iUnit, :, iBin, 1, index] = y2
                data[iUnit, :, iBin, 2, index] = y3
                data[iUnit, :, iBin, 3, index] = y4

            #
            # if nSpikes[iUnit] == 20346:
            #     pass

    #
    binIndices = np.where(np.logical_and(
        tPoints >= baselineWindow[0],
        tPoints <= baselineWindow[1]
    ))[0]
    sigma = data[:, binIndices, 0, 0, 0].std(1).reshape(-1, 1)

    #
    indicesToDelete = np.where(np.vstack([
        np.isnan(sigma).all(1),
        np.isnan(data[:, :, 0, 0, 0]).all(axis=1),
        np.isnan(data[:, :, 0, 0, 1]).all(axis=1)
    ]).any(0))[0]
    data = np.delete(data, indicesToDelete, axis=0)
    sigma = np.delete(sigma, indicesToDelete, axis=0)

    #
    # for iBin in range(binCenters.size):
    #     for iTerm in range(4):
    #         for iPeth in range(2):
    #             data[:, :, iBin, iTerm, iPeth] /= sigma

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