import h5py
import numpy as np
from scipy.stats import sem
from matplotlib import pylab as plt
from myphdlib.general.toolkit import smooth

def getTimePointsForHistogram(
    hdf,
    protocol='dg'
    ):
    """
    """

    #
    with h5py.File(hdf, 'r') as stream:
        t = np.array(stream[f'rProbe/{protocol}/left'].attrs['t'])

    return t

def getTimePointsForPerisaccadicWindow(
    hdf,
    protocol='dg'
    ):
    """
    """
    #
    with h5py.File(hdf, 'r') as stream:
        edges = np.array(stream[f'rMixed/{protocol}/left'].attrs['edges'])
        t = edges.mean(1)

    return t

def loadUnitLabels(
    hdf,
    ):
    """
    """

    with h5py.File(hdf, 'r') as stream:
        unitLabels = np.array(stream['unitLabel'])

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
        binIndicesForBaselineWindow = np.where(np.logical_and(
            tPoints >= baselineWindow[0],
            tPoints <= baselineWindow[1]
        ))[0]
        mu = round(rProbe[probeDirection][iUnit][binIndicesForBaselineWindow].mean(), 3)
        sigma = round(rProbe[probeDirection][iUnit][binIndicesForBaselineWindow].std(), 3)

        # Filter out units with low baselines
        if mu < minimumBaselineActivity:
            continue

        # Find the maximum of the PETH
        binIndicesForResponseWindow = np.where(np.logical_and(
            tPoints >= responseWindow[0],
            tPoints <= responseWindow[1]
        ))[0]
        iPeak[iUnit] = np.argmax(coeff * (rProbe[probeDirection][iUnit][binIndicesForResponseWindow] - mu)) + np.sum(tPoints < responseWindow[0])
        responseAmplitude = (coeff * (rProbe[probeDirection][iUnit] - mu))[int(iPeak[iUnit])]

        # Filter out units with low amplitude resopnses
        if responseAmplitude < minimumResponseAmplitude:
            continue

        #
        for probeDirection in ('left', 'right'):

            #
            if directionPreference[iUnit] == -1 and probeDirection == 'left':
                iPeth = 0
            elif directionPreference[iUnit] == -1 and probeDirection == 'right':
                iPeth = 1
            elif directionPreference[iUnit] == +1 and probeDirection == 'left':
                iPeth = 1
            elif directionPreference[iUnit] == +1 and probeDirection == 'right':
                iPeth = 0
            else:
                continue

            # Get the baseline FR for the visual-only PETH
            binIndices = np.where(np.logical_and(
                tPoints >= baselineWindow[0],
                tPoints <= baselineWindow[1]
            ))[0]
            mu = round(rProbe[probeDirection][iUnit][binIndices].mean(), 3)

            #
            for iBin in range(binCenters.size):

                # Reflect PSTHs around the baseline for negatively-signed responses
                if responseSign[iUnit] == -1:
                    y1 = -1 * (rProbe[probeDirection][iUnit] - mu) + mu
                    y2 = -1 * (rSaccade[probeDirection][iUnit, :, iBin] - mu) + mu
                    y3 = -1 * (rMixed[probeDirection][iUnit, :, iBin] - mu) + mu
                
                # Do nothing for positively-signed responses
                else:
                    y1 = rProbe[probeDirection][iUnit]
                    y2 = rSaccade[probeDirection][iUnit, :, iBin]
                    y3 = rMixed[probeDirection][iUnit, :, iBin]

                #
                y4 = y3 - y2
                y4 -= y4[binIndicesForBaselineWindow].mean()

                #
                data[iUnit, :, iBin, 0, iPeth] = y1 - y1[binIndicesForBaselineWindow].mean()
                data[iUnit, :, iBin, 1, iPeth] = y2 - y2[binIndicesForBaselineWindow].mean()
                data[iUnit, :, iBin, 2, iPeth] = y3 - y3[binIndicesForBaselineWindow].mean()
                data[iUnit, :, iBin, 3, iPeth] = y4
                
    #
    file.close()

    #
    sigma = data[:, binIndicesForResponseWindow, 0, 0, 0].std(1).reshape(-1, 1)

    #
    indices = np.where(np.vstack([
        np.isnan(sigma).all(1),
        np.isnan(data[:, :, 0, 0, 0]).all(axis=1),
        np.isnan(data[:, :, 0, 0, 1]).all(axis=1)
    ]).any(0))[0]
    data = np.delete(data, indices, axis=0)
    sigma = np.delete(sigma, indices, axis=0)

    return data, sigma, indices

class SaccadicModulationAnalysis():
    """
    """

    def __init__(
        self,
        ):
        """
        """

        self.data = None
        self.labels = None
        self.sigma = None
        self.t = None

        return
    
    def loadData(
        self,
        hdf
        ):
        """
        """

        self.data, self.sigma, indices = loadPeths(hdf)
        self.labels = loadUnitLabels(hdf)
        self.labels = np.delete(self.labels, indices, axis=0)
        self.tHist = getTimePointsForHistogram(hdf)
        self.tWin = getTimePointsForPerisaccadicWindow(hdf)

        return
    
    def plotModulationCurves(
        self,
        responseWindow=(0, 0.3),
        figsize=(5, 5),
        fig=None,
        ):
        """
        """

        if fig is None:
            fig, ax = plt.subplots()
        else:
            ax = fig.axes[0]
        nBins = self.data.shape[2]
        uniqueLabels = np.unique(self.labels)
        uniqueLabels = np.delete(uniqueLabels, np.isnan(uniqueLabels))
        binIndices = np.where(np.logical_and(
            self.t >= responseWindow[0],
            self.t <= responseWindow[1]
        ))[0]
        y0 = list()
        for iLabel, unitLabel in enumerate(uniqueLabels):
            y1 = list()
            for iUnit in np.where(self.labels == unitLabel)[0]:
                y2 = list()
                for iBin in range(nBins):
                    rProbeExpected = self.data[iUnit, :, iBin,  0, 0] / self.sigma[iUnit]
                    rProbeComputed = self.data[iUnit, :, iBin, -1, 0] / self.sigma[iUnit]
                    iPeak = np.argmax(rProbeExpected[binIndices]) + binIndices.min()
                    y2.append(rProbeComputed[iPeak] - rProbeExpected[iPeak])
                y1.append(y2)
            y1 = np.array(y1)
            y0.append(y1)
            color = f'C{iLabel}'
            ax.plot(self.tWin, y1.mean(0), color=color, label=f'C{iLabel + 1}')
            ax.vlines(
                self.tWin,
                y1.mean(0) - sem(y1, axis=0),
                y1.mean(0) + sem(y1, axis=0),
                color=color
            )

        #
        ax.set_ylabel('Modulation')
        ax.set_xlabel('Time from saccade (sec)')
        ax.legend()

        return fig, ax, y0
    
    def plotAveragePeths(
        self,
        figsize=(10, 8),
        fig=None
        ):
        """
        """

        uniqueLabels = np.unique(self.labels)
        uniqueLabels = np.delete(uniqueLabels, np.isnan(uniqueLabels))
        nBins = self.data.shape[2]
        if fig is None:
            fig, axs = plt.subplots(nrows=len(uniqueLabels), ncols=nBins, sharex=True, sharey=False)
        else:
            axs = fig.axes

        #
        for labelIndex, unitLabel in enumerate(uniqueLabels):
            if np.isnan(unitLabel):
                continue
            color = f'C{labelIndex}'
            labelMask = np.ravel(self.labels == unitLabel)
            for iBin in range(nBins):
                rProbeExpected = self.data[labelMask, :, iBin,  0, 0]
                rProbeComputed = self.data[labelMask, :, iBin, -1, 0]
                axs[labelIndex, iBin].plot(self.tHist, np.nanmean(rProbeExpected / self.sigma[labelMask], axis=0), color=color, alpha=0.7)
                axs[labelIndex, iBin].plot(self.tHist, np.nanmean(rProbeComputed / self.sigma[labelMask], axis=0), color='k', alpha=0.3)

            #
            y1, y2 = np.inf, -np.inf
            for ax in axs[labelIndex, :]:
                ylim = ax.get_ylim()
                if ylim[0] < y1:
                    y1 = ylim[0]
                if ylim[1] > y2:
                    y2 = ylim[1]
            for i, ax in enumerate(axs[labelIndex, :]):
                if i != 0:
                    ax.set_yticks([])
                # ax.set_ylim([y1, y2])
                ax.set_ylim([-1, 3])

        #
        for i, ax in enumerate(axs[:, 0]):
            ax.set_yticks([])
            ax.set_ylabel(f'C{i + 1}', rotation='horizontal', ha='right', va='center')

        #
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, axs