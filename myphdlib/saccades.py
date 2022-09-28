import os
import re
from venv import create
import numpy as np
import pathlib as pl
from matplotlib import lines
from matplotlib import pylab as plt
from matplotlib import widgets as wid
from datetime import datetime
from sklearn.neural_network import MLPClassifier
from scipy.optimize import curve_fit as fitCurve
from scipy.signal import find_peaks as findPeaks
from scipy.stats import ttest_1samp as tTestOneSample
from myphdlib.toolkit import smooth
from myphdlib import pipeline
from myphdlib.factory import SessionFactory
from myphdlib.functions import sigmoid, relu

class SaccadeLabelingGUI():
    """
    """

    def __init__(
        self,
        xlim=(-4, 4),
        side='both'
        ):
        """
        """

        #
        self.xlim = xlim
        self.side = side
        self.labels = (
            'Left',
            'Right',
            'Noise',
            'Unscored'
        )

        #
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(left=0.35)
        self.line = lines.Line2D([0], [0], color='k')
        self.midline = lines.Line2D(self.xlim, [0, 0], color='r')
        self.ax.add_line(self.line)
        self.ax.add_line(self.midline)
        self.sampleIndex = 0

        # Checkbox panel
        self.checkboxPanel = wid.RadioButtons(
            plt.axes([0.02, 0.5, 0.2, 0.2]),
            labels=self.labels,
            active=3
        )
        self.checkboxPanel.on_clicked(self.onCheckboxClicked)

        # Previous button
        self.previousButton = wid.Button(
            plt.axes([0.02, 0.4, 0.15, 0.05]), 
            label='Previous',
            color='white',
            hovercolor='grey'
        )
        self.previousButton.on_clicked(self.onPreviousButtonClicked)

        # Next button
        self.nextButton = wid.Button(
            plt.axes([0.02, 0.3, 0.15, 0.05]),
            label='Next',
            color='white',
            hovercolor='grey'
        )
        self.nextButton.on_clicked(self.onNextButtonClicked)

        # Exit button
        self.exitButton = wid.Button(
            plt.axes([0.02, 0.2, 0.15, 0.05]),
            label='Exit',
            color='white',
            hovercolor='grey'
        )
        self.exitButton.on_clicked(self.onExitButtonClicked)

        return

    def collectSamples(
        self,
        datasetNames=['Realtime'],
        randomizeSamples=True,
        ):
        """
        """

        self.sampleIndex = 0

        #
        xTrain = list()
        factory = SessionFactory()
        for datasetName in datasetNames:
            for obj, animal, date, session in pipeline.iterateSessions(datasetName):
                if hasattr(obj, 'saccadeWaveformsPutative'):
                    if self.side == 'both':
                        keys = ['left', 'right']
                    elif self.side == 'left':
                        keys = ['left']
                    elif self.side == 'right':
                        keys = ['right']
                    for key in keys:
                        for wave in obj.saccadeWaveformsPutative[key]:
                            xTrain.append(wave)

        # Cast to numpy array
        self.xTrain = np.array(xTrain)

        # Randomize samples
        if randomizeSamples:
            np.random.shuffle(self.xTrain)

        # Create a new array for labels
        self.y = np.full([self.xTrain.shape[0], 1], np.nan)

        # Draw the first sample
        self.ylim = 0, self.xTrain.shape[1]
        midpoint = (self.ylim[1] - 1) / 2
        self.midline.set_data(self.xlim, [midpoint, midpoint])
        self.updatePlot()
        plt.show()

        return

    def updatePlot(self):
        """
        """

        wave = np.take(self.xTrain, self.sampleIndex, mode='wrap', axis=0)
        self.line.set_data(wave, np.arange(wave.size))
        self.ax.set_ylim(self.ylim)
        self.ax.set_xlim(self.xlim)
        self.fig.canvas.draw()

        return

    def updateCheckboxPanel(self):
        """
        """

        currentLabel = np.take(self.y, self.sampleIndex, axis=0, mode='wrap')
        if currentLabel == -1:
            self.checkboxPanel.set_active(0)
        elif currentLabel == +1:
            self.checkboxPanel.set_active(1)
        elif currentLabel == 0:
            self.checkboxPanel.set_active(2)
        elif np.isnan(currentLabel):
            self.checkboxPanel.set_active(3)

        return

    def onCheckboxClicked(self, buttonLabel):
        """
        """

        checkboxIndex = np.where(np.array(self.labels) == buttonLabel)[0].item()
        newLabel = np.array([-1, 1, 0, np.nan])[checkboxIndex]
        sampleIndex = np.take(np.arange(self.y.size), self.sampleIndex, mode='wrap')
        self.y[sampleIndex] = newLabel

        return

    def onNextButtonClicked(self, event):
        """
        """

        self.sampleIndex += 1
        self.updatePlot()
        self.updateCheckboxPanel()

        return

    def onPreviousButtonClicked(self, event):
        """
        """

        self.sampleIndex -= 1
        self.updatePlot()
        self.updateCheckboxPanel()

        return

    def onExitButtonClicked(self, event):
        """
        """

        plt.close(self.fig)

        return

    @property
    def trainingData(self):
        """
        """

        mask = np.invert(np.isnan(self.y.flatten()))
        X = np.diff(self.xTrain[mask, :], axis=1)
        y = self.y[mask, :]

        return X, y

def labelPutativeSaccades(
    datasetNames=['Realtime'],
    side='both',
    xlim=(-3, 3),
    ):
    """
    """

    plt.ion()
    gui = SaccadeLabelingGUI(xlim, side)
    gui.collectSamples(datasetNames)
    while plt.fignum_exists(gui.fig.number):
        plt.pause(0.05)
        continue

    #
    user = os.environ['USER']
    result = list(pl.Path(f'/media/{user}/').glob('JH-DATA-*'))
    if len(result) == 1:
        eHDD = result.pop()
    saccadesFolderPath = eHDD.joinpath('Saccades')
    if saccadesFolderPath.exists() == False:
        raise Exception('Could not locate output folder')
    
    #
    today = datetime.today().strftime('%Y-%m-%d')
    sessionNumber = 1
    for file in saccadesFolderPath.iterdir():
        if bool(re.search(f'.*{today}.*', file.name)):
            sessionNumber += 1
            
    identifier = str(sessionNumber).zfill(3)
    name = f'{today}_labeledSaccadeWaveforms_session{identifier}'
    file = saccadesFolderPath.joinpath(name)
    
    #
    X, y = gui.trainingData
    stacked = np.hstack([X, y])
    np.save(str(file), stacked)

    return gui.trainingData

def createTrainingDataset(
    specificDates=['2022-06-27'],
    dateRange=None
    ):
    """
    """

    #
    user = os.environ['USER']
    result = list(pl.Path(f'/media/{user}/').glob('JH-DATA-*'))
    if len(result) == 1:
        eHDD = result.pop()
    saccadesFolderPath = eHDD.joinpath('Saccades')
    if saccadesFolderPath.exists() == False:
        raise Exception('Could not locate output folder')
    
    #
    files = list()
    for file in saccadesFolderPath.iterdir():
        for date in specificDates:
            if bool(re.search(f'.*{date}.*', file.name)):
                files.append(str(file))
    
    # TODO: For each sample, make sure its not a duplicate
    X, y = list(), list()
    for file in files:
        stacked = np.load(file)
        for sample in stacked[:, :-1]:
            X.append(sample)
        for label in stacked[:, -1]:
            y.append(label)
    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    return X, y

def determineSaccadeOnset(
    saccadeWaveform,
    saccadeWaveformSample,
    deviationThreshold=0.2,
    sampleTimeWindow=(0, 0.1),
    acquisitionFramerate=200,
    modelFunction=None,
    samplesAfterPeakVelocity=5,
    ):
    """
    """

    #
    x = np.arange(saccadeWaveform.size)
    yTrue = saccadeWaveform

    if modelFunction == 'Sigmoid':
        f = sigmoid
        popt, pcov = fitCurve(f, x, yTrue, method='trf', maxfev=1000000)
        yFit = sigmoid(x, *popt)

        #
        dy = np.diff(yFit)
        pvalues = list()
        for index, value in enumerate(dy):
            t, p = tTestOneSample(dy, value)
            pvalues.append(p)

        #
        peakIndices, peakProperties = findPeaks(pvalues, height=0.2)
        if len(peakIndices) != 2:
            raise Exception('Failed to determine saccade onset')
        saccadeOnset = peakIndices[0] + 1
        sampleOffset = saccadeOnset - int(saccadeWaveform.size / 2)

    #
    elif modelFunction == 'ReLU':
        stop = int(yTrue.size / 2) + samplesAfterPeakVelocity
        f = relu
        y = yTrue[:stop]
        x = x[:stop]
        popt, pcov = fitCurve(f, x, y, method='trf', maxfev=1000000)
        m, x0, lb = popt
        saccadeOnset = round(x0) + 1
        sampleOffset = saccadeOnset - int(saccadeWaveform.size / 2)

    #
    elif modelFunction == 'Sigmoid-Exponential':
        pass

    # Try to determine the onset algorithmically
    elif modelFunction == None:
        start, stop = tuple(map(round, np.array(sampleTimeWindow) * acquisitionFramerate))
        dx = np.diff(saccadeWaveformSample[:, start: stop], axis=1).flatten()
        if dx[int((dx.size + 1) / 2)] < 0:
            dx *= -1
        mu, sigma = dx.mean(), dx.std()
        for frameIndex, saccadeOnset in enumerate(np.arange(0, int(saccadeWaveform.size / 2), 1)[::-1]):
            value = (saccadeWaveform[saccadeOnset] - mu) / sigma
            if value < deviationThreshold:
                sampleOffset = -1 * (frameIndex + 1)
                break

    return saccadeOnset, sampleOffset