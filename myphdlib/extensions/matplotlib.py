import numpy as np
from matplotlib import pylab as plt
from matplotlib import widgets as wid
from matplotlib import lines

class VerticalLineBuilder():
    """
    """

    def __init__(self, data, yLimits=(0, 1)):
        self.fig, self.ax = plt.subplots()
        self.ax.plot(data)
        self.yLimits = yLimits
        self.ax.set_ylim(self.yLimits)
        self.fig.canvas.callbacks.connect('button_press_event', self.onClick)
        return

    def onClick(self, event):
        """
        """

        # Create new line
        if event.button == 1 and event.key == 'shift':
            line = self.ax.plot([event.xdata, event.xdata], self.yLimits, color='r')[0]

        # Delete the closest line
        elif event.button == 3 and event.key == 'shift':
            if len(self.ax.lines) == 1:
                return
            dists = list()
            for line in self.ax.lines[1:]:
                x1 = event.xdata
                x2 = line.get_data()[0][0]
                dist = np.linalg.norm(x2 - x1)
                dists.append(dist)
            closest = np.argmin(np.abs(dists)) + 1
            self.ax.lines.remove(self.ax.lines[closest])

        #
        self.fig.canvas.draw()

        return

def placeVerticalLines(data, yMarginFraction=0.1):
    """
    """

    yRange = data.max() - data.min()
    yMargin = yRange * yMarginFraction
    yLimits = [data.min() - yMargin, data.max() + yMargin]
    builder = VerticalLineBuilder(data, yLimits=yLimits)
    plt.show(block=True)
    xPositions = list()
    for line in builder.ax.lines[1:]:
        xdata, ydata = line.get_data()
        xPositions.append(xdata[0])

    return np.array(xPositions)

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

    # TODO: Recode this method
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