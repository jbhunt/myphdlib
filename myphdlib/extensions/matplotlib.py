import numpy as np
from matplotlib import pylab as plt
from matplotlib import widgets as wid
from matplotlib import lines
from matplotlib.backend_bases import MouseButton

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
        figsize=(5, 5)
        ):
        """
        """

        #
        self.xlim = (-1, 1)
        self.ylim = (-1, 1)
        self.labels = (
            'Left',
            'Right',
            'Noise',
            'Unscored'
        )

        #
        self.fig, self.ax = plt.subplots()
        self.fig.subplots_adjust(left=0.35)
        self.fig.set_figwidth(figsize[0])
        self.fig.set_figheight(figsize[1])
        self.wave = lines.Line2D([0], [0], color='k')
        self.cross = {
            'vertical': lines.Line2D([0, 0], self.ax.get_ylim(), color='k', alpha=0.3),
            'horizontal': lines.Line2D(self.ax.get_xlim(), [0, 0], color='k', alpha=0.3)
        }
        self.ax.add_line(self.wave)
        for line in self.cross.values():
            self.ax.add_line(line)

        #
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

    def inputSamples(self, samples, gain=1.3, randomizeSamples=False):
        """
        """

        #
        self.xTrain = samples
        if randomizeSamples:
            np.random.shuffle(self.xTrain)
        nSamples, nFeatures = self.xTrain.shape
        self.y = np.full([self.xTrain.shape[0], 1], np.nan)

        #
        self.ylim = np.array([0, nFeatures - 1])
        self.xlim = np.array([
            self.xTrain.min() * gain,
            self.xTrain.max() * gain
        ])

        #
        if nFeatures % 2 == 0:
            center = nFeatures / 2
        else:
            center = (nFeatures - 1) / 2
        self.cross['vertical'].set_data([0, 0], self.ylim)
        self.cross['horizontal'].set_data(self.xlim, [center, center])

        #
        self.updatePlot()
        plt.show()

        return

    def updatePlot(self):
        """
        """

        wave = np.take(self.xTrain, self.sampleIndex, mode='wrap', axis=0)
        self.wave.set_data(wave, np.arange(wave.size))
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

    def isRunning(self):
        """
        """

        return plt.fignum_exists(self.fig.number)

    @property
    def trainingData(self):
        """
        """

        mask = np.invert(np.isnan(self.y.flatten()))
        X = np.diff(self.xTrain[mask, :], axis=1)
        y = self.y[mask, :]

        return X, y

class MarkerPlacer():
    """
    """

    def __init__(self, image=None, ax=None, fig=None, color='r'):
        """
        """

        #
        self.markers = None
        self.points = list()
        if ax is None and fig is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.fig, self.ax = fig, ax
        if image is not None:
            self.image = self.ax.imshow(image, cmap='binary_r')
        else:
            self.image = None
        self.color = color
        self.fig.show()
        self.fig.canvas.callbacks.connect('button_press_event', self.onClick)
        self.fig.canvas.callbacks.connect('key_press_event', self.onPress)

        return

    def drawPoints(self):
        """
        """

        #
        if self.markers is not None:
            self.markers.remove()
        
        #
        x = [point[0] for point in self.points]
        y = [point[1] for point in self.points]
        self.markers = self.ax.scatter(x, y, color=self.color)

        #
        self.fig.canvas.draw()

        return

    def onClick(self, event):
        """
        """

        if event.button == MouseButton.LEFT and event.key == 'shift':
            point = (event.xdata, event.ydata)
            self.points.append(point)
        self.drawPoints()

        return

    def onPress(self, event):
        """
        """
        
        if event.key == 'ctrl+z':
            if self.points:
                point = self.points.pop()
        self.drawPoints()

        return

def placeMarkers(image=None):
    """
    """

    return

class BodypartLabelingGUI():
    """
    """

    def __init__(self, bodyparts=('nasal', 'temporal'), figsize=(5, 5)):
        """
        """

        #
        self.fig, self.ax = plt.subplots()
        self.fig.subplots_adjust(left=0.35)
        self.fig.set_figwidth(figsize[0])
        self.fig.set_figheight(figsize[1])

        #
        self.imageList = list()
        self.imageIndex = 0
        self.imageObject = None
        self.bodypartPositions = list()
        self.bodyparts = bodyparts
        self.markers = [self.ax.scatter([0], [0], alpha=0) for i in range(len(self.bodyparts))]

        # Checkbox panel
        self.checkboxPanel = wid.RadioButtons(
            plt.axes([0.02, 0.5, 0.2, 0.2]),
            labels=bodyparts,
            active=0
        )
        self.checkboxPanel.on_clicked(self.onCheckboxClick)

        # Previous button
        self.previousButton = wid.Button(
            plt.axes([0.02, 0.4, 0.15, 0.05]), 
            label='Previous',
            color='white',
            hovercolor='grey'
        )
        self.previousButton.on_clicked(self.onPreviousButtonClick)

        # Next button
        self.nextButton = wid.Button(
            plt.axes([0.02, 0.3, 0.15, 0.05]),
            label='Next',
            color='white',
            hovercolor='grey'
        )
        self.nextButton.on_clicked(self.onNextButtonClick)

        # Exit button
        self.exitButton = wid.Button(
            plt.axes([0.02, 0.2, 0.15, 0.05]),
            label='Exit',
            color='white',
            hovercolor='grey'
        )
        self.exitButton.on_clicked(self.onExitButtonClick)

        #
        self.fig.canvas.callbacks.connect('button_press_event', self.onMouseClick)
        self.fig.canvas.callbacks.connect('key_press_event', self.onKeyPress)

        #
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.show()

        return

    def onKeyPress(self, event):
        """
        """

        # ctrl+z

        # shift+Arrows

        return

    def onMouseClick(self, event):
        """
        """

        # shift+left click
        if event.button == MouseButton.LEFT and event.key == 'shift':
            self.bodypartPositions[self.imageIndex][self.bodypart] = (event.xdata, event.ydata)

        #
        offsets = list()
        alphas = list()
        for bodypart in self.bodyparts:
            entry = self.bodypartPositions[self.imageIndex]
            if entry[bodypart] is not None:
                offsets.append([entry[bodypart]])
                alphas.append(1)
            else:
                offsets.append([(0, 0)])
                alphas.append(0)

        #
        for marker, offset, alpha in zip(self.markers, offsets, alphas):
            marker.set_offsets(offset)
            marker.set_alpha(alpha)

        #
        self.updatePlot()

        return

    def onCheckboxClick(self, bodypart):
        """
        """

        self.bodypart = bodypart

        return

    def onNextButtonClick(self, event):
        """
        """

        self.imageIndex = np.take(np.arange(len(self.imageList)), self.imageIndex + 1, mode='wrap')
        self.updatePlot()

        return

    def onPreviousButtonClick(self, event):
        """
        """

        self.imageIndex = np.take(np.arange(len(self.imageList)), self.imageIndex - 1, mode='wrap')
        self.updatePlot()

        return

    def onExitButtonClick(self, event):
        """
        """

        plt.close(self.fig)

        return

    def updatePlot(self):
        """
        """

        image = self.imageList[self.imageIndex]
        if self.imageObject is None:
            self.imageObject = self.ax.imshow(image)
        else:
            self.imageObject.set_data(image)
        self.fig.canvas.draw()

        return

    def inputImages(self, images, resetImageList=False):
        """
        """

        if resetImageList:
            self.bodypartPositions = list()
            self.imageList = list()

        for image in images:
            self.imageList.append(image)
            self.bodypartPositions.append({bodypart: None for bodypart in self.bodyparts})

        #
        self.updatePlot()

        return
    