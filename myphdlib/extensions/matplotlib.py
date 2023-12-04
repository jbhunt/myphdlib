import numpy as np
from matplotlib import pylab as plt
from matplotlib import widgets as wid
from matplotlib import lines
from matplotlib.backend_bases import MouseButton
from matplotlib.colors import LinearSegmentedColormap
import matplotlib as mpl

def removeArrowKeyBindings():
    """
    """
    
    pairs = (
        ('back', 'left'),
        ('forward', 'right')
    )
    for action, key in pairs:
        if key in mpl.rcParams[f'keymap.{action}']:
            mpl.rcParams[f'keymap.{action}'].remove(key)

    return

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

    yMin, yMax = np.nanmin(data), np.nanmax(data)
    yRange = yMax - yMin
    yMargin = yRange * yMarginFraction
    yLimits = [yMin - yMargin, yMax + yMargin]
    builder = VerticalLineBuilder(data, yLimits=yLimits)
    plt.show(block=True)
    xPositions = list()
    for line in builder.ax.lines[1:]:
        xdata, ydata = line.get_data()
        xPositions.append(xdata[0])

    return np.array(xPositions)

class SaccadeDirectionLabelingGUI():
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
        removeArrowKeyBindings()
        self.fig.canvas.callbacks.connect('key_press_event', self.onKeyPress)

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
            np.nanmin(self.xTrain) * gain,
            np.nanmax(self.xTrain) * gain
        ])
        if np.any(np.isnan(self.xlim)):
            raise Exception('Fuck')

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

    def cycleRadioButtons(self, direction=-1):
        """
        """

        buttonLabel = self.checkboxPanel.value_selected
        currentButtonIndex = np.where(np.array(self.labels) == buttonLabel)[0].item()
        nextButtonIndex = np.take(np.arange(4), currentButtonIndex + direction, mode='wrap')
        self.checkboxPanel.set_active(nextButtonIndex)

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

    def updatePlot(self):
        """
        """

        wave = np.take(self.xTrain, self.sampleIndex, mode='wrap', axis=0)
        self.wave.set_data(wave, np.arange(wave.size))
        self.ax.set_ylim(self.ylim)
        self.ax.set_xlim(self.xlim)
        self.fig.canvas.draw()

        return

    def onKeyPress(self, event):
        """
        """

        if event.key in ('up', 'down', 'left', 'right'):
            if event.key == 'up':
                self.cycleRadioButtons(-1)
            if event.key == 'down':
                self.cycleRadioButtons(+1)
            self.updateCheckboxPanel()

        if event.key == 'enter':
            self.sampleIndex += 1
            self.updatePlot()
            self.updateCheckboxPanel()

        return

    @property
    def trainingData(self):
        """
        """

        mask = np.invert(np.isnan(self.y.flatten()))
        X = self.xTrain[mask, :]
        y = self.y[mask, :]

        return X, y

class SaccadeEpochLabelingGUI():
    """
    """

    def __init__(
        self,
        figsize=(6, 6)
        ):
        """
        """

        #
        self.xlim = (-1, 1)
        self.ylim = (-1, 1)
        self.t = np.array([])
        self.tc = 0

        #
        self.labels = (
            'Start',
            'Stop',
        )
        self.label = 'Start'

        #
        self.fig, self.ax = plt.subplots()
        self.fig.subplots_adjust(left=0.35)
        self.fig.set_figwidth(figsize[0])
        self.fig.set_figheight(figsize[1])
        self.lines = {
            'wave': lines.Line2D([0], [0], color='k', alpha=0.5),
            'start': lines.Line2D([0, 0], self.ax.get_ylim(), color='r', alpha=0.5),
            'stop': lines.Line2D([0, 0], self.ax.get_ylim(), color='b', alpha=0.5)
        }
        for line in self.lines.values():
            self.ax.add_line(line)
        self.line = self.lines['start']

        #
        self.sampleIndex = 0

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
            plt.axes([0.02, 0.2, 0.1, 0.1]),
            label='Exit',
            color='white',
            hovercolor='grey'
        )
        self.exitButton.on_clicked(self.onExitButtonClicked)

        # Checkbox panel
        self.checkboxPanel = wid.RadioButtons(
            plt.axes([0.02, 0.5, 0.2, 0.2]),
            labels=self.labels,
            active=0
        )
        self.checkboxPanel.on_clicked(self.onCheckboxClicked)

        #
        removeArrowKeyBindings()
        self.fig.canvas.callbacks.connect('key_press_event', self.onKeyPress)
        self.fig.canvas.callbacks.connect('button_press_event', self.onButtonPress)

    def updatePlot(self):
        """
        """

        wave = np.take(self.xTrain, self.sampleIndex, mode='wrap', axis=0)
        self.lines['wave'].set_data(self.t, wave)
        x1, x2 = np.take(self.y, self.sampleIndex, mode='wrap', axis=0)
        if np.isnan([x1, x2]).all():
            x1, x2 = self.tc, self.tc
        self.lines['start'].set_data([x1, x1], self.ylim)
        self.lines['stop'].set_data([x2, x2], self.ylim)
        self.fig.canvas.draw()

        return

    def resetAxesLimits(self):
        """
        """

        self.ax.set_ylim(self.ylim)
        self.ax.set_xlim(self.xlim)

        return

    def inputSamples(self, samples, labels, gain=1.5, randomizeSamples=False):
        """
        """

        #
        self.xTrain = samples
        if randomizeSamples:
            np.random.shuffle(self.xTrain)
        nSamples, nFeatures = self.xTrain.shape
        self.y = np.full([self.xTrain.shape[0], 2], np.nan)
        self.z = labels

        #
        self.xlim = np.array([0, nFeatures - 1])
        self.ylim = np.array([
            np.nanmin(self.xTrain) * gain,
            np.nanmax(self.xTrain) * gain
        ])

        #
        self.t = np.arange(nFeatures)
        self.tc = nFeatures / 2
        for key, line in self.lines.items():
            if key == 'wave':
                continue
            line.set_data([self.tc, self.tc], self.ylim)
        self.xlim = np.array([self.t.min(), self.t.max()])

        #
        self.resetAxesLimits()
        self.updatePlot()
        plt.show()

        return

    def onNextButtonClicked(self, event):
        """
        """

        self.sampleIndex += 1
        self.resetAxesLimits()
        self.updatePlot()
        self.label = 'Start'
        self.updateCheckboxPanel()

        return

    def onPreviousButtonClicked(self, event):
        """
        """

        self.sampleIndex -= 1
        self.resetAxesLimits()
        self.updatePlot()
        self.label = 'Start'
        self.updateCheckboxPanel()

        return

    def onExitButtonClicked(self, event):
        """
        """

        plt.close(self.fig)

        return

    def updateCheckboxPanel(self):
        """
        """

        if self.label == 'Start':
            self.checkboxPanel.set_active(0)
        else:
            self.checkboxPanel.set_active(1)

        return

    def onCheckboxClicked(self, buttonLabel):
        """
        """

        if buttonLabel == 'Start':
            self.line = self.lines['start']
            self.label = 'Start'
        else:
            self.line = self.lines['stop']
            self.label = 'Stop'

        return

    def onKeyPress(self, event):
        """
        """

        # Move the epoch boundary left or right
        clicked = False
        if event.key in ('left', 'shift+left', 'right', 'shift+right'):
            clicked = True

        if clicked:
            if event.key in  ('left', 'right'):
                offset = 0.05
            elif event.key in ('shift+left', 'shift+right'):
                offset = 1
            x, y = self.line.get_data()
            if 'left' in event.key:
                xp = [
                    x[0] - offset,
                    x[1] - offset
                ]
                self.line.set_data(xp, y)
            elif 'right' in event.key:
                xp = [
                    x[0] + offset,
                    x[1] + offset
                ]
                self.line.set_data(xp, y)
            self.y[self.sampleIndex, 0] = self.lines['start'].get_data()[0][0]
            self.y[self.sampleIndex, 1] = self.lines['stop'].get_data()[0][0]
            self.updatePlot()

        # Toggle epoch boundaries
        clicked = False
        if event.key in ('up', 'down'):
            clicked = True

        if clicked:
            if self.label == 'Start':
                self.label = 'Stop'
                self.updateCheckboxPanel()
            elif self.label == 'Stop':
                self.label = 'Start'
                self.updateCheckboxPanel()


    def onButtonPress(self, event):
        """
        """

        # Move the epoch boundary where a click was made
        clicked = False
        if hasattr(event, 'button'):
            clicked = True
        
        if clicked:
            if event.button == 1 and event.key == 'shift':
                xf = event.xdata
                self.line.set_data([xf, xf], self.ylim)
                self.y[self.sampleIndex, 0] = self.lines['start'].get_data()[0][0]
                self.y[self.sampleIndex, 1] = self.lines['stop'].get_data()[0][0]
                self.updatePlot()

        return

    def isRunning(self):
        """
        """

        return plt.fignum_exists(self.fig.number)

    @property
    def labeledSamplesMask(self):
        return np.invert(np.isnan(self.y).all(axis=1))

    @property
    def trainingData(self):
        """
        """

        X = self.xTrain[self.labeledSamplesMask, :]
        y = np.around(self.y[self.labeledSamplesMask, :] - self.tc, 3)
        z = self.z[self.labeledSamplesMask, :]

        return X, y, z


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

parulaColorspace = [
    [0.2081, 0.1663, 0.5292],
    [0.2116238095, 0.1897809524, 0.5776761905], 
    [0.212252381, 0.2137714286, 0.6269714286],
    [0.2081, 0.2386, 0.6770857143], 
    [0.1959047619, 0.2644571429, 0.7279],
    [0.1707285714, 0.2919380952, 0.779247619],
    [0.1252714286, 0.3242428571, 0.8302714286], 
    [0.0591333333, 0.3598333333, 0.8683333333],
    [0.0116952381, 0.3875095238, 0.8819571429],
    [0.0059571429, 0.4086142857, 0.8828428571], 
    [0.0165142857, 0.4266, 0.8786333333],
    [0.032852381, 0.4430428571, 0.8719571429],
    [0.0498142857, 0.4585714286, 0.8640571429], 
    [0.0629333333, 0.4736904762, 0.8554380952],
    [0.0722666667, 0.4886666667, 0.8467],
    [0.0779428571, 0.5039857143, 0.8383714286], 
    [0.079347619, 0.5200238095, 0.8311809524],
    [0.0749428571, 0.5375428571, 0.8262714286],
    [0.0640571429, 0.5569857143, 0.8239571429], 
    [0.0487714286, 0.5772238095, 0.8228285714],
    [0.0343428571, 0.5965809524, 0.819852381],
    [0.0265, 0.6137, 0.8135],
    [0.0238904762, 0.6286619048, 0.8037619048],
    [0.0230904762, 0.6417857143, 0.7912666667], 
    [0.0227714286, 0.6534857143, 0.7767571429],
    [0.0266619048, 0.6641952381, 0.7607190476],
    [0.0383714286, 0.6742714286, 0.743552381], 
    [0.0589714286, 0.6837571429, 0.7253857143], 
    [0.0843, 0.6928333333, 0.7061666667],
    [0.1132952381, 0.7015, 0.6858571429], 
    [0.1452714286, 0.7097571429, 0.6646285714],
    [0.1801333333, 0.7176571429, 0.6424333333],
    [0.2178285714, 0.7250428571, 0.6192619048], 
    [0.2586428571, 0.7317142857, 0.5954285714],
    [0.3021714286, 0.7376047619, 0.5711857143],
    [0.3481666667, 0.7424333333, 0.5472666667], 
    [0.3952571429, 0.7459, 0.5244428571],
    [0.4420095238, 0.7480809524, 0.5033142857],
    [0.4871238095, 0.7490619048, 0.4839761905], 
    [0.5300285714, 0.7491142857, 0.4661142857],
    [0.5708571429, 0.7485190476, 0.4493904762],
    [0.609852381, 0.7473142857, 0.4336857143], 
    [0.6473, 0.7456, 0.4188],
    [0.6834190476, 0.7434761905, 0.4044333333], 
    [0.7184095238, 0.7411333333, 0.3904761905], 
    [0.7524857143, 0.7384, 0.3768142857],
    [0.7858428571, 0.7355666667, 0.3632714286],
    [0.8185047619, 0.7327333333, 0.3497904762], 
    [0.8506571429, 0.7299, 0.3360285714],
    [0.8824333333, 0.7274333333, 0.3217], 
    [0.9139333333, 0.7257857143, 0.3062761905],
    [0.9449571429, 0.7261142857, 0.2886428571],
    [0.9738952381, 0.7313952381, 0.266647619], 
    [0.9937714286, 0.7454571429, 0.240347619],
    [0.9990428571, 0.7653142857, 0.2164142857],
    [0.9955333333, 0.7860571429, 0.196652381], 
    [0.988, 0.8066, 0.1793666667],
    [0.9788571429, 0.8271428571, 0.1633142857], 
    [0.9697, 0.8481380952, 0.147452381],
    [0.9625857143, 0.8705142857, 0.1309], 
    [0.9588714286, 0.8949, 0.1132428571],
    [0.9598238095, 0.9218333333, 0.0948380952],
    [0.9661, 0.9514428571, 0.0755333333], 
    [0.9763, 0.9831, 0.0538]
]

def getParulaColormap(N=256):
    """
    """

    cmap = LinearSegmentedColormap.from_list('parula', parulaColorspace, N=N)

    return cmap