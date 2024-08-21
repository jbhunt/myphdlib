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

isoluminantRainbowColorspace = [
    (0.847, 0.057, 0.057),
    (0.527, 0.527, 0),
    (0, 0.592, 0),
    (0, 0.559, 0.559),
    (0.316, 0.316, 0.991),
    (0.718, 0, 0.718)
]

def getIsoluminantRainbowColormap(N=256):
    """
    """

    cmap = LinearSegmentedColormap.from_list('isoluminant_rainbow', isoluminantRainbowColorspace, N=N)

    return cmap

isoluminantBlueGreenOrangeColorspace = [
[0.21566, 0.71777, 0.92594],
[0.21805, 0.71808, 0.92254],
[0.2204, 0.71839, 0.91913],
[0.22272, 0.7187, 0.91573],
[0.22499, 0.71901, 0.91232],
[0.22727, 0.71931, 0.90891],
[0.2295, 0.71962, 0.9055],
[0.23174, 0.71992, 0.90208],
[0.23392, 0.72022, 0.89866],
[0.23611, 0.72051, 0.89524],
[0.23825, 0.72081, 0.89181],
[0.24038, 0.7211, 0.88838],
[0.2425, 0.72139, 0.88495],
[0.2446, 0.72168, 0.88151],
[0.24668, 0.72196, 0.87807],
[0.24877, 0.72225, 0.87462],
[0.25081, 0.72253, 0.87117],
[0.25284, 0.72281, 0.86772],
[0.25488, 0.72309, 0.86426],
[0.25687, 0.72336, 0.8608],
[0.25887, 0.72363, 0.85734],
[0.26085, 0.7239, 0.85387],
[0.26281, 0.72417, 0.8504],
[0.26477, 0.72444, 0.84692],
[0.26672, 0.7247, 0.84344],
[0.26866, 0.72496, 0.83996],
[0.27061, 0.72522, 0.83647],
[0.2725, 0.72548, 0.83297],
[0.27442, 0.72573, 0.82948],
[0.27635, 0.72598, 0.82598],
[0.27824, 0.72622, 0.82247],
[0.28012, 0.72647, 0.81896],
[0.28203, 0.72671, 0.81545],
[0.28389, 0.72694, 0.81193],
[0.28576, 0.72718, 0.8084],
[0.28764, 0.72741, 0.80487],
[0.28953, 0.72764, 0.80134],
[0.2914, 0.72787, 0.7978],
[0.29325, 0.72809, 0.79426],
[0.29511, 0.72831, 0.79071],
[0.29699, 0.72853, 0.78716],
[0.29885, 0.72875, 0.7836],
[0.3007, 0.72896, 0.78003],
[0.30255, 0.72917, 0.77647],
[0.30443, 0.72937, 0.77289],
[0.30631, 0.72958, 0.76931],
[0.30815, 0.72978, 0.76573],
[0.31004, 0.72997, 0.76213],
[0.31189, 0.73016, 0.75854],
[0.31378, 0.73035, 0.75494],
[0.31565, 0.73053, 0.75133],
[0.31753, 0.73071, 0.74772],
[0.31943, 0.73089, 0.74409],
[0.32131, 0.73106, 0.74047],
[0.32321, 0.73123, 0.73684],
[0.32511, 0.7314, 0.7332],
[0.32703, 0.73156, 0.72956],
[0.32897, 0.73172, 0.72591],
[0.33088, 0.73187, 0.72225],
[0.33284, 0.73202, 0.71859],
[0.33479, 0.73217, 0.71492],
[0.33674, 0.73231, 0.71125],
[0.33871, 0.73244, 0.70756],
[0.34069, 0.73257, 0.70388],
[0.34269, 0.7327, 0.70018],
[0.34469, 0.73282, 0.69648],
[0.34672, 0.73293, 0.69278],
[0.34875, 0.73304, 0.68906],
[0.35081, 0.73315, 0.68534],
[0.35286, 0.73325, 0.68161],
[0.35496, 0.73334, 0.67788],
[0.35706, 0.73343, 0.67413],
[0.35916, 0.73351, 0.67039],
[0.3613, 0.73359, 0.66663],
[0.36345, 0.73366, 0.66287],
[0.36562, 0.73373, 0.6591],
[0.36781, 0.73379, 0.65532],
[0.37003, 0.73384, 0.65155],
[0.37227, 0.73388, 0.64775],
[0.37454, 0.73392, 0.64396],
[0.37683, 0.73395, 0.64016],
[0.37914, 0.73398, 0.63635],
[0.38148, 0.73399, 0.63254],
[0.38384, 0.734, 0.62872],
[0.38622, 0.734, 0.62489],
[0.38865, 0.734, 0.62106],
[0.3911, 0.73398, 0.61722],
[0.39357, 0.73396, 0.61337],
[0.39609, 0.73392, 0.60953],
[0.39863, 0.73388, 0.60567],
[0.40121, 0.73383, 0.60181],
[0.40382, 0.73377, 0.59795],
[0.40646, 0.7337, 0.59409],
[0.40914, 0.73361, 0.59021],
[0.41186, 0.73352, 0.58635],
[0.41461, 0.73342, 0.58247],
[0.41741, 0.7333, 0.5786],
[0.42024, 0.73318, 0.57472],
[0.4231, 0.73304, 0.57084],
[0.42602, 0.73289, 0.56698],
[0.42898, 0.73273, 0.5631],
[0.43199, 0.73255, 0.55923],
[0.43504, 0.73236, 0.55536],
[0.43813, 0.73216, 0.5515],
[0.44125, 0.73194, 0.54764],
[0.44444, 0.73171, 0.54379],
[0.44765, 0.73146, 0.53995],
[0.45093, 0.73119, 0.53612],
[0.45426, 0.73092, 0.53231],
[0.45762, 0.73062, 0.5285],
[0.46104, 0.73032, 0.52471],
[0.46451, 0.72999, 0.52094],
[0.468, 0.72965, 0.51719],
[0.47157, 0.72928, 0.51346],
[0.47518, 0.7289, 0.50975],
[0.47882, 0.7285, 0.50606],
[0.48251, 0.72809, 0.50241],
[0.48625, 0.72766, 0.49879],
[0.49004, 0.7272, 0.4952],
[0.49386, 0.72673, 0.49165],
[0.49773, 0.72625, 0.48813],
[0.50164, 0.72574, 0.48464],
[0.50557, 0.72521, 0.48121],
[0.50955, 0.72466, 0.47782],
[0.51357, 0.72409, 0.47449],
[0.51761, 0.72351, 0.4712],
[0.52167, 0.7229, 0.46795],
[0.52578, 0.72228, 0.46477],
[0.5299, 0.72164, 0.46164],
[0.53404, 0.72098, 0.45857],
[0.53822, 0.7203, 0.45556],
[0.5424, 0.71961, 0.45262],
[0.5466, 0.7189, 0.44973],
[0.55081, 0.71817, 0.4469],
[0.55503, 0.71743, 0.44415],
[0.55926, 0.71667, 0.44145],
[0.5635, 0.71589, 0.43882],
[0.56773, 0.71509, 0.43627],
[0.57197, 0.71428, 0.43376],
[0.57622, 0.71346, 0.43134],
[0.58045, 0.71262, 0.42898],
[0.58468, 0.71178, 0.42669],
[0.5889, 0.71092, 0.42445],
[0.59311, 0.71004, 0.42224],
[0.5973, 0.70917, 0.42009],
[0.60146, 0.70828, 0.41796],
[0.60561, 0.70738, 0.41587],
[0.60975, 0.70647, 0.41382],
[0.61386, 0.70556, 0.4118],
[0.61796, 0.70463, 0.40983],
[0.62204, 0.7037, 0.40789],
[0.62612, 0.70276, 0.406],
[0.63017, 0.70181, 0.40413],
[0.63421, 0.70085, 0.4023],
[0.63822, 0.69988, 0.40055],
[0.64223, 0.69891, 0.3988],
[0.64621, 0.69792, 0.39711],
[0.65019, 0.69693, 0.39547],
[0.65415, 0.69593, 0.39385],
[0.65809, 0.69492, 0.39229],
[0.66202, 0.6939, 0.39078],
[0.66593, 0.69288, 0.3893],
[0.66983, 0.69184, 0.38787],
[0.67371, 0.69079, 0.38648],
[0.67758, 0.68974, 0.38515],
[0.68143, 0.68868, 0.38386],
[0.68527, 0.6876, 0.38261],
[0.6891, 0.68652, 0.38142],
[0.69291, 0.68544, 0.38026],
[0.6967, 0.68433, 0.37915],
[0.70047, 0.68324, 0.37809],
[0.70424, 0.68212, 0.37708],
[0.70798, 0.681, 0.37611],
[0.71172, 0.67987, 0.37518],
[0.71544, 0.67874, 0.37432],
[0.71914, 0.6776, 0.37349],
[0.72282, 0.67644, 0.37271],
[0.7265, 0.67528, 0.37197],
[0.73016, 0.67411, 0.37128],
[0.7338, 0.67294, 0.37065],
[0.73742, 0.67175, 0.37006],
[0.74103, 0.67057, 0.36951],
[0.74462, 0.66937, 0.36902],
[0.74821, 0.66816, 0.36856],
[0.75177, 0.66694, 0.36815],
[0.75531, 0.66573, 0.36778],
[0.75884, 0.6645, 0.36746],
[0.76236, 0.66327, 0.36719],
[0.76586, 0.66203, 0.36696],
[0.76934, 0.66077, 0.36678],
[0.77281, 0.65952, 0.36664],
[0.77626, 0.65826, 0.36654],
[0.7797, 0.65699, 0.36649],
[0.78312, 0.65572, 0.36647],
[0.78653, 0.65443, 0.3665],
[0.78991, 0.65314, 0.36657],
[0.79329, 0.65185, 0.36668],
[0.79664, 0.65055, 0.36682],
[0.79999, 0.64925, 0.36701],
[0.80331, 0.64792, 0.36723],
[0.80662, 0.64661, 0.3675],
[0.80991, 0.64528, 0.36781],
[0.81318, 0.64395, 0.36816],
[0.81645, 0.64261, 0.36854],
[0.81969, 0.64126, 0.36896],
[0.82293, 0.63992, 0.36941],
[0.82614, 0.63856, 0.36989],
[0.82935, 0.6372, 0.37041],
[0.83253, 0.63583, 0.37097],
[0.8357, 0.63446, 0.37156],
[0.83885, 0.63308, 0.37219],
[0.84199, 0.63169, 0.37285],
[0.84511, 0.6303, 0.37355],
[0.84822, 0.6289, 0.37427],
[0.85131, 0.6275, 0.37502],
[0.85439, 0.62609, 0.3758],
[0.85745, 0.62468, 0.37663],
[0.86051, 0.62326, 0.37746],
[0.86354, 0.62183, 0.37833],
[0.86656, 0.62041, 0.37923],
[0.86956, 0.61897, 0.38016],
[0.87256, 0.61752, 0.38112],
[0.87554, 0.61607, 0.38209],
[0.8785, 0.61462, 0.38312],
[0.88145, 0.61316, 0.38414],
[0.88438, 0.6117, 0.3852],
[0.8873, 0.61023, 0.38628],
[0.89021, 0.60876, 0.3874],
[0.8931, 0.60728, 0.38854],
[0.89598, 0.60578, 0.38969],
[0.89885, 0.60429, 0.39088],
[0.9017, 0.60279, 0.39208],
[0.90454, 0.60128, 0.3933],
[0.90737, 0.59978, 0.39455],
[0.91018, 0.59826, 0.39583],
[0.91298, 0.59674, 0.3971],
[0.91577, 0.59521, 0.39842],
[0.91854, 0.59368, 0.39974],
[0.92131, 0.59215, 0.40111],
[0.92406, 0.59059, 0.40246],
[0.92679, 0.58904, 0.40386],
[0.92952, 0.58748, 0.40527],
[0.93223, 0.58593, 0.40669],
[0.93493, 0.58436, 0.40813],
[0.93763, 0.58278, 0.4096],
[0.9403, 0.5812, 0.41108],
[0.94297, 0.57962, 0.41258],
[0.94562, 0.57802, 0.41408],
[0.94826, 0.57644, 0.41561],
[0.95089, 0.57482, 0.41716],
[0.95351, 0.57322, 0.41871],
[0.95612, 0.57159, 0.42029],
[0.95872, 0.56997, 0.42188],
[0.9613, 0.56834, 0.42348],
[0.96388, 0.56671, 0.42511],
[0.96644, 0.56505, 0.42674],
]

def getCetI1Colormap(N=256):
    cmap = LinearSegmentedColormap.from_list('cet_i1', isoluminantBlueGreenOrangeColorspace, N=N)
    return cmap