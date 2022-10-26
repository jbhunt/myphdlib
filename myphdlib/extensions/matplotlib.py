import numpy as np
from matplotlib import pylab as plt

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