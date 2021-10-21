import numpy as np
from matplotlib import pylab as plt
from matplotlib import widgets as wid

class SaccadeLabelingGUI():
    """
    """

    NOISE = 0
    NASAL = 1
    TEMPORAL = 2
    UNSCORED = np.nan

    def __init__(self, samples, sacmat, slice=None, scaling='static', xlim=(-25, 25), randomize=False):
        """
        """

        nsamples, nfeatures = samples.shape

        #
        if slice is None:
            self.X = samples
            self.pom = None
            self.xdata = np.arange(nfeatures, dtype=np.int)
        else:
            start, stop = slice
            self.X = samples[:, start: stop]
            self.pom = (stop - start) / 2
            self.xdata = (np.arange(self.pom * 2) + start).astype(np.int)

        #
        self.sacmat = sacmat

        #
        if randomize:
            index = np.random.choice(np.arange(nsamples), size=nsamples, replace=True)
            self.X = self.X[index, :]
            self.sacmat = sacmat[index, :, :]

        self.xsacs = self.sacmat[:, :, 0]
        self.ysacs = self.sacmat[:, :, 1]
        nsaccades, nsamples, ncoords = self.sacmat.shape
        self.isample = 0
        self.nsamples = nsamples
        self.y = np.full(nsaccades, np.nan)
        self.running = True
        self.scaling = scaling
        self.xlim = xlim

        # init the GUI
        self.fig, (self.ax1, self.ax2) = plt.subplots(ncols=2, sharey=True)
        self.ax1.set_title('X position')
        self.ax2.set_title('Y position')
        plt.subplots_adjust(left=0.35)

        ax3 = plt.axes([0.02, 0.5, 0.15, 0.2])
        ax4 = plt.axes([0.02, 0.4, 0.1, 0.05])
        ax5 = plt.axes([0.02, 0.3, 0.1, 0.05])
        ax6 = plt.axes([0.02, 0.2, 0.1, 0.05])

        self.check = wid.RadioButtons(ax3, ('Nasal', 'Temporal', 'Noise', 'Unscored'), active=3)
        self.check.on_clicked(self.onCheckClicked)
        self.prev = wid.Button(ax4, 'Previous', color='white', hovercolor='grey')
        self.prev.on_clicked(self.onPreviousClicked)
        self.next = wid.Button(ax5, 'Next', color='white', hovercolor='grey')
        self.next.on_clicked(self.onNextClicked)
        self.exit = wid.Button(ax6, 'Exit', color='white', hovercolor='grey')
        self.exit.on_clicked(self.onExitClicked)

        # plot the very first sample
        xsac = np.take(self.xsacs, self.isample, axis=0, mode='wrap')
        xsac = xsac - xsac[0]
        ysac = np.take(self.ysacs, self.isample, axis=0, mode='wrap')
        ysac = ysac - ysac[0]
        xwave = xsac[self.xdata]
        ywave = ysac[self.xdata]

        # x pupil position
        self.xback = self.ax1.plot(xsac, np.arange(self.nsamples), color='k', alpha=0.3).pop()
        self.xline = self.ax1.plot(xwave, self.xdata, color='r').pop()
        self.ax1.hlines((self.nsamples - 1) / 2, self.xlim[0], self.xlim[1], color='b')

        # y pupil position
        self.yback = self.ax2.plot(ysac, np.arange(self.nsamples), color='k', alpha=0.3).pop()
        self.yline = self.ax2.plot(ywave, self.xdata, color='r').pop()
        self.ax2.hlines((self.nsamples - 1) / 2, self.xlim[0], self.xlim[1], color='b')

        # draw the first sample
        self.updatePlot()

        # start the event loop
        plt.show()

        return

    def onCheckClicked(self, label):
        """
        Update the label for the current sample
        """

        isample = np.take(np.arange(self.y.size), self.isample, mode='wrap')
        if label == 'Unscored':
            self.y[isample] = self.UNSCORED
        elif label == 'Noise':
            self.y[isample] = self.NOISE
        elif label == 'Nasal':
            self.y[isample] = self.NASAL
        elif label == 'Temporal':
            self.y[isample] = self.TEMPORAL

        return

    def onNextClicked(self, event):
        """
        Plot the next sample
        """

        # plot next
        self.isample += 1
        self.updatePlot()

        # update the radio buttons
        self.check.eventson = False
        score = np.take(self.y, self.isample, mode='wrap')
        if np.isnan(score):
            self.check.set_active(3)
        elif score == self.NOISE:
            self.check.set_active(2)
        elif score == self.TEMPORAL:
            self.check.set_active(1)
        elif score == self.NASAL:
            self.check.set_active(0)
        self.check.eventson = True

        return

    def onPreviousClicked(self, event):
        """
        Plot the previous sample
        """

        # update the sample index
        self.isample -= 1
        self.updatePlot()

        # update the radio buttons
        self.check.eventson = False
        score = np.take(self.y, self.isample, mode='wrap')
        if np.isnan(score):
            self.check.set_active(3)
        elif score == self.NOISE:
            self.check.set_active(2)
        elif score == self.TEMPORAL:
            self.check.set_active(1)
        elif score == self.NASAL:
            self.check.set_active(0)
        self.check.eventson = True

        return

    def onExitClicked(self, event):
        """
        Close the figure and set the running flag to False
        """

        plt.close(self.fig)
        self.running = False
        return

    def updatePlot(self):
        """
        """

        # collect the data for the current sample
        xsac = np.take(self.xsacs, self.isample, axis=0, mode='wrap')
        xsac = xsac - xsac[0]
        ysac = np.take(self.ysacs, self.isample, axis=0, mode='wrap')
        ysac = ysac - ysac[0]
        xwave = xsac[self.xdata]
        ywave = ysac[self.xdata]

        # set the line data
        self.xback.set_data(xsac, np.arange(self.nsamples))
        self.xline.set_data(xwave, self.xdata)
        self.yback.set_data(ysac, np.arange(self.nsamples))
        self.yline.set_data(ywave, self.xdata)

        # rescale if necessary
        if self.scaling == 'dynamic':
            xdata = self.xline.get_data()
            ydata = self.yline.get_data()
            self.ax1.set_xlim([xdata.min() - xdata.std(), xdata.max() + xdata.std()])
            self.ax2.set_xlim([ydata.min() - ydata.std(), ydata.max() + ydata.std()])

        elif self.scaling == 'static':
            self.ax1.set_xlim(self.xlim)
            self.ax2.set_xlim(self.xlim)

        # label nasal and temporal directions
        for ax in [self.ax1, self.ax2]:
            xticks = ax.get_xticks()
            ax.set_xticks([xticks.min(), 0, xticks.max()])
            ax.set_xticklabels(['T', '0', 'N'])

        # draw the new lines
        self.fig.canvas.draw()

        return
