from sacnet.nets import TwoSacs
from matplotlib.pylab import figure
from matplotlib.patches import Rectangle
from neuron import h
import numpy as np

h.load_file('stdrun.hoc')

class TwoSacsConnecivityTest():
    """
    """

    def __init__(self):
        """
        """

        # create network
        self.net = TwoSacs()
        self.net.build()
        # self.net.plot()

        # turn off the bipolar cell inputs
        for sac in self.net.sacs:
            for bip in sac.inputs:
                bip.g1max = 0
                bip.g2max = 0

        # set the membrane potential of each segment in the second cell to 0 mV
        for dend in self.net.sac2.dends:
            for seg in dend:
                seg.v = 5
                try:
                    seg.eca = 100
                except:
                    continue
        for seg in self.net.sac2.soma:
            seg.v = 5

        # setup the voltage clamp for the first cell
        self.vclamp1 = h.SEClamp(self.net.sac1.soma(0.5))
        self.vclamp1.dur1 = 200; self.vclamp1.dur2 = 500; self.vclamp1.dur3 = 400
        self.vclamp1.amp1 = -65
        self.vclamp1.amp2 = -65
        self.vclamp1.amp3 = -65

        # setup the voltage clamp for the second cell
        self.vclamp2 = h.SEClamp(self.net.sac2.soma(0.5))
        self.vclamp2.dur1 = 1100
        self.vclamp2.amp1 = 5

        # init the vectors
        self.t = h.Vector().record(h._ref_t)
        self.i = h.Vector().record(self.vclamp2._ref_i)

        return

    def simulate(self, vpre=-65):
        """
        """

        istart = int(100 / h.dt)

        self.vclamp1.amp2 = vpre

        h.finitialize()
        h.fcurrent()

        while h.t < 1100:
            h.fadvance()

        t = self.t.as_numpy()[istart:]
        i = self.i.as_numpy()[istart:]

        return (t,i)

# TODO: create a Reichardt detector using the Kv channel
# NOTE: the RD will be the cell-intrinsic mehanism of DS

class ReichardtDetector():
    """
    """

    def __init__(self):
        """
        """

        return

    def _build_detector(self):
        """
        """

        return

from sacnet.cells import StarburstAmacrineCell

class MovingBarSingleCell():
    """
    simulated moving bar of light across a single SAC
    """

    def __init__(self, theta=0, velocity=100, contrast=0.9, width=50):
        """
        keywords
        --------
        theta
            angle of the stimulus in radians
        velocity
            speed of the stimulus in um/sec
        contrast
            contrast of the stimulus (ranges from 0 to 1)
        width
            width of the bar in microns
        """

        # construct the cell
        self.cell = StarburstAmacrineCell()
        self.cell.build()
        self.cell.placebips()

        # this records the somatic membrane potential
        self._v = h.Vector().record(self.cell.soma(0.5)._ref_v)

        # the duration of each phase (in ms) - light off, on, and off
        self.t1 = 500 # this is the time before the leading edge enters the RF
        self.t2 = None # this is the time it takes the stimulus to cross the RF
        self.t3 = 500 # this it the time after the trailing edge exits the RF

        # set the stimulus parameters
        self._theta = theta
        self._velocity = velocity
        self._contrast = contrast
        self._width = width

        #
        self._init_params()
        self._update_params()

        return

    def _init_params(self):
        """
        """

        for synapse in self.cell.inputs:
            synapse.bip.g1max = synapse.bip.g1max * self._contrast
            synapse.bip.g2max = synapse.bip.g2max * self._contrast

        return

    def _update_params(self):
        """
        """

        # time it takes for the stimulus to cross the RF
        self.t2 = (2 * self.cell.radius + self._width) / self._velocity * 1000

        # duration of the stimulus
        t = self._width / self._velocity * 1000

        # point of origin for the stimulus
        x = self.cell.radius * np.cos(np.pi + self._theta)
        y = self.cell.radius * np.sin(np.pi + self._theta)
        self.origin = (x,y)

        # compute the delay and duration for each synapse
        for synapse in self.cell.inputs:
            dx = (synapse.x - x) * np.cos(self._theta)
            dy = (synapse.y - y) * np.sin(self._theta)
            synapse.bip.delay =  (dx + dy) / self._velocity * 1000 + self.t1
            synapse.bip.dur = t

    def simulate(self):
        """
        """

        # time variables
        tref = h.Vector().record(h._ref_t)
        tstop = sum([self.t1,self.t2,self.t3])

        # initialize the simulation
        h.finitialize(-60)
        h.fcurrent()

        # main simulation loop
        while h.t < tstop:
            h.fadvance()

        return tref.as_numpy(), self.v

    def plot(self, t=0, ax=None):
        """
        """

        if ax is None:
            self.ax = figure().add_subplot()
        else:
            self.ax = ax

        self.cell.plot(skeleton=True,ax=self.ax)

        cx = (self._velocity * t - self._width - (self.t1 / 1000 * self._velocity)) * np.cos(self._theta) + self.origin[0]
        cy = (self._velocity * t - self._width - (self.t1 / 1000 * self._velocity)) * np.sin(self._theta) + self.origin[1]
        x = self.cell.radius * np.cos(self._theta - 2 * np.pi / 4) + cx
        y = self.cell.radius * np.sin(self._theta - 2 * np.pi / 4) + cy

        bar = Rectangle(xy=(x,y),
                        width=self._width,
                        height=2 * self.cell.radius,
                        angle=np.rad2deg(self._theta),
                        color='y',
                        alpha=0.25
                        )
        self.ax.add_artist(bar)

    @property
    def theta(self):
        """
        angle of the moving bar in radians
        """

        return self._theta

    @theta.setter
    def theta(self, value):
        """
        """

        self._theta = value
        self._update_params()

    @property
    def velocity(self):
        """
        velocity of the moving bar in um/sec
        """

        return self._velocity

    @velocity.setter
    def velocity(self, value):
        """
        """

        self._velocity = value
        self._update_params()

    @property
    def contrast(self):
        """
        contrast of the moving bar (ranges from 0 to 1)
        """

        return self._contrast

    @contrast.setter
    def contrast(self, value):
        """
        """

        self._contrast = value
        self._update_params()

    @property
    def width(self):
        """
        contrast of the moving bar (ranges from 0 to 1)
        """

        return self._width

    @width.setter
    def width(self, value):
        """
        """

        self._width = value
        self._update_params()

    @property
    def v(self):
        """
        somatic membrane potential in mV
        """

        return(np.array(self._v))

    @v.setter
    def v(self, value):
        """
        """

        self._v = value
