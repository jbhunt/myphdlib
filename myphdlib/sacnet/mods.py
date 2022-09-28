import numpy as np
from neuron import h

class Varicosity():
    """
    a simulated starburst amacrine cell varicosity
    """

    def __init__(self, dendrites, locs):
        """
        keywords
        --------
        dendrites : tuple or list
            the pre- and post-synaptic dendrites
        locations : tuple or list
            the normalized location of the varicosity for each dendrite
        """

        self.dend1, self.dend2 = dendrites
        self.loc1, self.loc2 = locs

        # compute the global coordinates of the varicosity
        self.x = self.dend1.x3d(0) + (self.dend1.x3d(1) - self.dend1.x3d(0)) * self.loc1
        self.y = self.dend1.y3d(0) + (self.dend1.y3d(1) - self.dend1.y3d(0)) * self.loc1

        # compute the euclidean distance from the center of the cell
        pt1 = np.array([self.x,self.y])
        pt2 = np.array([self.dend1.cell().x,self.dend1.cell().y])
        self.d = np.linalg.norm(pt2 - pt1)

        # these are the vectors which record conductance and currents
        self._ica = None
        self._icl = None
        self._gcl = None

        # place the point-processes
        self._insert_clc() # NOTE: this method must be called first
        self._insert_cav()

        self._set_params()

        return

    def _set_params(self):
        """
        set the default parameters for the Cav and Cl channels
        """

        # Cav channel parameters
        for seg in self.presyndend:
            seg.gbar_calrgc = 0.008
            seg.shift_calrgc = 15

        # Cl channel parameters
        self.clc.amp = 8
        self.clc.thres = 0.07
        self.clc.tau1 = 10
        self.clc.tau2 = 3
        self.clc.g = 0.015

        return

    def _insert_cav(self):
        """
        insert the voltage-gated calcium channel in the pre-synaptic dendrite
        """

        # split the pre-synaptic dendrite at the point of intersection
        sec1,sec2 = self.dend1.split(locs=[self.loc1])

        # replace the pre-synaptic dendrite
        self.dend1.replace([sec1,sec2])

        #
        self.presyndend = sec1

        # insert the calcium ion density mechanisms (if they don't already exist)
        if not hasattr(sec1(1),'calrgc'):
            sec1.insert('calrgc')
        if not hasattr(sec1(1),'cadiff'):
            sec1.insert('cadiff')

        # double-check that the chloride channel has been placed
        try:
            assert hasattr(self,'clc')
        except:
            self._insert_clc()

        # set the reference for the chloride channel in the post-synaptic dendrite
        h.setpointer(sec1(1)._ref_cai,'capre',self.clc)

        # create a vector which records the calcium current in the pre-synaptic dendrite
        self._ica = h.Vector().record(sec1(1)._ref_ica)

        return

    def _insert_clc(self):
        """
        insert the chloride channel in the post-synaptic dendrite
        """

        # split the post-synaptic dendrite
        sec1,sec2 = self.dend2.split(locs=[self.loc2])

        # replace the post-synaptic dendrite
        self.dend2.replace([sec1,sec2])

        #
        self.postsyndend = sec1

        # place the chloride channel in the post-synaptic dendrite
        self.clc = h.ComplexCl2(sec1(1))

        # record the chloride conductance and current
        self._gcl = h.Vector().record(self.clc._ref_g)
        self._icl = h.Vector().record(self.clc._ref_i)

        return

    @property
    def ica(self):
        """
        return the calcium current as a numpy array
        """

        return np.array(self._ica)

    @ica.setter
    def ica(self, value):
        self._ica = value

    @property
    def icl(self):
        """
        return the chloride current as a numpy array
        """

        return np.array(self._icl)

    @icl.setter
    def icl(self, value):
        self._icl = value

    @property
    def gcl(self):
        """
        return the chloride conductance as a numpy array
        """

        return np.array(self._gcl)

    @gcl.setter
    def gcl(self, value):
        self._gcl = value

class BipolarCellSynapse():
    """
    simulated bipolar cell synapse
    """

    def __init__(self, seg, gratio=3, gscale=1, tau=20):
        """
        keywords
        --------
        seg : nrn.Segment
            the segment where the synapse will be placed
        gratio :
            the ratio of transient to sustained conductance
        gscale :
            conductance multiplier
        tau :
            time-constant for the conductance

        notes
        -----
        the conductance for each compenent of the synapse's conductane, transient
        or sustained, is as follows:

        transient = gratio * gscale

        sustained = 1 * gscale
        """

        # these are the parameters of the synaptic input
        self.gratio = gratio
        self.gscale = gscale
        self.tau = tau

        # compute the global position of the synapse
        self.bip = h.Bip(seg)
        self.loc = seg.x # normalized position on the dendrite seg
        self.x = self.bip.locx = seg.sec.x3d(0) + (seg.sec.x3d(1) - seg.sec.x3d(0)) * self.loc
        self.y = self.bip.locy = seg.sec.y3d(0) + (seg.sec.y3d(1) - seg.sec.y3d(0)) * self.loc

        # compute the euclidean distance from the center of the cell
        pt1 = np.array([self.x,self.y])
        pt2 = np.array([seg.sec.cell().x,seg.sec.cell().y])
        self.d = np.linalg.norm(pt2 - pt1)

        # these vectors record the conductance and current, respectively
        self._g = h.Vector().record(self.bip._ref_g) # stores the conductance
        self._i = h.Vector().record(self.bip._ref_i) # stores the current

        # set the parameters of the synaptic input
        self._set_params()

        return

    def _set_params(self):
        """
        set the parameters of the synaptic input
        """

        self.bip.e = 0 # reversal potential
        self.bip.g1max = self.gratio * self.gscale
        self.bip.g2max = 1 * self.gscale
        self.bip.tau = self.tau

        # flag the syanpse as being active
        self._isactive = True

        return

    # properties

    @property
    def isactive(self):
        """
        returns True if the synapse is active; otherwise, False
        """

        return self._isactive

    @isactive.setter
    def isactive(self, value):
        """
        """

        if value is True or value == 1:
            self.bip.g1max = self.gratio * self.gscale
            self.bip.g2max = 1 * self.gscale
            self._isactive = True

        elif value is False or value == 0:
            self.bip_g1max = 0
            self.bip.g2max = 0
            self._isactive = False

        else:
            raise ValueError('The isactive property must be either True (or 1) or False (or 0).')
            self._isactive = False

        return

    @property
    def i(self):
        """
        returns the current vector as a numpy array
        """

        return np.array(self._i)

    @i.setter
    def i(self, value):
        self._i = value

    @property
    def g(self):
        """
        returns the conductance vector as a numpy array
        """

        return np.array(self._g)

    @g.setter
    def g(self, value):
        self._g = value
