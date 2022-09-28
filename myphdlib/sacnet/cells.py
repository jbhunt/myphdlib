import os
import yaml
import itertools
import numpy as np
from matplotlib.pylab import *
from shapely.geometry import LineString
from neuron import h, hclass, load_mechanisms
from sacnet.mods import BipolarCellSynapse, Varicosity

# this is the relative path to the package
PKG_PATH = os.path.dirname(__file__)

# load the custom mod files
result = load_mechanisms(os.path.join(PKG_PATH,'data/mechanisms'))
if result is False:
    print('Warning: failed to load custom mechanisms ...')

# load the cell-type-specific model parameters
with open(os.path.join(PKG_PATH,'data/cells/sac.yaml'),'r') as stream:
    SAC_PARAMS = yaml.full_load(stream)

with open(os.path.join(PKG_PATH,'data/cells/rgc.yaml'),'r') as stream:
    RGC_PARAMS = yaml.full_load(stream)

def compute_dendrite_spacing():
    """
    compute the angle of each dendrite for each dendrite generation

    returns
    -------
    angles : list
        list of angles of each dendrite in each dendrite generation

    notes
    -----
    The order of generations is innermost to outermost
    """

    # angle of a single dendrite arm
    theta = 2 * np.pi / SAC_PARAMS['morphology']['dends']['narms']

    angles = list()

    for igen in range(SAC_PARAMS['morphology']['dends']['ngens'])[::-1]:

        # for the outermost generation ...
        if igen == (SAC_PARAMS['morphology']['dends']['ngens'] - 1):
            c = np.linspace(theta / (2 ** igen - 1) / 2, theta - (theta / (2 ** igen - 1) / 2),2 ** igen)

        # for the innermost generation ...
        elif igen == 0:
            c = np.array([theta / 2])

        # for all generations inbetween ...
        else:
            c = (c[:-1] + np.diff(c / 2))[::2]

        angles.append(c)

    # re-order the list of dendrite spacing from innermost to outermost
    angles = angles[::-1]

    return angles

class SacDendrite(hclass(h.Section)):
    """
    a model SAC dendrite
    """

    def __init__(self, iarm=0, igen=0, idend=0, **kwargs):
        """
        keyword
        -------
        iarm : int (default is 0)
            cell arm index
        igen : int (default is 0)
            dendrite generation index
        idend : int (default is 0)
            dendrite index

        notes
        -----
        The cell body is a special case of this class so I use pass -1,-1,-1 to the constructor method.
        """

        super().__init__(**kwargs)

        self.iarm = iarm    # cell arm index
        self.igen = igen    # dendrite generation index
        self.idend = idend  # dendrite index

        self._set_params()

        self.deleteme = False

    def _set_params(self):
        """
        set the axial resistance and passive membrane conductance
        """

        # set axial resistance
        self.Ra = SAC_PARAMS['physiology']['Ra']

        # set passive membrane conductance
        self.insert('pas')

        for seg in self:
            seg.pas.g = SAC_PARAMS['physiology']['pas']['g']
            seg.pas.e = SAC_PARAMS['physiology']['pas']['e']
            seg.v = seg.pas.e

        return

    def distance(self, loc=0.5):
        """
        find the euclidean distance from a point on the dendrite to the cell body

        keywords
        --------
        loc : float (default is 0.5)
            normalized location on the dendrite (0 to 1)

        returns
        -------
        dist
            distance from the point to the cell body
        """

        x = loc * (self.x3d(1) - self.x3d(0)) + self.x3d(0)
        y = loc * (self.y3d(1) - self.y3d(0)) + self.y3d(0)
        pt1 = np.array([x,y])
        pt2 = np.array([self.cell().x,self.cell().y])

        dist = np.linalg.norm(pt2 - pt1)

        return dist

    def intersects(self, partner):
        """
        find the point of intersection with another dendrite

        keywords
        --------
        partner : sacnet.cells.SacDendrite
            the post-synaptic dendrite

        returns
        -------
        result : bool
            True if a point of intersection is found otherwise False
        loc1 : float
            the normalized position of the intersection for the pre-synaptic dendrite
        loc2 : float
            the normalized position of the intersection for the post-synaptic dendrite
        """

        # find the intersection of the two dendrites
        line1 = LineString([(self.x3d(0),self.y3d(0)),(self.x3d(1),self.y3d(1))])
        line2 = LineString([(partner.x3d(0),partner.y3d(0)),(partner.x3d(1),partner.y3d(1))])

        if not line1.crosses(line2):
            return False, None, None

        cross = line1.intersection(line2)
        x,y = cross.x, cross.y

        # find the position on pre-synaptic dendrite
        loc1 = np.sqrt((self.y3d(0) - y) ** 2 + (self.x3d(0) - x) ** 2) / self.L

        # find the position on post-synaptic dendrite
        loc2 = np.sqrt((partner.y3d(0) - y) ** 2 + (partner.x3d(0) - x) ** 2) / partner.L

        return True, loc1, loc2

    def name(self):
        """
        """

        print('dend[{}]'.format(self.idend))

    def split(self, locs=[0.5]):
        """
        split the dendrite into a series of connected sections at the locations
        specified by the locs keyword argument

        keywords
        --------
        locs : list
            a list of normalized locations at which the dendrite will be split

        returns
        -------
        series : list
             a series of new dendrites
        """

        # this list holds the series of new dendrites
        series = list()

        # for each location ...
        for iloc,loc in enumerate(locs):

            # create a new dendrite
            dend = SacDendrite(self.iarm,
                               self.igen,
                               idend=None,
                               cell=self.cell(),
                               )

            x1 = loc * (self.x3d(1) - self.x3d(0)) + self.x3d(0)
            y1 = loc * (self.y3d(1) - self.y3d(0)) + self.y3d(0)

            # for the very first dendrite in the series ...
            if iloc == 0:
                x0 = self.x3d(0)
                y0 = self.y3d(0)

            # for the intermediate dendrites in the series ...
            else:
                parent = series[(iloc - 1)] # identify the parent dendrite
                dend.connect(parent)
                x0 = parent.x3d(1)
                y0 = parent.y3d(1)
                x1 = loc * (self.x3d(1) - self.x3d(0)) + self.x3d(0)
                y1 = loc * (self.y3d(1) - self.y3d(0)) + self.y3d(0)

            dend.pt3dadd(x0,y0,0,self.diam)
            dend.pt3dadd(x1,y1,0,self.diam)
            series.append(dend)

        # for the last dendrite in the series ...
        dend = SacDendrite(self.iarm,
                           self.igen,
                           idend=None,
                           cell=self.cell(),
                           )
        parent = series[-1]
        x0,y0 = parent.x3d(1),parent.y3d(1)
        x1,y1 = self.x3d(1),self.y3d(1)
        dend.pt3dadd(x0,y0,0,self.diam)
        dend.pt3dadd(x1,y1,0,self.diam)
        dend.connect(parent)
        series.append(dend)

        return series

    def replace(self, series):
        """
        replaces the dendrite with a series of sections

        keywords
        --------
        series : list
            a list of connected dendrites
        """

        # grab the parent and children of the old dendrite
        parent = self.parentseg().sec
        children = self.children()

        # disconnect the old dendrite
        for child in children:
            h.disconnect(sec=child)
        h.disconnect(sec=self)

        # connect the series of new dendrites
        for child in children:
            child.connect(series[-1])
        series[0].connect(parent)

        # position of the dendrite to be replaced
        cell = self.cell()
        idx = cell.dends.index(self)

        # increment all other whole dendrites by the length of the dendrite series minus 1
        for sec in cell.dends[(idx + 1):]:
            if sec.idend is not None:
                sec.idend += (len(series) - 1)

        # remove the old dendrite
        cell.dends.pop(idx)

        # insert the series of new dendrites in the place of the old dendrite
        for isec,sec in enumerate(series):
            cell.dends.insert(idx + isec,sec)
            sec.idend = self.idend + isec

        return

    def insertbips(self, locs):
        """
        insert a bipolar cell syanpse at one or more locations

        notes
        -----
        This method works by splitting the dendrite up into individual sections
        at the location(s) provided in the locs keyword argument. The point
        processes are then inserted at the 1-end of each individual section in
        this series of sections (except for the very last section). This series
        of new dendrites replaces the old dendrite in the 'dends' attribute of
        the dendrite's owner (i.e., an instance of StarburstAmacrineCell).
        """

        processes = list()

        series = self.split(locs)
        for dend in series[:-1]:
            synapse = BipolarCellSynapse(dend(1),gscale=10)
            processes.append(synapse)

        self.replace(series)

        return processes

# TODO: move the code for generating a single arm from the StarburstAmacrineCell
# class into the SacArm class

class SacArm():
    """
    """

    def __init__(self):
        """
        """

        return

class StarburstAmacrineCell():
    """
    this is a model starburst amacrine cell
    """

    def __init__(self, center=(0,0)):
        """
        """

        # list of biplar cell synapses
        self.inputs = list()

        # list of varicosities
        self.outputs = list()

        # global position of the cell
        self.x, self.y = center

        # cell body membrane potential
        self._vm = None

        self._set_params()

        return

    def _set_params(self):
        """
        set the parameters used for building the cell morphology
        """

        # randomly jitter the origin of the cell
        if SAC_PARAMS['morphology']['cell']['jitter'] is True:
            low = 0 - SAC_PARAMS['morphology']['cell']['jrange'] / 2
            high = 0 + SAC_PARAMS['morphology']['cell']['jrange'] / 2
            xjit = np.random.uniform(low,high,1)
            yjit = np.random.uniform(low,high,1)

            # update the origin
            self.x = self.x + xjit
            self.y = self.y + yjit

        # randomly rotate the cell
        if SAC_PARAMS['morphology']['cell']['rotate'] is True:
            self.rotation = np.random.uniform(0,2 * np.pi,1)
        else:
            self.rotation = 0

        # this is the number of dendrites in a single arm
        self.armsize = sum([2 ** n for n in range(SAC_PARAMS['morphology']['dends']['ngens'])])

        # cell radius in microns
        self.radius = SAC_PARAMS['morphology']['cell']['radius']

        # number of dendrite generations
        self.ngens = SAC_PARAMS['morphology']['dends']['ngens']

        # determine the parameters of each dendrite generation
        self.gendata = list()

        # dendrite diameter increase per generation
        dstep = (SAC_PARAMS['morphology']['dends']['dmax'] - SAC_PARAMS['morphology']['dends']['dmin']) / (self.ngens - 1)

        #
        angles = compute_dendrite_spacing()

        # for each genration ...
        for igen in np.arange(SAC_PARAMS['morphology']['dends']['ngens']):
            r = self.radius / self.ngens * (igen + 1)
            d = dstep * igen + SAC_PARAMS['morphology']['dends']['dmin']
            phis = angles[igen] + self.rotation
            self.gendata.append({'r':r,'d':d,'phis':phis})

        return

    def build(self):
        """
        construct the cell and place the inputs
        """

        # initialize the cell body
        self.soma = SacDendrite(-1,-1,-1,name='soma',cell=self)
        self.soma.diam = self.soma.L = SAC_PARAMS['morphology']['soma']['size']
        self.soma.pt3dadd(-1 * self.soma.diam / 2 + self.x,self.y,0,self.soma.diam)
        self.soma.pt3dadd(self.soma.diam / 2 + self.x,self.y,0,self.soma.diam)

        #
        self._vm = h.Vector().record(self.soma(0.5)._ref_v)

        # list of dendrite sections
        # self.dends = list()
        self.dends = list()

        # dendrite index counter
        idend = 0

        # for each branch (and branch angle) ...
        for iarm,theta in enumerate(np.arange(0,2 * np.pi,2 * np.pi / SAC_PARAMS['morphology']['dends']['narms'])):

            # for each dendrite generation ...
            for igen in np.arange(SAC_PARAMS['morphology']['dends']['ngens']):

                # angles of the dendrites in the current arm and generation
                phis = self.gendata[igen]['phis']

                # for each angle in the appropriate spacing for this generation ...
                for phi in phis:

                    # identify the target dendrite
                    dend = SacDendrite(iarm,igen,idend,cell=self)

                    # define some basic geometry and physiology
                    dend.diam = self.gendata[igen]['d'] # set the appropriate segment diameter

                    # this condition is met for each of the initial dendrite segments of each branch
                    if idend % self.armsize == 0:
                        parent = self.soma
                        xori, yori = (self.x,self.y)
                        dend.connect(self.soma(0.5))

                    # for all other dendrites in the branch determine the appropriate parent dendrite
                    else:
                        iparent = int(np.floor(((idend - iarm) - 1) / 2) - iarm * ((self.armsize - 1) / 2) + (iarm * self.armsize))
                        parent = self.dends[iparent]
                        xori = parent.psection()['morphology']['pts3d'][-1][0]
                        yori = parent.psection()['morphology']['pts3d'][-1][1]
                        dend.connect(parent)

                    xend = self.gendata[igen]['r'] * np.cos(phi + theta) + self.x
                    yend = self.gendata[igen]['r'] * np.sin(phi + theta) + self.y

                    dend.pt3dadd(xori,yori,0,dend.diam)
                    dend.pt3dadd(xend,yend,0,dend.diam)

                    # store the section
                    self.dends.append(dend)

                    # increment the counter
                    idend += 1

        return

    def placebips(self):
        """
        place evenly spaced bipolar cell synapses on dendrites that qualify

        notes
        -----
        This method works by first identifying the dendrites which qualify for
        inputs from bipolar cells and the locations of each input on the
        dendrite. This information is stored in a queue until all dendrites have
        been considered. The queue is necessary because placing a point-process
        like a synaptic input is accomplished by splitting the dendrite up into
        smaller sections at the location of the point-process. The new series of
        dendrites replace the old single dendrite in the list of dendrites
        stored in the 'dends' attribute of the cell which increases the length
        of this list. The queue avoids changing the length of this list while it
        is being iterated over.
        """

        # maximum distance from soma which qualifies for a bipolar cell synapse
        maxdist = SAC_PARAMS['synapses']['inputs']['excitatory']['dmax'] * SAC_PARAMS['morphology']['cell']['radius']

        # density of excitatory inputs (in synapses / um)
        density = SAC_PARAMS['synapses']['inputs']['excitatory']['density']

        # this stores the dendrites that qualify for bipolar cell input and the
        # location of the inputs in a tuple
        queue = list()

        # for each of the cell's dendrites ...
        for dend in self.dends:

            # compute the position of each putative synapse
            nsyn = np.around(dend.L / density)
            locs = np.linspace(1 / nsyn / 2,1 - (1 / nsyn / 2),nsyn)
            mask = np.array([dend.distance(loc) for loc in locs]) < maxdist

            # check that at least one of the putative synapses qualifies
            if mask.sum() == 0:
                continue

            # select just the synapses that qualify
            locs = locs[mask]

            # queue up the dendrite and locations of synaptic input
            queue.append((dend,locs))

        # iterate over the queue
        for dend,locs in queue:

            # insert the bipolar cell synapses at each point that qualified
            synapses = dend.insertbips(locs)

            # store the synapses returned by the insertbips method
            for synapse in synapses:
                self.inputs.append(synapse)

    def connect(self, partner):
        """
        connects the two cells at the geometric intersections of their dendrites
        """

        # minimum distance from the pre-syanptic cell soma which qualifies for a varicosity
        thresh1 = SAC_PARAMS['synapses']['outputs']['inhibitory']['dmin'] * SAC_PARAMS['morphology']['cell']['radius']

        # maximum distance from the post-syanptic cell soma which qualifies for a chloride channel
        thresh2 = SAC_PARAMS['synapses']['inputs']['inhibitory']['dmax'] * SAC_PARAMS['morphology']['cell']['radius']

        print('Connecting ...')
        connecting = True
        while connecting:

            # cartesian product of the two cells' dendrites
            for dend1,dend2 in itertools.product(self.dends,partner.dends):

                result, loc1, loc2 = dend1.intersects(dend2)

                if result is False:
                    continue

                # check that the position on the pre-synaptic cell qualifies for a varicosity
                dist1 = dend1.distance(loc1)
                if dist1 < thresh1:
                    continue

                # check that the position on the post-synaptic cell qualifies for an inhibitory synaptic connection
                dist2 = dend2.distance(loc2)
                if dist2 > thresh2:
                    continue

                dendrites = (dend1,dend2)
                locations = (loc1,loc2)
                self.outputs.append(Varicosity(dendrites,locations))

                break

            # this case is met when all dendrites have been considered
            if dend1 == self.dends[-1]:
                connecting = False

        print('All done!')

        return

    def plot(self, ax=None, skeleton=False):
        """
        plots the cell

        keywords
        --------
        ax : matplotlib.axes._subplots.AxesSubplot or None
        skeleton : bool (default is False)
            if True, plots just the cell skeleton: otherwise, plots the inputs and outputs
        """

        self.ax = ax

        if ax is None:
            self.ax = figure().add_subplot()

        for dend in self.dends:
            x = (dend.x3d(0),dend.x3d(1))
            y = (dend.y3d(0),dend.y3d(1))
            plot(x,y,color='k',alpha=0.25,lw=1.5,zorder=-1)

        # only plot the skeleton
        if skeleton is True:
            return

        # color the zone for excitatory and inhibitory synaptic input
        radius = SAC_PARAMS['synapses']['inputs']['excitatory']['dmax'] * SAC_PARAMS['morphology']['cell']['radius']
        zone = Circle((self.x,self.y),radius,color='g',alpha=0.1)
        self.ax.add_artist(zone)

        # color the varicose zone
        radii = [SAC_PARAMS['synapses']['outputs']['inhibitory']['dmin'] * SAC_PARAMS['morphology']['cell']['radius'],
                 SAC_PARAMS['morphology']['cell']['radius']
                 ]
        theta = np.linspace(0,2 * np.pi,100)
        xs = np.outer(radii, np.cos(theta)) + self.x
        ys = np.outer(radii, np.sin(theta)) + self.y
        xs[1,:] = xs[1,::-1]
        ys[1,:] = ys[1,::-1]
        ring = self.ax.fill(xs.ravel(),ys.ravel(),color='r')[0]
        ring.set_alpha(0.1)

        # plot the bipolar cell synapses
        for syn in self.inputs:
            plot(syn.x,syn.y,marker='o',markerfacecolor='b',markeredgecolor='b',markersize=2.5)

        # plot the varicosities
        for varicosity in self.outputs:
            plot(varicosity.x,varicosity.y,marker='o',markerfacecolor='r',markeredgecolor='r',markersize=2.5)

    @property
    def vm(self):
        """
        returns the cell body membrane potential as a numpy array
        """

        return np.array(self._vm)

    @vm.setter
    def vm(self, value):
        """
        """

        self._vm = value

# TODO: finish making the RGC class

class RetinalGanglionCell():
    """
    a very simple model retinal ganglion cell
    """

    def __init__(self, plot=False, ax=None):
        """
        """

        yml = os.path.join(PKG_PATH,'data/cells/rgc.yaml')
        with open(yml,'r') as stream:
            params = yaml.full_load(stream)

        self.morphology = params['morphology']
        self.physiology = params['physiology']
        self.synapses = params['synapses']

        self.plot = plot
        if self.plot is True and ax is None:
            self.ax = figure().add_subplot()
        elif self.plot is True and ax is not None:
            self.ax = ax

        return

    def _build(self):
        """
        """

        return

    def _init_morph(self):
        """
        """

        return

    def _init_phys(self):
        """
        """

        return

    def connect(self):
        """
        """

        return
