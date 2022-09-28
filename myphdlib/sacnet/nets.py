import itertools
from sacnet.cells import StarburstAmacrineCell

class TwoSacs():
    """
    a very simple network made of two reciprocally connected SACs
    """
    
    def __init__(self):
        """
        """
        
        return
    
    def build(self, overlap=0.5):
        """
        construct the network
        
        keyword
        -------
        overlap : float
            percent of overlap of the cells' dendritic fields
        """
        
        # init the two cells
        self.sac1 = StarburstAmacrineCell()
        self.sac2 = StarburstAmacrineCell()
        
        # determine the position of the second cell given the desired % overlap
        self.sac2.x = self.sac1.radius * 2 - (self.sac1.radius * 2 * overlap)
        
        # build the cells
        self.sac1.build()
        self.sac2.build()
        
        # connect the two cells
        self.sacs = [self.sac1,self.sac2]
        for (sac1,sac2) in itertools.combinations(self.sacs,2):
            sac1.connect(sac2)
            sac2.connect(sac1)
        
        return
    
    def plot(self):
        """
        """
        
        self.sac1.plot()
        self.sac2.plot(ax=self.sac1.ax)
        
        return