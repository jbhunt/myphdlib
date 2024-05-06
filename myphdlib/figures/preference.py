import numpy as np
from myphdlib.figures.analysis import AnalysisBase

class DirectionSectivityAnalysis(
    AnalysisBase,
    ):
    """
    """

    def __init__(self, **kwargs):
        """
        """

        super().__init__(**kwargs)

        return

    def loadNamespace(
        self,
        ):
        """
        """

        return

    def saveNamespace(
        self,
        ):
        """
        """

        return

    def measureDirectionSelectivityForProbes(
        self,
        ):
        """
        """

        for ukey in self.ukeys():
            self.ukey = ukey
            for gratingMotion in (-1, 1):
                trialIndices = np.where()

        return

    def measureDirectionSelectivityForSaccades(
        self,
        ):
        """
        """

        return