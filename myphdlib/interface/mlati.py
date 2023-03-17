import numpy as np
from myphdlib.interface.session import SessionBase
from myphdlib.general.labjack import loadLabjackData, filterPulsesFromPhotologicDevice

class MlatiSession(SessionBase):
    """
    """

    def _processStimuliMetadata(self):
        """
        """

        data = loadLabjackData(self.folders.labjack)
        signal = filterPulsesFromPhotologicDevice(data[:, 7], minimumPulseWidthInSeconds=0.0167)
        stateTransitionIndices = np.where(
            np.abs(np.diff(signal)) > 0.5
        )[0]
        longestIntervalIndices = np.where(
            np.diff(stateTransitionIndices) / 1000 > 2
        )[0][:5]
        stack = np.vstack([
            stateTransitionIndices[longestIntervalIndices + 1],
            stateTransitionIndices[longestIntervalIndices]
        ]).T
        interBlockIndices = np.around(stack.mean(1), 0).astype(np.int)

        return interBlockIndices