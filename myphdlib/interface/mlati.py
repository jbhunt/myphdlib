import numpy as np
from myphdlib.interface.session import SessionBase
from myphdlib.general.labjack import loadLabjackData, filterPulsesFromPhotologicDevice

labjackChannelMapping = {
    'barcode': 0
}

class MlatiSession(SessionBase):
    """
    """

    def __init__(self, sessionFolder):
        """
        """

        super().__init__(sessionFolder)

        return

def extractEvents(session):
    """
    """

    return