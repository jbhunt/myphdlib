from myphdlib.toolkit.labjack import loadLabjackData, extractLabjackEvent

def extractLabjackEvents(sessionObject):
    """
    """

    labjackDataMatrix = loadLabjackData(sessionObject.labjackFolder)

    return