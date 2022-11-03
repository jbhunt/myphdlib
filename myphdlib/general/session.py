import os
import string
import pickle
import pathlib as pl

def saveSessionData(sessionObject, name, data, createOutputFile=True):
    """
    """

    #
    if sessionObject.outputFilePath.exists() == False:
        if createOutputFile:
            with open(str(sessionObject.outputFilePath), 'wb') as stream:
                pass
        else:
            raise Exception('Could not locate output file')

    #
    with open(str(sessionObject.outputFilePath), 'rb') as stream:
        try:
            dataContainer = pickle.load(stream)
        except EOFError:
            dataContainer = dict()
    sessionObject.outputFilePath.unlink() # TODO: Wait to delete the output file until it passes a check

    #
    try:
        dataContainer.update({name: data})
        with open(str(sessionObject.outputFilePath), 'wb') as stream:
            pickle.dump(dataContainer, stream)
    except:
        import pdb; pdb.set_trace() # TODO: Remove this

    return

def loadSessionData(sessionObject, name):
    """
    """

    if sessionObject.outputFilePath.exists() == False:
        raise Exception('Could not locate output file')

    with open(sessionObject.outputFilePath, 'rb') as stream:
        try:
            dataContainer = pickle.load(stream)
        except EOFError:
            raise Exception('Output file is empty') from None

    if name not in dataContainer.keys():
        raise Exception(f'Invalid data key: {name}')
    else:
        return dataContainer[name]

def locateFactorySource(hdd, alias, source=None):
    """
    """

    rootFolderPath = None

    #
    if source is not None:
        rootFolderPath = pl.Path(source)

    #
    elif os.name == 'posix':
        user = os.environ['USER']
        rootFolderPath = pl.Path(f'/media/{user}').joinpath(hdd, alias)
        if rootFolderPath.exists() == False:
            rootFolderPath = None
    
    #
    elif os.name == 'nt':
        for driveLetter in string.ascii_uppercase:
            rootFolderPath = pl.WindowsPath().joinpath(f'{driveLetter}:/', alias)
            if rootFolderPath.exists():
                rootFolderPath = rootFolderPath
                break

    #
    if rootFolderPath is None:
        raise Exception('Could not locate root folder')
    else:
        return rootFolderPath