import os
import pathlib as pl
import subprocess as sp

def locatMatlabAddonsFolder(
    ):
    """
    """

    if os.name == 'nt':
        matlabAddonsFolder = 'C:/Users/NeuroPixel1/Documents/MATLAB'
    elif os.name == 'posix':
        matlabAddonsFolder = '/home/jbhunt/Documents/MATLAB'

    return matlabAddonsFolder

def locateMatlabExecutable(
    matlabVersion='2021a',
    ):
    """
    """

    _matlabExecutableFile = None
    if os.name == 'nt':
        _matlabExecutableFile = pl.Path(f'C:/Program Files/MATLAB/R{matlabVersion}/bin/matlab.exe')
    elif os.name == 'posix':
        _matlabExecutableFile = pl.Path(f'/usr/bin/matlab')
        
    return _matlabExecutableFile

def runMatlabScript(
    scriptFilePath,
    nogui=True,
    ):
    """
    """
    matlabExecutableFile = locateMatlabExecutable()
    if matlabExecutableFile.exists() == False:
        raise Exception('Could not locate matlab executable')
    if os.name == 'nt':
        matlabFunction = f'run(""{scriptFilePath}"")' # NOTE: For some reason double-quotes are necessary on Windows
    elif os.name == 'posix':
        matlabFunction = f'run("{scriptFilePath}")'

    command = [
        str(matlabExecutableFile),
        f'-r',
        matlabFunction
    ]
    if nogui:
        command.insert(1, '-nodesktop')
    p = sp.Popen(command)
    output, errors = p.communicate() # This should block until MATLAB closes (works on Ubuntu but not Windows)
    return