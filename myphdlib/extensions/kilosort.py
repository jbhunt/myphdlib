import re
import os
import shutil
import pathlib as pl
import subprocess as sp
try:
    import matlab.engine as me
except ImportError:
    me = None

# Constants
kilosortInstallFolder = None
numpyInstallFolder = None
matlabExecutableFile = None

def locateSoftwareDependencies(
    matlabAddonsFolder=None,
    kilosortVersion='2.0',
    matlabVersion='2021a'
    ):
    """
    Identifies the filepaths to the Kilosort and npy-matlab installations
    """

    # Kilosort
    global kilosortInstallFolder

    # Runs some checks
    if matlabAddonsFolder is None:
        raise Exception('Path to MATLAB addons not specified')

    if kilosortVersion not in ('2.0', '2.5-branch'):
        raise Exception(f'Kilosort-{kilosortVersion} is not supported yet')

    #
    _kilosortInstallFolder = pl.Path(matlabAddonsFolder).joinpath(f'Kilosort-{kilosortVersion}')
    if _kilosortInstallFolder.exists() == False:
        raise Exception(f'Could not locate Kilosort installation folder')
    else:
        kilosortInstallFolder = _kilosortInstallFolder

    # npy-matlab package
    global numpyInstallFolder

    _numpyInstallFolder = pl.Path(matlabAddonsFolder).joinpath(f'npy-matlab-master', 'npy-matlab')
    if _numpyInstallFolder.exists() == False:
        raise Exception(f'Could not locate the npy-matlab installation folder')
    else:
        numpyInstallFolder = _numpyInstallFolder

    # MATALB executable
    global matlabExecutableFile

    _matlabExecutableFile = None
    if os.name == 'nt':
        _matlabExecutableFile = pl.Path(f'C:/Program Files/MATLAB/R{matlabVersion}/bin/matlab.exe')
    elif os.name == 'posix':
        _matlabExecutableFile = pl.Path(f'/usr/local/bin/matlab')
    if _matlabExecutableFile is not None and _matlabExecutableFile.exists():
        matlabExecutableFile = _matlabExecutableFile

    return

def generateMatlabScripts(
    workingDirectory=None,
    saveRezFile=False
    ):
    """
    Creates a custom MATLAB script the will execute the automatic spike-sorting

    TODO: Provide the option to change the most important spike-sorting parameters
    """

    #
    global kilosortInstallFolder
    global numpyInstallFolder

    if kilosortInstallFolder is None:
        raise Exception('Kilosort installation folder is undetermined')
    if numpyInstallFolder is None:
        raise Exception('npy-matlab installation folder is undetermined')

    # Copy the channel map data into the working directory
    channelMapFilePath1 = kilosortInstallFolder.joinpath('configFiles', 'neuropixPhase3A_kilosortChanMap.mat')
    workingDirectoryPath = pl.Path(workingDirectory)
    channelMapFilePath2 = workingDirectoryPath.joinpath('kilosortChannelMap3a.mat')
    shutil.copy(channelMapFilePath1, channelMapFilePath2)    

    # Copy the default sorting config file into the working directory
    configFilePath1 = kilosortInstallFolder.joinpath('configFiles', 'StandardConfig_MOVEME.m')    
    with open(configFilePath1, 'r') as stream:
        lines = stream.readlines()
    configFilePathLocal = workingDirectoryPath.joinpath('kilosortConfigScript.m')
    with open(configFilePathLocal, 'w') as stream:
        for line in lines:
            if bool(re.search('D:.*kilosortChanMap.mat', line)):
                replacement = re.sub('D:.*kilosortChanMap.mat', str(channelMapFilePath2.as_posix()), line)
                stream.write(replacement)
            else:
                stream.write(line)

    # Copy the main sorting script into the working directory
    mainScriptFilePathSource = kilosortInstallFolder.joinpath('main_kilosort.m')
    mainScriptFilePathLocal = workingDirectoryPath.joinpath('kilosortMainScript.m')
    with open(mainScriptFilePathSource, 'r') as stream:
        lines = stream.readlines()
    with open(mainScriptFilePathLocal, 'w') as stream:
        for line in lines:

            #
            if bool(re.search('%% if you want to save the results to a Matlab file...\n', line)):
                if saveRezFile == False:
                    break

            #
            replacement = None
            if bool(re.search('D:.*KiloSort2', line)):
                replacement = re.sub('D:.*KiloSort2', str(kilosortInstallFolder.as_posix()), line)
            if bool(re.search('D:.*npy-matlab', line)):
                replacement = re.sub(
                    'D:.*npy-matlab',
                    str(numpyInstallFolder.as_posix()),
                    line
                )
            if bool(re.search('G:.*test', line)):
                replacement = re.sub('G:.*test', str(workingDirectoryPath.as_posix()), line)
            if bool(re.search("'H:.*';", line)):
                path = str(workingDirectoryPath.as_posix())
                repl = f"'{path}';"
                replacement = re.sub("'H:.*';", repl, line)
            if bool(re.search('D:.*configFiles', line)):
                replacement = re.sub(
                    'D:.*configFiles',
                    str(workingDirectoryPath.as_posix()),
                    line
                )
            if bool(re.search('neuropixPhase3A_kilosortChanMap.mat', line)):
                replacement = re.sub('neuropixPhase3A_kilosortChanMap.mat', 'kilosortChannelMap3a.mat', line)
            if bool(re.search('configFile384.m', line)):
                replacement = re.sub('configFile384.m', 'kilosortConfigScript.m', line)
            if replacement is None:
                stream.write(line)
            else:
                stream.write(replacement)

        # Append this code block to the end of the main script
        stream.write(f'% Kill MATLAB and clean up\n')
        stream.write(f'fprintf("All done!");\n')
        stream.write(f'close all;\n')
        stream.write(f'quit;\n')         

    return mainScriptFilePathLocal

def checkAutosortingResults(workingDirectory):
    """
    Check if the automatic spike-sorting was successful
    """

    expectedFilenames = (
        'cluster_group.tsv',
        'cluster_ContamPct.tsv',
        'cluster_KSLabel.tsv',
        'cluster_Amplitude.tsv',
    )

    flags = list()

    # Check for cluster files
    for expectedFilename in expectedFilenames:
        fileList = [file.name for file in pl.Path(workingDirectory).glob('cluster*')]
        if expectedFilename in fileList:
            flags.append(True)
        else:
            flags.append(False)

    # Check for phy params.py file
    if 'params.py' in [file.name for file in pl.Path(workingDirectory).iterdir()]:
        flags.append(True)
    else:
        flags.append(False)

    # Check if all files were found
    result = all(flags)

    return result

def updateSortingParameters(workingDirectory, **kwargs):
    """
    """

    #
    if 'batchSize' in kwargs.keys():
        if kwargs['batchSize'] % 32 != 0:
            raise Exception(f'Batch size must be a multiple of 32')

    # Find config script
    kilosortConfigScript = None
    for file in pl.Path(workingDirectory).iterdir():
        if file.name == 'kilosortConfigScript.m':
            kilosortConfigScript = file
    if kilosortConfigScript is None:
        raise Exception(f'Could not find the config script for kilosort')

    # Parse lines looking for kwarg keys
    with open(kilosortConfigScript, 'r') as stream:
        lines = stream.readlines()
    kilosortConfigScript.unlink()
    with open(kilosortConfigScript, 'w') as stream:
        for line in lines:
            replacement = None
            for parameter, value in kwargs.items():
                if bool(re.search(f'ops\.{parameter}.*=.*;', line)):
                    replacement = re.sub(f'ops\.{parameter}.*=.*;', f'ops.{parameter} = {value};', line)
                    break
                elif parameter == 'batchSize':
                    if line.startswith('ops.NT'):
                        replacement = re.sub(f'\*1024\+', f'*{value}+', line)

            if replacement is None:
                stream.write(line)
            else:
                stream.write(replacement)

    return

def startKilosortProcess(
    sourceDirectory=None,
    workingDirectory=None,
    deleteAfterSorting=False,
    useMatlabEngine=False,
    **sortingParameters
    ):
    """
    Runs the automatic spike-sorting in the working directory then
    return the spike-sorting results to the source directory
    """

    # Check that all the software dependencies were located
    global matlabExecutableFile
    global numpyInstallFolder
    global kilosortInstallFolder
    if any([
        matlabExecutableFile is None,
        numpyInstallFolder is None,
        kilosortInstallFolder is None
    ]):
        raise Exception('Please call the "locateSoftwareDependencies" function')

    #
    mainScriptFilePath = generateMatlabScripts(workingDirectory)
    updateSortingParameters(workingDirectory, **sortingParameters)

    #
    binaryDataTransferred = False
    for file in pl.Path(workingDirectory).iterdir():
        if file.name == 'continuous.dat':
            binaryDataTransferred = True
    
    if binaryDataTransferred == False:
        sourceDirectoryPath = pl.Path(sourceDirectory)
        for file in sourceDirectoryPath.iterdir():
            if bool(re.search('continuous.dat', file.name)):
                shutil.copy(
                    str(file),
                    str(pl.Path(workingDirectory).joinpath('continuous.dat'))
                )

    # Execute main script
    if useMatlabEngine:
        if me is None:
            raise Exception(f'Python engine for MATLAB is not installed')
        else:
            engine = me.start_matlab()
            engine.run(str(mainScriptFilePath), nargout=0)
            engine.quit()

    else:
        if os.name == 'nt':
            matlabFunction = f'run(""{mainScriptFilePath}"")' # NOTE: For some reason double-quotes are necessary on Windows
        elif os.name == 'posix':
            matlabFunction = f'run("{mainScriptFilePath}")'
        command = [
            str(matlabExecutableFile),
            f'-r',
            matlabFunction
        ]
        p = sp.Popen(command)
        output, errors = p.communicate() # This should block until MATLAB closes (works on Ubuntu but not Windows)

    # NOTE: Blocking until the sorting is complete has to be done manually on Windows
    if os.name == 'nt':
        while True:
            result = checkAutosortingResults(workingDirectory)
            if result:
                break

    # Copy sorting results back to the source directory and clean up
    for file in pl.Path(workingDirectory).iterdir():
        if file.name not in ('continuous.dat', 'temp_wh.dat', 'rez.mat'):
            shutil.copy(
                str(file),
                str(pl.Path(sourceDirectory).joinpath(file.name))
            )
        if deleteAfterSorting:
            file.unlink()

    return