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

def locateSoftwareDependencies(matlabAddonsFolder='/home/jbhunt/Code', kilosortVersion='2.0'):
    """
    Identifies the filepaths to the Kilosort and npy-matlab installations
    """

    global kilosortInstallFolder

    _kilosortInstallFolder = pl.Path(matlabAddonsFolder).joinpath(f'Kilosort-{kilosortVersion}')
    if _kilosortInstallFolder.exists() == False:
        raise Exception(f'Could not locate Kilosort installation folder')
    else:
        kilosortInstallFolder = _kilosortInstallFolder

    global numpyInstallFolder

    _numpyInstallFolder = pl.Path(matlabAddonsFolder).joinpath(f'npy-matlab-master', 'npy-matlab')
    if _numpyInstallFolder.exists() == False:
        raise Exception(f'Could not locate the npy-matlab installation folder')
    else:
        numpyInstallFolder = _numpyInstallFolder

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
            if bool(re.search('%% if you want to save the results to a Matlab file...\n', line)) and saveRezFile == False:
                stream.write(f'% Kill MATLAB')
                stream.write(f'fprintf("All done!");\n')
                stream.write(f'quit;\n')
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

    return mainScriptFilePathLocal

def autosortNeuralRecording(
    sourceDirectory=None,
    workingDirectory=None,
    deleteAfterSorting=False,
    useMatlabEngine=False
    ):
    """
    Runs the automatic spike-sorting in the working directory then
    return the spike-sorting results to the source directory
    """

    #
    mainScriptFilePath = generateMatlabScripts(workingDirectory)

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
            # engine.run('/home/jbhunt/Desktop/untitled.m', nargout=0)
            engine.run(str(mainScriptFilePath), nargout=0)
            engine.quit()

    else:
        command = [
            f'/usr/local/bin/matlab',
            f'-r',
            f'run("{mainScriptFilePath}")'
        ]
        print(command)
        p = sp.run(command)
        # p = sp.Popen(command)
        # out, errs = p.communicate()

    # Copy sorting results back to the source directory and clean up
    for file in pl.Path(workingDirectory).iterdir():
        if file.name in ('continuous.dat', 'temp_wh.dat', 'rez.mat'):
            if deleteAfterSorting:
                file.unlink()
            continue
        shutil.copy(
            str(file),
            str(pl.Path(sourceDirectory).joinpath(file.name))
        )
        if deleteAfterSorting:
            file.unlink()
    if deleteAfterSorting:
        pl.Path(workingDirectory).unlink()

    return