import os
import re
import sys
import yaml
import time
import logging
import contextlib
import numpy as np
import pandas as pd
import pathlib as pl
import subprocess as sp
from matplotlib import pylab as plt
import shutil
try:
    import deeplabcut as dlc
except ImportError:
    dlc = None

configFilePath = None

def changeWorkingNetwork(network='Gazer'):
    """
    """

    if network == 'Gazer':
        global configFilePath
        configFilePath = pl.Path('/media/jbhunt/JH-DATA-00B/Networks/Gazer-Josh-2022-05-24/config.yaml')
        if configFilePath.exists() == False:
            raise Exception('Could not locate config file')

    return

def deleteLabeledFolders(config):
    """
    """

    with open(config, 'rb') as stream:
        data = yaml.load(stream, Loader=yaml.SafeLoader)

    projectFolderPath = pl.Path(data['project_path'])
    labeledDataFolderPath = projectFolderPath.joinpath('labeled-data')
    for folder in labeledDataFolderPath.iterdir():
        if folder.name.endswith('_labeled'):
            print(f'Deleting {folder.name} ...')
            for file in folder.iterdir():
                os.remove(str(file))
            folder.rmdir()

    return

def loadBodypartData(csv, bodypart='pupilCenter', feature='likelihood'):
    """
    """

    frame = pd.read_csv(csv, header=list(range(3)), index_col=0)
    network = frame.columns[0][0]

    return np.array(frame[network, bodypart, feature])

def checkGpuMemory(refresh=1):
    """
    """

    try:
        while True:
            text = sp.check_output('nvidia-smi').decode()
            fan, gpu = re.findall('\d*%', text)
            value = round(float(gpu.rstrip('%')), 2)
            print(f'GPU Usage: {value} %', end='\r')
            time.sleep(refresh)

    except KeyboardInterrupt as error:
        print('\n', end='\r')

    return

def visualizeGpuMemory(refresh=1, head=5, window=30, **kwargs):
    """
    """

    fig, ax = plt.subplots(num=1)
    ax.set_title('GPU Usage (%)')
    ax.set_xlabel('Time (sec)')
    ax.set_ylabel('% Usage')
    ax.set_ylim([-5, 105])
    ax.set_xlim([-1, window])
    lines = {
        'head': ax.plot([0], [0], color='r')[0],
        'tail': ax.plot([0], [0], color='k')[0]
    }
    data = {
        'x': np.arange(window),
        'y': np.full(window, np.nan),
    }
    plt.ion()

    #
    while plt.fignum_exists(1):
        try:
            
            #
            t1 = time.time()
            text = sp.check_output('nvidia-smi').decode()
            fan, gpu = re.findall('\d*%', text)
            value = round(float(gpu.rstrip('%')), 0)
            ax.set_title(f'GPU Usage ({value:.0f}%)')

            #
            y = np.roll(data['y'], shift=-1)
            y[-1] = value
            lines['tail'].set_data(data['x'][:window - head], data['y'][:window - head])
            lines['head'].set_data(data['x'][window - head - 1:], data['y'][window - head - 1:])
            data['y'] = y
            fig.canvas.draw()
            plt.pause(0.001)

            #
            t2 = time.time()
            elapsed = t2 - t1
            time.sleep(refresh - (elapsed))
        
        except KeyboardInterrupt as error:
            plt.close(fig)
            break

    return

def analyzeVideosQuietly(*args, **kwargs):
    """
    Call DeepLabCut's analyze_videos function but suppress messaging
    """

    with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        logging.disable(logging.CRITICAL)  # Disable logging
        try:
            return dlc.analyze_videos(*args, **kwargs)
        finally:
            logging.disable(logging.NOTSET)  # Re-enable logging

def transferAndExtractPose(
    configFile,
    targetDirectory,
    videoList,
    skipExisting=True,
    timeout=30,
    ):
    """
    """

    for src in videoList:

        #
        try:

            # Check for existing pose estimates
            src = pl.Path(src)
            skip = False
            for file in src.parent.iterdir():
                if 'sacnet' in file.name and skipExisting == True:
                    skip = True
            if skip:
                print(f'INFO: Skipping {src}')
                continue

            # Copy
            dst = targetDirectory.joinpath(src.name)
            print(f'INFO: Copying {src} --> {dst}')
            shutil.copy2(
                src,
                targetDirectory.joinpath(src.name)
            )

            #
            print(f'INFO: Analyzing {dst}')
            analyzeVideosQuietly(
                configFile,
                [str(dst),],
                save_as_csv=True,
            )

            #
            for file in targetDirectory.iterdir():
                if file.suffix == '.csv':
                    dst2 = src.parent.joinpath(file.name)
                    print(f'INFO: Copying {file} --> {dst2}')
                    os.system(f'cp "{file}" "{dst2}"') # NOTE: shutil freaks out trying to copy to a networked drive
                    t1 = time.time()
                    while dst2.exists() != True: # Wait 30 seconds
                        dt = time.time() - t1
                        if dt >= timeout:
                            break
                        continue
                    if dst2.exists() == False:
                        raise Exception()

        except KeyboardInterrupt:
            break

        except:
            print(f'ERROR: Pose estimation failed for {dst}')
            pass
        
        # Clean up
        print(f'INFO: Cleaning up target directory')
        dst.unlink()
        for file in targetDirectory.iterdir():
            file.unlink()

    return