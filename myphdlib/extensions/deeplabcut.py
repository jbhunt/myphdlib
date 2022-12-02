import os
import yaml
import numpy as np
import pandas as pd
import pathlib as pl

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