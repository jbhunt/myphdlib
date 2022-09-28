import os
import re
import pathlib as pl

CONFIG = None

def changeWorkingNetwork(network='Gazer'):
    """
    """

    if network == 'Gazer':
        global CONFIG
        CONFIG = '/media/jbhunt/JH-DATA-00B/Networks/Gazer-Josh-2022-05-24/config.yaml'
        if pl.Path(CONFIG).exists() == False:
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
