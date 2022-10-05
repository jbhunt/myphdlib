# Imports
import re
import pathlib as pl

# Class definitions
class Session():
    """
    """

    def __init__(self, sessionFolder):
        """
        """

        # Paths
        self.sessionFolderPath = pl.Path(sessionFolder)
        self.labjackFolderPath = self.sessionFolderPath.joinpath('labjack')
        self.ephysFolderPath = self.sessionFolderPath.joinpath('ephys')
        self.videosFolderPath = self.sessionFolderPath.joinpath('videos')

        # Determine the animal, date, and treatment
        self.notesFilePath = self.sessionFolderPath.joinpath('notes.txt')
        self.animal, self.date, self.treatment = None, None, None
        if self.notesFilePath.exists():
            with open(self.notesFilePath, 'r') as stream:
                lines = stream.readlines()
            for line in lines:
                for attribute in ('animal', 'date', 'treatment'):
                    if bool(re.search(f'{attribute}*', line.lower())):
                        value = line.lower().split(': ')[-1]
                        setattr(self, attribute, value)

        return

class SessionFactory():
    """
    """

    def __init__(self, hdd='JH-DATA-01', alias='Suppression2'):
        """
        """

        self.rootFolder = pl.Path('/media').joinpath(hdd, alias)
        if self.rootFolder.exists() == False:
            raise Exception('Could not locate data')

        return

    def produce(self):
        return

    def __iter__(self):
        return

    def __next__(self):
        return