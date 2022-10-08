# Imports
import os
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
                for attribute in ('animal', 'date', 'experiment'):
                    if bool(re.search(f'{attribute}*', line.lower())) and line.startswith('-') == False:
                        value = line.lower().split(': ')[-1].rstrip('\n')
                        setattr(self, attribute, value)

        return

class SessionFactory():
    """
    """

    def __init__(self, hdd='JH-DATA-01', alias='Suppression2'):
        """
        """

        user = os.environ['USER']
        self.rootFolder = pl.Path(f'/media/{user}').joinpath(hdd, alias)
        if self.rootFolder.exists() == False:
            raise Exception('Could not locate data')

        self.sessionFolders = None

        return

    def produce(self):
        return

    def __iter__(self):
        self.sessionFolders = list()
        for date in self.rootFolder.iterdir():
            for animal in date.iterdir():
                self._sessionFolders.append(str(animal))
        self._listIndex = 0
        return self

    def __next__(self):
        if self._listIndex < len(self.sessionFolders):
            sessionFolder = self.sessionFolders[self._listIndex]
            self._listIndex += 1
            return Session(sessionFolder)
        else:
            raise StopIteration