import re
from myphdlib.general.session import (
    saveSessionData,
    locateFactorySource,
    SessionBase
)

class Session(SessionBase):
    """
    """

    def __init__(self, sessionFolder):
        """
        """

        #
        super().__init__(sessionFolder)

        #
        self.videosFolderPath = self.sessionFolderPath.joinpath('videos')

        # Determine the animal, date, and treatment
        self.notesFilePath = self.sessionFolderPath.joinpath('notes.txt')
        self.animal, self.date, self.treatment = None, None, None
        if self.notesFilePath.exists():
            with open(self.notesFilePath, 'r') as stream:
                lines = stream.readlines()
            for line in lines:
                for attribute in ('animal', 'date', 'experiment', 'treatment'):
                    if bool(re.search(f'{attribute}*', line.lower())) and line.startswith('-') == False:
                        value = line.lower().split(': ')[-1].rstrip('\n')
                        if attribute == 'date':
                            value = dt.strptime(value, '%Y-%m-%d')
                        setattr(self, attribute, value)

        return