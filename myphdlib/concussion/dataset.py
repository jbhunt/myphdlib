import re
import numpy as np
import pathlib as pl
from myphdlib.toolkit import reflectVideo

rootFolder = '/media/jbhunt/JH-DATA-00/Concussion'

class ConcussionSession():
    """
    """

    def __init__(self, sessionFolder):
        """
        """

        self.sessionFolderPath = pl.Path(sessionFolder)

        return

    @property
    def videosFolder(self):
        for folder in self.sessionFolderPath.iterdir():
            result = np.any([
                bool(re.search('videos', folder.name)),
                bool(re.search('Videos', folder.name)),
                bool(re.search('session\d{3}', folder.name))
            ])
            if result:
                return str(folder)
        return None

class Dataset():
    """
    """

    def __init__(self, rootFolder):
        """
        """

        self.rootFolderPath = pl.Path(rootFolder)

        return

    def getVideoFilenames(self):
        """
        """

        result = list()

        for sessionFolder in self.sessionFolders:
            videos = list(pl.Path(sessionFolder).rglob('*.mp4'))
            for video in videos:
                if bool(re.search('(reflected)', video.name)) or bool(re.search('rightCam', video.name)):
                    result.append(str(video))

        return result

    @property
    def sessions(self):
        """
        """

        sessions = list()
        for sessionFolder in self.sessionFolders:
            session = Session(sessionFolder)
            sessions.append(session)

        return sessions

    @property
    def sessionFolders(self):
        """
        """

        sessionFolders = list()
        for date in self.rootFolderPath.iterdir():
            if re.search('\d{4}-\d{2}-\d{2}', date.name):
                for animal in date.iterdir():
                    result = np.any([
                        re.search('blast\d{1}', animal.name),
                        re.search('dreadd\d{1}', animal.name),
                    ])
                    if result:
                        sessionFolders.append(str(animal))

        return sessionFolders