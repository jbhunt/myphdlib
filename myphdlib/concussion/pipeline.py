import numpy as np
import pathlib as pl
from myphdlib.toolkit import reflectVideo

def reflectVideos(dataset, eye='left'):
    """
    """

    for session in dataset.sessions:
        videos = list(pl.Path(session.videosFolder).rglob('*.mp4'))
        for video in videos:
            if f'{eye}Cam' in video.name and 'reflected' not in video.name:
                print(video.name)
                reflectVideo(str(video))

    return

class Pipeline():
    """
    """

    def __init__(self):
        """
        """

        self.modules = list()
        self.kwargs = list()

        return

    def addModule(self, f, **kwargs):
        """
        """

        self.modules.append(f)
        self.kwargs.append(kwargs)

        return

    def process(self, dataset):
        """
        """

        for f, kwargs in zip(self.modules, self.kwargs):
            f(dataset, **kwargs)

        return

    