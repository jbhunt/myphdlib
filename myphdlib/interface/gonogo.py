import yaml
import numpy as np
import pandas as pd
import pathlib as pl
from myphdlib.interface.session import SessionBase

class GonogoSession(SessionBase):
    """
    """

    def __init__(self, sessionFolder):
        """
        """

        super().__init__(sessionFolder)

        return

    @property
    def fps(self):
        """
        """

        result = list(self.sessionFolderPath.joinpath('videos').glob('*_metadata.yaml'))
        if len(result) != 1:
            raise Exception('Could not locate video acquisition metadata file')
        with open(result.pop(), 'r')  as stream:
            metadata = yaml.full_load(stream)

        for key in metadata.keys():
            if key in ('cam1', 'cam2'):
                if metadata[key]['ismaster']:
                    fps = int(metadata[key]['framerate'])

        return fps
    @property
    def probeMetadata(self):
        """
        """

        result = list(self.sessionFolderPath.joinpath('videos').glob('*ProbeMetadata.txt'))
        if len(result) != 1:
            raise Exception('Could not locate the probe metadata')
        else:
            return result.pop()

    @property
    def rightCameraMovie(self):
        """
        """

        result = list(self.sessionFolderPath.joinpath('videos').glob('*_rightCam-0000.mp4'))
        if len(result) != 1:
            raise Exception('Could not locate the right camera movie')
        else:
            return result.pop()

    @property
    def leftEyePose(self):
        """
        """

        result = list(self.sessionFolderPath.joinpath('videos').glob('*pupilsizeFeb6shuffle1*'))
        if len(result) != 1:
            raise Exception('Could not locate the left eye pose estimate')
        else:
            return result.pop()
    @property
    def tonguePose(self):
        """
        """

        result = list(self.sessionFolderPath.joinpath('videos').glob('*licksNov3shuffle1*'))
        if len(result) != 1:
            raise Exception('Could not locate the tongue pose estimate')
        else:
            return result.pop()

    @property
    def rightCameraTimestamps(self):
        """
        """

        result = list(self.sessionFolderPath.joinpath('videos').glob('*rightCam_timestamps.txt'))
        if len(result) != 1:
            raise Exception('Could not locate the right camera timestamps')
        else:
            return result.pop()

    @property
    def leftCameraTimestamps(self):
        """
        """

        result = list(self.sessionFolderPath.joinpath('videos').glob('*leftCam_timestamps.txt'))
        if len(result) != 1:
            raise Exception('Could not locate the left camera timestamps')
        else:
            return result.pop()

    @property
    def labjackFolder(self):
        """
        """

        result = list(self.sessionFolderPath.joinpath('labjack').glob('*test*'))
        if len(result) != 1:
            raise Exception('Could not locate the Labjack folder')
        else:
            return result.pop()