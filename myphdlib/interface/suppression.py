import yaml
import numpy as np
import pandas as pd
import pathlib as pl
from myphdlib.interface.session import SessionBase

class SuppressionSession(SessionBase):
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

        if self.cohort == 1:
            tail = '_metadata.yaml'
        result = list(self.folders.videos.glob(f'*{tail}'))
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
    def leftEyePose(self):
        """
        """

        if self.cohort == 1:
            tail = 'rightCam-0000DLC_resnet50_GazerMay24shuffle1_1030000.csv'
        result = list(self.sessionFolderPath.joinpath('videos').glob(f'*{tail}'))
        if len(result) != 1:
            return None
        else:
            return result.pop()
    
    @property
    def rightEyePose(self):
        """
        """

        if self.cohort == 1:
            tail = 'leftCam-0000_reflectedDLC_resnet50_GazerMay24shuffle1_1030000.csv'
        result = list(self.sessionFolderPath.joinpath('videos').glob(f'*{tail}'))
        if len(result) != 1:
            return None
        else:
            return result.pop()
        
    @property
    def leftCameraTimestamps(self):
        """
        """

        if self.cohort == 1:
            tag = 'rightCam_timestamps'
        result = list(self.folders.videos.glob(f'*{tag}*'))
        if len(result) != 1:
            return None
        else:
            return result.pop()
    
    @property
    def rightCameraTimestamps(self):
        """
        """
        
        if self.cohort == 1:
            tag = 'leftCam_timestamps'
        result = list(self.sessionFolderPath.joinpath('videos').glob(f'*{tag}*'))
        if len(result) != 1:
            return None
        else:
            return result.pop()