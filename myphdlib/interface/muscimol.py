import yaml
import numpy as np
import pandas as pd
import pathlib as pl
from myphdlib.interface.session import SessionBase

def readExperimentLog(log, animal, date, letter=None, key='treatment'):
    """
    """

    #
    if type(date) != str:
        date = str(date)

    #
    try:
        sheet = pd.read_excel(log, sheet_name=animal)
    except ValueError as error:
        raise Exception(f'{animal} is not a valid sheet name') from None
    
    #
    if letter is None:
        row = sheet[sheet.Date == date].squeeze()
    else:
        mask = np.logical_and(sheet.Date == date, sheet.Letter == letter)
        row = sheet[mask].squeeze()

    #
    if key in row.keys():
        return row[key]
    
    else:
        raise Exception(f'{key} is not a valid key')

class MuscimolSession(SessionBase):
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
            tail = 'video-acquisition-metadata.yml'
        elif self.cohort == 2:
            tail = 'metadata.yaml'
        result = list(self.sessionFolderPath.joinpath('videos').glob(f'*{tail}'))
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
    def leftCameraMovie(self):
        """
        """

        if self.cohort == 1:
            tag = 'left-camera-movie'
        elif self.cohort == 2:
            tag = 'leftCam-0000_reflected'
        result = list(self.sessionFolderPath.joinpath('videos').glob(f'*{tag}.mp4'))
        if len(result) != 1:
            return None
        else:
            return result.pop()
    
    @property
    def rightCameraMovie(self):
        """
        """

        if self.cohort == 1:
            tag = 'right-camera-movie'
        elif self.cohort == 2:
            tag = 'rightCam-0000'
        result = list(self.sessionFolderPath.joinpath('videos').glob(f'*{tag}*.mp4'))
        if len(result) != 1:
            return None
        else:
            return result.pop()   
    
    @property
    def leftEyePose(self):
        """
        """

        if self.cohort == 1:
            tail = 'left-camera-movieDLC_resnet50_GazerMay24shuffle1_1030000.csv'
        elif self.cohort == 2:
            tail = 'leftCam-0000_reflectedDLC_resnet50_GazerMay24shuffle1_1030000.csv'
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
            tail = 'right-camera-movie-reflectedDLC_resnet50_GazerMay24shuffle1_1030000.csv'
        elif self.cohort == 2:
            tail = 'rightCam-0000DLC_resnet50_GazerMay24shuffle1_1030000.csv'
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
            tag = 'left-camera-timestamps'
        elif self.cohort == 2:
            tag = 'leftCam_timestamps'
        result = list(self.sessionFolderPath.joinpath('videos').glob(f'*{tag}*'))
        if len(result) != 1:
            return None
        else:
            return result.pop()
    
    @property
    def rightCameraTimestamps(self):
        """
        """
        
        if self.cohort == 1:
            tag = 'right-camera-timestamps'
        elif self.cohort == 2:
            tag = 'rightCam_timestamps'
        result = list(self.sessionFolderPath.joinpath('videos').glob(f'*{tag}*'))
        if len(result) != 1:
            return None
        else:
            return result.pop()
