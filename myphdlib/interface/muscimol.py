import yaml
import numpy as np
import pandas as pd
import pathlib as pl
from myphdlib.interface.session import SessionBase

def readExperimentLog(log, animal, date, side='left', key='treatment'):
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
    sheet.date = sheet.date.interpolate(method='ffill')

    #
    mask = np.logical_and(
        sheet.date == date,
        sheet.side == side
    )
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

        result = list(self.sessionFolderPath.joinpath('videos').glob('*video-acquisition-metadata*'))
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

        result = list(self.sessionFolderPath.joinpath('videos').glob('*left-camera-movie*.csv'))
        if len(result) != 1:
            return None
        else:
            return result.pop()
    
    @property
    def rightEyePose(self):
        """
        """

        result = list(self.sessionFolderPath.joinpath('videos').glob('*right-camera-movie*.csv'))
        if len(result) != 1:
            return None
        else:
            return result.pop()
    
    @property
    def leftCameraTimestamps(self):
        """
        """

        result = list(self.sessionFolderPath.joinpath('videos').glob('*left-camera-timestamps*'))
        if len(result) != 1:
            return None
        else:
            return result.pop()
    
    @property
    def rightCameraTimestamps(self):
        """
        """
        
        result = list(self.sessionFolderPath.joinpath('videos').glob('*right-camera-timestamps*'))
        if len(result) != 1:
            return None
        else:
            return result.pop()
