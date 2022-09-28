import re
import itertools
import numpy as np
import pandas as pd
import pathlib as pl
from scipy.stats import pearsonr
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

from .. import toolkit

class MuscimolSession():
    """
    """

    def __init__(self, root):
        """
        """

        self._root = pl.Path(root)

        return

    def pupil_center_coords(self, normalize=True, hanning_window_length=5, ndigits=3):
        """
        """

        data = {
            'left': {
                'x': None,
                'y': None,
                'pc1': None,
                'pc2': None,
            },
            'right': {
                'x': None,
                'y': None,
                'pc1': None,
                'pc2': None,
            }
        }

        #
        csvs = {
            'left': list(self.root.rglob('*left-eye-score-2.csv')).pop(),
            'right': list(self.root.rglob('*right-eye-score-2.csv')).pop()
        }

        for eye, csv in csvs.items():

            # Extract coordinates for the pupil center in pixels
            score = pd.read_csv(str(csv), header=[0, 1, 2, 3], index_col=0)
            network = score.keys()[0][0]
            likelihood = np.array(score[network, 'pupil-center', 'likelihood'])
            pupil_center_coords = pd.concat([
                score[network, 'pupil-center', 'x'],
                score[network, 'pupil-center', 'y']
            ], axis=1)
            condition = np.hstack([likelihood < 0.99, likelihood < 0.99])
            pupil_center_coords.mask(condition, inplace=True)
            pupil_center_coords.interpolate(method='polynomial', order=3, axis=0, inplace=True)
            smoothed = toolkit.smooth(np.array(pupil_center_coords), hanning_window_length, axis=0)
            data[eye]['x'] = smoothed[:, 0]
            data[eye]['y'] = smoothed[:, 1]

            # Find the average position of the nasal and temporal corners of the eye
            nasal_corner_coords = pd.concat([
                score[network, 'nasal-corner', 'x'],
                score[network, 'nasal-corner', 'y']
            ], axis=1)

            temporal_corner_coords = pd.concat([
                score[network, 'temporal-corner', 'x'],
                score[network, 'temporal-corner', 'y']
            ], axis=1)

            nasal_corner_likelihood = np.array(score[network, 'nasal-corner', 'likelihood'])
            nasal_corner_condition = np.hstack([
                nasal_corner_likelihood < 0.99,
                nasal_corner_likelihood < 0.99
            ])
            nasal_corner_coords.mask(nasal_corner_condition, inplace=True)

            temporal_corner_likelihood = np.array(score[network, 'temporal-corner', 'likelihood'])
            temporal_corner_condition = np.hstack([
                temporal_corner_likelihood < 0.99,
                temporal_corner_likelihood < 0.99
            ])
            temporal_corner_coords.mask(temporal_corner_condition, inplace=True)

            # Decompose the pupil center position
            model = PCA(n_components=2).fit(smoothed)
            decomposed = model.transform(smoothed)
            pc1, pc2 = decomposed[:, 0], decomposed[:, 1]

            # Determine the coef
            r, p = pearsonr(pc1, np.array(score[network, 'pupil-center', 'x']))
            if r <= 0:
                coef = -1
            else:
                coef = +1

            if eye == 'right':
                coef *= -1

            #
            nasal_corner_point = model.transform(np.atleast_2d(np.nanmean(nasal_corner_coords, axis=0))).flatten()
            temporal_corner_point = model.transform(np.atleast_2d(np.nanmean(temporal_corner_coords, axis=0))).flatten()
            pc1_value_range = np.array([nasal_corner_point[0], temporal_corner_point[0]]).reshape(-1, 1)
            pc2_value_range = np.array([nasal_corner_point[1], temporal_corner_point[1]]).reshape(-1, 1)
            pc1_axis_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(pc1_value_range)
            pc2_axis_scaler = MinMaxScaler(feature_range=(-1, 1)).fit(pc2_value_range)

            #
            pc1_rescaled = pc1_axis_scaler.transform(pc1.reshape(-1, 1)).flatten()
            pc2_rescaled = pc2_axis_scaler.transform(pc2.reshape(-1, 1)).flatten()
            data[eye]['pc1'] = pc1_rescaled * coef
            data[eye]['pc2'] = pc2_rescaled * coef

        columns = list(itertools.product(['left', 'right'], ['x', 'y', 'pc1', 'pc2']))
        reformed = {
            (eye, feature): np.around(data[eye][feature], ndigits)
                for eye, feature in columns
        }

        return pd.DataFrame(reformed, columns=pd.MultiIndex.from_tuples(columns))

    @property
    def root(self):
        return self._root

class MuscimolDataset():
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
                if bool(re.search('reflected', video.name)) or bool(re.search('left-camera', video.name)):
                    result.append(str(video))

        return result

    @property
    def sessionFolders(self):
        """
        """

        sessionFolders = list()
        for animal in self.rootFolderPath.iterdir():
            if bool(re.search('musc\d{1}', animal.name)):
                for date in animal.iterdir():
                    if re.search('\d{4}-\d{2}-\d{2}', date.name):
                        sessionFolders.append(str(date))

        return sessionFolders
