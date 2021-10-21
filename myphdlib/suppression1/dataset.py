import re
import numpy as np
import pandas as pd
import pathlib as pl
from scipy import stats
from functools import wraps
from sklearn import decomposition as dec

from .  import constants as const
from .. import toolkit as tk
from .  import activity

# Stops a performance warning raised by pandas
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

class ExperimentError(Exception):
    pass

class DatasetError(Exception):
    pass

class timestamp():
    def __init__(self, event=None):
        self.event = event

    def __call__(self):
        return

def timestamps(event_code, event_phase='onset', saccade_direction='ipsi', probe_level='low'):
    """
    """

    def callable(method):

        @wraps(method)
        def wrapped(self):

            #
            try:
                mat = np.load(self.folders['analysis'].joinpath('event-metadata-and-timestamps.npy'))
            except FileNotFoundError:
                return np.array([])

            #
            event_mask = mat[:, 0] == event_code

            #
            if event_code == const.TARGET_EVENT_SACCADE:
                if saccade_direction == 'ipsi':
                    direction_mask = mat[event_mask, 1] == const.CONJUGATE_SACCADE_DIRECTION_IPSILATERAL
                    timestamps = mat[event_mask][direction_mask, 2 if event_phase == 'onset' else 3]
                elif saccade_direction == 'contra':
                    direction_mask = mat[event_mask, 1] == const.CONJUGATE_SACCADE_DIRECTION_CONTALATERAL
                    timestamps = mat[event_mask][direction_mask, 2 if event_phase == 'onset' else 3]
            elif event_code == const.TARGET_EVENT_PROBE:
                if probe_level == 'low':
                    level_mask = mat[event_mask, 1] == const.PROBE_CONTRAST_LOW
                    timestamps = mat[event_mask][level_mask, 2 if event_phase == 'onset' else 3]
                elif probe_level == 'medium':
                    level_mask = mat[event_mask, 1] == const.PROBE_CONTRAST_MEDIUM
                    timestamps = mat[event_mask][level_mask, 2 if event_phase == 'onset' else 3]
                else:
                    level_mask = mat[event_mask, 1] == const.PROBE_CONTRAST_HIGH
                    timestamps = mat[event_mask][level_mask, 2 if event_phase == 'onset' else 3]

            return timestamps

        return wrapped

    return callable

class DriftingGratingWithProbeExperiment():
    """
    """

    def __init__(self, root, load_neural_data=False):
        """
        keywords
        --------
        """

        self._root = pl.Path(root)
        if not self._root.exists():
            raise DatasetError('Experiment directory does not exist')

        self._folders = {
            'videos'      : self.root.joinpath('Videos'),
            'labjack'     : self.root.joinpath('LabJack'),
            'analysis'    : self.root.joinpath('Analysis'),
            'neuropixels' : self.root.joinpath('Neuropixels')
        }

        # some metadata
        self._date = re.findall('\d{4}-\d{2}-\d{2}', str(self.root)).pop()
        self._animal = re.findall('pixel\d', str(self.root)).pop()
        self._first_ephys_sample = None

        # Pupil center coordinates in pixels and PCs
        self._pupil_center_coords = None

        #
        if load_neural_data:
            self._population = activity.Population(self)
        else:
            self._population = None

        return

    @property
    def first_ephys_sample(self):
        """
        Offset (in Neuropixels samples) that marks the begging of the recording
        """

        if self._first_ephys_sample is None:
            result = list(self.root.rglob('*sync_messages.txt'))
            if len(result) == 1:
                sync_messages = result.pop()
            else:
                raise Exception('No sync messages file found')
            with open(sync_messages, 'r') as stream:
                lines = stream.readlines()
            for line in lines:
                result = re.findall('\d+@30000Hz', line)
                if len(result) != 0:
                    self._first_ephys_sample = int(re.findall('\d+@', result.pop()).pop().rstrip('@'))

        return self._first_ephys_sample

    @property
    def pupil_center_coords(self):
        """Pupil coordinates (in pixels and along the first 2 PCs)"""

        if self._pupil_center_coords is None:

            #
            left_eye_score = str(list(self.folders['videos'].rglob('*left*.csv')).pop())
            right_eye_score = str(list(self.folders['videos'].rglob('*right*.csv')).pop())

            #
            for eye, score in zip(['left', 'right'], [left_eye_score, right_eye_score]):
                scorer = re.findall('DLC*\w+.csv', score).pop().rstrip('.csv')
                df = pd.read_csv(score, index_col=0, header=list(range(4)))
                coords = pd.concat([
                    df[scorer, 'pupil-c', 'x'],
                    df[scorer, 'pupil-c', 'y']],
                    axis=1
                )
                likelihood = np.array(df[scorer, 'pupil-c', 'likelihood']).flatten()
                mask = likelihood < 0.98
                coords.loc[mask, :] = pd.DataFrame(np.full((mask.sum(), 2), np.nan))

                # Interpolate NaN values
                coords.interpolate(method='polynomial', order=3, axis=0, inplace=True)

                # Smooth the signal
                coords_smoothed = np.empty(coords.shape)
                for icol in range(2):
                    coords_smoothed[:, icol] = tk.smooth(coords.iloc[:, icol].to_numpy())

                # Store the result
                if eye == 'left':
                    left_pupil_coords = coords_smoothed
                else:
                    right_pupil_coords = coords_smoothed

            #
            nframes = left_pupil_coords.shape[0] \
                if left_pupil_coords.shape[0] <= right_pupil_coords.shape[0] \
                    else right_pupil_coords.shape[0]
            left_pupil_coords = left_pupil_coords[:nframes, :]
            right_pupil_coords = right_pupil_coords[:nframes, :]

            # Decompose
            pca = dec.PCA(n_components=2)
            left_pupil_PCs = pca.fit_transform(left_pupil_coords)
            right_pupil_PCs = pca.fit_transform(right_pupil_coords)

            # Determine the sign of the PCs (flip if correlation is negative)
            for coords, PCs in zip([left_pupil_coords, right_pupil_coords], [left_pupil_PCs, right_pupil_PCs]):
                PC1 = PCs[:, 0]
                y = coords[:, 1]
                r, p = stats.pearsonr(PC1, y)
                if r < 0:
                    PCs *= -1

            data = np.hstack([
                left_pupil_coords,
                left_pupil_PCs,
                right_pupil_coords,
                right_pupil_PCs
            ])
            labels = [
                ('left',  'x'),
                ('left',  'y'),
                ('left',  'pc1'),
                ('left',  'pc2'),
                ('right', 'x'),
                ('right', 'y'),
                ('right', 'pc1'),
                ('right', 'pc2')
            ]
            self._pupil_center_coords = pd.DataFrame(data, columns=pd.MultiIndex.from_tuples(labels))

        return self._pupil_center_coords

    @property
    def root(self):
        return self._root

    @property
    def folders(self):
        return self._folders

    @property
    def animal(self):
        return self._animal

    @property
    def date(self):
        return self._date

    @property
    def population(self):
        return self._population

    @property
    def probe_onset_timestamps(self):
        """
        """

        container = {'low': np.array([]), 'medium': np.array([]), 'high': np.array([])}

        #
        try:
            mat = np.load(self.folders['analysis'].joinpath('event-metadata-and-timestamps.npy'))
        except FileNotFoundError:
            return container

        #
        event_mask = mat[:, 0] == const.TARGET_EVENT_PROBE

        for level in ['low', 'medium', 'high']:
            if level == 'low':
                level_mask = mat[event_mask, 1] == const.PROBE_CONTRAST_LOW
                timestamps = mat[event_mask][level_mask, 2]
            elif level == 'medium':
                level_mask = mat[event_mask, 1] == const.PROBE_CONTRAST_MEDIUM
                timestamps = mat[event_mask][level_mask, 2]
            elif level == 'high':
                level_mask = mat[event_mask, 1] == const.PROBE_CONTRAST_HIGH
                timestamps = mat[event_mask][level_mask, 2]
            container[level] = timestamps

        return container

    @property
    def saccade_onset_timestamps(self):
        """
        """

        container = {'ipsi': np.array([]), 'contra': np.array([])}

        #
        try:
            mat = np.load(self.folders['analysis'].joinpath('event-metadata-and-timestamps.npy'))
        except FileNotFoundError:
            return container

        #
        event_mask = mat[:, 0] == const.TARGET_EVENT_SACCADE

        #
        for direction in ['ipsi', 'contra']:
            if direction == 'ipsi':
                direction_mask = mat[event_mask, 1] == const.CONJUGATE_SACCADE_DIRECTION_IPSILATERAL
                timestamps = mat[event_mask][direction_mask, 2]
            elif direction == 'contra':
                direction_mask = mat[event_mask, 1] == const.CONJUGATE_SACCADE_DIRECTION_CONTALATERAL
                timestamps = mat[event_mask][direction_mask, 2]
            container[direction] = timestamps

        return container

    @property
    @timestamps(const.TARGET_EVENT_SACCADE, saccade_direction='ipsi')
    def saccade_onset_ipsi(self):
        return

    @property
    @timestamps(const.TARGET_EVENT_SACCADE, saccade_direction='ipsi', event_phase='offset')
    def saccade_offset_ipsi(self):
        return

    @property
    @timestamps(const.TARGET_EVENT_SACCADE, saccade_direction='contra')
    def saccade_onset_contra(self):
        return

    @property
    @timestamps(const.TARGET_EVENT_SACCADE, saccade_direction='contra', event_phase='offset')
    def saccade_offset_contra(self):
        return

    @property
    @timestamps(const.TARGET_EVENT_PROBE, probe_level='low')
    def probe_onset_low(self):
        return

    @property
    @timestamps(const.TARGET_EVENT_PROBE, probe_level='medium')
    def probe_onset_medium(self):
        return

    @property
    @timestamps(const.TARGET_EVENT_PROBE, probe_level='high')
    def probe_onset_high(self):
        return

class WholeDataset():
    def __init__(self, root, exclude_sessions=(('pixel3', '2021-05-27'),)):
        if type(root) != pl.Path:
            root = pl.Path(root)
        dirs = list()
        for date in root.iterdir():
            if date.name == 'Supplement':
                continue
            for animal in date.iterdir():
                exclude = False
                for animal_, date_ in exclude_sessions:
                    if animal_ == animal.name and date_ == date.name:
                        exclude = True
                if exclude:
                    continue
                dirs.append(str(animal))

        self.sessions = [
            DriftingGratingWithProbeExperiment(dir)
                for dir in dirs
        ]
        return
