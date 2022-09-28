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

class SuppressionSession():
    """
    """

    def __init__(self, root, load_neural_data=True):
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
                df = df.sort_index(level=1, axis=1)
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

            #

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
    def grating_onset_timestamps(self):
        """
        """

        container = {
            'cw': {
                'static': list(),
                'motion': list(),
            },
            'ccw': {
                'static': list(),
                'motion': list(),
            }
        }

        #
        try:
            mat = np.load(self.folders['analysis'].joinpath('event-metadata-and-timestamps.npy'))
        except FileNotFoundError:
            return container

        motion_event_mask = mat[:, 0] == const.TARGET_EVENT_MOTION
        appearance_event_mask = mat[:, 0] == const.TARGET_EVENT_GRATING

        #
        for start, direction in zip([0, 1], ['cw', 'ccw']):

            grating_motion_onset_timestamps = mat[motion_event_mask, 2][start::2]
            container[direction]['motion'] = grating_motion_onset_timestamps

            grating_appearance_onset_timestamps = mat[appearance_event_mask, 2][start::2]
            container[direction]['static'] = grating_appearance_onset_timestamps

        return container

    @property
    def grating_onset_timestamps(self):
        """
        """

        data = {
            'ipsi': list(),
            'contra': list()
        }

        #
        try:
            mat = np.load(self.folders['analysis'].joinpath('event-metadata-and-timestamps.npy'))
        except FileNotFoundError:
            return container

        target_event_mask = mat[:, 0] == const.TARGET_EVENT_GRATING

        for start, direction in zip([0, 1], ['ipsi', 'contra']):
            data[direction] = mat[target_event_mask, 2][start::2]

        return data

    @property
    def grating_motion_timestamps(self):
        """
        """

        data = {
            'ipsi': list(),
            'contra': list()
        }

        #
        try:
            mat = np.load(self.folders['analysis'].joinpath('event-metadata-and-timestamps.npy'))
        except FileNotFoundError:
            return container

        target_event_mask = mat[:, 0] == const.TARGET_EVENT_MOTION

        for start, direction in zip([0, 1], ['ipsi', 'contra']):
            data[direction] = mat[target_event_mask, 2][start::2]

        return data

    @property
    def grating_offset_timestamps(self):
        """
        """

        data = {
            'ipsi': list(),
            'contra': list()
        }

        #
        try:
            mat = np.load(self.folders['analysis'].joinpath('event-metadata-and-timestamps.npy'))
        except FileNotFoundError:
            return container

        target_event_mask = mat[:, 0] == const.TARGET_EVENT_STOP

        for start, direction in zip([0, 1], ['ipsi', 'contra']):
            data[direction] = mat[target_event_mask, 2][start::2]

        return data

    @property
    def binocular_saccade_waveforms(self):

        window = (-0.5, 0.7)
        start = int(np.floor(window[0] * const.SAMPLING_RATE_CAMERAS))
        stop = int(np.ceil(window[1] * const.SAMPLING_RATE_CAMERAS))

        data = {
            'contra': {
                'left' : np.full((self.saccade_onset_timestamps['contra'].size, stop - start), np.nan),
                'right': np.full((self.saccade_onset_timestamps['contra'].size, stop - start), np.nan),
                'indices': np.full(self.saccade_onset_timestamps['contra'].size, np.nan)
            },
            'ipsi': {
                'left' : np.full((self.saccade_onset_timestamps['ipsi'].size, stop - start), np.nan),
                'right': np.full((self.saccade_onset_timestamps['ipsi'].size, stop - start), np.nan),
                'indices': np.full(self.saccade_onset_timestamps['ipsi'].size, np.nan)
            }
        }

        #
        try:
            mat = np.load(self.folders['analysis'].joinpath('true-saccades-classification-results.npy'))
        except:
            return data

        for direction in ['contra', 'ipsi']:
            if direction == 'ipsi':
                saccade_direction_mask = mat[:, -1] == const.CONJUGATE_SACCADE_DIRECTION_IPSILATERAL
            else:
                saccade_direction_mask = mat[:, -1] == const.CONJUGATE_SACCADE_DIRECTION_CONTALATERAL
            for eye in ['left', 'right']:
                for irow, saccade_onset_index in enumerate(mat[saccade_direction_mask, 0]):
                    # start, stop = saccade_onset_index - 25, saccade_onset_index + 12 + 25 + 1
                    waveform = self.pupil_center_coords[eye, 'pc1'][saccade_onset_index + start: saccade_onset_index + stop]
                    data[direction][eye][irow, :] = waveform
                    data[direction]['indices'][irow] = saccade_onset_index

        #
        for direction in ['ipsi', 'contra']:
            data[direction]['indices'] = data[direction]['indices'].astype(int)

        return data

    @property
    def perisaccadic(self):
        """
        """

        # Get all saccade onset timestamps
        saccadeOnsetTimestamps = np.concatenate([
            self.saccade_onset_timestamps['ipsi'],
            self.saccade_onset_timestamps['contra']
        ])
        saccadeOnsetTimestamps.sort()

        #
        perisaccadic = {
            'low': np.zeros(self.probe_onset_timestamps['low'].size).astype(bool),
            'medium': np.zeros(self.probe_onset_timestamps['medium'].size).astype(bool),
            'high': np.zeros(self.probe_onset_timestamps['high'].size).astype(bool)
        }

        #
        for level in ['low', 'medium', 'high']:
            probeOnsetTimestamps = self.probe_onset_timestamps[level]
            for index, probeOnsetTimestamp in enumerate(probeOnsetTimestamps):
                latencies = probeOnsetTimestamp - saccadeOnsetTimestamps
                closest = np.argmin(abs(latencies))
                latency = latencies[closest]
                if np.all([latency >= -0.05, latency <= 0.15]):
                    perisaccadic[level][index] = True
                else:
                    perisaccadic[level][index] = False

        return perisaccadic

    @property
    def extrasaccadic(self):
        """
        """

        # Get all saccade onset timestamps
        saccadeOnsetTimestamps = np.concatenate([
            self.saccade_onset_timestamps['ipsi'],
            self.saccade_onset_timestamps['contra']
        ])
        saccadeOnsetTimestamps.sort()

        #
        extrasaccadic = {
            'low': np.zeros(self.probe_onset_timestamps['low'].size).astype(bool),
            'medium': np.zeros(self.probe_onset_timestamps['medium'].size).astype(bool),
            'high': np.zeros(self.probe_onset_timestamps['high'].size).astype(bool)
        }

        #
        for level in ['low', 'medium', 'high']:
            probeOnsetTimestamps = self.probe_onset_timestamps[level]
            for index, probeOnsetTimestamp in enumerate(probeOnsetTimestamps):
                latencies = probeOnsetTimestamp - saccadeOnsetTimestamps
                closest = np.argmin(abs(latencies))
                latency = latencies[closest]
                if np.all([latency >= -0.05, latency <= 0.15]):
                    extrasaccadic[level][index] = False
                else:
                    extrasaccadic[level][index] = True

        return extrasaccadic

class SuppressionDataset():
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
                check1 = not bool(re.search('labeled', video.name))
                check2 = np.any([
                    bool(re.search('reflected', video.name)),
                    bool(re.search('rightCam', video.name)) 
                ])
                if np.all([check1, check2]):
                    result.append(str(video))

        return result

    @property
    def sessionFolders(self):
        """
        """

        sessionFolders = list()
        for date in self.rootFolderPath.iterdir():
            if re.search('\d{4}-\d{2}-\d{2}', date.name):
                for animal in date.iterdir():
                    if bool(re.search('pixel\d{1}', animal.name)):
                        sessionFolders.append(str(animal))

        return sessionFolders
