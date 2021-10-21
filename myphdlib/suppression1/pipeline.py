import re
import os
import yaml
import numpy as np
import pandas as pd
import pathlib as pl

# scipy
from scipy import stats
from scipy import signal as sig

# sci-kit learn
from sklearn import neighbors as ne
from sklearn import decomposition as dec
from sklearn import neural_network as nn
from sklearn import model_selection as ms

# Relative imports
from .  import constants as const
from .  import labjack as lj
from .. import toolkit as tk

# Stops a performance warning raised by pandas
import warnings
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# load the saccade training dataset
try:
    user = os.environ['USER']
    SACCADE_TRAINING_DATASET_SAMPLES = np.load(f'/media/{user}/JH-DATA-00/Suppression/Supplement/saccade-training-dataset-samples.npy')
    SACCADE_TRAINING_DATASET_LABELS  = np.load(f'/media/{user}/JH-DATA-00/Suppression/Supplement/saccade-training-dataset-labels.npy')
    SACCADE_TRAINING_DATASET_LOADED  = True

except FileNotFoundError:
    SACCADE_TRAINING_DATASET_LOADED  = False

# load the visual stimulus metadata
try:
    VISUAL_STIMULUS_METADATA_FILEPATH = f'/media/{user}/JH-DATA-00/Suppression/Supplement/visual-stimulus-metadata.txt'
    with open(VISUAL_STIMULUS_METADATA_FILEPATH, 'r') as stream:
        VISUAL_STIMULUS_METADATA = [
            line.rstrip('\n').split(', ')
                for line in stream.readlines()
        ]
except FileNotFoundError:
    VISUAL_STIMULUS_METADATA = list()

class PipelineError(Exception):
    def __init__(self, message):
        super().__init__(message)

class PipelineModule():
    def __init__(self, session, filename=None):
        self.output = None
        self.session = session
        self.filename = filename
        return

    def save(self):
        if self.filename is None:
            raise PipelineError('No filename given')
        dst = str(self.session.folders['analysis'].joinpath(self.filename + '.npy'))
        np.save(dst, self.output)
        return

class ExtractPupilPositionModule(PipelineModule):
    """
    """

    def __init__(self, session, filename='pupil-center-position-for-both-eyes'):
        super().__init__(session, filename)

    def execute(
        self,
        input=None,
        n_decimal_places=3,
        likelihood_threshold=0.99,
        smooth_position_data=True,
        pupil_center_marker='pupil-c',
        ):
        """
        """

        # get the file paths for the DLC scores
        videos_folder = pl.Path(self.session.folders['videos'])
        csvs = [str(csv) for csv in
            list(videos_folder.rglob('*left*.csv')) + list(videos_folder.rglob('*right*.csv'))
        ]
        if len(csvs) != 2:
            raise PipelineError(f'DLC processing incomplete ({len(csvs)} of 2 scores detected)')

        #
        self.output = list()

        # iterate through each score (left and right eyes)
        for csv, side in zip(csvs, ['left', 'right']):

            # extract the x and y coordinate for the pupil center
            scorer = re.findall('DLC*\w+.csv', csv).pop().rstrip('.csv')
            df = pd.read_csv(csv, index_col=0, header=list(range(4)))
            xc = np.array(df[scorer, pupil_center_marker, 'x']).flatten()
            yc = np.array(df[scorer, pupil_center_marker, 'y']).flatten()

            # mask unlikely points
            likelihood = np.array(df[scorer, pupil_center_marker, 'likelihood']).flatten()
            mask = likelihood < likelihood_threshold
            xc[mask] = np.nan
            yc[mask] = np.nan

            # interpolate NaN values
            xc = np.array(pd.Series(xc).interpolate(method='polynomial', order=3))
            yc = np.array(pd.Series(yc).interpolate(method='polynomial', order=3))

            # Smooth
            if smooth_position_data:
                xc = tk.smooth(xc)
                yc = tk.smooth(yc)

            self.output.append(xc.tolist())
            self.output.append(yc.tolist())

        #
        list_lengths = list(map(len, self.output))
        if len(set(list_lengths)) != 1:
            target_list_length = min(list_lengths)
            for ilst, lst in enumerate(self.output):
                if len(lst) != target_list_length:
                    self.output.remove(lst)
                    difference = len(lst) - target_list_length
                    self.output.insert(ilst, lst[:len(lst) - difference])

        self.output = np.around(np.array(self.output), n_decimal_places).T

        return

class DetectPutativeSaccadesModule(PipelineModule):
    """
    Detect all putative saccades for each eye
    """

    def __init__(self, session, filename='putative-saccade-detection-results'):
        super().__init__(session, filename)

    def execute(self, input, **kwargs):
        """
        """

        # peak detection parameters
        peak_detection_params = {
            'minimum_halfpeak_distance': 2,
            'maximum_halfpeak_distance': 10,
            'minimum_interpeak_distance': 30,
            'standard_deviation_mulitple': 4,
        }
        for k, v in kwargs.items():
            if k in peak_detection_params.keys():
                peak_detection_params[k] = v

        # definitions of temporal epochs
        time_window_indices = {
            'presaccadic' : (-35 , -5  ),
            'perisaccadic': (-100, 101),
            'postsaccadic': ( 5  , 35 )
        }
        for k, v in kwargs.items():
            if k in time_window_indices.keys():
                time_window_indices[k] = v

        # data containers
        saccades, indices, eyes = list(), list(), list()

        #
        for side in ['left', 'right']:

            # find all putative saccades
            coords = np.hstack([
                np.array(input[:, 0 if side == 'left' else 2]).reshape(-1, 1),
                np.array(input[:, 1 if side == 'left' else 3]).reshape(-1, 1)
            ])
            dxy = np.sqrt((np.diff(coords, axis=0) ** 2).sum(axis=1))
            height = dxy.mean() + dxy.std() * peak_detection_params['standard_deviation_mulitple']
            peaks, props = sig.find_peaks(dxy, height, None, peak_detection_params['minimum_interpeak_distance'])
            perisaccadic_window_length = np.diff(time_window_indices['perisaccadic'])

            #
            for ipeak in peaks:

                # determine the onset of the putative saccade
                onset = None
                baseline = dxy[ipeak + time_window_indices['presaccadic'][0]: ipeak + time_window_indices['presaccadic'][-1]]
                for iback in np.arange(ipeak + time_window_indices['perisaccadic'][0], ipeak, 1)[::-1]:
                    nsamples = ipeak - iback
                    if nsamples < peak_detection_params['minimum_halfpeak_distance']:
                        continue
                    elif nsamples > peak_detection_params['maximum_halfpeak_distance']:
                        break
                    elif dxy[iback] <= baseline.mean() + baseline.std() * peak_detection_params['standard_deviation_mulitple']:
                        onset = iback
                        onset += 1 # plus 1 to account for n - 1 of the derivative
                        break

                # determine the offset of the putative saccade (this is optional)
                offset = np.nan
                baseline = dxy[ipeak + time_window_indices['postsaccadic'][0]: ipeak + time_window_indices['postsaccadic'][-1]]
                threshold = baseline.mean() + baseline.std() * peak_detection_params['standard_deviation_mulitple']
                for iforward in np.arange(ipeak, ipeak + peak_detection_params['maximum_halfpeak_distance'], 1):
                    if dxy[iforward] <= threshold:
                        offset = iforward
                        offset += 1

                # discard saccades with an undetermined onset index
                if onset is None:
                    continue

                # slice out the x and y position of the saccade
                start, stop = time_window_indices['perisaccadic']
                xsac = np.array(input[:, 0 if side == 'left' else 2])[onset + start: onset + stop]
                ysac = np.array(input[:, 1 if side == 'left' else 3])[onset + start: onset + stop]

                # discard incomplete putative saccades
                if len(xsac) < perisaccadic_window_length or len(ysac) < perisaccadic_window_length:
                    continue

                #
                saccades.append(np.hstack([xsac.reshape(-1, 1), ysac.reshape(-1, 1)]))
                indices.append(np.array([onset, offset]))
                eyes.append(np.array([0 if side == 'right' else 1, 0 if side == 'left' else 1]))

        #
        saccades = np.array(saccades)
        indices = np.array(indices)
        eyes = np.array(eyes)

        self.output = np.zeros((saccades.shape[0], saccades.shape[1] + 2, 2))
        self.output[:, 2:, :] = saccades
        self.output[:, 1 , :] = indices
        self.output[:, 0 , :] = eyes

        return

class IdentifyTrueSaccadesModule(PipelineModule):
    """
    """

    def __init__(self, session, filename='true-saccade-detection-results'):
        super().__init__(session, filename)

    def execute(self, input, **kwargs):
        """
        """

        #
        global SACCADE_TRAINING_DATASET_LOADED
        if SACCADE_TRAINING_DATASET_LOADED is False:
            raise PipelineError('Failed to load saccade training dataset')

        global SACCADE_TRAINING_DATASET_LABELS
        global SACCADE_TRAINING_DATASET_SAMPLES

        # train the classifier
        classifier_parameters = {
            'n_neighbors': np.arange(2, 31, 1),
            'weights'    : ['uniform', 'distance'],
            'metric'     : ['euclidean', 'manhattan']
        }

        #
        for k, v in kwargs.items():
            if k in classifier_parameters.keys():
                classifier_parameters[k] = v

        #
        gs = ms.GridSearchCV(
            ne.KNeighborsClassifier(),
            classifier_parameters
        )

        results = gs.fit(
            SACCADE_TRAINING_DATASET_SAMPLES,
            SACCADE_TRAINING_DATASET_LABELS.flatten()
        )
        clf = results.best_estimator_

        #
        nsamples, nfeatures = SACCADE_TRAINING_DATASET_SAMPLES.shape

        #
        output = list()

        # main loop
        for side in ['left', 'right']:

            # Get only the saccades for the target eye
            position = 0 if side == 'left' else 1
            mask = np.array([
                True if row[position] == 1 else False for row in input[:, 0, :]
            ])
            target_eye_input = input[mask, :, :]

            # Combine and randomly order the coordinates across saccades (maybe not necessary)

            # Flip the sign of PC1 to match the direction of eye movement
            pca = dec.PCA(n_components=1).fit(np.array(self.session.pupil_center_coords[side].iloc[:, :2]))
            pc1 = pca.transform(np.array(self.session.pupil_center_coords[side, 'y']).reshape(-1, 1)).flatten()
            r, p = stats.pearsonr(pc1, np.array(self.session.pupil_center_coords[side, 'y']))
            if r < 0:
                coef = -1
            else:
                coef = +1

            # Decompose the saccade waveforms
            target = np.array([
                pca.transform(target_eye_input[i, 2:, :]).flatten()
                    for i in np.arange(mask.sum())
            ])

            # slice out the peri-saccadic epoch
            middle = (target.shape[1] - 1) / 2
            nfeatures = SACCADE_TRAINING_DATASET_SAMPLES.shape[1]
            start = int(middle - ((nfeatures - 1) / 2))
            stop = int(middle + ((nfeatures - 1) / 2) + 1)

            # zero and multiple by coefficient
            target = np.array([
                (sample[start: stop] - sample[start: stop][0]) * coef
                    for sample in target
            ])

            # Predict true and false saccades
            predictions = clf.predict(target)

            # Save the true saccades
            for i, prediction in enumerate(predictions):
                if prediction in [const.MONOCULAR_SACCADE_DIRECTION_NASAL, const.MONOCULAR_SACCADE_DIRECTION_TEMPORAL]:

                    # Empty container
                    data = np.zeros((nfeatures + 3, 2))

                    # Direction of the saccade (1 = nasal, 2 = temporal)
                    data[0, :] = np.array([prediction, np.nan])

                    # Eye
                    data[1, :] = np.array([
                        1 if side == 'left'  else 0,
                        1 if side == 'right' else 0
                    ])

                    # Saccade onset and offset
                    data[2, :] = target_eye_input[i, 1, :]

                    # PC1
                    data[3:, :] = np.hstack([target[i, :].reshape(-1, 1), np.full((nfeatures, 1), np.nan)]) # Store PC1 for the saccade

                    output.append(data)

        # sort by saccade onset timestamp
        output = np.array(output)
        order = output[:, 2, 0].argsort()
        self.output = output[order, :, :]

        return

class ClassifyConjugateSaccadesModule(PipelineModule):
    """
    Classify saccades into contralateral, ipsilateral, divergent, or convergent
    """

    def __init__(self, session, filename='true-saccades-classification-results'):
        super().__init__(session, filename)

    def execute(self, input, conjugate_saccade_window=(-0.015, 0.015)):
        """
        """

        output = list()

        # Masks for the left and right eye
        left_eye_mask = np.array([
            np.array_equal([1, 0], saccade[1, :])
                for saccade in input
        ])
        right_eye_mask = ~left_eye_mask

        #
        for side in ['left', 'right']:

            # Target eye data
            target_eye = side
            target_eye_mask = left_eye_mask if side == 'left' else right_eye_mask
            target_eye_input = input[target_eye_mask, :, :]

            # Opposite eye data
            opposite_eye = 'left' if target_eye == 'right' else 'right'
            opposite_eye_mask = left_eye_mask if side == 'right' else right_eye_mask
            opposite_eye_input = input[opposite_eye_mask, :, :]

            #
            for saccade in target_eye_input:

                #
                target_eye_direction = 'nasal' if saccade[0, 0] == const.MONOCULAR_SACCADE_DIRECTION_NASAL else 'temporal'
                target_eye_onset, target_eye_offset = saccade[2, :]
                target_eye_pc1 = saccade[3:, :]

                #
                candidates = list()
                for opposite_eye_index, (opposite_eye_onset, opposite_eye_offset) in enumerate(opposite_eye_input[:, 2, :]):
                    distance = opposite_eye_onset - target_eye_onset
                    if conjugate_saccade_window[0] <= distance <= conjugate_saccade_window[1]:
                        candidates.append(opposite_eye_index)

                # No candidate conjugate saccade detected
                if len(candidates) == 0:
                    continue

                # Exactly one candidate saccade found
                if len(candidates) == 1:
                    opposite_eye_index = candidates.pop()
                    opposite_eye_onset, opposite_eye_offset = opposite_eye_input[opposite_eye_index, 2, :]
                    opposite_eye_direction = 'nasal' \
                        if opposite_eye_input[opposite_eye_index, 0, 0] == const.MONOCULAR_SACCADE_DIRECTION_NASAL \
                            else 'temporal'

                # More than one candidate saccade detected (shouldn't happen often)
                else:
                    continue

                # Vergent eye movements (agnostic of target eye)
                if target_eye_direction == opposite_eye_direction:
                    if target_eye_direction == 'nasal':
                        relative_saccade_direction = const.CONJUGATE_SACCADE_DIRECTION_DIVERGENT
                    elif target_eye_direction == 'temporal':
                        relative_saccade_direction = const.CONJUGATE_SACCADE_DIRECTION_CONVERGENT

                # Relative to left eye
                elif target_eye == 'left':
                    if target_eye_direction == 'nasal' and opposite_eye_direction == 'temporal':
                        relative_saccade_direction = const.CONJUGATE_SACCADE_DIRECTION_CONTALATERAL
                    elif target_eye_direction == 'temporal' and opposite_eye_direction == 'nasal':
                        relative_saccade_direction = const.CONJUGATE_SACCADE_DIRECTION_IPSILATERAL

                # Relative to right eye
                elif target_eye == 'right':
                    if target_eye_direction == 'nasal' and opposite_eye_direction == 'temporal':
                        relative_saccade_direction = const.CONJUGATE_SACCADE_DIRECTION_IPSILATERAL
                    elif target_eye_direction == 'temporal' and opposite_eye_direction == 'nasal':
                        relative_saccade_direction = const.CONJUGATE_SACCADE_DIRECTION_CONTALATERAL

                # Combine target data
                earliest_onset_index = np.min([target_eye_onset, opposite_eye_onset])
                earliest_offset_index = np.min([target_eye_offset, opposite_eye_offset])
                data = (earliest_onset_index, earliest_offset_index, relative_saccade_direction)

                #
                if data not in output:
                    output.append(data)

            self.output = np.array(output, dtype=np.int)

        return

class ComputeEventTimestampsModule(PipelineModule):
    """
    """

    def __init__(self, session, filename='event-metadata-and-timestamps'):
        super().__init__(session, filename)

    def execute(self, input):
        """
        """

        #
        output = list()

        #
        yml = str(self.session.folders['analysis'].joinpath('manually-collected-metadata.yml'))
        with open(yml, 'r') as stream:
            metadata = yaml.full_load(stream)

        # Get the sample indices for the sync signal on LJ's side
        data = lj.load_labjack_data(self.session.folders['labjack'])
        lj_sync_sig, lj_sync_idxs = lj.extract_labjack_event(
            data,
            iev=metadata['signaling']['labjack']['sync']['index'],
            analog=metadata['signaling']['labjack']['sync']['analog'],
        )

        # Get the sample indices for the sync signal on NP's side
        np_sync_idxs = np.load(list(self.session.folders['neuropixels'].rglob('*TTL_1*')).pop().joinpath('timestamps.npy'))
        start = metadata['signaling']['neuropixels']['sync']['start']
        stop  = metadata['signaling']['neuropixels']['sync']['stop']
        if start is None:
            start = 0
        if stop is None:
            stop = np_sync_idxs.size
        np_sync_idxs = np_sync_idxs[start: stop]

        #
        if lj_sync_idxs.size != np_sync_idxs.size:
            raise PipelineError(f'Different number of sync pulses recorded (NP={np_sync_idxs.size}, LJ={lj_sync_idxs.size})')

        # Compute saccade onset and offset timestamps
        lj_exposure_sig, lj_exposure_idxs = lj.extract_labjack_event(
            data,
            iev=metadata['signaling']['labjack']['exposure']['index'],
            analog=metadata['signaling']['labjack']['exposure']['analog']
        )
        np_exposure_idxs = lj.convert_labjack_indices(lj_exposure_idxs, lj_sync_idxs, np_sync_idxs, version=2) - self.session.first_ephys_sample

        # Compute the timestamp for each saccade (onset and offset)
        for frame_index_onset, frame_index_offset, direction in input:
            saccade_onset_timestamp = np.around(np_exposure_idxs[frame_index_onset] / const.SAMPLING_RATE_NEUROPIXELS, 3)
            saccade_offset_timestamp = np.around(np_exposure_idxs[frame_index_offset] / const.SAMPLING_RATE_NEUROPIXELS, 3)
            row = [const.TARGET_EVENT_SACCADE, direction, saccade_onset_timestamp, saccade_offset_timestamp]
            output.append(row)

        # Compute probe onset and offset timestamps
        lj_stim_sig, lj_stim_idxs = lj.extract_labjack_event(
            data,
            edge='both',
            iev=metadata['signaling']['labjack']['stimulus']['index'],
            analog=metadata['signaling']['labjack']['stimulus']['analog']
        )
        start = metadata['signaling']['labjack']['stimulus']['start']
        stop  = metadata['signaling']['labjack']['stimulus']['stop']
        if start is None:
            start = 0
        if stop is None:
            stop = lj_stim_idxs.size
        lj_stim_idxs = lj_stim_idxs[start: stop]


        # Parse the probe pulses if more than the target number are detected
        if lj_stim_idxs.size != len(VISUAL_STIMULUS_METADATA):
            ikeep = np.where(np.diff(lj_stim_idxs) > 0.9 * const.SAMPLING_RATE_LABJACK)[0] + 1
            if ikeep.size != len(VISUAL_STIMULUS_METADATA):
                raise PipelineError(f'Too many stimulus TTL pulses detected (actual={ikeep.size}, target=1120)')
            else:
                lj_stim_idxs = lj_stim_idxs[ikeep]

        #
        np_stim_idxs = lj.convert_labjack_indices(lj_stim_idxs, lj_sync_idxs, np_sync_idxs, version=2) - self.session.first_ephys_sample

        #
        for iev, (event, level, phase) in enumerate(VISUAL_STIMULUS_METADATA[::2]):
            ts1 = np.around(np_stim_idxs[int(iev * 2    )] / const.SAMPLING_RATE_NEUROPIXELS, 3)
            ts2 = np.around(np_stim_idxs[int(iev * 2 + 1)] / const.SAMPLING_RATE_NEUROPIXELS, 3)
            if event == 'probe':
                if level == 'low':
                    row = [const.TARGET_EVENT_PROBE, const.PROBE_CONTRAST_LOW, ts1, ts2]
                elif level == 'medium':
                    row = [const.TARGET_EVENT_PROBE, const.PROBE_CONTRAST_MEDIUM, ts1, ts2]
                else:
                    row = [const.TARGET_EVENT_PROBE, const.PROBE_CONTRAST_HIGH, ts1, ts2]
            elif event == 'start':
                row = [const.TARGET_EVENT_GRATING, np.nan, ts1, ts2]
            elif event == 'motion':
                row = [const.TARGET_EVENT_MOTION, np.nan, ts1, ts2]
            elif event == 'stop':
                row = [const.TARGET_EVENT_STOP, np.nan, ts1, ts2]
            output.append(row)

        # Sort the output by the time of event onset
        output = np.array(output)
        onset_sorted_index = output[:, 2].argsort()
        self.output = output[onset_sorted_index, :]

        return

class PreprocessingPipeline():
    """
    """

    def __init__(self):
        """
        """

        self.modules = (
            ExtractPupilPositionModule,
            DetectPutativeSaccadesModule,
            IdentifyTrueSaccadesModule,
            ClassifyConjugateSaccadesModule,
            ComputeEventTimestampsModule,
        )

        return

    def run(self, session):
        """
        """

        print(f'processing session for {session.animal} on {session.date}', end=' ')

        self.outputs = list()

        for i, M in enumerate(self.modules):
            m = M(session)
            if i is 0:
                m.execute()
                m.save()
                data = m.output
                self.outputs.append(data)
            else:
                m.execute(data)
                m.save()
                data = m.output
                self.outputs.append(data)

        print('Done!', end=' ')

        return
