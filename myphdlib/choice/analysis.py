import os
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.stats import sem, mannwhitneyu
from .models import Model1, Model2

class Session():
    """
    """

    def __init__(self, raw_input, iv='x_light', dv='x_choice', score=None, include_oe=True, ok2_extract_meta_data='josh'):
        """
        keywords
        --------

        examples
        --------
        >>> session_ids = [session1,session2,session3] # maybe this was pre-saline, muscimol, post-saline
        >>> iv = 'x_other' # the x_other column is used to code other experimental manipulations.
        >>> score = (0,1,0)
        >>> session = Session(session_ids, iv, score)
        """

        # attributes.
        self.iv = iv # target predictor variable
        self.dv = dv # target outcome variable
        self.include_oe = include_oe # include or exclude out-early trial types
        self.session_id = raw_input # filename(s)
        self.extract = ok2_extract_meta_data

        # accepts a string for a single file ...
        if type(raw_input) is str:
            self.data= self._mat_to_data(raw_input,include_oe=self.include_oe)

        # ... or a list or tuple of filenames.
        elif (type(raw_input) is list) | (type(raw_input) is tuple):
            self._combine_multiple_sessions(raw_input, score)

        else:
            msg = "The raw_input argument must be a filepath or a list of filepaths for multiple sessions"
            raise ValueError(msg)

        if iv not in ['x_light','x_other']:
            msg = "invalid input: iv must be one of ['x_light','x_other']"
            raise ValueError(msg)

        # do some preprocessing.
        self._preprocess_session()

    def _decode_odor_params(self, odor_params):
        """
        identifies the pattern used to encode the odor mixtures

        keywords
        --------
        odor_params : numpy array

        """

        frac_odor_right = []
        for odor in odor_params[0][0]['odors'][0][0][0]:
            odor = odor[0][0]
            if type(odor) is str: # sometimes the NaN values come out as empty strings.
                frac_odor_right.append(np.nan)
            elif odor > 0:
                frac_odor_right.append(odor)
            elif odor < 0:
                frac_odor_right.append(odor + 100)
            else:
                frac_odor_right.append(np.nan)

        frac_odor_right = np.array(frac_odor_right).reshape(-1,1)
        frac_odor_left = np.full_like(frac_odor_right,100) - frac_odor_right
        ids = np.arange(0,frac_odor_left.size).reshape(-1,1) + 1
        params = np.concatenate([frac_odor_left,ids],1)
        params_df = pd.DataFrame(params,columns=['frac_odor_left','id'])
        params_df.dropna(inplace=True) # drop rows with NaN values.
        odor_code_dict = dict(zip(params_df['frac_odor_left'] / 100,params_df['id']))

        return odor_code_dict

    def _mat_to_data(self, mat, key='taskbase', include_oe=True):
        """
        translates the raw experiment (taskbase) files into a pandas dataframe

        keywords
        --------
        mat : str
            full file path for the taskbase file (.mat)
        key : str
            specifies which key to use for accessing the taskbase
        include_oe : bool
            flag which determines whether out-early trial types are included in the analysis
        """

        mat_open = loadmat(mat)
        if key not in mat_open.keys():
            key = 'ans'
        taskbase = mat_open[key]

        # extract all behavioral events
        stim = taskbase['stimID'][0][0]
        choice = taskbase['choice'][0][0]
        choice_ee = taskbase['SideAFOE'][0][0]
        cue_on = taskbase['cue'][0][0]
        odor_on = taskbase['DIO'][0][0]
        odor_off = taskbase['OdorOff'][0][0]
        odor_poke_out = taskbase['OdorPokeOut'][0][0]
        water_poke_in = taskbase['WaterPokeIn'][0][0]
        light_on = taskbase['fib_lit'][0][0]
        req_odor_sample_time = taskbase['req_time'][0][0]

        # choice.

        # include trials in which the animal committed an out-early error, but still made a choice.
        if include_oe:
            choice_oe = taskbase['SideAFOE'][0][0] # choice immediately after an out-early trial
            mask_choice_oe = ~np.isnan(choice_oe)
            choice[mask_choice_oe] = choice_oe[mask_choice_oe]

        # exclude trials with an Nan value for cue_on, odor_on, or odor_poke_out
        mask_choice_made = (choice == 1) | (choice == 2)
        mask_complete_trial = (~np.isnan(cue_on)) & (~np.isnan(odor_on) & (~np.isnan(odor_poke_out)))
        idx_choice = np.where(np.logical_and(mask_choice_made,mask_complete_trial) == True)[0]
        x_choice = choice[idx_choice] - 1 # 0: left | 1: right
        x_choice = 1 - x_choice           # 1: left | 0: right
        x_choice = x_choice.reshape(-1,1) # reshape as a column

        # used for shaping empty fields.
        x_for_shape = np.full_like(x_choice, np.nan)

        # reaction time.

        # the mean time between odor cue and go tone
        mean_odor_sample_time = req_odor_sample_time.mean()

        # the minimum time between odor cue and go tone
        min_odor_sample_time = req_odor_sample_time.min()

        x_reaction_time = []
        for i in idx_choice:

            # trial-by-trial event timestamps
            i_cue_on = cue_on[i][0]
            i_odor_poke_out = odor_poke_out[i][0]
            i_odor_on = odor_on[i][0]

            # earliest possible time at which the go cue could be presented
            # i_min_odor_sample_time = i_odor_on + mean_odor_sample_time - 0.055
            i_min_odor_sample_time = i_odor_on + min_odor_sample_time

            # animal waited for the go cue to exit the odor port
            if i_odor_poke_out >= i_cue_on:
                rt = i_odor_poke_out - i_cue_on

            # animal exited odor port early during the period when it was possible to get the go cue
            elif i_odor_poke_out >= i_min_odor_sample_time:
                # i_cue_on = (i_odor_poke_out + i_max_odor_sample_time) / 2
                mask = req_odor_sample_time > (i_odor_poke_out - i_odor_on)
                i_cue_on = i_odor_on + np.mean(req_odor_sample_time[mask])
                rt = i_odor_poke_out - i_cue_on

            # animal exited odor port early outside of the period when it was possible to get the go cue
            elif i_odor_poke_out < i_min_odor_sample_time:
                i_cue_on = i_odor_on + mean_odor_sample_time
                rt =  i_odor_poke_out - i_cue_on

            # no valid RT for trial
            else:
                print('WARNING: No valid definition of reaction time.')
                rt = np.nan

            x_reaction_time.append(rt)

        x_reaction_time = np.array(x_reaction_time).reshape(-1,1)

        # movement time.
        x_movement_time = water_poke_in - odor_poke_out
        x_movement_time = x_movement_time[idx_choice]

        # stimulus identity

        odor_params = taskbase['odor_params']
        x_frac_odor_left = np.full(choice.size,0.)
        odor_code_dict = self._decode_odor_params(odor_params)

        # loop through each unique odor and fill in the x_frac_odor_left array
        for odor_frac, odor_id in odor_code_dict.items():
            odor_id_idx_i,odor_id_idx_j = np.where(stim == int(odor_id))
            x_frac_odor_left[odor_id_idx_i] = odor_frac

        # fraction of odors left and right.
        x_frac_odor_left = x_frac_odor_left[idx_choice].reshape(-1,1)
        x_frac_odor_right = 1 - x_frac_odor_left

        # X odors left and right
        x_odor_left = (x_frac_odor_left - 0.5) / 0.5
        x_odor_right = (x_frac_odor_right - 0.5) / 0.5
        x_odor_left[x_odor_left<0] = 0
        x_odor_right[x_odor_right<0] = 0

        # light.
        try:
            x_light = light_on[idx_choice].reshape(-1,1)
        except:
            x_light = np.zeros_like(x_for_shape)

        x_other = np.zeros_like(x_for_shape)
        x_trial = np.arange(x_other.size).reshape(-1,1).astype(int)

        # build the pandas dataframe
        all_trial_feats = [x_trial,
                           x_frac_odor_left,
                           x_frac_odor_right,
                           x_odor_left,
                           x_odor_right,
                           x_reaction_time,
                           x_movement_time,
                           x_choice,
                           x_light,
                           x_other
                           ]

        data_raw = np.concatenate(all_trial_feats,axis=1)

        labels = ('x_trial',
                  'x_frac_odor_left',
                  'x_frac_odor_right',
                  'x_odor_left',
                  'x_odor_right',
                  'x_reaction_time',
                  'x_movement_time',
                  'x_choice',
                  'x_light',
                  'x_other'
                  )

        data = pd.DataFrame(data_raw, columns=labels)

        # define the data type for each trial feature
        dtypes = ('int','float','float','float','float','float','float','int','int','int')
        for (i,col) in enumerate(data.columns):
            data[col] = data[col].astype(dtypes[i])
        data.set_index('x_trial', inplace=True)

        return data

    def _combine_multiple_sessions(self, session_ids, score=None):
        """
        """

        n_sessions = len(session_ids)
        if score is None:
            score = tuple(map(int,'0' * n_sessions))

        labels = ('x_frac_odor_left',
                  'x_frac_odor_right',
                  'x_odor_left',
                  'x_odor_right',
                  'x_reaction_time',
                  'x_movement_time',
                  'x_choice',
                  'x_light',
                  'x_other'
                  )

        data = pd.DataFrame(dict(),columns=labels)

        for (i,s) in enumerate(score):
            data_temp = self._mat_to_data(session_ids[i], include_oe=self.include_oe)
            n_trials = data_temp.shape[0]
            x_other = np.full((n_trials, 1), s)
            if score is None:
                data_temp['x_light'] = x_other
            else:
                data_temp['x_other'] = x_other
            data = data.append(data_temp, ignore_index=True)

        self.data = data

    def _preprocess_session(self):
        """
        """

        self.rxn_time_data = {'all_times_manip_off_left':self.data.loc[(self.data[self.iv]==0)&(self.data['x_choice']==1)]['x_reaction_time'],
                              'all_times_manip_on_left':self.data.loc[(self.data[self.iv]==1)&(self.data['x_choice']==1)]['x_reaction_time'],
                              'all_times_manip_off_right':self.data.loc[(self.data[self.iv]==0)&(self.data['x_choice']==0)]['x_reaction_time'],
                              'all_times_manip_on_right':self.data.loc[(self.data[self.iv]==1)&(self.data['x_choice']==0)]['x_reaction_time'],
                              }

        # take the median value of the reaction times for each condition
        self.rxn_time_data['med_time_manip_off_left'] = self.rxn_time_data['all_times_manip_off_left'].median()
        self.rxn_time_data['med_time_manip_on_left'] = self.rxn_time_data['all_times_manip_on_left'].median()
        self.rxn_time_data['med_time_manip_off_right'] = self.rxn_time_data['all_times_manip_off_right'].median()
        self.rxn_time_data['med_time_manip_on_right'] = self.rxn_time_data['all_times_manip_on_right'].median()

        # perform choice-specific Mann-Whitney U-test for reaction times

        # left choice
        samp1,samp2 = self.rxn_time_data['all_times_manip_off_left'],self.rxn_time_data['all_times_manip_on_left']
        res = mannwhitneyu(samp1,samp2,alternative='two-sided')
        self.rxn_time_data['mwu_result_left_u'] = res.statistic
        self.rxn_time_data['mwu_result_left_p'] = res.pvalue

        # right choice
        samp1,samp2 = self.rxn_time_data['all_times_manip_off_right'],self.rxn_time_data['all_times_manip_on_right']
        res = mannwhitneyu(samp1,samp2,alternative='two-sided')
        self.rxn_time_data['mwu_result_right_u'] = res.statistic
        self.rxn_time_data['mwu_result_right_p'] = res.pvalue

        # create a dictionary for the session meta data

        if self.extract in self.session_id:

            # dictionary keys
            labels = ('home_dir',
                      'ftype',
                      'protocol',
                      'experimenter',
                      'animal_id',
                      'side',
                      'stim_params',
                      'laser_intensity',
                      'laser_wavelength',
                      'date'
                      )

            homedir,filename = os.path.split(self.session_id)
            raw_session_info = filename.split('_')
            if len(raw_session_info) != 9:
                print("WARNING: A file that violates the standard naming convention was detected.")
                self.sesh_meta_data = None
                return

            # load values into the dictionary
            values = [homedir] + raw_session_info
            self.sesh_meta_data = dict(zip(labels,values))

            # define the animal class and number
            animal_id = self.sesh_meta_data['animal_id']
            pos = list(animal_id).index(next(filter(lambda s: s.isdigit(),list(animal_id))))
            animal_n = int(animal_id[pos:])
            animal_class = animal_id[:pos]
            self.sesh_meta_data['animal_n'] = animal_n
            self.sesh_meta_data['animal_class'] = animal_class

            # define the program and version
            program, version = self.sesh_meta_data['protocol'].split('-')
            self.sesh_meta_data['program'] = program
            self.sesh_meta_data['version'] = version

            # parse the photo-stimulation parameters.
            stim_on_time, pulse_on_width, pulse_off_width = self.sesh_meta_data['stim_params'].split('-')
            stim_on_time, pulse_on_width, pulse_off_width = list(map(int, [stim_on_time, pulse_on_width, pulse_off_width]))
            laser_frequency = (1000 / (1000 - stim_on_time)) * (stim_on_time / (pulse_on_width+pulse_off_width))
            laser_intensity = float(self.sesh_meta_data['laser_intensity'][:-2])
            laser_wavelength = int(self.sesh_meta_data['laser_wavelength'][:-2])

            # input the optogenetic stimulation parameters into the session meta data dict
            self.sesh_meta_data['stim_on_time'] = stim_on_time
            self.sesh_meta_data['pulse_on_width'] = pulse_on_width
            self.sesh_meta_data['laser_frequency'] = laser_frequency
            self.sesh_meta_data['laser_intensity'] = laser_intensity
            self.sesh_meta_data['laser_wavelength'] = laser_wavelength

    def fit_choice_data(self):
        """
        """

        # fit a sigmoid curve to the choice data
        self.model1 = Model1(self.data)
        try:
            self.b1 = self.model1.fit(self.iv)
        except:
            print("WARNING: Curve-fitting failed.")
            self.b1 = np.nan

        # use logistic regression to estimate the effect of the experimental manipulation
        self.model2 = Model2(self.data)
        try:
            self.b2 = self.model2.fit_with_sklearn(iv=self.iv)
        except:
            print("WARNING: Logistic regression failed.")
            self.b2 = np.nan

    def test_h0(self, metric='b2', n_iters=5000):
        """
        performs a bootstrap hypothesis test

        keywords
        --------
        metric : str (default is 'b2')
            target metric
        n_iters : int (default is 5000)
            size of the sample for comparison

        returns
        -------
        tv : float
            test value - true effect of the experimental manipulation
        sample: numpy.ndarray
            sample generated by shuffling and re-fitting
        p : float
            proportion of the sample more extreme than the test value
        """

        sample = []
        data = self.data.copy()

        if metric == 'b1': # (a/b+0.5)*100

            tv = self.model1.bias

            for i in range(n_iters):
                bias = self.model1.shuffle_and_refit()
                sample.append(bias)

        elif metric == 'b2': # beta light.

            tv = self.model2.bias

            for i in range(n_iters):
                bias = self.model2.shuffle_and_refit()
                sample.append(bias)

        else:
            msg = "metric must be one of ['b1','b2']"
            raise ValueError(msg)

        # compute p-value
        sample = np.array(sample)
        sample = sample[~np.isnan(sample)] # drop values where regression failed.
        p_right = np.array(sample)[sample>=tv].size / n_iters
        p_left = np.array(sample)[sample<=tv].size / n_iters

        if p_left < p_right:
            p = p_left
        elif p_right < p_left:
            p = p_right
        else:
            p = p_left

        # test results remain stored until method is called again.
        self.tv = tv
        self.sample = sample
        self.p = p

        return (tv, sample, p)

class AnimalError(Exception):
    """
    """

    def __init__(self, msg):
        """
        """

        super().__init__(msg)

        return

class Animal(Session):
    """
    """

    def __init__(self, animal, side='ipsi', checklist=[], screen=False, **kwargs):
        """
        """

        self.animal = animal
        self.side   = side

        # flags
        if 'mew' in animal:
            self.side     = None
            self.jacki    = True
            self.muscimol = True
        elif 'abw' in animal:
            self.jacki    = True
            self.muscimol = False
        else:
            self.jacki    = False
            self.muscimol = False

        # collect taskbase files
        self.mats = list()
        for root, folders, files in os.walk('/home/jbhunt/Dropbox/Datasets/M2L/'):
            for file in files:

                # filter on checklist
                if len(checklist) != 0:
                    if file not in checklist:
                        continue

                # session metadata in filename
                metadata = file.split('_')
                if animal not in metadata:
                    continue

                # filter by side
                if self.jacki and not self.muscimol:
                    if self.side == 'contra' and 'contra' not in metadata:
                        continue
                    elif self.side == 'ipsi' and 'contra' in metadata:
                        continue

                elif self.jacki and self.muscimol:
                    pass

                else:
                    if self.side not in metadata:
                        continue

                mat = os.path.join(root, file)
                self.mats.append(mat)

        # screen for quality
        if screen:
            mats, rmvd = self._screenSessions()
            self.mats = mats

        #
        if len(self.mats) == 0:
            raise AnimalError(f'no sessions found for {animal} with stim on the {side} side')
            return

        #
        if self.muscimol:

            # create a score for the saline and muscimol sessions
            score = list()
            for mat in self.mats:
                if 'saline' in mat:
                    score.append(0)
                elif 'mu' in mat:
                    score.append(1)

            super().__init__(self.mats, iv='x_other', score=score, **kwargs)

        else:
            super().__init__(self.mats, iv='x_light', score=None, **kwargs)

        return

    def _screenSessions(self, min_trial_count=150, min_trial_count_with_stim=15):
        """
        """

        if self.muscimol:
            iv = 'x_other'
        else:
            iv = 'x_light'

        idx = list()
        for imat, mat in enumerate(self.mats):
            s = Session(mat)
            if s.data.shape[0] < min_trial_count:
                idx.append(imat)
            elif s.data[s.data[iv] == 1].shape[0] < min_trial_count_with_stim:
                idx.append(imat)
            else:
                continue

        #
        mask = np.zeros(len(self.mats), dtype='bool')
        mask[idx] = True
        rmvd = list(np.array(self.mats)[ mask])
        mats = list(np.array(self.mats)[~mask])

        return mats, rmvd

class Dataset():
    """
    """

    def __init__(self):
        """
        """

        self.fileset = list()
        self.sessions = list()
        self.excluded_files = list()
        self.n_sessions_excluded = 0
        self.b1 = list()
        self.b2 = list()
        self.loaded = False

    def _collect_gad2_cre_with_chr2(self, base_dir, _params):
        """
        """

        iv = 'x_light'
        for root, folders, files in os.walk(base_dir):
            for file in files:
                if _params['gad2_tag'] in file:

                    # init Session object
                    tb = os.path.join(root, file)
                    s = Session(tb,iv,include_oe=self.include_oe)

                    # check total trial count
                    # n_trials_total = s.data.index.size
                    # if n_trials_total < _params['n_trials_total_min']:
                    #     self.excluded_files.append(file)
                    #     print('WARNING: A file was excluded b/c the animal only performed {} trials.'.format(data.index.size))
                    #     continue
                    #
                    # check number of trials with the experimental manipulation
                    # n_trials_manip = s.data[s.data[iv]==1].index.size
                    # if n_trials_manip < _params['n_trials_manip_min']:
                    #     self.excluded_files.append(file)
                    #     print('WARNING: A file was excluded b/c the animal only performed {} trials with the experimental manipulation'.format(n_trials_manip))
                    #     continue

                    self.sessions.append(s)
                    self.fileset.append(tb)

        return

    def _collect_muscimol_experiment(self, base_dir, _params):
        """
        """

        iv = 'x_other'
        try:
            idx = pd.read_excel(_params['mu_log'])
        except:
            msg = "Please provide the muscimol experiment log as the 'mu_log' keyword argument."
            raise TypeError(msg)

        triplets_raw = [list(idx.iloc[i:i+3,1]) for i in range(0,idx.shape[0],3)]
        triplets = []

        for trip_raw in triplets_raw:
            trip = []
            for file in trip_raw:
                trip.append(os.path.join(base_dir,file+'.mat'))
            triplets.append(trip)
            s = Session(trip,iv,score=(0,1,0),include_oe=self.include_oe)
            self.sessions.append(s)

        self.fileset = triplets

        return

    def _collect_all_other_experiments(self, base_dir, _params):
        """
        """

        iv = 'x_light'
        fileset_raw = []

        for root, folders, files in os.walk(base_dir):
            for file in files:
                if 'tb' in file:
                    tb = os.path.join(root, file)
                    fileset_raw.append(tb)

        for file in fileset_raw:

            # init a Session object
            s = Session(file,iv,include_oe=self.include_oe)

            # continue to next file if the current file isn't a member session of the target experiment
            if s.sesh_meta_data['animal_class'] != _params['label']:
                continue # don't record these as excluded files

            # continue to next file if the current file isn't a member of a particular set of animals
            if _params['animal_ids'] is not None:
                if s.sesh_meta_data['animal_id'] not in _params['animal_ids']:
                    continue # don't record these as excluded files

            # quick-fix: overwrite params extracted from filename when both are requested
            if _params['side'] == None:
                s.sesh_meta_data['side'] = None

            if _params['version'] == None:
                s.sesh_meta_data['version'] = None

            # drop sessions which don't meet the specified session parameter criteria.
            if ((s.sesh_meta_data['side']              !=  _params['side'])           |
                (s.sesh_meta_data['version']           !=  _params['version'])        |
                (s.sesh_meta_data['laser_intensity']   <=  _params['laser_int'][0])   |
                (s.sesh_meta_data['laser_intensity']   >=  _params['laser_int'][1])   |
                (s.sesh_meta_data['laser_frequency']   <=  _params['laser_freq'][0])  |
                (s.sesh_meta_data['laser_frequency']   >=  _params['laser_freq'][1])  |
                (s.sesh_meta_data['pulse_on_width']    <=  _params['laser_pow'][0])   |
                (s.sesh_meta_data['pulse_on_width']    >=  _params['laser_pow'][1])   |
                (s.sesh_meta_data['laser_wavelength']  !=  _params['laser_wav'])
                ):
                continue

            # the code below filters sessions based on task performance

            # check total trial count
            n_trials_total = s.data.index.size
            if n_trials_total < _params['n_trials_total_min']:
                self.excluded_files.append(file)
                self.n_sessions_excluded += 1
                continue

            # check number of trials with the experimental manipulation
            n_trials_manip = s.data[s.data[iv]==1].index.size
            if n_trials_manip < _params['n_trials_manip_min']:
                self.excluded_files.append(file)
                self.n_sessions_excluded += 1
                continue

            self.fileset.append(file)
            self.sessions.append(s)

    def _collect(self, base_dir, **params):
        """
        collects and filters sessions

        keywords
        --------
        base_dir : str
            where the program begins its search for taskbase files
        """

        # labels for the session meta data dict

        # default parameters.
        _params = {'label':'gluCTL',               # experiment label
                   'side':None,                    # side of brain in which photo-stimulation was delivered
                   'version':None,                 # protocol version (can be 'old', 'new', or 'both')
                   'animal_ids':None,

                   # laser parameters
                   'laser_int':(0,15),             # range of laser intensities (mW) to include
                   'laser_freq':(0,100),           # range of laser frequencies (Hz) to include
                   'laser_pow':(0,501),            # range of laser pulse-on widths (ms) to include
                   'laser_wav':473,                # laser wavelength (nm)

                   # for identifying files with not standard file naming
                   'mu':False,                     # muscimol experiment flag
                   'mu_log':None,                  # muscimol experiment log (used to identify which sessions were pre-, treatment, or post-)
                   'gad2':False,                   # Gad2-cre + ChR2 experiment flag
                   'gad2_tag':'clean',             # a string used to identify the files for this experiment
                   'josh_tag':'josh',              # a string that identifies files from experiments other than the mu and gad2 experiments

                   # performance filters
                   'n_trials_total_min':100,       # minimum total number of trials
                   'n_trials_manip_min':15,        # minimum number of trials with the experimental manipulation
                   }

        # parse kwargs.
        for kwarg in params.keys():
            if kwarg in _params.keys():
                _params[kwarg] = params[kwarg]

        # stores the files that were filtered out
        self.excluded_files = []

        # muscimol experiments.
        if _params['mu'] is True:
            self._collect_muscimol_experiment(base_dir, _params)

        # Gad2-cre + ChR2 experiments.
        elif _params['gad2'] is True:
            self._collect_gad2_cre_with_chr2(base_dir, _params)

        # all other experiments
        else:
            self._collect_all_other_experiments(base_dir, _params)

        if self.n_sessions_excluded > 0:
            msg = "\nWARNING: {} sessions were excluded due to inadequate behavioral performance.".format(self.n_sessions_excluded)
            print(msg)

        # extract data from each session
        self.rt_iv_0_left =         [s.rxn_time_data['med_time_manip_off_left'] for s in self.sessions]
        self.rt_iv_1_left =         [s.rxn_time_data['med_time_manip_on_left'] for s in self.sessions]
        self.rt_iv_0_right =        [s.rxn_time_data['med_time_manip_off_right'] for s in self.sessions]
        self.rt_iv_1_right =        [s.rxn_time_data['med_time_manip_on_right'] for s in self.sessions]
        self.rt_effect_left_p =     [s.rxn_time_data['mwu_result_left_p'] for s in self.sessions]
        self.rt_effect_left_u =     [s.rxn_time_data['mwu_result_left_u'] for s in self.sessions]
        self.rt_effect_right_p =    [s.rxn_time_data['mwu_result_right_p'] for s in self.sessions]
        self.rt_effect_right_u =    [s.rxn_time_data['mwu_result_right_u'] for s in self.sessions]

    def test(self, metric='b2', n_iters=5000):
        """
        run the bootstrap hypothesis testing for each session in the dataset
        """

        self.p_values = list()
        for s in self.sessions:
            tv,sample,p = s.test_h0(metric,n_iters)
            self.p_values.append(p)

    def load(self, genotype='Vlgut2-cre', transgene='none', laser='blue', include_oe=True, **kwargs):
        """
        uses default parameters to load a pre-defined dataset

        keywords
        --------
        genotype : str (default is 'Vglut2-cre')
            animal genotype
        transgene : str (default is 'none')
            specifies which experiment
        laser : str (default is 'blue')
            specifies what laser was used for the Gad2-cre control experiments

        notes
        -----
        use genotype='wildtype', transgene='none', and laser='none' to load the muscimol dataset
        """

        if self.loaded:
            self.__init__()

        # flag for including or excluding out-early trial-types
        self.include_oe = include_oe

        valid_combos = (('VGlut2-cre','ChR2','blue'),
                        ('VGlut2-cre','none','blue'),
                        ('Gad2-cre','ChR2','blue'),
                        ('Gad2-cre','Arch','green'),
                        ('Gad2-cre','none','blue'),
                        ('Gad2-cre','none','green'),
                        ('wildtype','none','none')
                        )

        combo = (genotype,transgene,laser)

        if combo not in valid_combos:
            msg = '{}, {}, and {} is not a valid combination'.format(genotype, transgene, laser)
            raise ValueError(msg)

        # root_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
        pkg_path = os.path.abspath(os.path.dirname(__file__))
        root_dir = os.path.join(pkg_path,'data')
        print('\nlooking in {}'.format(root_dir))

        print('\nChecking for combo: {} + {} + {}'.format(combo[0], combo[1], combo[2]))

        #
        if combo == ('Gad2-cre','ChR2','blue'):
            iv = 'x_light'
            root_dir = os.path.join(root_dir,'jacki/gad2/')
            self._collect(root_dir, gad2=True, laser_wav=473)

        #
        elif combo == ('Gad2-cre','none','blue'):
            iv = 'x_light'
            root_dir = os.path.join(root_dir,'josh/')
            self._collect(root_dir, label='gadCTL', laser_wav=473,**kwargs)

        #
        elif combo == ('Gad2-cre','none','green'):
            iv = 'x_light'
            root_dir = os.path.join(root_dir,'josh/')
            self._collect(root_dir, label='gadCTL', laser_wav=532,**kwargs)

        #
        elif combo == ('VGlut2-cre','ChR2','blue'):
            iv = 'x_light'
            root_dir = os.path.join(root_dir,'josh/')
            self._collect(root_dir, label='glu',**kwargs)

        #
        elif combo == ('VGlut2-cre','none','blue'):
            iv = 'x_light' # default iv
            root_dir = os.path.join(root_dir,'josh/')
            self._collect(root_dir, label='gluCTL',**kwargs)

        #
        elif combo == ('Gad2-cre','Arch','green'):
            iv = 'x_light' # default iv
            root_dir = os.path.join(root_dir,'josh/')
            self._collect(root_dir,label='arch',laser_wav=532,**kwargs)

        #
        elif combo == ('Gad2-cre','none','green'):
            iv = 'x_light' # default iv
            root_dir = os.path.join(root_dir,'josh/')
            self._collect(root_dir, label='gadCTL', laser_wav=532, side='both',**kwargs)

        #
        elif combo == ('wildtype','none','none'):
            iv = 'x_other'
            root_dir = os.path.join(root_dir,'jacki/mu/')
            log = os.path.join(root_dir, 'log.xlsx')
            if not os.path.exists(log):
                msg = "No log file detected for the muscimol experiment."
                raise ValueError(msg)
            print('\nMuscimol experiment log file detected: {}'.format(log))
            self._collect(root_dir, mu=True, mu_log=log)

        else:
            msg = '{}, {}, and {} is not a valid combination'.format(genotype, transgene, laser)
            raise ValueError(msg)

        # fit the choice data for each session
        for s in self.sessions:
            s.fit_choice_data()

        self.b1 = np.array([s.b1 for s in self.sessions])
        self.b2 = np.array([s.b2 for s in self.sessions])

        self.loaded = True

        print('\nDone!')

    # def get(self, attribute):
    #     """
    #     for interacting with MATLAB
    #
    #     keywords
    #     --------
    #     attribute : str
    #         the desired Dataset class attribute
    #     """
    #
    #     attr_dict = {'b2': self.b2,
    #                  'rt_iv_0': self.rt_iv_0,
    #                  'rt_iv_1': self.rt_iv_1,
    #                  'rt_iv_0_left': self.rt_iv_0_left,
    #                  'rt_iv_0_right': self.rt_iv_0_right,
    #                  'rt_iv_1_left': self.rt_iv_1_left,
    #                  'rt_iv_1_right': self.rt_iv_1_right
    #                  }
    #
    #     if attribute in attr_dict.keys():
    #         return attr_dict[attribute]

class ModelSession():
    """
    """

    def __init__(self, xls, sheet_name='Sheet1', iv='x_light'):
        """
        """

        self.iv = iv
        self.sheet_name = sheet_name
        self.data = self._xls_to_data(xls, sheet_name)
        self.fit()

    def _xls_to_data(self, xls, sheet_name='Sheet1'):
        """
        extracts model data from an excel sheet

        keywords
        --------
        xls : pandas.ExcelFile
            opened excel file
        sheet_name : str (default is 'Sheet1')
            name of the sheet to be processed

        returns
        -------
        data : pandas.DataFrame
            processed data
        """

        df_in = xls.parse(sheet_name,skiprows=1)
        cols = [ctitle.replace("'",'') for ctitle in df_in.columns]
        df_in.columns = cols

         # choice
        choice = np.array(df_in['Choice'])
        idx_choice_left = np.where(choice==1)[0] # left.
        idx_choice_right = np.where(choice==2)[0] # right.
        idx_choice = sorted(np.concatenate([idx_choice_left,idx_choice_right]))
        x_choice = 1 - (choice[idx_choice].reshape(-1,1) - 1) # 1: left; 0: right.

        # odors
        x_frac_odor_left = np.array(df_in['Left Odor']/100).reshape(-1,1)
        x_frac_odor_right = np.array(df_in['Right Odor']/100).reshape(-1,1)
        x_odor_left = (x_frac_odor_left-0.5)/0.5
        x_odor_right = (x_frac_odor_right-0.5)/0.5
        x_odor_left[x_odor_left<0] = 0
        x_odor_right[x_odor_right<0] = 0

        # light and other fields
        x_light = np.array(df_in['fib_lit']).reshape(-1,1)
        x_trial = np.array(df_in.index).reshape(-1,1)
        x_temp = np.full_like(x_trial, np.nan).reshape(-1,1) # used for shaping empty fields.
        x_reaction_time = np.full_like(x_temp,0)
        x_movement_time = np.full_like(x_temp,0)
        x_other = np.full_like(x_temp,0)

        # build raw data structure
        fields = (x_trial,
                  x_frac_odor_left,
                  x_frac_odor_right,
                  x_odor_left,
                  x_odor_right,
                  x_reaction_time,
                  x_movement_time,
                  x_choice,
                  x_light,
                  x_other
                  )
        data_raw = np.concatenate(fields, axis=1)

        # convert to pandas dataframe
        labels = ('x_trial',
                  'x_frac_odor_left',
                  'x_frac_odor_right',
                  'x_odor_left',
                  'x_odor_right',
                  'x_reaction_time',
                  'x_movement_time',
                  'x_choice',
                  'x_light',
                  'x_other'
                  )
        data = pd.DataFrame(data_raw,columns=labels)

        dtypes = ('int','float','float','float','float','float','float','int','int','int')
        for (i,col) in enumerate(data.columns):
            data[col] = data[col].astype(dtypes[i])
        data.set_index('x_trial', inplace=True)

        return data

    def fit(self):
        """
        """

        # fit sigmoid curve
        self.model1 = Model1(self.data)
        try:
            self.model1.fit(self.iv)
            self.b1 = self.model1.bias
        except:
            self.b1 = np.nan

        # logistic regression
        self.model2 = Model2(self.data)
        try:
            self.model2.fit_with_sklearn(self.iv)
            self.b2 = self.model2.bias
        except:
            self.b2 = np.nan

    def test_h0(self, dv='b2', n_iters=5000):
        """
        """

        sample = []
        data = self.data.copy()

        if dv == 'b1': # (a/b+0.5)*100

            # test value
            tv = self.model11.bias

            for i in range(n_iters):
                bias = self.model1.shuffle_and_refit()
                sample.append(bias)

        elif dv == 'b2': # beta light.

            # test value
            tv = self.model2.bias

            for i in range(n_iters):
                bias = self.model2.shuffle_and_refit()
                sample.append(bias)

        else:
            msg = "invalid argument: dv must be one of ['b1','b2']"
            raise ValueError(msg)

        # build sample of bias estimation from shuffled data
        sample = np.array(sample)
        sample = sample[~np.isnan(sample)] # drop values where regression failed.
        p_right = np.array(sample)[sample>tv].size / n_iters
        p_left = np.array(sample)[sample<tv].size / n_iters

        if p_left < p_right:
            p = p_left
        elif p_right < p_left:
            p = p_right
        else:
            p = p_left

        # test results remain stored until method is called again.
        self.tv = tv
        self.sample = sample
        self.p = p

        return (tv, sample, p)

class ModelDataset():
    """
    """

    def __init__(self, xlsx):
        """
        """

        self.xlsx = xlsx
        self.xls = pd.ExcelFile(xlsx)
        self.sheet_names = self.xls.sheet_names
        if 'Sheet' in self.sheet_names:
            self.sheet_names.remove('Sheet')
        self.process()

    def process(self):
        """
        """

        self.sessions = [ModelSession(self.xls,sheet_name) for sheet_name in self.sheet_names]

        self.beta_light = [s.b2 for s in self.sessions]
        self.beta_odor_left = [s.model2.coefs['x_odor_left'] for s in self.sessions]
        self.beta_odor_right = [s.model2.coefs['x_odor_right'] for s in self.sessions]

        # run the boostrap hypothesis testing
        for s in self.sessions:
            s.test_h0()

        self.p_values = [s.p for s in self.sessions]

    def _update_excel_file(self):
        """
        """

        book = load_workbook(self.xlsx)
        writer = pd.ExcelWriter(self.xlsx, engine='openpyxl')
        writer.book = book

        if 'analysis' in book.sheetnames:
            print('WARNING: {} has already been analyzed.')
            return

            print('WARNING: {} has already been analyzed. Reanalyzing...'.format(self.xlsx))
            book.remove(book['analysis'])

        if 'Sheet' in book.sheetnames:
            print('WARNING: Empty first sheet detected. Deleting.')
            book.remove(book['Sheet'])

        # dct = {sheet_name:weight for (sheet_name,weight) in zip(self.sheet_names,self.b2)}
        dct = {'sheet':self.sheet_names,
               'weight_light':self.beta_light,
               'weight_odor_left':self.beta_odor_left,
               'weight_odor_right':self.beta_odor_right,
               'p':self.p_values}
        df = pd.DataFrame(dct)
        df.set_index('sheet', inplace=True)

        df.to_excel(writer,sheet_name='analysis')
        writer.save()
        writer.close()
