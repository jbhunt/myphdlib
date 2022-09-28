import pickle
import numpy as np
from .. import toolkit as tk
from . import constants as const

class EphysError(Exception):
    pass

class SingleUnit():
    """
    An object which represents a single unit
    """

    def __init__(self, uid, session):
        """
        """

        self._uid = uid
        self._session = session
        self._sorted = None

        #
        self._load_sorting_data()

        return

    def _load_sorting_data(self):
        """
        Load the spike sorting results
        """

        spike_times_list = list(self.session.folders['neuropixels'].rglob('*spike_times.npy'))
        spike_clusters_list = list(self.session.folders['neuropixels'].rglob('*spike_clusters.npy'))
        cluster_sorted_list = list(self.session.folders['neuropixels'].rglob('*cluster_sorted.tsv'))
        cluster_labels_list = list(self.session.folders['neuropixels'].rglob('*cluster_KSLabel.tsv'))
        cluster_info_list = list(self.session.folders['neuropixels'].rglob('*cluster_info.tsv'))

        lists = [
            spike_times_list,
            spike_clusters_list,
            cluster_sorted_list,
            cluster_labels_list,
            cluster_info_list,
        ]

        for lst, lid in zip(lists, ['spike times', 'spike clusters', 'sorted flags', 'kilosort labels', 'cluster info']):
            if len(lst) != 1:
                raise EphysError(f'Spike sorting results incomplete (missing {lid} file)')

        # Load the spike timestamps
        all_spike_times = np.load(spike_times_list.pop())
        all_spike_clusters = np.load(spike_clusters_list.pop())
        unit_mask = all_spike_clusters == self.uid
        self._timestamps = np.around(all_spike_times[unit_mask].flatten() / const.SAMPLING_RATE_NEUROPIXELS, 3)

        # Load the sorted flag
        with open(cluster_info_list[0], 'r') as stream:
            lines = stream.readlines()
        entries = [
            line.rstrip('\n').split('\t')
                for line in lines[1:]
        ]
        uids = np.array([int(entry[0]) for entry in entries])
        flags = [entry[-1] for entry in entries]

        flag = flags[np.where(uids == self.uid)[0].item()]

        if flag == 'yes':
            self._sorted = True
        else:
            self._sorted = False

        # Load the Kilosort label
        with open(cluster_info_list[0], 'r') as stream:
            lines = stream.readlines()
        entries = [
            line.rstrip('\n').split('\t')
                for line in lines
        ]
        try:
            labels = [entry[-1] for entry in entries]

            self._label = labels[np.where(uids == self.uid)[0].item()]
        except:
            import pdb; pdb.set_trace()

        #
        with open(cluster_info_list[0], 'r') as stream:
            lines = stream.readlines()

        entries = [
            line.rstrip('\n').split('\t')
                for line in lines[1:]
        ]
        positions = np.array([float(entry[6]) for entry in entries])
        self._position = positions[np.where(uids == self.uid)[0].item()]

        return

    def load_stable_spiking_epochs(self):
        """
        """

        lst = list(self.session.folders['analysis'].rglob('*stable-spiking-epochs.pkl'))
        if len(lst) == 0:
            raise Exception('Missing stable spiking epochs dictionary')
        else:
            with open(lst.pop(), 'rb') as stream:
                dct = pickle.load(stream)

        #
        if self.uid not in dct.keys():
            epochs = None,
            stable = {
                'probes': self.session.probe_onset_timestamps,
                'saccades': self.session.saccade_onset_timestamps
            }
            return epochs, stable
            raise Exception('Unit is unstable')
        else:
            epochs = np.array(dct[self.uid])

        #
        stable = {
            'probes': {
                level: list()
                    for level in ['low', 'medium', 'high']
            },
            'saccades': {
                direction: list()
                    for direction in ['ipsi', 'contra']
            }
        }

        for start, stop in epochs:
            for direction, timestamps in self.session.saccade_onset_timestamps.items():
                mask = np.logical_and(
                    timestamps >= start,
                    timestamps <= stop,
                )
                stable['saccades'][direction] += timestamps[mask].tolist()

        for start, stop in epochs:
            for level, timestamps in self.session.probe_onset_timestamps.items():
                mask = np.logical_and(
                    timestamps >= start,
                    timestamps <= stop,
                )
                stable['probes'][level] += timestamps[mask].tolist()

        for event, dct in stable.items():
            for k, v in dct.items():
                stable[event][k] = np.array(v)
                stable[event][k].sort()

        return epochs, stable

    def estimate_baseline_activity(self, binsize=0.01, window=(-1, -0.5), stable_spiking_epochs=True):
        """
        """

        #
        if stable_spiking_epochs:
            epochs, stable = self.load_stable_spiking_epochs()
            event_onset_timestamps = np.concatenate([
                stable['saccades']['ipsi'],
                stable['saccades']['contra'],
                stable['probes']['low'],
                stable['probes']['medium'],
                stable['probes']['high']
            ])

        #
        else:
            event_onset_timestamps = np.concatenate([
                self.session.saccade_onset_timestamps['ipsi'],
                self.session.saccade_onset_timestamps['contra'],
                self.session.probe_onset_timestamps['low'],
                self.session.probe_onset_timestamps['medium'],
                self.session.probe_onset_timestamps['high'],
            ])

        #
        event_onset_timestamps.sort()

        #
        edges, M = tk.psth(
            event_onset_timestamps,
            self.timestamps,
            window=window,
            binsize=binsize
        )

        mu, sigma = M.flatten().mean() / binsize, M.flatten().std() / binsize

        return mu, sigma

    @property
    def uid(self):
        return self._uid

    @property
    def session(self):
        return self._session

    @property
    def timestamps(self):
        return self._timestamps

    @property
    def label(self):
        return self._label

    @property
    def sorted(self):
        return self._sorted

    @property
    def position(self):
        return self._position

class Population():
    """
    """

    def __init__(self, session, labels=['mua', 'good'], exclude_unsorted_units=True):
        """
        """

        cluster_info_list = list(session.folders['neuropixels'].rglob('*cluster_info.tsv'))
        if len(cluster_info_list) != 1:
            raise EphysError('Could not locate cluster_info.tsv file')

        with open(cluster_info_list.pop(), 'r') as stream:
            lines = stream.readlines()

        self._units = list()
        for line in lines[1:]:

            #
            elements = line.rstrip('\n').split('\t')
            uid = int(elements[0])
            label = elements[3]
            sorted = True if elements[-1] == 'yes' else False

            #
            if label in labels and sorted is True:
                # print(f'Loading unit {uid} ...')
                unit = SingleUnit(uid, session)
                self._units.append(unit)

        return

    def __len__(self):
        return len(self._units)

    def __iter__(self):
        return (unit for unit in self._units)

    def __getitem__(self, index):
        return self._units[index]
