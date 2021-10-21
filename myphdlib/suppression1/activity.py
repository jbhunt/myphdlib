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
        self._timestamps = all_spike_times[unit_mask].flatten() / const.SAMPLING_RATE_NEUROPIXELS

        # Load the sorted flag
        with open(cluster_sorted_list.pop(), 'r') as stream:
            lines = stream.readlines()
        entries = [
            line.rstrip('\n').split('\t')
                for line in lines[1:]
        ]
        uids = np.array([int(entry[0]) for entry in entries])
        flags = [entry[1] for entry in entries]

        flag = flags[np.where(uids == self.uid)[0].item()]
        if flag == 'yes':
            self._sorted = True
        else:
            self._sorted = False

        # Load the Kilosort label
        with open(cluster_labels_list.pop(), 'r') as stream:
            lines = stream.readlines()
        entries = [
            line.rstrip('\n').split('\t')
                for line in lines
        ]
        labels = [entry[1] for entry in entries]

        self._label = labels[np.where(uids == self.uid)[0].item()]

        #
        with open(cluster_info_list.pop(), 'r') as stream:
            lines = stream.readlines()

        entries = [
            line.rstrip('\n').split('\t')
                for line in lines[1:]
        ]
        positions = np.array([float(entry[6]) for entry in entries])
        self._position = positions[np.where(uids == self.uid)[0].item()]

        return

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

        cluster_group_list = list(session.folders['neuropixels'].rglob('*cluster_group.tsv'))
        if len(cluster_group_list) != 1:
            raise EphysError('Could not locate cluster_group.tsv file')

        with open(cluster_group_list.pop(), 'r') as stream:
            lines = stream.readlines()

        uids, labels = list(), list()
        for line in lines[1:]:
            uid, label = line.rstrip('\n').split('\t')
            uids.append(int(uid))
            labels.append(label)

        #
        self._units = list()
        for uid in uids:
            unit = SingleUnit(uid, session)
            if unit.label not in labels:
                continue
            if exclude_unsorted_units and unit.sorted is False:
                continue
            self._units.append(unit)

        return

    def __len__(self):
        return len(self._units)

    def __iter__(self):
        return (unit for unit in self._units)

    def __getitem__(self, index):
        return self._units[index]
