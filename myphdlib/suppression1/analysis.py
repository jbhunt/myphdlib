import numpy as np
import statsmodels.api as sm
from .. import toolkit
from .  import helpers

class PerisaccadicModulationVersion1():
    """
    """

    def __init__(self, session, uids):
        """
        """

        self.session = session
        self.uids = uids

        return

    def run(self, **params_):
        """
        """

        params = {
            'binsize': 0.01,
            'coincidence': (-1, 1),
            'normalize': True,
            'around_saccades_window': (-0.06, 0.12),
            'visual_response_window': (0.05 , 0.15)
        }
        params.update(params_)

        #
        data = {level: list() for level in ['low', 'medium', 'high']}

        #
        mapping = {
            u.uid: i
                for i, u in enumerate(self.session.population)
        }

        #
        for uid in self.uids:
            iunit = mapping[uid]
            unit = self.session.population[iunit]

            # Baseline FR in spikes / second
            try:
                baseline, variance = helpers.estimate_baseline_activity(
                    self.session,
                    unit,
                    binsize=params['binsize']
                )
                epochs, stable = unit.load_stable_spiking_epochs()
            except Exception:
                continue

            # Timestamps for all probes (that occur within a stable spiking epoch)
            all_probe_onset_timestamps = np.concatenate([
                stable['probes']['low'],
                stable['probes']['medium'],
                stable['probes']['high']
            ])

            # Timestamps for all saccades
            all_saccade_onset_timestamps = np.concatenate([
                stable['saccades']['ipsi'],
                stable['saccades']['contra']
            ])

            #
            for level, probe_onset_timestamps in stable['probes'].items():

                # ==========
                # Average probe-only activity
                # ==========

                # Mask for probes that occur outside of the perisaccadic window
                extrasaccadic = np.invert(helpers.coincident2(
                    probe_onset_timestamps,
                    all_saccade_onset_timestamps,
                    coincidence=params['coincidence']
                ))

                # Compute the average probe-only response for the target contrast level (Rp)
                t, M = toolkit.psth(
                    probe_onset_timestamps[extrasaccadic],
                    unit.timestamps,
                    window=params['visual_response_window'],
                    binsize=params['binsize']
                )

                # Average and smooth
                Rp = np.mean(M.mean(0) / params['binsize'] - baseline)

                # ==========
                # Average integrated activity
                # ==========

                perisaccadic = helpers.coincident2(
                    probe_onset_timestamps,
                    all_saccade_onset_timestamps,
                    coincidence=params['around_saccades_window']
                )

                if perisaccadic.sum() == 0:
                    continue

                edges, M = toolkit.psth(
                    probe_onset_timestamps[perisaccadic],
                    unit.timestamps,
                    window=params['visual_response_window'],
                    binsize=params['binsize']
                )

                Rsp = np.mean(M.mean(0) / params['binsize'] - baseline)

                # ==========
                # Average, time-shifted saccade-related activity
                # ==========

                # Compute the latencies for all perisaccadic probes
                latencies = list()
                for probe_onset_timestamp in probe_onset_timestamps[perisaccadic]:
                    relative = probe_onset_timestamp - all_saccade_onset_timestamps
                    imin = np.abs(relative).argmin()
                    latency = relative[imin]
                    latencies.append(latency)

                #
                extrastimulus = np.invert(helpers.coincident2(
                    all_saccade_onset_timestamps,
                    all_probe_onset_timestamps,
                    coincidence = params['coincidence']
                ))

                #
                M = list()
                for latency in latencies:
                    window = np.around(latency + np.array(params['visual_response_window']), 2)
                    edges, m = toolkit.psth(
                        all_saccade_onset_timestamps[extrastimulus],
                        unit.timestamps,
                        window=window,
                        binsize=params['binsize']
                    )
                    trial = m.mean(0)
                    M.append(trial)

                M = np.array(M)
                Rs = np.mean(M.mean(0) / params['binsize'] - baseline)

                # ==========
                #
                # ==========

                error = Rsp - (Rs + Rp)
                if params['normalize']:
                    error /= baseline
                data[level].append(error)

        return data

class PerisaccadicModulationVersion2():
    """
    """

    def __init__(self, session, uids):
        """
        """

        self.session = session
        self.uids = uids

        return

    def run(self, **params_):
        """
        """

        params = {
            'binsize': 0.01,
            'normalize': True,
            'coincidence': (-1, 1),
            'around_saccades_window': (-0.06, 0.12),
            'visual_response_window': (0.05 , 0.15)
        }
        params.update(params_)

        #
        data = {level: list() for level in ['low', 'medium', 'high']}

        #
        N = {level: list() for level in ['low', 'medium', 'high']}

        #
        mapping = {
            u.uid: i
                for i, u in enumerate(self.session.population)
        }

        #
        for uid in self.uids:

            #
            iunit = mapping[uid]
            unit = self.session.population[iunit]

            # Baseline FR and stable spiking epochs
            try:
                baseline, variance = helpers.estimate_baseline_activity(
                    self.session,
                    unit,
                    binsize=params['binsize']
                )
                epochs, stable = unit.load_stable_spiking_epochs()
            except Exception:
                continue

            # Timestamps for all probes (that occur within a stable spiking epoch)
            all_probe_onset_timestamps = np.concatenate([
                stable['probes']['low'],
                stable['probes']['medium'],
                stable['probes']['high']
            ])

            # Timestamps for all saccades
            all_saccade_onset_timestamps = np.concatenate([
                stable['saccades']['ipsi'],
                stable['saccades']['contra']
            ])

            # Split the probes on the direction of the drifting grating
            probes = helpers.parse_visual_probes(self.session)

            #
            for level, probe_onset_timestamps in stable['probes'].items():

                for direction, saccade_onset_timestamps in stable['saccades'].items():

                    # ==========
                    # Average probe-only activity
                    # ==========

                    # Mask for probes that occur outside of the perisaccadic window
                    extrasaccadic = np.invert(helpers.coincident2(
                        probes[direction][level],
                        all_saccade_onset_timestamps,
                        coincidence=params['coincidence']
                    ))

                    # Compute the average probe-only response for the target contrast level (Rp)
                    t, M = toolkit.psth(
                        np.array(probes[direction][level])[extrasaccadic],
                        unit.timestamps,
                        window=params['visual_response_window'],
                        binsize=params['binsize']
                    )

                    # Average and smooth
                    Rp = np.mean(M.mean(0) / params['binsize'] - baseline)

                    perisaccadic = helpers.coincident2(
                        probe_onset_timestamps,
                        all_saccade_onset_timestamps,
                        coincidence=params['around_saccades_window']
                    )

                    if perisaccadic.sum() == 0:
                        continue

                    # Compute the latencies for all perisaccadic probes
                    latencies = list()
                    for probe_onset_timestamp in probe_onset_timestamps[perisaccadic]:
                        relative = probe_onset_timestamp - all_saccade_onset_timestamps
                        imin = np.abs(relative).argmin()
                        latency = relative[imin]
                        latencies.append(latency)

                    #
                    extrastimulus = np.invert(helpers.coincident2(
                        saccade_onset_timestamps,
                        all_probe_onset_timestamps,
                        coincidence = params['coincidence']
                    ))

                    #
                    sample = list()

                    #
                    for latency, probe_onset_timestamp in zip(latencies, probe_onset_timestamps[perisaccadic]):

                        # Compute the time-shifted, average saccade-related activity
                        edges, M = toolkit.psth(
                            saccade_onset_timestamps[extrastimulus],
                            unit.timestamps,
                            window=np.array(params['visual_response_window']) + latency,
                            binsize=params['binsize']
                        )

                        Rs = np.mean(M.mean(0) / params['binsize'] - baseline)

                        #
                        edges, M = toolkit.psth(
                            [probe_onset_timestamp],
                            unit.timestamps,
                            window=params['visual_response_window'],
                            binsize=params['binsize']
                        )

                        Rsp = np.mean(M.flatten() / 0.01 - baseline)

                        #
                        mi = Rsp - (Rs + Rp)
                        if params['normalize']:
                            mi /= baseline
                        sample.append(mi)

                    #
                    data[level].append(np.mean(sample))
                    N[level].append(len(sample))

        return data

class PerisaccadicModulationVersion3():
    """
    """

    def __init__(self, session, uids):
        """
        """

        self.session = session
        self.uids = uids

        return

    def run(self, **params_):
        """
        """

        #
        params = {
            'binsize': 0.01,
            'normalize': True,
            'coincidence': (-1, 1),
            'around_saccades_window': (-0.06, 0.12),
            'visual_response_window': (0    , 0.2 ),
            'independent_variables' : ['x_probe_level', 'x_around_saccade'],
        }
        params.update(params_)

        #
        data = {
            iv: list()
                for iv in params['independent_variables']
        }

        #
        mapping = {
            u.uid: i
                for i, u in enumerate(self.session.population)
        }

        for uid in self.uids:

            #
            iunit = mapping[uid]
            unit = self.session.population[iunit]

            try:
                baseline, deviation = helpers.estimate_baseline_activity(
                    self.session,
                    unit,
                    binsize=params['binsize']
                )
                epochs, stable = unit.load_stable_spiking_epochs()
            except:
                continue

            # Timestamps for all saccades
            all_saccade_onset_timestamps = np.concatenate([
                stable['saccades']['ipsi'],
                stable['saccades']['contra']
            ])

            #
            X, y = list(), list()

            #
            for i, (level, probe_onset_timestamps) in enumerate(stable['probes'].items()):
                for probe_onset_timestamp in probe_onset_timestamps:

                    # X
                    relative = probe_onset_timestamp - all_saccade_onset_timestamps
                    imin = np.abs(relative).argmin()
                    saccade_latency = relative[imin]
                    start, stop = params['around_saccades_window']
                    if start <= saccade_latency <= stop:
                        around_saccade = +1
                    else:
                        around_saccade = -1
                    probe_level = np.linspace(-1, 1, 3)[i]

                    lst = list()
                    for k in data.keys():
                        if k == 'x_around_saccade':
                            lst.append(around_saccade)
                        if k == 'x_saccade_latency':
                            lst.append(saccade_latency)
                        if k == 'x_probe_level':
                            lst.append(probe_level)
                    X.append(lst)

                    # y
                    edges, M = toolkit.psth(
                        [probe_onset_timestamp],
                        unit.timestamps,
                        window=params['visual_response_window'],
                        binsize=params['binsize']
                    )
                    fr = np.mean(M.flatten() / params['binsize'])
                    y.append(fr)

            # Model fitting
            X, y = np.array(X), np.array(y).reshape(-1, 1)
            X = sm.add_constant(X)
            model = sm.OLS(y, X)
            result = model.fit()
            for k, v in zip(data.keys(), result.params[1:]):
                data[k].append(v)

        return data
