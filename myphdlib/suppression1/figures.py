import numpy as np
import itertools as it
from matplotlib import pylab as plt
from matplotlib import gridspec as gs
from decimal import Decimal
from scipy.interpolate import CubicSpline
from scipy.stats import sem
from . import helpers
from .. import toolkit as tk

class PopulationHeatmap():
    """
    Standardized neural population activity relative to a target event
    """

    def __init__(self):
        return

    def plot(
        self,
        session,
        event='saccade',
        window=(-0.1, 0.3),
        binsize=0.01,
        directions=['ipsi', 'contra'],
        levels=['low', 'medium', 'high'],
        interpolate=True,
        evalualtion_sample_size=1000,
        smooth=True,
        hanning_window_size=7,
        figsize=(8, 4),
        target_unit_ids=[],
        target_unit_colors=[],
        target_unit_marker='>',
        **pcolormesh_kwargs
        ):
        """
        """

        fig = plt.figure()

        #
        pcolormesh_kwargs_ = {
            'vmin': -1,
            'vmax':  1,
            'cmap': 'binary_r'
        }
        pcolormesh_kwargs_.update(pcolormesh_kwargs)

        #
        if event == 'saccade':
            axs = {
                direction: fig.add_subplot(1, len(directions), i + 1)
                    for i, direction in enumerate(directions)
            }

        elif event == 'probe':
            axs = {
                level: fig.add_subplot(1, len(levels), i + 1)
                    for i, level in enumerate(levels)
            }

        heatmaps = {
            condition: list()
                for condition in axs.keys()
        }

        # Sort by unit depth
        depth_sorted_index = np.array([
            unit.position
                for unit in session.population
        ]).argsort()

        #
        target_unit_yvalues = list()
        if len(target_unit_ids) != 0:
            uids = np.array([unit.uid for unit in session.population])
            target_unit_yvalues = list()
            for uid in target_unit_ids:
                index = np.where(uids == uid)[0].item()
                yvalue = depth_sorted_index.argsort()[index]
                target_unit_yvalues.append(yvalue)

        #
        for i, (condition, ax) in enumerate(axs.items()):

            #
            heatmap = list()

            # For each unit
            for unit in session.population:

                # Identify the target event
                if condition == 'ipsi':
                    target_event = session.saccade_onset_timestamps['ipsi']
                elif condition == 'contra':
                    target_event = session.saccade_onset_timestamps['contra']
                elif condition == 'low':
                    target_event = session.probe_onset_timestamps['low']
                elif condition == 'medium':
                    target_event = session.probe_onset_timestamps['medium']
                elif condition == 'high':
                    target_event = session.probe_onset_timestamps['high']

                #
                edges, M1 = tk.psth(target_event, unit.timestamps, window=window, binsize=binsize)

                # Standardize
                mu, sigma = helpers.estimate_baseline_activity(session, unit, binsize)
                fr = M1.mean(0) / binsize
                row = (fr - mu) / sigma

                #
                if smooth:
                    row = tk.smooth(row, hanning_window_size)

                if interpolate is True:
                    x1 = np.arange(row.size)
                    spline = CubicSpline(x1, row)
                    x2 = np.linspace(0, x1.max(), evalualtion_sample_size)
                    row = spline(x2)

                heatmap.append(row)

            heatmap = np.array(heatmap)
            heatmaps[condition] = heatmap

            # Plot
            mappable = ax.pcolormesh(heatmap[depth_sorted_index, :], **pcolormesh_kwargs_)
            ax.set_aspect('auto')
            ax.set_xticks(np.linspace(0, heatmap.shape[1], 5))

            if i == 0:
                ax.set_xticklabels(np.around(np.linspace(window[0], window[1], 5) * 1000).astype(int), rotation=45)

            else:
                ax.set_xticklabels([])
                ax.set_yticklabels([])

        # Mark the target units
        left_most_ax = axs[list(axs.keys())[0]]
        xlim = left_most_ax.get_xlim()
        for uid, color, y in zip(target_unit_ids, target_unit_colors, target_unit_yvalues):
            left_most_ax.scatter(-100, y + 0.5, marker=target_unit_marker, color=color, clip_on=False)
        left_most_ax.set_xlim(xlim)

        # Add a colorbar
        # fig.subplots_adjust(right=0.8)
        # cax = fig.add_axes([0.85, 0.2, 0.05, 0.6])
        # fig.colorbar(mappable, cax=cax)

        fig.set_figheight(figsize[0])
        fig.set_figwidth(figsize[1])
        fig.show()

        return fig, axs, heatmaps

class PerisaccadicVisualResponses2():
    """
    """

    def __init__(self, width=4, height=3, ymin=-15, ymax=15):
        self._figsize = (width, height)
        self._ylim = (ymin, ymax)
        return

    @property
    def figsize(self):
        return self._figsize

    @figsize.setter
    def figsize(self, value):
        self._figsize = value

    @property
    def ylim(self):
        return self._ylim

    @ylim.setter
    def ylim(self, value):
        self._ylim = value

    def plot(
        self,
        unit,
        session,
        nbins=14,
        normalize=True,
        visual_response_window=( 0.05, 0.15),
        around_saccades_window=(-1.5,  1.5 ),
        modulation_index_version=3,
        **kwargs_
        ):
        """
        """

        # PSTH keyword args
        kwargs = {
            'binsize': 0.01,
            'window' : (-0.1, 0.3),
            'coincidence': (-1, 1)
        }
        kwargs.update(kwargs_)

        # Mask for identifying the visual response
        edges = np.arange(
            kwargs['window'][0],
            kwargs['window'][1] + kwargs['binsize'],
            kwargs['binsize']
        )
        visual_response_filter = np.logical_and(
            edges[:-1] + kwargs['binsize'] / 2 >= visual_response_window[0],
            edges[:-1] + kwargs['binsize'] / 2 <= visual_response_window[1]
        )

        # Baseline FR in spikes / second
        baseline, variance = helpers.estimate_baseline_activity(
            session, unit, binsize=kwargs['binsize']
        )

        # Empty data container
        data = {
            level: {'actual': list(), 'fictive': list()}
                for level in ['low', 'medium', 'high']
        }

        # Apply stable spiking filter to probes
        epochs, stable = unit.load_stable_spiking_epochs()

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

            # =============
            # Actual probes
            # =============

            # Mask for probes that occur outside of the perisaccadic window
            extrasaccadic = np.invert(helpers.coincident2(
                probe_onset_timestamps,
                all_saccade_onset_timestamps,
                coincidence=kwargs['coincidence']
            ))

            # Compute the average probe-only response for the target contrast level (Rp)
            t, M = tk.psth(
                probe_onset_timestamps[extrasaccadic],
                unit.timestamps,
                window=visual_response_window,
                binsize=kwargs['binsize']
            )

            # Average and smooth
            average_actual_visual_response = M.mean(0) / kwargs['binsize']
            Rp = average_actual_visual_response.mean() - baseline

            # Mask for saccades that occur around the time of probes
            peristimulus = helpers.coincident2(
                all_saccade_onset_timestamps,
                all_probe_onset_timestamps,
                coincidence=kwargs['coincidence']
            )

            # Mask for saccades that don't occur around the time of probes
            extrastimulus = np.invert(peristimulus)

            #
            for saccade_onset_timestamp in all_saccade_onset_timestamps:

                # Look for probes that appeared within the target window
                perisaccadic = np.logical_and(
                    around_saccades_window[0] <= probe_onset_timestamps - saccade_onset_timestamp,
                    around_saccades_window[1] >= probe_onset_timestamps - saccade_onset_timestamp
                )

                # Skip if no probes detected within target window
                if perisaccadic.sum() == 0:
                    continue

                # For each probe that appears in the target window
                for probe_onset_timestamp in probe_onset_timestamps[perisaccadic]:

                    # Time from saccade to probe onset
                    latency = np.around(probe_onset_timestamp - saccade_onset_timestamp, 3)

                    # Compute the single trial visuomotor response (Rsp)
                    t, M = tk.psth(
                        [probe_onset_timestamp],
                        unit.timestamps,
                        binsize=kwargs['binsize'],
                        window=visual_response_window
                    )
                    single_trial_actual_visual_response = M.flatten() / kwargs['binsize']
                    Rsp = single_trial_actual_visual_response.mean() - baseline

                    # Compute the response that is attributable to the saccade (Rs)
                    time_shifted_window = (
                        float(Decimal(str(visual_response_window[0])) + Decimal(str(latency))),
                        float(Decimal(str(visual_response_window[1])) + Decimal(str(latency)))
                    )
                    t, M = tk.psth(
                        all_saccade_onset_timestamps[extrastimulus],
                        unit.timestamps,
                        binsize=kwargs['binsize'],
                        window=time_shifted_window
                    )

                    # Average, subtract baseline, and smooth
                    average_time_shifted_saccade_only_response = M.mean(0) / kwargs['binsize']
                    Rs = average_time_shifted_saccade_only_response.mean() - baseline

                    #
                    if modulation_index_version == 1:
                        value = (Rsp - (Rp + Rs)) / abs(Rp + Rs)
                    elif modulation_index_version == 2:
                        value = (Rsp - (Rs + Rp)) / (Rsp + (Rs + Rp))
                    elif modulation_index_version == 3:
                        value = (Rsp - (Rs + Rp))

                    #
                    if normalize:
                        value /= baseline
                    #
                    data[level]['actual'].append((latency, value))

        return data, list()

        for i in list():
            # ==============
            # Fictive probes
            # ==============

            # Compute the average fictive visual response looking outside of the perisaccadic window
            stop = unit.timestamps.max() + float(Decimal(str(1.0)) - Decimal(str(unit.timestamps.max() % 1)))
            fictive_probe_onset_timestamps = np.arange(0, stop, 1)
            perisaccadic = helpers.coincident2(
                fictive_probe_onset_timestamps,
                all_saccade_onset_timestamps,
                coincidence=(-0.05, 0.11)
            )
            peristimulus = helpers.coincident2(
                fictive_probe_onset_timestamps,
                all_probe_onset_timestamps,
                coincidence=(-1, 1)
            )
            extraevent = np.invert(np.vstack([
                perisaccadic,
                peristimulus
            ]).any(axis=0))
            extraevent_fictive_probe_onset_timestamps = fictive_probe_onset_timestamps[extraevent]

            #
            t, M = tk.psth(
                extraevent_fictive_probe_onset_timestamps,
                unit.timestamps,
                window=kwargs['window'],
                binsize=kwargs['binsize']
            )

            # Average and smooth
            average_fictive_visual_response = M.mean(0) / kwargs['binsize'] - baseline

            # Compute the response magnitude given the target metric
            Ro = helpers.extract_response_amplitude(
                average_fictive_visual_response,
                visual_response_filter,
                metric='mean'
            )

            # Sample from across the perisaccadic window excluding actual events
            perisaccadic_fictive_probe_onset_timestamps = list()
            for saccade_onset_timestamp in all_saccade_onset_timestamps:
                start = saccade_onset_timestamp + around_saccades_window[0]
                stop  = saccade_onset_timestamp + around_saccades_window[1]
                timestamps = np.arange(start, stop, 0.1)
                extrastimulus = np.invert(helpers.coincident2(
                    timestamps,
                    all_probe_onset_timestamps,
                    coincidence=(-1, 1)
                ))
                perisaccadic_fictive_probe_onset_timestamps += timestamps[extrastimulus].tolist()
            perisaccadic_fictive_probe_onset_timestamps = np.array(perisaccadic_fictive_probe_onset_timestamps)

            for direction, saccade_onset_timestamps in session.saccade_onset_timestamps.items():

                # Extrastimulus saccades
                extrastimulus = np.invert(helpers.coincident2(
                    session.saccade_onset_timestamps[direction],
                    all_probe_onset_timestamps,
                    coincidence=(-1, 1)
                ))

                #
                for saccade_onset_timestamp in saccade_onset_timestamps:

                    # Look for probes that appeared within the target window
                    perisaccadic = np.logical_and(
                        around_saccades_window[0] <= perisaccadic_fictive_probe_onset_timestamps - saccade_onset_timestamp,
                        around_saccades_window[1] >= perisaccadic_fictive_probe_onset_timestamps - saccade_onset_timestamp
                    )

                    # Skip if no probes detected within target window
                    if perisaccadic.sum() == 0:
                        continue

                    # For each probe that occurs around the time of this saccade ...
                    for fictive_probe_onset_timestamp in perisaccadic_fictive_probe_onset_timestamps[perisaccadic]:

                        # Time from saccade to probe onset
                        latency = np.around(fictive_probe_onset_timestamp - saccade_onset_timestamp, 3)

                        # Compute the single trial visuomotor response (Rsp)
                        t, M = tk.psth(
                            np.array([fictive_probe_onset_timestamp]),
                            unit.timestamps,
                            binsize=kwargs['binsize'],
                            window=kwargs['window']
                        )
                        single_trial_fictive_visual_response = M.flatten() / kwargs['binsize'] - baseline

                        # Compute the response magnitude given the target metric
                        Rso = helpers.extract_response_amplitude(
                            single_trial_fictive_visual_response,
                            visual_response_filter,
                            metric='mean'
                        )

                        # Compute the response that is attributable to the saccade (Rs)
                        time_shifted_window = (
                            float(Decimal(str(kwargs['window'][0])) + Decimal(str(latency))),
                            float(Decimal(str(kwargs['window'][1])) + Decimal(str(latency)))
                        )
                        t, M = tk.psth(
                            session.saccade_onset_timestamps[direction][extrastimulus],
                            unit.timestamps,
                            binsize=kwargs['binsize'],
                            window=time_shifted_window
                        )

                        # Average, subtract baseline, and smooth
                        average_time_shifted_saccade_only_response = M.mean(0) / kwargs['binsize'] - baseline

                        # Compute the response magnitude given the target metric
                        Rs = helpers.extract_response_amplitude(
                            average_time_shifted_saccade_only_response,
                            visual_response_filter,
                            metric='mean'
                        )

                        #
                        if modulation_index_version == 1:
                            value = (Rso - (Ro + Rs)) / abs(Ro + Rs)
                        elif modulation_index_version == 2:
                            value = (Rso - (Rs + Ro)) / (Rso + (Rs + Ro))
                        elif modulation_index_version == 3:
                            value = (Rso - (Rs + Ro)) / baseline

                        #
                        data[level]['fictive'].append((latency, value))

        # ========
        # Plotting
        # ========

        # Setup the sub-figures
        grid = gs.GridSpec(nrows=100, ncols=1)
        figures = [plt.figure() for i in range(3)]
        for level, fig in zip(['low', 'medium', 'high'], figures):

            #
            actual = np.array(data[level]['actual'])
            fictive = np.array(data[level]['fictive'])

            #
            ax1 = fig.add_subplot(grid[:97, :])
            times, heights, points, binsize = helpers.bin_and_average_data(actual, around_saccades_window, nbins)
            ax1.plot(times, heights, color='k', lw=1.5)

            #
            for t, y in zip(times, points):
                n = len(y)
                ax1.scatter(np.full(n, t), y, color='k', marker='.', s=70, alpha=0.3, edgecolor='none')

            #
            ax1.set_xlim(around_saccades_window)
            ax1.set_xticks([])
            ax1.spines['bottom'].set_visible(False)

            #
            ax2 = fig.add_subplot(grid[97:, :])
            cycler = it.cycle(['k', 'w'])
            for ibin in range(nbins):
                x1 = times[ibin] - binsize / 2
                x2 = x1 + binsize
                ax2.fill_betweenx([0, 1], x1, x2, color=next(cycler))

            ax2.set_xlim(around_saccades_window)
            ax2.set_yticks([])

            width, height = self.figsize
            fig.set_figheight(height)
            fig.set_figwidth(width)

        return data, figures

class PerisaccadicVisualResponses():
    """
    """

    def __init__(self):
        return

    def plot(
        self,
        session,
        unit,
        metric='mean',
        window=(-2, 2),
        nbins=20,
        levels=['low', 'medium', 'high'],
        visual_response_window=(0.05, 0.15),
        modulation_index_version=3,
        box_and_whiskers=False,
        hanning_window_length=5,
        actual_condition_color='k',
        fictive_condition_color='r',
        master_line_weight=1.5,
        figsize=(4, 8),
        ylim='dynamic',
        **psth_kwargs
        ):
        """
        """

        # Peri-stimulus time histogram keywords
        psth_kwargs_ = {
            'binsize': 0.01,
            'window': (-0.1, 0.3)
        }
        psth_kwargs_.update(psth_kwargs)

        # Mask for identifying the visual response
        psth_edges = np.arange(
            psth_kwargs_['window'][0],
            psth_kwargs_['window'][1] + psth_kwargs_['binsize'],
            psth_kwargs_['binsize']
        )
        response_window_mask = np.logical_and(
            psth_edges[:-1] + psth_kwargs_['binsize'] / 2 >= visual_response_window[0],
            psth_edges[:-1] + psth_kwargs_['binsize'] / 2 <= visual_response_window[1]
        )

        # Baseline FR in spikes / second
        mu, sigma = helpers.estimate_baseline_activity(session, unit, binsize=psth_kwargs_['binsize'])
        baseline = mu / psth_kwargs_['binsize']

        # Timestamps for all saccades
        all_saccade_onset_timestamps = np.concatenate([
            session.saccade_onset_timestamps['ipsi'],
            session.saccade_onset_timestamps['contra']
        ])

        # Timestamps for all probes
        all_probe_onset_timestamps = np.concatenate([
            session.probe_onset_timestamps['low'],
            session.probe_onset_timestamps['medium'],
            session.probe_onset_timestamps['high']
        ])

        # Empty data container
        data = {
            level: {'actual': list(), 'fictive': list()}
                for level in ['low', 'medium', 'high']
        }

        #
        for level, probe_onset_timestamps in session.probe_onset_timestamps.items():

            # =============
            # Actual probes
            # =============

            # Mask for probes that occur outside of the perisaccadic window
            extrasaccadic = np.invert(helpers.create_coincidence_mask(
                session.probe_onset_timestamps[level],
                all_saccade_onset_timestamps,
                window=(-2, 2)
            ))

            # Compute the average probe-only response for the target contrast level (Rp)
            t, M = tk.psth(
                session.probe_onset_timestamps[level][extrasaccadic],
                unit.timestamps,
                window=psth_kwargs_['window'],
                binsize=psth_kwargs_['binsize']
            )

            # Average and smooth
            average_actual_visual_response = M.mean(0) / psth_kwargs_['binsize'] - baseline

            # Compute the response magnitude given the target metric
            Rp = helpers.extract_response_amplitude(
                average_actual_visual_response,
                response_window_mask,
                metric=metric
            )

            for direction, saccade_onset_timestamps in session.saccade_onset_timestamps.items():

                # Mask for saccades that occur around the time of probes
                peristimulus = helpers.create_coincidence_mask(
                    session.saccade_onset_timestamps[direction],
                    all_probe_onset_timestamps,
                    window=(-1, 1)
                )

                # Mask for saccades that don't occur around the time of probes
                extrastimulus = np.invert(peristimulus)

                #
                for saccade_onset_timestamp in saccade_onset_timestamps:

                    # Look for probes that appeared within the target window
                    within_window_mask = np.logical_and(
                        window[0] <= probe_onset_timestamps - saccade_onset_timestamp,
                        window[1] >= probe_onset_timestamps - saccade_onset_timestamp
                    )

                    # Skip if no probes detected within target window
                    if within_window_mask.sum() == 0:
                        continue

                    #
                    for probe_onset_timestamp in probe_onset_timestamps[within_window_mask]:

                        # Time from saccade to probe onset
                        latency = np.around(probe_onset_timestamp - saccade_onset_timestamp, 3)

                        # Compute the single trial visuomotor response (Rsp)
                        t, M = tk.psth(
                            [probe_onset_timestamp],
                            unit.timestamps,
                            **psth_kwargs_
                        )
                        single_trial_actual_visual_response = M.flatten() / psth_kwargs_['binsize'] - baseline

                        # Compute the response amplitude for this single trial
                        Rsp = helpers.extract_response_amplitude(
                            single_trial_actual_visual_response,
                            response_window_mask,
                            metric=metric
                        )

                        # Compute the response that is attributable to the saccade (Rs)
                        time_shifted_window = (
                            float(Decimal(str(psth_kwargs_['window'][0])) + Decimal(str(latency))),
                            float(Decimal(str(psth_kwargs_['window'][1])) + Decimal(str(latency)))
                        )
                        t, M = tk.psth(
                            session.saccade_onset_timestamps[direction][extrastimulus],
                            unit.timestamps,
                            binsize=psth_kwargs_['binsize'],
                            window=time_shifted_window
                        )

                        # Average, subtract baseline, and smooth
                        average_time_shifted_saccade_only_response = M.mean(0) / psth_kwargs_['binsize'] - baseline

                        # Compute the response magnitude given the target metric
                        Rs = helpers.extract_response_amplitude(
                            average_time_shifted_saccade_only_response,
                            response_window_mask,
                            metric=metric
                        )

                        #
                        if modulation_index_version == 1:
                            value = (Rsp - (Rp + Rs)) / abs(Rp + Rs)
                        elif modulation_index_version == 2:
                            value = (Rsp - (Rs + Rp)) / (Rsp + (Rs + Rp))
                        elif modulation_index_version == 3:
                            value = (Rsp - (Rs + Rp))

                        #
                        data[level]['actual'].append((latency, value))

            # ==============
            # Fictive probes
            # ==============

            # Compute the average fictive visual response looking outside of the perisaccadic window
            stop = unit.timestamps.max() + float(Decimal(str(1.0)) - Decimal(str(unit.timestamps.max() % 1)))
            fictive_probe_onset_timestamps = np.arange(0, stop, 1)
            perisaccadic = helpers.create_coincidence_mask(
                fictive_probe_onset_timestamps,
                all_saccade_onset_timestamps,
                window=(-2, 2)
            )
            peristimulus = helpers.create_coincidence_mask(
                fictive_probe_onset_timestamps,
                all_probe_onset_timestamps,
                window=(-1, 1)
            )
            extraevent = np.invert(np.vstack([
                perisaccadic,
                peristimulus
            ]).any(axis=0))
            extraevent_fictive_probe_onset_timestamps = fictive_probe_onset_timestamps[extraevent]

            #
            t, M = tk.psth(
                extraevent_fictive_probe_onset_timestamps,
                unit.timestamps,
                window=psth_kwargs_['window'],
                binsize=psth_kwargs_['binsize']
            )

            # Average and smooth
            average_fictive_visual_response = M.mean(0) / psth_kwargs_['binsize'] - baseline

            # Compute the response magnitude given the target metric
            Ro = helpers.extract_response_amplitude(
                average_fictive_visual_response,
                response_window_mask,
                metric=metric
            )

            # Sample from across the perisaccadic window excluding actual events
            perisaccadic_fictive_probe_onset_timestamps = list()
            for saccade_onset_timestamp in all_saccade_onset_timestamps:
                start = saccade_onset_timestamp - 2
                stop  = saccade_onset_timestamp + 2
                timestamps = np.arange(start, stop, 0.1)
                extrastimulus = np.invert(helpers.create_coincidence_mask(
                    timestamps,
                    all_probe_onset_timestamps,
                    window=(-1, 1)
                ))
                perisaccadic_fictive_probe_onset_timestamps += timestamps[extrastimulus].tolist()
            perisaccadic_fictive_probe_onset_timestamps = np.array(perisaccadic_fictive_probe_onset_timestamps)

            for direction, saccade_onset_timestamps in session.saccade_onset_timestamps.items():

                # Extrastimulus saccades
                extrastimulus = np.invert(helpers.create_coincidence_mask(
                    session.saccade_onset_timestamps[direction],
                    all_probe_onset_timestamps,
                    window=(-1, 1)
                ))

                #
                for saccade_onset_timestamp in saccade_onset_timestamps:

                    # Look for probes that appeared within the target window
                    within_window_mask = np.logical_and(
                        window[0] <= perisaccadic_fictive_probe_onset_timestamps - saccade_onset_timestamp,
                        window[1] >= perisaccadic_fictive_probe_onset_timestamps - saccade_onset_timestamp
                    )

                    # Skip if no probes detected within target window
                    if within_window_mask.sum() == 0:
                        continue

                    # For each probe that occurs around the time of this saccade ...
                    for fictive_probe_onset_timestamp in perisaccadic_fictive_probe_onset_timestamps[within_window_mask]:

                        # Time from saccade to probe onset
                        latency = np.around(fictive_probe_onset_timestamp - saccade_onset_timestamp, 3)

                        # Compute the single trial visuomotor response (Rsp)
                        t, M = tk.psth(
                            np.array([fictive_probe_onset_timestamp]),
                            unit.timestamps,
                            **psth_kwargs_
                        )
                        single_trial_fictive_visual_response = M.flatten()/ psth_kwargs_['binsize'] - baseline

                        # Compute the response magnitude given the target metric
                        Rso = helpers.extract_response_amplitude(
                            single_trial_fictive_visual_response,
                            response_window_mask,
                            metric=metric
                        )

                        # Compute the response that is attributable to the saccade (Rs)
                        time_shifted_window = (
                            float(Decimal(str(psth_kwargs_['window'][0])) + Decimal(str(latency))),
                            float(Decimal(str(psth_kwargs_['window'][1])) + Decimal(str(latency)))
                        )
                        t, M = tk.psth(
                            session.saccade_onset_timestamps[direction][extrastimulus],
                            unit.timestamps,
                            binsize=psth_kwargs_['binsize'],
                            window=time_shifted_window
                        )

                        # Average, subtract baseline, and smooth
                        average_time_shifted_saccade_only_response = M.mean(0) / psth_kwargs_['binsize'] - baseline

                        # Compute the response magnitude given the target metric
                        Rs = helpers.extract_response_amplitude(
                            average_time_shifted_saccade_only_response,
                            response_window_mask,
                            metric=metric
                        )

                        #
                        if modulation_index_version == 1:
                            value = (Rso - (Ro + Rs)) / abs(Ro + Rs)
                        elif modulation_index_version == 2:
                            value = (Rso - (Rs + Ro)) / (Rso + (Rs + Ro))
                        elif modulation_index_version == 3:
                            value = (Rso - (Rs + Ro))

                        #
                        data[level]['fictive'].append((latency, value))

        # ========
        # Plotting
        # ========

        fig, axs = plt.subplots(ncols=3)
        left_bin_edges = np.linspace(window[0], window[1], nbins + 1)[:-1]
        right_bin_edges = np.linspace(window[0], window[1], nbins + 1)[1:]
        binsize = np.unique(np.around(right_bin_edges - left_bin_edges, 3)).item()
        timepoints = left_bin_edges + (binsize / 2)

        for ax, (level, dct) in zip(axs, data.items()):
            for condition, rows in dct.items():

                # Color
                color = 'k' if condition == 'actual' else fictive_condition_color

                # Bin the data
                arr = np.array(rows)
                all_bins_data = list()
                all_bins_mean = list()
                all_bins_error = list()

                # Bin the data
                for left_bin_edge, right_bin_edge in zip(left_bin_edges, right_bin_edges):
                    single_bin_data = list()
                    for ipt, latency in enumerate(arr[:, 0]):
                        if left_bin_edge <= latency < right_bin_edge:
                            single_bin_data.append(arr[ipt, 1])

                    all_bins_data.append(single_bin_data)
                    all_bins_mean.append(np.mean(single_bin_data))
                    all_bins_error.append(sem(single_bin_data))

                # Smooth
                if hanning_window_length is None:
                    all_bins_mean_smoothed = np.array(all_bins_mean)
                    all_bins_error_smoothed = np.array(all_bins_error)
                else:
                    all_bins_mean_smoothed = tk.smooth(np.array(all_bins_mean), hanning_window_length)
                    all_bins_error_smoothed = tk.smooth(np.array(all_bins_error), hanning_window_length)

                # Box and whiskers plot
                if box_and_whiskers:
                    if condition == 'fictive':
                        positions = (np.arange(timepoints.size * 2)[::2]) * binsize
                    else:
                        positions = (np.arange(timepoints.size * 2)[::2] + 1) * binsize
                    boxplot = ax.boxplot(all_bins_data, positions=positions, widths=binsize, showfliers=False, patch_artist=True)
                    for patch in boxplot['boxes']:
                        patch.set(facecolor=color, alpha=0.3)
                    for whisker in boxplot['whiskers']:
                        whisker.set(color='k')
                    for median in boxplot['medians']:
                        median.set(color='k')
                    for cap in boxplot['caps']:
                        cap.set(color='k')

                # Simple line plot
                else:
                    if condition == 'actual':
                        ax.plot(arr[:, 0], arr[:, 1], color='none', alpha=0.5, ms=4, marker='o', mec='none', mfc=color)
                    ax.plot(timepoints, all_bins_mean_smoothed, color=color, lw=master_line_weight)
                    ax.fill_between(
                        timepoints,
                        all_bins_mean_smoothed - all_bins_error_smoothed,
                        all_bins_mean_smoothed + all_bins_error_smoothed,
                        color=color,
                        alpha=0.1
                    )

        #
        if ylim == 'dynamic':
            ylim = [0, 0]
            for ax in axs:
                ymin, ymax = ax.get_ylim()
                if ymin < ylim[0]:
                    ylim[0] = ymin
                if ymax > ylim[1]:
                    ylim[1] = ymax

        for ax in axs:
            ax.set_ylim(ylim)
            for spine in ['top', 'right']:
                ax.spines[spine].set_visible(False)

        for ax in axs[1:]:
            ax.set_yticklabels([])

        for label, ax in zip(['Low', f'Medium', 'high'], axs):
            ax.set_title(label)
        axs[0].set_ylabel('Modulation index')
        axs[1].set_xlabel('Time from saccade onset (seconds)')

        fig.set_figheight(figsize[0])
        fig.set_figwidth(figsize[1])
        fig.tight_layout()

        return data

class SingleUnitActivityTable():

    def plot(
        self,
        session,
        unit,
        nbins=5,
        binsize=0.6,
        margin=0.05,
        direction='contra',
        standardize=False,
        smooth=True,
        hanning_window_length=7,
        return_data_containers=False,
        line_plot_alpha=0.5,
        figsize=(5, 10),
        legend_location=1,
        legend_handle_length=1,
        ylim_margin=(0, 0),
        line_width=2.5,
        probe_only_color='C0',
        saccade_only_color='C1',
        probe_and_saccade_color='C2',
        alpha=0.7,
        **psth_kwargs
        ):
        """
        """

        # Keyword arguments for the psth function
        psth_kwargs_ = {
            'binsize': 0.01,
            'window' : (-0.15, 0.35)
        }
        psth_kwargs_.update(psth_kwargs)

        # Mean and standard deviation of spike count in a single psth bin
        if standardize:
            mu, sigma = unit.estimate_baseline_activity(psth_kwargs_['binsize'])

        # =====================
        # Pure visual responses
        # =====================

        probe_only_data = {
            level: None for level in ['low', 'medium', 'high']
        }

        for level, probe_onset_timestamps in session.probe_onset_timestamps.items():
            coincidence_mask = helpers.create_coincidence_mask(
                probe_onset_timestamps,
                np.concatenate([session.saccade_onset_timestamps['ipsi'], session.saccade_onset_timestamps['contra']]),
                window=(-0.05, 0.65)
            )
            probe_onset_timestamps = probe_onset_timestamps[~coincidence_mask]
            t, M = tk.psth(probe_onset_timestamps, unit.timestamps, **psth_kwargs_)
            probe_only_data[level] = np.array([
                row for row in M
            ])

        # ======================
        # Pure premotor activity
        # ======================

        saccade_only_data = {
            direction: None for direction in ['ipsi', 'contra']
        }

        for direction, saccade_onset_timestamps in session.saccade_onset_timestamps.items():
            coincidence_mask = helpers.create_coincidence_mask(
                saccade_onset_timestamps,
                np.concatenate([probe_onset_timestamps for probe_onset_timestamps in session.probe_onset_timestamps.values()]),
                window=(-0.05, 0.65)
            )
            saccade_onset_timestamps = saccade_onset_timestamps[~coincidence_mask]
            t, M = tk.psth(saccade_onset_timestamps, unit.timestamps, **psth_kwargs_)
            saccade_only_data[direction] = np.array([
                row for row in M
            ])

        # ===================
        # Visuomotor activity
        # ===================

        combined_visuomotor_data = {
            'ipsi' : {
                'low'    : {ibin: {'actual': list(), 'summation': list(), 'subtraction': list()} for ibin in range(nbins)},
                'medium' : {ibin: {'actual': list(), 'summation': list(), 'subtraction': list()} for ibin in range(nbins)},
                'high'   : {ibin: {'actual': list(), 'summation': list(), 'subtraction': list()} for ibin in range(nbins)},
            },
            'contra': {
                'low'    : {ibin: {'actual': list(), 'summation': list(), 'subtraction': list()} for ibin in range(nbins)},
                'medium' : {ibin: {'actual': list(), 'summation': list(), 'subtraction': list()} for ibin in range(nbins)},
                'high'   : {ibin: {'actual': list(), 'summation': list(), 'subtraction': list()} for ibin in range(nbins)},
            }
        }

        # Collect visual reponses from each time bin across the perisaccadic window
        for direction, saccade_onset_timestamps in session.saccade_onset_timestamps.items():

            # Compute the time points for the edges of each time bin around the saccades
            bins = list()
            for saccade_onset_timestamp in saccade_onset_timestamps:
                edges = list()
                for ibin in np.linspace(-1 * (nbins - 1) / 2, (nbins - 1) / 2, nbins).astype(int):
                    start = ibin * (2 * margin + binsize) + saccade_onset_timestamp
                    stop  = start + (2 * margin + binsize)
                    edges.append((start, stop))
                bins.append(edges)
            bins = np.array(bins)

            # Iterate through each level of the probe
            for level, probe_onset_timestamps in session.probe_onset_timestamps.items():

                # Iterate through each saccade
                for saccade_onset_timestamp, edges in zip(saccade_onset_timestamps, bins):

                    # Iterate through each time bin around the saccade
                    for ibin, (start, stop) in enumerate(edges):

                        # Look for probes which appear within the target time bin
                        intra_bin_mask = np.array([
                            True if start <= probe_onset_timestamp < stop else False
                                for probe_onset_timestamp in probe_onset_timestamps
                        ])

                        # No probes detected within the time bin
                        if intra_bin_mask.sum() == 0:
                            # npts = int(((psth_kwargs_['window'][1] - psth_kwargs_['window'][0]) / psth_kwargs_['binsize']))
                            # empty = np.full(npts, np.nan)
                            # for condition in ['actual', 'summation', 'subtraction']:
                            #     combined_visuomotor_data[direction][level][ibin][condition].append(empty)
                            continue

                        # One probe detected within the time bin
                        elif intra_bin_mask.sum() == 1:

                            # Target probe timestamp
                            probe_onset_timestamp = probe_onset_timestamps[intra_bin_mask]

                            # Iterate through each condition
                            for condition in ['actual', 'summation', 'subtraction']:

                                # Actual activity
                                if condition == 'actual':
                                    t, M = tk.psth(
                                        [probe_onset_timestamp],
                                        unit.timestamps,
                                        **psth_kwargs_
                                    )
                                    response = M.flatten() # Save the spike counts

                                # Predicted activity
                                else:

                                    # Get the mask which identifies saccades which happen outside of visual events
                                    coincidence_mask = helpers.create_coincidence_mask(
                                        saccade_onset_timestamps,
                                        np.concatenate([probe_onset_timestamps for probe_onset_timestamps in session.probe_onset_timestamps.values()]),
                                        window=(-0.05, 0.65)
                                    )

                                    # Compute the time-shifted average premotor activity
                                    offset = probe_onset_timestamp - saccade_onset_timestamp
                                    t, M = tk.psth(saccade_onset_timestamps[~coincidence_mask] + offset, unit.timestamps, **psth_kwargs_)
                                    average_premotor_activity = M.mean(0)
                                    average_visual_response = probe_only_data[level].mean(0)

                                    # Summation of average visual and motor responses
                                    if condition == 'summation':
                                        response = average_visual_response + average_premotor_activity

                                    # Difference of average visual and motor responses
                                    elif condition == 'subtraction':
                                        response = average_visual_response - average_premotor_activity

                                combined_visuomotor_data[direction][level][ibin][condition].append(response)

        # Convert to numpy array
        for direction in ['ipsi', 'contra']:
            for level in ['low', 'medium', 'high']:
                for ibin in np.arange(nbins):
                    for condition in ['actual', 'summation', 'subtraction']:
                        combined_visuomotor_data[direction][level][ibin][condition] = np.array(combined_visuomotor_data[direction][level][ibin][condition])

        # Stop here and check data if necessary
        if return_data_containers:
            return probe_only_data, saccade_only_data, combined_visuomotor_data


        # ========
        # Plotting
        # ========

        fig, axs = plt.subplots(ncols=nbins, nrows=3)

        npts = int((psth_kwargs_['window'][1] - psth_kwargs_['window'][0]) / psth_kwargs_['binsize'])
        time = np.linspace(
            psth_kwargs_['window'][0] + psth_kwargs_['binsize'] / 2,
            psth_kwargs_['window'][1] - psth_kwargs_['binsize'] / 2,
            npts
        )

        # Probe only activity
        probe_only_responses = {
            level: None for level in ['low', 'medium', 'high']
        }
        for irow, (level, trials) in enumerate(probe_only_data.items()):
            for icol in range(nbins):
                tax = axs[irow, icol]
                if standardize:
                    response = (trials.mean(0) - mu) / sigma
                else:
                    response = (trials.mean(0) / psth_kwargs_['binsize'])
                if smooth:
                    response = tk.smooth(response, hanning_window_length)
                tax.plot(time, response, color=probe_only_color, alpha=alpha, label=f'n={trials.shape[0]}', lw=line_width)
                probe_only_responses[level] = response

        # Saccade-only activity
        saccade_only_responses = {
            direction: None for direction in ['ipsi', 'contra']
        }
        for irow in range(3):
            for direction, trials in saccade_only_data.items():
                if direction == 'ipsi':
                    continue
                for icol in range(nbins):
                    tax = axs[irow, icol]
                    if standardize:
                        response = (trials.mean(0) - mu) / sigma
                    else:
                        response = (trials.mean(0) / psth_kwargs_['binsize'])
                    if smooth:
                        response = tk.smooth(response, hanning_window_length)
                    tax.plot(time, response, color=saccade_only_color, alpha=alpha, label=f'n={trials.shape[0]}', lw=line_width)
                    saccade_only_responses[direction] = response

        # Visuomotor activity
        for irow, level in enumerate(['low', 'medium', 'high']):
            for icol in range(nbins):
                for color, condition in zip(['C0', 'C1', 'C2'], ['actual', 'summation', 'subtraction']):
                    if condition in ['summation', 'subtraction']:
                        continue
                    trials = combined_visuomotor_data[direction][level][icol][condition]
                    if len(trials) == 0:
                        continue
                    tax = axs[irow, icol]
                    if standardize:
                        response = (trials.mean(0) - mu) / sigma
                    else:
                        response = (trials.mean(0) / psth_kwargs_['binsize'])
                    if smooth:
                        response = tk.smooth(response, hanning_window_length)
                    if condition == 'actual':
                        tax.plot(time, response, color=probe_and_saccade_color, alpha=alpha, linestyle='-', label=f'n={len(trials)}', lw=line_width)
                    else:
                        tax.plot(time, response, color='k', linestyle='dotted', alpha=alpha, lw=line_width)
                    # tax.legend(handlelength=legend_handle_length, loc=legend_location)
                    # tax.fill_between(time, response, probe_only_responses[level], color='r', alpha=0.2)

        #
        ymin = np.array([ax.get_ylim()[0] for ax in axs.flatten()]).min() + ylim_margin[0]
        ymax = np.array([ax.get_ylim()[1] for ax in axs.flatten()]).max() + ylim_margin[1]
        for ax in axs.flatten():
            ax.set_ylim([ymin, ymax])
            ax.fill_betweenx([ymin, ymax], 0, 3 * (1/ 60), color='y', alpha=0.2, edgecolor='none')

        #
        for ax in axs[:, 1:].flatten():
            ax.set_yticklabels([])

        #
        for ax in axs[:2, :].flatten():
            ax.set_xticklabels([])

        #
        fig.set_figheight(figsize[0])
        fig.set_figwidth(figsize[1])
        fig.tight_layout()
