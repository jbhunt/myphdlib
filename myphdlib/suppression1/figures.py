import numpy as np
from matplotlib import pylab as plt
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
        binsize=0.02,
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
        for k, v in pcolormesh_kwargs.items():
            if k in pcolormesh_kwargs_.keys():
                pcolormesh_kwargs_[k] = v

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
                    target_event = session.saccade_onset_ipsi
                elif condition == 'contra':
                    target_event = session.saccade_onset_contra
                elif condition == 'low':
                    target_event = session.probe_onset_low
                elif condition == 'medium':
                    target_event = session.probe_onset_medium
                elif condition == 'high':
                    target_event = session.probe_onset_high

                #
                edges, M1 = tk.psth(target_event, unit.timestamps, window=window, binsize=binsize)

                # Standardize
                psuedo_window = (-1 * binsize / 2, binsize / 2)
                M2 = unit.compute_psuedo_response(N=1000, window=psuedo_window)
                if M2.sum() == 0:
                    if interpolate:
                        row = np.full(evalualtion_sample_size, np.nan)
                    else:
                        row = np.full(M1.shape[1], np.nan)
                    heatmap.append(row)
                    continue
                mu, sigma = M2.mean(), M2.std()
                row = (M1.mean(0) - mu) / sigma

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
        fig.subplots_adjust(right=0.8)
        cax = fig.add_axes([0.85, 0.2, 0.05, 0.6])
        fig.colorbar(mappable, cax=cax)
        fig.set_figheight(figsize[0])
        fig.set_figwidth(figsize[1])
        fig.show()

        return fig, axs, heatmaps

class PerisaccadicVisualResponses():
    """
    """

    def __init__(self):
        return

    def plot(
        self,
        session,
        unit,
        metric='auc',
        direction='contra',
        window=(-1.45, 2.05),
        nbins=15,
        levels=['low', 'medium', 'high'],
        visual_response_window=(0.05, 0.25),
        hanning_window_length=11,
        figsize=(4, 8),
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

        #
        mu, sigma = helpers.estimate_baseline_activity(session, unit, binsize=psth_kwargs_['binsize'])

        # Empty data container
        data = {
            level: list() for level in ['low', 'medium', 'high']
        }

        #
        for level, probe_onset_timestamps in session.probe_onset_timestamps.items():

            # Mask for probes that occur around the time of saccades
            perisaccadic_probes_mask = helpers.create_coincidence_mask(
                session.probe_onset_timestamps[level],
                np.concatenate([
                    session.saccade_onset_timestamps['ipsi'],
                    session.saccade_onset_timestamps['contra']
                ]),
                window=(-1, 1)
            )

            # Mask for probes that don't occur around the time of saccades
            extrasacadic_probes_mask = np.invert(perisaccadic_probes_mask)

            # Compute the average probe-only response for the target contrast level (Rp)
            t, M = tk.psth(
                session.probe_onset_timestamps[level][extrasacadic_probes_mask],
                unit.timestamps,
                window=psth_kwargs_['window'],
                binsize=psth_kwargs_['binsize']
            )

            # Average and smooth
            average_visual_response = tk.smooth(M.mean(0), hanning_window_length)

            # Compute the response magnitude given the target metric
            if metric == 'auc':
                Rp = np.trapz(average_visual_response[response_window_mask])
            elif metric == 'peak':
                Rp = average_visual_response[response_window_mask].max()
            elif metric == 'count':
                Rp = average_visual_response[response_window_mask].sum()

            for direction, saccade_onset_timestamps in session.saccade_onset_timestamps.items():

                # Mask for saccades that occur around the time of probes
                peristimulus_saccades_mask = helpers.create_coincidence_mask(
                    session.saccade_onset_timestamps[direction],
                    np.concatenate([
                        session.probe_onset_timestamps['low'],
                        session.probe_onset_timestamps['medium'],
                        session.probe_onset_timestamps['high']
                    ]),
                    window=(-1, 1)
                )

                # Mask for saccades that don't occur around the time of probes
                extrastimulus_saccades_mask = np.invert(peristimulus_saccades_mask)

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
                        latency = np.around(probe_onset_timestamp - saccade_onset_timestamp, 2)

                        # Compute the single trial visuomotor response (Rsp)
                        t, M = tk.psth(
                            [probe_onset_timestamp],
                            unit.timestamps,
                            **psth_kwargs_
                        )
                        actual_visual_response = tk.smooth(M.flatten(), hanning_window_length)

                        # Compute the response magnitude given the target metric
                        if metric == 'auc':
                            Rsp = np.trapz(actual_visual_response[response_window_mask])
                        elif metric == 'peak':
                            Rsp = actual_visual_response[response_window_mask].max()
                        elif metric == 'count':
                            Rsp = actual_visual_response[response_window_mask].sum()

                        # Compute the response that is attributable to the saccade (Rs)
                        time_shifted_window = (
                            float(Decimal(str(psth_kwargs_['window'][0])) + Decimal(str(latency))),
                            float(Decimal(str(psth_kwargs_['window'][1])) + Decimal(str(latency)))
                        )
                        t, M = tk.psth(
                            session.saccade_onset_timestamps[direction][extrastimulus_saccades_mask],
                            unit.timestamps,
                            binsize=psth_kwargs_['binsize'],
                            window=time_shifted_window
                        )

                        # Average, subtract baseline, and smooth
                        average_motor_activity = tk.smooth(M.mean(0) - mu, hanning_window_length)

                        # Compute the response magnitude given the target metric
                        if metric == 'auc':
                            Rs = np.trapz(average_motor_activity[response_window_mask])
                        elif metric == 'peak':
                            Rs = average_motor_activity[response_window_mask].max()
                        elif metric == 'count':
                            Rs = average_motor_activity[response_window_mask].sum()

                        #
                        value = (Rsp - (Rp + Rs)) / (Rp + Rs)

                        #
                        data[level].append((latency, value))

        # Plotting
        fig, axs = plt.subplots(ncols=3)

        left_edges = np.linspace(window[0], window[1], nbins + 1)[:-1]
        right_edges = np.linspace(window[0], window[1], nbins + 1)[1:]
        bin_width = np.unique(np.around(right_edges - left_edges, 3)).item()
        time_points = left_edges + (bin_width / 2)

        for ax, color, marker, (level, rows) in zip(axs, ['C0', 'C1', 'C2'], ['o', 's', '^'], data.items()):
            arr = np.array(rows)
            ax.scatter(arr[:, 0], arr[:, 1], color='k', alpha=0.1, s=15, marker='o')

            bin_heights = list()
            bin_errors  = list()

            # Bin the scatterplot
            for left_edge, right_edge in zip(left_edges, right_edges):
                binned_data = list()
                for ipt, latency in enumerate(arr[:, 0]):
                    if left_edge <= latency < right_edge:
                        binned_data.append(arr[ipt, 1])

                bin_heights.append(np.mean(binned_data))
                bin_errors.append(sem(binned_data))

            bin_heights_smoothed = tk.smooth(np.array(bin_heights), 5)
            bin_errors_smoothed = tk.smooth(np.array(bin_errors), 5)
            ax.plot(time_points, bin_heights_smoothed, color='k')
            ax.fill_between(
                time_points,
                bin_heights_smoothed - bin_errors_smoothed,
                bin_heights_smoothed + bin_errors_smoothed,
                color='k',
                alpha=0.1
            )

        # Show the mean probe only response for each contrast level
        # for iax, (level, probe_onset_timestamps) in enumerate(session.probe_onset_timestamps.items()):
        #     coincidence_mask = helpers.create_coincidence_mask(
        #         probe_onset_timestamps,
        #         np.concatenate([session.saccade_onset_timestamps['ipsi'], session.saccade_onset_timestamps['contra']]),
        #         window=(-1, 1)
        #     )
        #
        #     t, M = tk.psth(probe_onset_timestamps[~coincidence_mask], unit.timestamps, **psth_kwargs_)
        #     response_window_mask = np.logical_and(
        #         t[:-1] + psth_kwargs_['binsize'] / 2 >= visual_response_window[0],
        #         t[:-1] + psth_kwargs_['binsize'] / 2 <= visual_response_window[1]
        #     )
        #     average_visual_response = M.mean(0)
        #     smoothed = tk.smooth(average_visual_response, hanning_window_length)
        #     if metric == 'auc':
        #         magnitude = np.trapz(smoothed[response_window_mask])
        #     elif metric == 'peak':
        #         magnitude = smoothed[response_window_mask].max()
        #     elif metric == 'count':
        #         magnitude = smoothed[response_window_mask].sum()
        #     axs[iax].hlines(magnitude, time_points[0], time_points[-1], color='k', linestyle='dotted')

        #
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

        fig.set_figheight(figsize[0])
        fig.set_figwidth(figsize[1])
        fig.tight_layout()

        # for ax in axs:
        #     for phase, ibin in zip([1, 0, 1, 0, 1], np.linspace(-1 * 4 / 2, 4 / 2, 5)):
        #         left_edge = np.around(ibin * (2 * 0.05 + 0.6) - 0.05, 3)
        #         right_edge = np.around(left_edge + (2 * 0.05 + 0.6), 3)
        #         color = 'k' if ibin != 0 else 'r'
        #         if phase is 1:
        #            ax.fill_betweenx((ymin, ymax), left_edge, right_edge, color=color, alpha=0.1)

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
                tax.plot(time, response, color='k', alpha=0.7, label=f'n={trials.shape[0]}', lw=2)
                probe_only_responses[level] = response

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
                        tax.plot(time, response, color='C0', alpha=0.7, label=f'n={len(trials)}', lw=2)
                    else:
                        tax.plot(time, response, color='k', linestyle='dotted', alpha=0.8, lw=2)
                    # tax.legend(handlelength=legend_handle_length, loc=legend_location)
                    # tax.fill_between(time, response, probe_only_responses[level], color='r', alpha=0.2)

        #
        ymin = np.array([ax.get_ylim()[0] for ax in axs.flatten()]).min() + ylim_margin[0]
        ymax = np.array([ax.get_ylim()[1] for ax in axs.flatten()]).max() + ylim_margin[1]
        for ax in axs.flatten():
            ax.set_ylim([ymin, ymax])
            ax.fill_betweenx([ymin, ymax], 0, 3 * (1/ 60), color='y', alpha=0.2)

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
