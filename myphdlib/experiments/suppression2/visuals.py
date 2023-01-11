import numpy as np
import pathlib as pl
from matplotlib import pylab as plt
from scipy.stats import sem
from myphdlib.general.toolkit import smooth, psth2

def plotHistogramsByTrialType(
    result,
    metadata,
    factory,
    experiment='saline',
    window=(-0.2, 0.5),
    binsize=0.02,
    outputFolder=None,
    pmax=0.03,
    minimumTrialCount=5,
    targetNeuronIndex=None,
    error='sem',
    ):
    """
    """

    mask = np.logical_or(
        result[experiment]['ipsi']['p'] < pmax,
        result[experiment]['contra']['p'] < pmax,
    )
    counter = 0
    for index, flag in enumerate(mask):
        if flag:

            if targetNeuronIndex is not None:
                if counter < targetNeuronIndex:
                    counter += 1
                    continue
                elif counter > targetNeuronIndex:
                    return

            fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, sharex=True, gridspec_kw={'height_ratios': [1, 3, 1, 3]})

            #
            animal, date, cluster = metadata[experiment][index]
            session = factory.produce(animal, date)
            neuron = session.rez.search(cluster)
            probes = session.parseVisualProbes()
            saccades = session.saccadeOnsetTimestamps()
            ymax, ymin = (-np.inf, np.inf)
            
            #
            for motion, axMain, axHist, color in zip(['ipsi', 'contra'], [ax2, ax4], [ax1, ax3], ['r', 'b']):

                #
                for category, ls in zip(['extrasaccadic', 'perisaccadic'], ['-', '--']):

                    #
                    if category == 'extrasaccadic':
                        color = 'r'
                    else:
                        color = 'g'

                    #
                    probeOnsetTimestamps = probes[category][motion]['timestamps']
                    nProbes = probeOnsetTimestamps.size
                    probeOnsetLatencies = probes[category][motion]['latencies']

                    # Plot the visual responses
                    if nProbes >= minimumTrialCount:
                        t, M = psth2(
                            probeOnsetTimestamps,
                            neuron.timestamps,
                            window=window,
                            binsize=binsize
                        )
                        fr = smooth(M.mean(0) / binsize, 3)
                        if error == 'sem':
                            err = smooth(sem(M, axis=0) / binsize, 3)
                        elif error == 'std':
                            err = smooth(M.std(0) / binsize, 3)
                        axMain.plot(t, fr, color=color, label=f'{motion} motion\n({category}, n={nProbes})')
                        axMain.fill_between(t, fr - err, fr + err, color=color, alpha=0.1)
                        if np.max(fr + err) > ymax:
                            ymax = np.max(fr + err)
                        if np.min(fr - err) < ymin:
                            ymin = np.min(fr - err)

                        # Plot the timestamps for the peri-saccadic trials
                        if category == 'perisaccadic':
                            axHist.hist(probeOnsetLatencies, range=window, bins=50, color='w', edgecolor='k')
                            axHist.spines['top'].set_visible(False)
                            axHist.spines['right'].set_visible(False)

                    # Plot the saccade-related activity
                    if category == 'perisaccadic':
                        if motion == 'ipsi':
                            direction = 'contra'
                        elif motion == 'contra':
                            direction = 'ipsi'
                        saccadeOnsetTimestamps = saccades[direction]
                        nSaccades = saccadeOnsetTimestamps.size
                        if nSaccades >= minimumTrialCount:
                            t, M = psth2(
                                saccadeOnsetTimestamps,
                                neuron.timestamps,
                                window=window,
                                binsize=binsize
                            )
                            fr = smooth(M.mean(0) / binsize, 3)
                            if error == 'sem':
                                err = smooth(sem(M, axis=0) / binsize, 3)
                            elif error == 'std':
                                err = smooth(M.std(0) / binsize, 3)
                            if np.max(fr + err) > ymax:
                                ymax = np.max(fr + err)
                            if np.min(fr - err) < ymin:
                                ymin = np.min(fr - err)

                            axMain.plot(t, fr, color='b', label=f'{direction} saccades\n(n={nSaccades})')
                            axMain.fill_between(t, fr - err, fr + err, color='b', alpha=0.1)

                    # Shrink current axis by 20%
                    for ax in (axMain, axHist):
                        box = ax.get_position()
                        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])

                    # Put a legend to the right of the current axis
                    axMain.legend(loc='center left', bbox_to_anchor=(1, 0.5), fancybox=True, shadow=True)

            #
            ax4.set_xlabel('Time from event onset (sec)')
            ax2.set_ylabel('FR (spikes/sec)')
            ax4.set_ylabel('FR (spikes/sec)')

            #
            yrange = ymax - ymin
            yoffset = yrange * 0.05
            for ax in (ax2, ax4):
                ax.set_ylim([ymin - yoffset, ymax + yoffset])

            #
            ymax2 = np.max([np.max(ax1.get_ylim()), np.max(ax3.get_ylim())])
            for ax in (ax1, ax3):
                ax.set_ylim([0, ymax2])

            #
            if outputFolder is None:
                plt.close(fig)

            else:
                filename = pl.Path(outputFolder).joinpath(f'{animal}_{date}_{cluster}.png')
                fig.savefig(filename, dpi=300)
                plt.close(fig)

    return