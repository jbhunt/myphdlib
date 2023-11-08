from matplotlib import pyplot as plt
import numpy as np
def filterUnits(
    sessions,
    probeMotion=-1
    ):
    """
    """

    probeDirection = 'left' if probeMotion == -1 else 'right'

    subsets = {
        ('p0', 'p'): list(),
        ('p0', 'f'): list(),
        ('p1', 'p'): list(),
        ('p1', 'f'): list(),
        ('p2', 'p'): list(),
        ('p2', 'f'): list(),
        ('p3', 'p'): list(),
        ('p3', 'f'): list(),
    }

    # Phase 0
    for session in sessions:
        for unit in session.population:
            subsets[('p0', 'p')].append(unit)

    # Phase 1 - Event-related activity
    for unit in subsets[('p0', 'p')]:
        if unit.visualResponseProbability[probeDirection] > 0.999:
            subsets[('p1', 'p')].append(unit)
        else:
            subsets[('p1', 'f')].append(unit)

    # Phase 2 - Clustering quality
    for unit in subsets[('p1', 'p')]:
        flags = (
            unit.amplitudeCutoff <= 0.1,
            unit.presenceRatio >= 0.9,
            unit.refractoryPeriodViolationRate <= 0.5
        )
        if all(flags):
            subsets[('p2', 'p')].append(unit)
        else:
            subsets[('p2', 'f')].append(unit)

    # Phase 3 - Response amplitude
    for unit in subsets[('p2', 'p')]:
        flags = (
            unit.visualResponseAmplitude[probeDirection] >= 5,
            unit.visualResponseLatency[probeDirection] >= 0.05,
        )
        if all(flags):
            subsets[('p3', 'p')].append(unit)
        else:
            subsets[('p3', 'f')].append(unit)

    return subsets

def visualizeFilteringProcedure(
    sessions,
    probeMotion=-1,
    ):
    """
    """

    subsets = filterUnits(sessions, probeMotion)
    fig = plt.figure(constrained_layout=False)
    nTotal = len(subsets[('p0', 'p')])
    gs = fig.add_gridspec(nrows=nTotal, ncols=4, hspace=0.1)
    for i in range(4):
        phase = f'p{i}'
        nPassing = len(subsets[(phase, 'p')])
        nFailing = len(subsets[(phase, 'f')])
        nBlanked = nTotal - (nPassing + nFailing)

        #
        

        #
        rows = np.array([
            [0, nPassing],
            [nPassing, nPassing + nFailing],
            [nPassing + nFailing, nPassing + nFailing + nBlanked]
        ])
        for iRow, (start, stop) in enumerate(rows):

            #
            if stop - start == 0:
                continue
            ax = fig.add_subplot(gs[start: stop, i])
            if iRow != len(rows):
                ax.set_xticks([])

            #
            if iRow == 0:
                key = 'p'
            elif iRow == 1:
                key = 'f'
            else:
                continue
            R = list()
            for unit in subsets[(phase, key)]:
                trialIndices = np.where(unit.session.filterProbes(
                    trialType='es',
                    probeDirections=(probeMotion,)
                ))[0]
                t, fr = unit.peth(
                    unit.session.probeTimestamps[trialIndices],
                )
                R.append(fr)
            ax.pcolormesh(R, vmin=-3, vmax=3, cmap='binary_r')

    return fig
