import numpy as np
from scipy.signal import find_peaks as findPeaks
from myphdlib.general.toolkit import smooth, psth2, computeAngleFromStandardPosition
from scipy.interpolate import Akima1DInterpolator

def _estimateDirectionTuningForSingleUnit(
    unit,
    preference='max',
    normalize=False,
    baselineWindowSize=1,
    ):
    """
    """

    # Load stimulus metadata
    movingBarOrientations = unit.session.load('stimuli/mb/orientation')
    barOnsetTimestamps = unit.session.load('stimuli/mb/onset/timestamps')
    barOffsetTimestamps = unit.session.load('stimuli/mb/offset/timestamps')
    movingBarTimestamps = np.hstack([
        barOnsetTimestamps.reshape(-1, 1),
        barOffsetTimestamps.reshape(-1, 1)
    ])

    #
    vertices = list()

    #
    uniqueOrientations = np.unique(movingBarOrientations)
    uniqueOrientations.sort()

    # Estimate a baseline response distribution
    if normalize:
        t, M = psth2(
            movingBarTimestamps[:, 0],
            unit.timestamps,
            window=(0 - baselineWindowSize, 0),
            binsize=None,
        )
        bl = M.flatten() / baselineWindowSize
        mu, sigma = bl.mean(), bl.std()
        if sigma == 0:
            return np.full([9, 2], np.nan), np.nan, np.nan, np.nan

    #
    for orientation in uniqueOrientations:

        #
        trialIndices = np.where(movingBarOrientations == orientation)[0]
        R = list()
        for trialIndex in trialIndices:
            t1, t2 = movingBarTimestamps[trialIndex, :]
            dt = t2 - t1
            t, ri = psth2(
                np.array([t1]),
                unit.timestamps,
                window=(0, dt),
                binsize=None
            )
            fr = ri.sum() / dt
            if normalize:
                z = (fr - mu) / sigma
                R.append(z)
            else:
                R.append(fr)

        #
        amplitude = np.mean(R)

        # Compute the coordinates of the polar plot vertices
        theta = np.deg2rad(orientation)
        vertex = (
            amplitude * np.cos(theta),
            amplitude * np.sin(theta)
        )
        vertices.append(vertex)

    #
    vertices.append(vertices[0])
    vertices = np.around(np.array(vertices), 2)
    amplitudes = np.array([
        np.linalg.norm(vertex - np.array([0, 0]))
            for vertex in vertices[:-1]
    ])

    # Estimate the null and preferred directions
    if preference == 'max':
        iMax = np.argmax(amplitudes)
        pd = uniqueOrientations[iMax]

    # Compute PD as average vector
    # TODO: Figure out why the response in the null direction is sometimes larger than the response to the preferred direction
    elif preference == 'mean':
        ep = np.nanmean(vertices[:-1, :], axis=0)
        pd = round(computeAngleFromStandardPosition(ep), 2)

    #
    else:
        raise Exception(f'{preference} is not a valid metric for measuring preferred direction')

    # Preferred direction could not be determined
    if np.isnan(pd):
        return vertices, np.nan, np.nan, np.nan
    
    # Estimate null direction
    nd = round(np.mod(pd + 180, 360), 3)

    # Compute DSI
    xf = np.concatenate([uniqueOrientations, [360]])
    xp = np.concatenate([amplitudes, [amplitudes[0]]])
    rPreferred = round(np.interp(pd, xf, xp), 3)
    rNull = round(np.interp(nd, xf, xp), 3)
    if rPreferred == 0 or rNull == 0:
        rNull += 1.0
        rPreferred += 1.0
    dsi = round(1 - (rNull / rPreferred), 3)
    if dsi < 0:
        import pdb; pdb.set_trace()

    return vertices, dsi, nd, pd

def measureDirectionSelectivity(
    session,
    preference='max',
    filters=('visual',)
    ):
    """
    """

    nUnits = len(session.population)
    data = {
        'dsi': np.full(nUnits, np.nan),
        'nd': np.full(nUnits, np.nan),
        'pd': np.full(nUnits, np.nan)
    }

    #
    if len(filters) == 1:
        path = f'population/filters/{filters[0]}'
        filter_ = session.load(path)
    else:
        filter_ = np.logical_and(
            [session.load(f'population/filters/{f}') for f in filters]
        )
    if filter_.sum() == 0:
        session.log(f'No units made it through filtering', level='warning')
        return

    #
    for iUnit, (unit, flag) in enumerate(zip(session.population, filter_)):
        if flag:
            vertices, dsi, nd, pd = _estimateDirectionTuningForSingleUnit(
                unit,
                preference,
            )
            # vertices, dsi, nd, pd, ep = _estimateDirectionTuningForSingleUnit2(
            #     unit,
            # )
            data['dsi'][iUnit] = dsi
            data['nd'][iUnit] = nd
            data['pd'][iUnit] = pd
        else:
            data['dsi'][iUnit] = np.nan
            data['dsi'][iUnit] = np.nan
            data['dsi'][iUnit] = np.nan

    #
    sample = data['dsi'][filter_]
    if np.isnan(sample).sum() == sample.size:
        session.log(f'No units made it through filtering', level='warning')
        return
    average = np.nanmean(sample)
    session.log(f'Average DSI for {filter_.sum()} visual units is {average:.2f}')
    
    #
    for key in data.keys():
        session.save(f'population/tuning/{key}', data[key])

    return

class DirectionSelectivityAnalysis():
    """
    """

    def run(self):
        """
        """

        return

