import numpy as np
from myphdlib.general.toolkit import psth2, smooth

def summarizeEvokedActivity(
    session,
    eventTimestampsPath='stimuli/dg/timestamps',
    window=(-0.2, 0.5),
    binsize=0.02,
    baseline=(-0.2, 0),
    subtractBaselineActivity=True,
    minimumSpikeCount=1000,
    ):
    """
    """

    #
    if session.isAutosorted == False:
        raise Exception('Spike-sorting is not complete')
    
    #
    event = session.load(eventTimestampsPath)
    
    #
    R = list()
    for unit in session.population:

        #
        if unit.timestamps.size < minimumSpikeCount:
            continue

        #
        t, M = psth2(
            event,
            unit.timestamps,
            window=window,
            binsize=binsize
        )
        fr = M.mean(0) / binsize

        #
        # if subtractBaselineActivity:
        #     t_, M_ = psth2(
        #         event,
        #         unit.timestamps,
        #         window=baseline,
        #         binsize=binsize
        #     )
        #     bl = np.mean(M_.mean(1) / binsize)
        #    fr -= bl

        #
        # R.append(fr)
        mu, sigma = unit.describe(event=event, window=baseline)
        if sigma != 0:
            z = (fr - mu) / sigma
            R.append(z)

    return np.array(R)

def estimateReceptiveFieldsWithSparseNoise(session, window=(0, 0.2), phase='on'):
    """
    """

    rfs = list()
    fields = session.load(f'stimuli/sn/pre/fields')
    nDots, nRows, nCols = fields.shape

    #
    for unit in session.population:
        stack = np.full([nRows, nCols], 0.0)
        for block in ('pre', 'post'):
            fields = session.load(f'stimuli/sn/{block}/fields')
            if fields is None:
                continue
            if phase == 'on':
                timestamps = session.load(f'stimuli/sn/{block}/timestamps')[0::2]
            elif phase == 'off':
                timestamps = session.load(f'stimuli/sn/{block}/timestamps')[1::2]
                
            for field, timestamp in zip(fields, timestamps):
                i, j = np.where(field != -1)
                t, M = psth2(
                    np.array([timestamp]),
                    unit.timestamps,
                    window=window,
                    binsize=None,
                )
                nSpikes = M.sum()
                stack[i, j] += nSpikes
        rf = stack / 6
        rfs.append(rf)

    return np.array(rfs)