import numpy as np
from myphdlib.general.toolkit import psth2

def estimateReceptiveFieldsWithSparseNoise(session, window=(0, 0.2), phase='on'):
    """
    """

    rfs = list()
    fields = session.load(f'stimuli/sn/pre/fields')
    if fields is None:
        fields = session.load(f'stimuli/sn/post/fields')
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

def ReceptiveFieldMappingAnalysis():
    """
    """

    def run(self):
        """
        """

        return