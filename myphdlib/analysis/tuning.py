import numpy as np
from scipy.signal import find_peaks as findPeaks
from myphdlib.general.toolkit import smooth, psth2

def computeDirectionSelectivityForSingleUnit(
    session,
    unit,
    window=(0, 3),
    binsize=0.02,
    version=1
    ):
    """
    """

    movingBarOrientations = session.load('stimuli/mb/orientation')
    movingBarTimestamps = session.load('stimuli/mb/timestamps')
    vertices = list()
    iterable = np.unique(movingBarOrientations)
    iterable.sort()

    #
    for orientation in iterable:

        #
        trialIndices = np.where(movingBarOrientations == orientation)[0]
        t, M = psth2(
            movingBarTimestamps[trialIndices],
            unit.timestamps,
            window=window,
            binsize=binsize
        )

        # Compute the response amplitude
        theta = np.deg2rad(orientation)

        # Simple spike count
        if version == 1:
            amplitude = M.sum()

        # Peak response amplitude
        elif version == 2:
            fr = M.mean(0) / binsize
            y = smooth(fr, 7)
            peakIndices, peakProps = findPeaks(y)
            largest = np.argmax(y[peakIndices])
            amplitude = y[peakIndices[largest]]

        # F1

        vertex = (
            amplitude * np.cos(theta),
            amplitude * np.sin(theta)
        )
        vertices.append(vertex)

    #
    vertices.append(vertices[0])
    vertices = np.array(vertices)

    # Compute DSI
    amplitudes = np.array([
        np.linalg.norm(vertex - np.array([0, 0]))
            for vertex in vertices
    ])
    if amplitudes.min() == 0:
        dsi = np.nan
    else:
        dsi = round(1 - (amplitudes.min() / amplitudes.max()), 2)

    return vertices, dsi

def computeDirectionSelectivityForPopulation(
    session,
    window=(0, 3),
    binsize=0.02,
    version=1,
    ):
    """
    """

    DSI = list()
    mask = session.load('analysis/typing/visual')
    for unit, flag in zip(session.population, mask):
        if flag:
            vertices, dsi = computeDirectionSelectivityForSingleUnit(
                session,
                unit,
                window,
                binsize,
                version
            )
        else:
            dsi = np.nan
        DSI.append(dsi)
    
    #
    DSI = np.array(DSI)
    session.save('analysis/population/dsi', DSI)

    return