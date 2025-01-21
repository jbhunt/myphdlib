import numpy as np
import matplotlib.pyplot as plt
from myphdlib.figures.analysis import AnalysisBase

def determineSaccadeDirection(eye, saccadeLabel, manipulationSide):
    """
    """

    
    # Determine the saccade direction in the case of the left eye
    if eye == 'left' and saccadeLabel == -1:
        saccadeDirection = 'ipsi'
    if eye == 'left' and saccadeLabel == +1:
        saccadeDirection = 'contra'
    if eye == 'right' and saccadeLabel == -1:
        saccadeDirection = 'contra'
    if eye == 'right' and saccadeLabel == +1:
        saccadeDirection = 'ipsi'
    
    # Flip the saccade direction in the case of the right eye
    if manipulationSide == 'left':
        pass
    else:
        saccadeDirection = 'ipsi' if saccadeDirection == 'contra' else 'contra'

    return saccadeDirection

class SaccadeTrajectoryAnalysis(AnalysisBase):
    """
    """

    def __init__(self, ukey=None, hdf=None, **kwargs_):
        """
        """

        kwargs = {
            'experiment': ('Muscimol',),
            'animals': ('dreadd1', 'dreadd2', 'dreadd3', 'dreadd4',),
            'tag': 'Satoru',
            'mount': False,
        }
        kwargs.update(kwargs_)

        super().__init__(ukey, hdf, **kwargs)

        return

    def _measureSaccadeTrajectories(
        self,
        ):
        """
        """

        data = list()
        for s in self.sessions:
            saccadeLabels = s.load(f'saccades/predicted/{s.eye}/labels')
            saccadeDirections = [
                determineSaccadeDirection(s.eye, l, s.hemisphere)
                    for l in saccadeLabels
            ]
            columnIndex = 0 if s.eye == 'left' else 2
            eyePosition = s.load('pose/filtered')[:, columnIndex]
            # TODO: Subtract off the mean eye position/find a way to normalize eye position across animals
            saccadeEpochs = s.load(f'saccades/predicted/{s.eye}/epochs')
            saccadeTrajectories = list()
            for f1, f2 in saccadeEpochs:
                p1, p2 = np.interp(np.array([f1, f2]), np.arange(eyePosition.size), eyePosition)
                saccadeTrajectories.append([p1, p2])
            saccadeTrajectories = np.array(saccadeTrajectories)
            ipsiSaccadeMask = np.array([True if d == 'ipsi' else False for d in saccadeDirections])
            contraSaccadeMask = np.invert(ipsiSaccadeMask)
            entry = [
                np.nanmean(saccadeTrajectories[ipsiSaccadeMask, 0]),
                np.nanmean(saccadeTrajectories[contraSaccadeMask, 0]),
                np.nanmean(saccadeTrajectories[ipsiSaccadeMask, 1]),
                np.nanmean(saccadeTrajectories[contraSaccadeMask, 1]),
                float(s.animal.strip('dreadd')),
                0 if s.treatment == 'saline' else 1,
            ]
            data.append(entry)

        return np.array(data)

    def plot(self, data):
        """
        """

        fig, ax = plt.subplots()
        nSal = np.sum(data[:, -1] == 0)
        nMus = data.shape[0] - nSal

        samples = list()
        for treatment, offset in zip([0, 1], [-0.3, +0.3]):
            mask = data[:, -1] == treatment
            for x1, columnIndex in zip([0, 1, 2, 3], [0, 2, 1, 3]):
                if treatment == 0:
                    x2 = np.full(nSal, x1) + offset
                else:
                    x2 = np.full(nMus, x1) + offset
                x3 = np.copy(x2)
                for i, o in zip(range(4), (np.arange(4) - 1.5) / (4 * 3)):
                    x3[data[mask, 4] == (i + 1)] += o
                    ya = data[:, columnIndex][np.logical_and(mask, data[:, 4] == (i + 1))].mean()
                    ax.scatter(x2[0] + o, ya, color='r', s=15, edgecolor='none')
                y = data[:, columnIndex][mask]
                ax.scatter(x3, y, color='k', alpha=0.3, s=15, edgecolor='none')
                # ax.boxplot(y, positions=[x1 + offset,], widths=[0.3,])
                samples.append(y)

        return fig, ax, samples