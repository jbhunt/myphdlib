import numpy as np

class SaccadeFrequencyAnalysis():
    """
    """

    def __init__(self):
        """
        """

        self.result = None

        return
    
    def run(self, sessions):
        """
        """

        #
        animals = np.unique([session.animal for session in sessions]).tolist()
        self.result = {
            animal: {
                'saline'  : {'ipsi': list(), 'contra': list()},
                'muscimol': {'ipsi': list(), 'contra': list()},
            }
                for animal in animals
        }

        #
        for session in sessions:
            for direction in ('ipsi', 'contra'):
                if direction == 'ipsi':
                    nSaccades = session.saccadeWaveformsIpsi.shape[0]
                elif direction == 'contra':
                    nSaccades = session.saccadeWaveformsContra.shape[1]
                nFrames = session.missingDataMask[session.eye].size - session.missingDataMask[session.eye].sum()
                duration = nFrames / session.fps
                frequency = nSaccades / duration
                self.result[session.animal][session.treatment][direction].append(frequency)

        return self.result
    
    def visualize(self):
        """
        """

        return