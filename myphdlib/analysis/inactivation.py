import numpy as np
from matplotlib import pylab as plt
from scipy import stats
from decimal import Decimal

class SaccadeWaveformAnalysis():
    """
    """

    def __init__(self):
        """
        """

        self.result = None

        return
    
    def _estimateSaccadeAmplitudeForSingleSession(
        self,
        session,
        zero=True,
        baselineWindowInSamples=(27, 37),
        troughToPeakIpsi=(36, 46),
        troughToPeakContra=(33, 51),
        returnAverageValue=True
        ):
        """
        """

        samples = {
            'ipsi': list(),
            'contra': list()
        }

        iterable = zip(
            ['ipsi', 'contra'],
            [session.saccadeWaveformsIpsi, session.saccadeWaveformsContra],
            [troughToPeakIpsi, troughToPeakContra]
        )
        for direction, waveforms, ttp in iterable:
            s1, s2 = ttp
            for waveform in waveforms:
                baseline = 0
                if zero:
                    baseline = waveform[baselineWindowInSamples[0]: baselineWindowInSamples[1]].mean()
                zeroed = waveform - baseline
                d2 = Decimal(str(zeroed[s2]))
                d1 = Decimal(str(zeroed[s1]))
                amplitude = abs(float(d2 - d1))
                samples[direction].append(amplitude)

        #
        if returnAverageValue == True:
            return np.mean(samples['ipsi']), np.mean(samples['contra'])
        else:
            return samples
    
    def _estimateSaccadeAmplitudeByAnimal(
        self,
        sessions,
        zero=True,
        baselineWindowInSamples=(27, 37),
        troughToPeakIpsi=(36, 46),
        troughToPeakContra=(33, 51)
        ):
        """
        """

        animals = np.unique([session.animal for session in sessions])
        averages = {
            animal: {
                'ipsi': None,
                'contra': None,
            }
                for animal in animals
        }
        for animal in animals:
            subset = [session for session in sessions if session.animal == animal and session.treatment == 'saline']
            samples = {
                'ipsi': list(),
                'contra': list()
            }
            for session in subset:
                samples_ = self._estimateSaccadeAmplitudeForSingleSession(
                    session,
                    zero,
                    baselineWindowInSamples,
                    troughToPeakIpsi,
                    troughToPeakContra,
                    returnAverageValue=False
                )
                for direction in ('ipsi', 'contra'):
                    for a in samples_[direction]:
                        samples[direction].append(a)

            #
            averages[animal]['ipsi'] = np.mean(samples['ipsi'])
            averages[animal]['contra'] = np.mean(samples['contra'])

        return averages

    def run(
        self,
        sessions,
        zero=True,
        baselineWindowInSamples=(27, 37),
        troughToPeakIpsi=(36, 46),
        troughToPeakContra=(33, 51)
        ):
        """
        """

        animals = np.unique([session.animal for session in sessions])
        self.result = {
            animal: {
                condition: {
                    'ipsi': list(),
                    'contra': list()
                }
                    for condition in ('a', 'b')
            }
                for animal in animals
        }

        #
        averages = self._estimateSaccadeAmplitudeByAnimal(
            sessions,
            zero,
            baselineWindowInSamples,
            troughToPeakIpsi,
            troughToPeakContra,
        )

        #
        for session in sessions:
            condition = 'a' if session.treatment == 'saline' else 'b'
            for direction, waveforms in zip(['ipsi', 'contra'], [session.saccadeWaveformsIpsi, session.saccadeWaveformsContra]):
                for waveform in waveforms:
                    baseline = 0
                    if zero:
                        baseline = waveform[baselineWindowInSamples[0]: baselineWindowInSamples[1]].mean()
                    zeroed = waveform - baseline
                    normed = zeroed / averages[session.animal][direction]
                    self.result[session.animal][condition][direction].append(normed)

        return self.result
    
    def visualize(self, animal=None):
        """
        """

        if animal is None:
            nAnimals = len(list(self.result.keys()))
            animals = list(self.result.keys())
        else:
            nAnimals = 1
            animals = [animal]
        fig, axs = plt.subplots(nrows=nAnimals, ncols=2, sharey=True)
        axs = np.atleast_2d(axs)

        for iRow, animal in enumerate(animals):
            for iCol, direction, c in zip([0, 1], ['ipsi', 'contra'], ['b', 'r']):

                #
                ax = axs[iRow, iCol]

                # Normalize to the peak saccade amplitude for saline sessions
                waveforms = np.array(self.result[animal]['a'][direction])
                maximumSaccadeAmplitude = np.abs(waveforms.mean(0)).max()

                #
                for color, linestyle, condition in zip(['k', 'k'], ['-', '--'], ['a', 'b']):
                    waveforms = np.array(self.result[animal][condition][direction])
                    print(condition, direction, waveforms.shape[0])
                    y = np.arange(waveforms.shape[1])
                    x = waveforms.mean(0) / maximumSaccadeAmplitude * -1
                    e = stats.sem(waveforms, axis=0)
                    ax.plot(x, y, color=color, linestyle=linestyle)
                    ax.fill_betweenx(y, x - e, x + e, color=color, alpha=0.1)

                #
                if iCol == 0:
                    ax.set_xlim([-1.3, 0.15])
                else:
                    ax.set_xlim([-0.15, 1.3])

        #
        axs[0, 0].set_title('Ipsi')
        axs[0, 1].set_title('Contra')
        axs[-1, 0].set_ylabel('Time (ms)')
        axs[-1, 0].set_xlabel('Eye position (normalized)')

        #
        for ax in axs[:, 0]:
            ax.set_yticks(np.arange(0, 80 + 20, 20))
            ax.set_yticklabels(np.array(np.arange(0, 80 + 20, 20) / 200 * 1000).astype(int))

        return
    
class SaccadeAmplitudeAnalysis():
    """
    """

    def __init__(self):
        """
        """

        self.result = None

        return
    
    def _organizeSessionsIntoPairs(self, sessions):
        """
        """

        animals = np.unique([session.animal for session in sessions])
        pairs = {
            animal: list()
                for animal in animals
        }

        for animal in animals:

            #
            B = [session for session in sessions if session.animal == animal and session.treatment != 'saline']
            A = [session for session in sessions if session.animal == animal and session.treatment == 'saline']

            #
            for b in B:

                # Look for the A session
                offsets = np.array([
                    abs(b.date - a.date)
                        for a in A
                            if a.date < b.date
                ])
                index = np.argmin(offsets)
                a = A[index]

                #
                pair = (a, b)
                pairs[animal].append(pair)

        return pairs
    
    def _estimateSaccadeAmplitudeForSingleSession(
        self,
        session,
        zero=True,
        baselineWindowInSamples=(27, 37),
        troughToPeakIpsi=(36, 46),
        troughToPeakContra=(33, 51),
        returnAverageValue=True
        ):
        """
        """

        samples = {
            'ipsi': list(),
            'contra': list()
        }

        iterable = zip(
            ['ipsi', 'contra'],
            [session.saccadeWaveformsIpsi, session.saccadeWaveformsContra],
            [troughToPeakIpsi, troughToPeakContra]
        )
        for direction, waveforms, ttp in iterable:
            s1, s2 = ttp
            for waveform in waveforms:
                baseline = 0
                if zero:
                    baseline = waveform[baselineWindowInSamples[0]: baselineWindowInSamples[1]].mean()
                zeroed = waveform - baseline
                d2 = Decimal(str(zeroed[s2]))
                d1 = Decimal(str(zeroed[s1]))
                amplitude = abs(float(d2 - d1))
                samples[direction].append(amplitude)

        #
        if returnAverageValue == True:
            return np.mean(samples['ipsi']), np.mean(samples['contra'])
        else:
            return samples
    
    def _estimateSaccadeAmplitudeByAnimal(
        self,
        sessions,
        zero=True,
        baselineWindowInSamples=(27, 37),
        troughToPeakIpsi=(36, 46),
        troughToPeakContra=(33, 51)
        ):
        """
        """

        animals = np.unique([session.animal for session in sessions])
        averages = {
            animal: {
                'ipsi': None,
                'contra': None,
            }
                for animal in animals
        }
        for animal in animals:
            subset = [session for session in sessions if session.animal == animal and session.treatment == 'saline']
            samples = {
                'ipsi': list(),
                'contra': list()
            }
            for session in subset:
                samples_ = self._estimateSaccadeAmplitudeForSingleSession(
                    session,
                    zero,
                    baselineWindowInSamples,
                    troughToPeakIpsi,
                    troughToPeakContra,
                    returnAverageValue=False
                )
                for direction in ('ipsi', 'contra'):
                    for a in samples_[direction]:
                        samples[direction].append(a)

            #
            averages[animal]['ipsi'] = np.mean(samples['ipsi'])
            averages[animal]['contra'] = np.mean(samples['contra'])

        return averages

    def run(
        self,
        sessions,
        zero=True,
        baselineWindowInSamples=(27, 37),
        troughToPeakIpsi=(36, 46),
        troughToPeakContra=(33, 51)
        ):
        """
        """

        animals = np.unique([session.animal for session in sessions])
        self.result = {
            animal: {'ipsi': list(), 'contra': list()}
                for animal in animals
        }

        pairs = self._organizeSessionsIntoPairs(sessions)
        averages = self._estimateSaccadeAmplitudeByAnimal(
            sessions,
            zero,
            baselineWindowInSamples,
            troughToPeakIpsi,
            troughToPeakContra,
        )

        #
        for animal in pairs.keys():

            #
            for A, B in pairs[animal]:
                print(A.date, B.date)
                print(A.treatment, B.treatment)
                aia, aca = self._estimateSaccadeAmplitudeForSingleSession(
                    A,
                    zero,
                    baselineWindowInSamples,
                    troughToPeakIpsi,
                    troughToPeakContra
                )
                aib, acb = self._estimateSaccadeAmplitudeForSingleSession(
                    B,
                    zero,
                    baselineWindowInSamples,
                    troughToPeakIpsi,
                    troughToPeakContra
                )
                xyi = np.array([aia, aib]) / averages[animal]['ipsi']
                xyc = np.array([aca, acb]) / averages[animal]['contra']
                self.result[animal]['ipsi'].append(xyi)
                self.result[animal]['contra'].append(xyc)

        #
        for animal in self.result.keys():
            for direction in ('ipsi', 'contra'):
                self.result[animal][direction] = np.array(self.result[animal][direction])

        return self.result
    
    def pairedplot(self, ylim=(0, 2), width=0.35):
        """
        """

        fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True)
        colors = [f'C{i}' for i in range(len(self.result.keys()))]
        samples = {
            'ipsi': list(),
            'contra': list()
        }
        for animal, color in zip(self.result.keys(), colors):
            for direction, ax in zip(['ipsi', 'contra'], [ax1, ax2]):
                pairs = self.result[animal][direction]
                for pair in pairs:
                    samples[direction].append(pair)
                    ax.plot([0, 1], pair, color=color, alpha=0.3)
                    ax.scatter([0, 1], pair, color=color, alpha=0.3, marker='.', s=30)

        #
        samples['ipsi'] = np.array(samples['ipsi'])
        samples['contra'] = np.array(samples['contra'])

        for direction, title, ax in zip(['ipsi', 'contra'], ['Ipsi', 'Contra'], [ax1, ax2]):
            sample = samples[direction]
            for iColumn, x in enumerate(range(2)):
                ax.boxplot(
                    sample[:, iColumn],
                    positions=[iColumn],
                    widths=[width],
                    showfliers=False,
                    medianprops={'color': 'k', 'lw': 1.5},
                    boxprops={'color': 'k', 'lw': 1.5},
                    whiskerprops={'color': 'k', 'lw': 1.5},
                    capprops={'color': 'k', 'lw': 1.5}
                )
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['S', 'M'])
            ax.set_title(title)
            ax.set_ylim(ylim)

        ax1.set_ylabel('Saccade amplitude (normalized)')

        return
    
    def scatterplot(self, figsize=(6, 3)):
        """
        """

        fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True)
        colors = [f'C{i}' for i in range(len(self.result.keys()))]
        for animal, color in zip(self.result.keys(), colors):
            samples = {
                'ipsi': list(),
                'contra': list()
            }
            for direction, ax in zip(['ipsi', 'contra'], [ax1, ax2]):
                coords = self.result[animal][direction]
                for coord in coords:
                    samples[direction].append(coord)
                ax.scatter(coords[:, 0], coords[:, 1], color=color, alpha=0.5, marker='.', s=50)

            # for direction, ax in zip(['ipsi', 'contra'], [ax1, ax2]):
            #     sample = np.array(samples[direction])
            #     c = sample.mean(0)
            #     ax.scatter(c[0], c[1], color=color, marker='^', s=50)

        for ax in (ax1, ax2):
            ax.set_xlim([0, 1.7])
            ax.set_ylim([0, 1.7])
            ax.plot([0, 1.7], [0, 1.7], color='k', alpha=0.7)
            ax.set_aspect('equal')
            ax.set_xlabel('Saccade amplitude (saline)')
        ax1.set_ylabel('Saccade amplitude (muscimol)')
        ax1.set_title('Ipsi')
        ax2.set_title('Contra')
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, (ax1, ax2)
    
    def test(self):
        """
        """

        samples = {
            'ipsi': {
                'A': list(),
                'B': list()
            },
            'contra': {
                'A': list(),
                'B': list()
            }
        }

        #
        for animal in self.result.keys():
            for direction in ('ipsi', 'contra'):
                for a in self.result[animal][direction][:, 0]:
                    samples[direction]['A'].append(a)
                for a in self.result[animal][direction][:, 1]:
                    samples[direction]['B'].append(a)

        #
        result = {
            'ipsi': None,
            'contra': None,
        }
        for direction in result.keys():
            n = len(samples[direction]['A'])
            t, p = stats.ttest_rel(
                samples[direction]['A'],
                samples[direction]['B']
            )
            result[direction] = (t, p, n)

        return result

# TODO: Implement this method
def _measureSaccadeFrequency(session):
    """
    Measure saccade frequency using a standard procedure
    """

    return

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
                    nSaccades = session.saccadeWaveformsContra.shape[0]
                nFrames = session.missingDataMask[session.eye].size - session.missingDataMask[session.eye].sum()
                duration = nFrames / session.fps
                frequency = round(nSaccades / duration, 3)
                self.result[session.animal][session.treatment][direction].append(frequency)

        return self.result
    
    def visualize(
        self,
        factor=0.05,
        colors=['r', 'b', 'xkcd:purple', 'xkcd:green'],
        figsize=(3.5, 4)
        ):
        """
        """

        #
        fig, (ax1, ax2) = plt.subplots(ncols=2, sharex=True, sharey=True)

        #
        nAnimals = len(self.result.keys())
        if len(colors) != nAnimals:
            colors = ['C' + str(i) for i in range(nAnimals)]
        offsets = (np.linspace(0, nAnimals, nAnimals) - (nAnimals / 2)) * factor
        for animal, color, offset in zip(self.result.keys(), colors, offsets):
            for direction, ax in zip(['ipsi', 'contra'], [ax1, ax2]):

                #
                data = {
                    'saline': {
                        'x': list(),
                        'y': list()
                    }, 
                    'muscimol': {
                        'x': list(),
                        'y': list()
                    }
                }

                #
                for x, treatment in zip([0, 1], ['saline', 'muscimol']):
                    frequency = self.result[animal][treatment][direction]
                    nPoints = len(frequency)
                    for iPoint in range(nPoints):
                        data[treatment]['x'].append(x)
                        data[treatment]['y'].append(frequency[iPoint])

                #
                for treatment, color_ in zip(['saline', 'muscimol'], ['k', 'r']):
                    x = np.array(data[treatment]['x']) + offset
                    ax.scatter(x, data[treatment]['y'], color=color, alpha=0.5, marker='.', s=25)

                #
                y = (
                    np.nanmean(data['saline']['y']),
                    np.nanmean(data['muscimol']['y'])
                )
                ax.plot([0, 1], y, color=color)

        #
        for ax, title in zip([ax1, ax2], ['Ipsi', 'Contra']):
            ax.set_xlim([-0.5, 1.5])
            ax.set_title(title)
            ax.set_xticks([0, 1])
            ax.set_xticklabels(['S', 'M'])
        ax1.set_ylabel('Frequency (saccades/sec)')
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, (ax1, ax2)