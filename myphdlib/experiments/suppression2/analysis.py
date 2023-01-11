import pickle
import numpy as np
from zetapy import getZeta
from decimal import Decimal
from matplotlib import pylab as plt
from joblib import delayed, Parallel
from scipy.ndimage import gaussian_filter as gaussianFilter
from myphdlib.general.toolkit import psth, psth2, smooth, DotDict
from myphdlib.general.ephys import SpikeSortingResults

def computeSRF(
    neuronObject, 
    sessionObject,
    binSize=0.01,
    gridShape=(10, 18),
    blur=True,
    kernelSpread=0.8,
    standardize=False,
    ):
    """
    Estimate the spatial receptive field from to the sparse noise stimulus
    """

    #
    data = sessionObject.load('visualStimuliData')
    stimulusTimestamps = np.hstack([
        data['sparseNoise']['t1'].reshape(-1, 1),
        data['sparseNoise']['t2'].reshape(-1, 1)
    ])
    spatialCoordinates = np.hstack([
        data['sparseNoise']['x'].reshape(-1, 1),
        data['sparseNoise']['y'].reshape(-1, 1)
    ])
    uniqueGridPoints = np.unique(spatialCoordinates, axis=0)

    # Left-to-right, top-to-bottom
    sortedIndex = np.lexsort([
        uniqueGridPoints[:, 0],
        uniqueGridPoints[:, 1][::-1],
    ])

    #
    resultMatrix = np.empty([gridShape[0], gridShape[1], 2])
    resultIndices = np.array([
        [i, j] for (i, j), v in np.ndenumerate(np.empty(gridShape))
    ])

    #
    if standardize:
        mu, sigma = neuronObject.baselineFiringRate

    #
    for (x, y), (i, j) in zip(uniqueGridPoints[sortedIndex, :], resultIndices):
        rowIndices = np.where([np.array_equal(row, np.array([x, y])) for row in spatialCoordinates])[0]
        stimulusOnsetTimestamps = stimulusTimestamps[rowIndices, 0]
        stimulusOffsetTimestamps = stimulusTimestamps[rowIndices, 1]
        edges, M = psth(stimulusOnsetTimestamps, neuronObject.timestamps)
        averageSpikeCount = M[:, 50:70].sum(1).mean()
        averageSpikeRate = averageSpikeCount / 0.2
        if standardize:
            averageSpikeRate = (averageSpikeRate - mu) / sigma
        resultMatrix[i, j, 0] = averageSpikeRate
        edges, M = psth(stimulusOffsetTimestamps, neuronObject.timestamps)
        averageSpikeCount = M[:, 50:100].sum(1).mean()
        averageSpikeRate = averageSpikeCount / 0.2
        if standardize:
            averageSpikeRate = (averageSpikeRate - mu) / sigma
        resultMatrix[i, j, 1] = averageSpikeRate

    if blur:
        resultMatrix = gaussianFilter(resultMatrix, kernelSpread)

    return resultMatrix

def computeSTA(sessionObject, neuronObject, timeWindow=(-1, 0.5), samplingRate=1000):
    """
    Compute a spike-triggered average of the noisy grating stimulus
    """

    data = sessionObject.load('visualStimuliData')['ng']
    trials = list()

    for timestamps, contrast in zip(data['t'], data['s']):

        #
        nSamples = round((timestamps.max() - timestamps.min()) * 1000)
        resampled = np.linspace(
            timestamps.min() * samplingRate,
            timestamps.max() * samplingRate,
            nSamples
        ) / samplingRate

        #
        stimulus = np.empty(resampled.size)
        s1 = 0
        for c, dt in zip(contrast[1:], np.diff(timestamps)):
            ds = round(dt * 1000)
            stimulus[s1: s1 + ds] = c
            s1 = s1 + ds

        #
        spikingMask = np.logical_and(
            neuronObject.timestamps > timestamps.min(),
            neuronObject.timestamps < timestamps.max()
        )

        #
        for t1 in neuronObject.timestamps[spikingMask]:
            relative = resampled - t1
            closest = np.argmin(np.abs(relative))
            start = closest + round(timeWindow[0] * samplingRate)
            stop = closest + round(timeWindow[1] * samplingRate)
            if start < 0:
                right = stimulus[:stop]
                left = np.full(abs(start), np.nan)
                trial = np.concatenate([left, right])
            elif stop > stimulus.size:
                left = stimulus[start:]
                right = np.full(stop - stimulus.size, np.nan)
                trial = np.concatenate([left, right])
            else:
                trial = stimulus[start: stop]

            trials.append(trial)

    return np.array(trials)

class SpikeTriggeredAverageAnalysis():
    """
    Computes a spike-triggered average during the presentation of the noisy grating stimulus
    """

    def __init__(self):
        self.result = None
        self.metadata = None
        return

    def _createStimulusTimeseries(self, session):
        """
        """

        return

    def run(
        self,
        factory
        ):

        return

class ProbeTimingAnalysis():
    """
    Counts the frequency of extra-/peri-saccadic probes by motion of the grating across sessions
    """

    def __init__(self):
        """
        """

        self.result = None
        self.metadata = None

        return

    def run(
        self,
        factory,
        experiment='saline',
        targetEye='left',
        perisaccadicTimeWindow=(-0.05, 0.1),
        histogramTimeWindow=(-3, 3),
        histogramBinSize=0.1,
        ):
        """
        """

        result = {
            'histograms': {
                'contra': list(),
                'ipsi': list()
            },
            'counts': {
                'perisaccadic': {
                    'contra': list(),
                    'ipsi': list()
                },
                'extrasaccadic': {
                    'contra': list(),
                    'ipsi': list()
                },
            }
        }

        #
        for session in factory:

            #
            if experiment is not None and session.experiment != experiment:
                continue

            #
            saccadeOnsetTimestamps = session.load('saccadeOnsetTimestamps')
            probeLatencySample = list()
            perisaccadicProbeCount = 0
            extrasaccadicProbeCount = 1

            # 
            for direction1 in ('nasal', 'temporal'):

                # Determine whether saccades are ipsi or contra
                if (targetEye == 'left' and direction1 == 'nasal') or (targetEye == 'right' and direction1 == 'temporal'):
                    direction2 = 'ipsi'
                else:
                    direction2 = 'contra'


                # Iterate through all probe presentations
                for probeOnsetTimestamp in session.probeOnsetTimestamps:

                    # Find the closest saccade in time
                    relative = probeOnsetTimestamp - saccadeOnsetTimestamps[targetEye][direction1]
                    mask = np.logical_and(
                        relative >= histogramTimeWindow[0],
                        relative <= histogramTimeWindow[1]
                    )
                    for latency in relative[mask]:
                        probeLatencySample.append(latency)
                    closest = np.argmin(np.abs(relative))

                    # Measure the difference in time
                    latency = probeOnsetTimestamp - saccadeOnsetTimestamps[targetEye][direction1][closest]

                    # Mark probe as peri- or extra-saccadic
                    if latency >= perisaccadicTimeWindow[0] and latency <= perisaccadicTimeWindow[1]:
                        perisaccadicProbeCount += 1
                    else:
                        extrasaccadicProbeCount += 1

                # Compute the peri-saccadic time histogram
                nBins = round(np.diff(histogramTimeWindow).item() / histogramBinSize)
                binEdges = np.linspace(histogramTimeWindow[0], histogramTimeWindow[1], nBins + 1)
                binCounts, binEdges = np.histogram(probeLatencySample, binEdges, range=histogramTimeWindow)

                #
                result['histograms'][direction2].append(binCounts)
                result['counts']['perisaccadic'][direction2].append(perisaccadicProbeCount)
                result['counts']['extrasaccadic'][direction2].append(extrasaccadicProbeCount)

        #
        for direction in ('ipsi', 'contra'):
            result['histograms'][direction] = np.array(result['histograms'][direction])

        for trialType in ('perisaccadic', 'extrasaccadic'):
            for direction in ('ipsi', 'contra'):
                result['counts'][trialType][direction] = np.array(result['counts'][trialType][direction])
        self.result = result

        return self.result

class PerisaccadicModulationAnalysis():
    """
    """

    def __init__(self):
        """
        """

        self.result = None
        self.metadata = None

        return

    def _initializeEmptyContainer(self):
        """
        """

        data = {
            'E': {
                's': {
                    'i': None,
                    'c': None,
                },
                'p': {
                    'i': {
                        'e': None,
                        'p': None
                    },
                    'c': {
                        'e': None,
                        'p': None
                    }
                }
            },
            'B': {
                's': {
                    'i': None,
                    'c': None,
                },
                'p': {
                    'i': {
                        'e': None,
                        'p': None,
                    },
                    'c': {
                        'e': None,
                        'p': None
                    }
                },
            },
            'P': {
                's': {
                    'i': None,
                    'c': None,
                },
                'p': {
                    'i': {
                        'e': None,
                        'p': None,
                    },
                    'c': {
                        'e': None,
                        'p': None
                    }
                },
            },
        }

        return data

    def _updateKeywordArguments(self, dct, session):
        """
        """

        kwargs = {
                'binsize': 0.005,
                'version': 1,
                'energyThreshold': None,
                'runZetaTest': False,
                'probeParsingResults': None,
                'visualResponseWindow': (0, 0.3),
                'saccadeOnsetTimestamps': None,
                'baselineResponseWindows': {
                    'stimuli': (-0.2, 0),
                    'saccades': (-0.3, -0.1),
                },
                'integrateVisuomotorResponses': False
            }
        kwargs.update(dct)
        if kwargs['probeParsingResults'] is None:
            kwargs['probeParsingResults'] = session.parseVisualProbes()
        if kwargs['saccadeOnsetTimestamps'] is None:
            kwargs['saccadeOnsetTimestamps'] = session.saccadeOnsetTimestamps()

        return kwargs

    def _estimateBaselineActivity(
        self,
        data,
        neuron,
        **kwargs
        ):
        """
        """

        # Compute the baseline level of activity prior to saccades
        for direction, kd in zip(['ipsi', 'contra'], ['i', 'c']):
            t, M = psth2(
                kwargs['saccadeOnsetTimestamps'][direction],
                neuron.timestamps,
                window=kwargs['baselineResponseWindows']['saccades'],
                binsize=kwargs['binsize']
            )
            data['B']['s'][kd] = round(np.mean(M.mean(1) / kwargs['binsize']), 3)

        # Compute the baseline level of activity prior to probes
        for motion, km in zip(['ipsi', 'contra'], ['i', 'c']):

            #
            for category, kc in zip(['extrasaccadic', 'perisaccadic'], ['e', 'p']):

                # Check to make sure there is at least one trial
                probeOnsetTimestamps = kwargs['probeParsingResults'][category][motion]['timestamps']
                nTrials = probeOnsetTimestamps.size
                if nTrials == 0:
                    data['B']['p'][km][kc] = np.nan
                    continue

                #
                t, M = psth2(
                    probeOnsetTimestamps,
                    neuron.timestamps,
                    window=kwargs['baselineResponseWindows']['stimuli'],
                    binsize=kwargs['binsize']
                )
                data['B']['p'][km][kc] = round(np.mean(M.mean(1) / kwargs['binsize']), 3)

        return data

    def _estimateEvokedActivity(
        self,
        data,
        neuron,
        nResamples=30,
        **kwargs
        ):
        """
        """

        # Iterate through each direction of motion (ipsi/contra)
        for motion, km in zip(['ipsi', 'contra'], ['i', 'c']):

            #
            nTrialsPerisaccadic = kwargs['probeParsingResults']['perisaccadic'][motion]['timestamps'].size
            nTrialsExtrasaccadic = kwargs['probeParsingResults']['extrasaccadic'][motion]['timestamps'].size
            if nTrialsPerisaccadic == 0:
                data['E']['p'][km]['p'] = np.nan
                data['E']['p'][km]['e'] = np.nan
                continue

            # Compute the average response energy for perisaccadic trials

            # Single-trial estimate
            if kwargs['version'] == 1:
                sample = np.full(nTrialsPerisaccadic, np.nan)
                for iTrial in range(nTrialsPerisaccadic):
                    probeOnsetTimestamp = np.array([
                        kwargs['probeParsingResults']['perisaccadic'][motion]['timestamps'][iTrial]
                    ])
                    t, M = psth2(
                        probeOnsetTimestamp,
                        neuron.timestamps,
                        window=kwargs['visualResponseWindow'],
                        binsize=kwargs['binsize']
                    )

                    #
                    fr = M.mean(0) / kwargs['binsize']
                    bl = data['B']['p'][km]['p']
                    sample[iTrial] = np.sqrt(np.sum(np.power(fr - bl, 2)))

                #
                data['E']['p'][km]['p'] = round(sample.mean(), 3)

            # Trial-averaged estimate
            else:
                t, M = psth2(
                    kwargs['probeParsingResults']['perisaccadic'][motion]['timestamps'],
                    neuron.timestamps,
                    window=kwargs['visualResponseWindow'],
                    binsize=kwargs['binsize']
                )

                #
                fr = M.mean(0) / kwargs['binsize']
                bl = data['B']['p'][km]['p']
                data['E']['p'][km]['p'] = round(np.sqrt(np.sum(np.power(fr - bl, 2))), 3)

            # Compute the average response energy for extrasaccadic trials

            # Single trial estimate
            if kwargs['version'] == 1:
                trialIndexSets = [np.array([iTrial]) for iTrial in range(nTrialsExtrasaccadic)]

            # Trial-averaged estimate
            else:
                trialIndexSets = list()
                for iSet in range(nResamples):
                    trialIndices = np.random.choice(
                        np.arange(0, nTrialsExtrasaccadic, 1),
                        size=nTrialsPerisaccadic
                    )
                    trialIndexSets.append(trialIndices)

            #
            sample = list()
            for trialIndices in trialIndexSets:
                t, M = psth2(
                    kwargs['probeParsingResults']['extrasaccadic'][motion]['timestamps'][trialIndices],
                    neuron.timestamps,
                    window=kwargs['visualResponseWindow'],
                    binsize=kwargs['binsize']
                )
                fr = M.mean(0) / kwargs['binsize']
                bl = data['B']['p'][km]['e']
                sample.append(np.sqrt(np.sum(np.power(fr - bl, 2))))

            #
            data['E']['p'][km]['e'] = round(np.mean(sample), 3)

        return data

    def _runZetaTest(self, data, neuron, **kwargs):
        """
        """

        for motion, km in zip(['ipsi', 'contra'], ['i', 'c']):
            probeOnsetTimestamps = kwargs['probeParsingResults']['extrasaccadic'][motion]['timestamps']
            p, t = getZeta(
                neuron.timestamps,
                probeOnsetTimestamps,
                tplRestrictRange=(0, 0.3)
            )
            data['P']['p'][km]['e'] = round(p, 3)

        return data

    def _estimateSaccadeComponent(
        self,
        data,
        neuron,
        latency=None,
        direction='ipsi',
        **kwargs
        ):
        """
        Estimate the time-shifted, single-trial saccade-related energy
        """

        nTrials = kwargs['saccadeOnsetTimestamps'][direction].size
        sample = np.full(nTrials, np.nan)
        for iTrial in range(nTrials):
            saccadeOnsetTimestamp = np.array([
                kwargs['saccadeOnsetTimestamps'][direction][iTrial]
            ])
            t, M = psth2(
                saccadeOnsetTimestamp,
                neuron.timestamps,
                window=kwargs['visualResponseWindow'] + latency,
                binsize=kwargs['binsize']
            )
            fr = M.flatten() / kwargs['binsize']
            bl = data['B']['s']['i' if direction == 'ipsi' else 'c']
            sample[iTrial] = np.sqrt(np.sum(np.power(fr - bl, 2)))

        return sample.mean()

    def _saveBatchAnaysisResults(self, batchResults, session):
        """
        """

        #
        if 'modulationAnalysisResults' in session.keys:
            modulationAnalysisResults = session.load('modulationAnalysisResults')
        else:
            modulationAnalysisResults = dict()


        # Iterate over the results for each unit
        for unitIndex, (animal, date, cluster) in enumerate(batchResults['metadata']):

            #
            if cluster in modulationAnalysisResults.keys():
                singleNeuronResults = modulationAnalysisResults[cluster]
            else:
                singleNeuronResults = {
                    'e': {'i': np.nan, 'c': np.nan},
                    'm': {'i': np.nan, 'c': np.nan},
                    'p': {'i': np.nan, 'c': np.nan},
                }

            #
            for k in ('e', 'm', 'p'):
                for motion, km in zip(['ipsi', 'contra'], ['i', 'c']):
                    value = batchResults[k][km][unitIndex]
                    if np.isnan(value):
                        continue
                    else:
                        singleNeuronResults[k][km] = value

            # Update the single neuron dictionary
            modulationAnalysisResults[cluster] = singleNeuronResults

        #
        session.save('modulationAnalysisResults', modulationAnalysisResults)

        return

    def runUnitAnalysis(
        self,
        neuron,
        session,
        **kwargs
        ):
        """
        """

        #
        output = {
            'm': {
                'i': np.nan,
                'c': np.nan,
            },
            'e': {
                'i': np.nan,
                'c': np.nan,
            },
            'p': {
                'i': np.nan,
                'c': np.nan,
            },
            'metadata': (
                session.animal,
                session.date,
                neuron.cluster
            )
        }

        # 
        kwargs = self._updateKeywordArguments(kwargs, session)

        # Populate container
        data = self._initializeEmptyContainer()
        data = self._estimateBaselineActivity(data, neuron, **kwargs)
        data = self._estimateEvokedActivity(data, neuron, **kwargs)
        if kwargs['runZetaTest']:
            data = self._runZetaTest(data, neuron, **kwargs)

        # Save the visual response energy and p-value from the ZETA test
        for km in ('i', 'c'):
            output['e'][km] = data['E']['p'][km]['e']
            output['p'][km] = data['P']['p'][km]['e']

        # Trial-by-trial analysis
        if kwargs['version'] == 1:
            for motion, km in zip(['ipsi', 'contra'], ['i', 'c']):

                # Skip if there is a zero-energy extra-saccadic response
                if np.isnan(data['E']['p'][km]['p']):
                    output['m'][km] = np.nan
                    continue

                #
                for category, kc in zip(['extrasaccadic', 'perisaccadic'], ['e', 'p']):

                    # TODO: Figure out if I want to analyze extra-saccadic trials
                    if category == 'extrasaccadic':
                        continue

                    #
                    nTrials = kwargs['probeParsingResults'][category][motion]['timestamps'].size
                    sample = np.full(nTrials, np.nan)
                    iterable = zip(
                        kwargs['probeParsingResults'][category][motion]['timestamps'],
                        kwargs['probeParsingResults'][category][motion]['directions'],
                        kwargs['probeParsingResults'][category][motion]['latencies'],
                    )

                    #
                    for ti, (timestamp, direction, latency) in enumerate(iterable):

                        # Compute the actual response
                        t, M = psth2(
                            np.array([timestamp]),
                            neuron.timestamps,
                            window=kwargs['visualResponseWindow'],
                            binsize=kwargs['binsize']
                        )
                        fr = M.flatten() / kwargs['binsize']
                        bl = data['B']['p'][km][kc]
                        actual = np.sqrt(np.sum(np.power(fr - bl, 2)))

                        # Estimate the expected response
                        expected = data['E']['p'][km]['e']
                        if kwargs['integrateVisuomotorResponses']:
                            Rs = self._estimateSaccadeComponent(data, neuron, latency, direction, **kwargs)
                            expected += Rs

                        # Compute the modulation
                        numerator = float(
                            round(Decimal(str(actual)) - Decimal(str(expected)), 3)
                        )
                        # TODO: Figure out why I'm getting warnings in the line below this comment
                        sample[ti] = round(numerator / expected, 3)

                    # NOTE: If I end up analyzing extra-saccadic trials,
                    # this needs to change
                    output['m'][km] = round(np.nanmean(sample), 3)
    
        # Mean response analysis
        # TODO: Add average time-shifted peri-saccadic time histogram to the expected response
        elif kwargs['version'] == 2:
            for motion, km in zip(['ipsi', 'contra'], ['i', 'c']):

                # This will be True if there were no peri-saccadic trials
                if np.isnan(data['E']['p'][km]['p']):
                    output['m'][km] = np.nan
                    continue

                # Compute the modulation index
                expected = data['E']['p'][km]['e'] # Extra-saccadic response
                actual = data['E']['p'][km]['p'] # Peri-saccadic response
                if expected == 0:
                    output['m'][km] = np.nan
                else:
                    numerator = float(
                        Decimal(str(actual)) - Decimal(str(expected))
                    )
                    output['m'][km] = round(numerator / expected, 3)

        return output

    def runBatchAnalysis(
        self,
        neurons,
        session,
        **kwargs
        ):
        """
        """

        nUnits = len(neurons)
        batchResults = {
            'm': {
                'i': np.full(nUnits, np.nan),
                'c': np.full(nUnits, np.nan),
            },
            'e': {
                'i': np.full(nUnits, np.nan),
                'c': np.full(nUnits, np.nan),
            },
            'p': {
                'i': np.full(nUnits, np.nan),
                'c': np.full(nUnits, np.nan),
            },
            'metadata': list()
        }

        for ui, neuron in enumerate(neurons):
            output = self.runUnitAnalysis(neuron, session, **kwargs)
            for metric in ('m', 'e', 'p'):
                for motion in ('i', 'c'):
                    batchResults[metric][motion][ui] = output[metric][motion]
            batchResults['metadata'].append(output['metadata'])

        return batchResults

    def run(
        self,
        factory,
        binsize=0.005,
        energyThreshold=None,
        visualResponseWindow=(0, 0.3),
        baselineResponseWindows={
            'stimuli': (-0.2, 0),
            'saccades': (-0.3, -0.1)
        },
        minimumSpikeCount=3000,
        version=1,
        batchsize=3,
        runZetaTest=False,
        integrateVisuomotorResponses=False,
        ):
        """
        """

        result = {
            'saline': {
                'ipsi': {
                    'e': list(),
                    'm': list(),
                    'p': list()
                },
                'contra': {
                    'e': list(),
                    'm': list(),
                    'p': list()
                },
            },
            'cno': {
                'ipsi': {
                    'e': list(),
                    'm': list(),
                    'p': list()
                },
                'contra': {
                    'e': list(),
                    'm': list(),
                    'p': list()
                },
            },
        }

        #
        metadata = {
            'saline': list(),
            'cno': list()
        }

        #
        for experiment in ('saline', 'cno'):

            #
            for session in factory:

                #
                if session.experiment != experiment:
                    continue

                # Determine which trials are peri-/extra-saccadic
                probeParsingResults = session.parseVisualProbes()
                saccadeOnsetTimestamps = session.saccadeOnsetTimestamps()

                #
                kwargs = {
                    'binsize': binsize,
                    'version': version,
                    'runZetaTest': runZetaTest,
                    'energyThreshold': energyThreshold,
                    'probeParsingResults': probeParsingResults,
                    'visualResponseWindow': visualResponseWindow,
                    'saccadeOnsetTimestamps': saccadeOnsetTimestamps,
                    'baselineResponseWindows': baselineResponseWindows,
                    'integrateVisuomotorResponses': integrateVisuomotorResponses

                }
                
                #
                neurons = [
                    neuron for neuron in session.rez
                        if neuron.timestamps.size >= minimumSpikeCount
                ]
                #
                print(f'INFO[animal={session.animal}, date={session.date}]: Estimating perisaccadic modulation of visual responses for {len(neurons)} neurons')
                batches = [
                    neurons[i: i + batchsize]
                    for i in range(0, len(neurons), batchsize)
                ]
                allBatchResults = Parallel(n_jobs=-1)(
                    delayed(self.runBatchAnalysis)(batch, session, **kwargs)
                        for batch in batches
                )
                for batchResults in allBatchResults:

                    #
                    for k in ('e', 'm', 'p'):
                        for motion, km in zip(['ipsi', 'contra'], ['i', 'c']):
                            nUnits = len(batchResults['m'][km])
                            for iUnit in range(nUnits):
                                m = batchResults[k][km][iUnit]
                                result[experiment][motion][k].append(m)
                    for entry in batchResults['metadata']:
                        metadata[experiment].append(entry)

                    # Save the results
                    self._saveBatchAnaysisResults(batchResults, session)

        #
        for experiment in ('saline', 'cno'):
            for motion in ('ipsi', 'contra'):
                for measure in ('e', 'm', 'p'):
                    result[experiment][motion][measure] = np.array(result[experiment][motion][measure])
        self.result = result
        self.metadata = metadata

        return result, metadata

    def save(self, filename):
        """
        """

        if self.result is None:
            raise Exception('No result available to save')

        with open(filename, 'rb') as stream:
            pickle.dump(self.result, stream)

        return