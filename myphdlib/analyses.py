import numpy as np
import pandas as pd
from myphdlib.toolkit import inrange
from myphdlib.pipeline import iterateSessions

def measureRealtimeSystemPerformance(
    datasetName='Realtime',
    targetEye='right',
    saccadeDirection='left',
    perisaccadicWindow=(-0.05, 0.11),
    metric='count',
    ):
    """
    """

    #
    output = list()
    tuples = list()

    #
    for obj, animal, date, session in iterateSessions(datasetName):

        # Combine animal and session
        if session != None:
            animal += session
    
        # Extract timesamps for saccades and probes
        try:
            labjackTimestamps = obj.labjackData[:, 0]
            probeOnsetTimestamps = obj.load('probeOnsetTimestamps')
            saccadeOnsetIndices = obj.saccadeOnsetIndicesClassified[targetEye][saccadeDirection]
            saccadeOnsetIndices.sort()  
            saccadeOnsetTimestamps = labjackTimestamps[obj.labjackIndicesAcquisition[saccadeOnsetIndices]]
        except:
            import pdb; pdb.set_trace()
            continue

        # Count the number of perisaccadic probes
        count = 0
        latencies = list()
        for probeOnsetTimestamp in probeOnsetTimestamps:
            dt = probeOnsetTimestamp - saccadeOnsetTimestamps
            closest = np.argsort(abs(dt))[0]
            latency = dt[closest]
            if inrange(latency, perisaccadicWindow[0], perisaccadicWindow[1]):
                count += 1
                latencies.append(latency)

        # Normalize by computing the percent of total saccades detected
        if metric == 'count':
            value = count
        elif metric == 'percent':
            value = np.around(count / saccadeOnsetTimestamps.size * 100, 2)
        elif metric == 'rate':
            value = np.around(count / (obj.frameCountGroundTruth[targetEye] / obj.fps) * 60, 3)

        # Save the result and tuple
        output.append((value, obj.threshold))
        tuples.append((date, animal))

    # Generate a dataframe from the results
    index = pd.MultiIndex.from_tuples(tuples, names=['Date', 'Animal'])
    frame = pd.DataFrame(output, index=index, columns=['Value', 'Threshold'])

    return frame

def compareSaccadeAmplitude(
    sessionIdentifiers=(),
    datasetName='Dreadd',
    ipsiversiveDirection='right',
    timeWindow=(-0.1, 0.1),
    ):
    """
    """

    iterable = list()
    for animal, date in sessionIdentifiers:
        for direction in ('ipsiversive', 'contraversive'):
            iterable.append((animal, date, direction))
    data = {(animal, date, direction): None for animal, date, direction in iterable}
    # contraversiveDirection = 'left' if ipsiversiveDirection == 'right' else 'right'

    for targetAnimal, targetDate in sessionIdentifiers:

        #
        sessionIdentified = False
        for obj, animal, date, session in iterateSessions(datasetName):
            if animal == targetAnimal and date == targetDate:
                sessionIdentified = True
                break
        if sessionIdentified == False:
            continue

        #
        samples = {'ipsiversive': list(), 'contraversive': list()}
        for direction in ('left', 'right'):
            if direction == 'left':
                eye = 'right'
            else:
                eye = 'left'
            saccadeWaveforms = obj.saccadeWaveformsClassified[eye][direction]
            start, stop = list(map(round, np.array(timeWindow) * obj.acquisitionFramerate + saccadeWaveforms.shape[1] / 2))
            for saccadeWaveform in saccadeWaveforms:
                positionRange = np.array([
                    np.min(saccadeWaveform[start: stop]),
                    np.max(saccadeWaveform[start: stop])
                ])
                amplitude = abs(np.diff(positionRange).item())
                if direction == ipsiversiveDirection:
                    samples['ipsiversive'].append(amplitude)
                else:
                    samples['contraversive'].append(amplitude)
        
        #
        for direction in ('ipsiversive', 'contraversive'):
            data[(animal, date, direction)] = [round(np.mean(samples[direction]), 3), round(np.std(samples[direction]), 3), len(samples[direction])]

    #
    index = pd.MultiIndex.from_tuples(list(data.keys()), names=['Animal', 'Date', 'Direction'])
    try:
        frame = pd.DataFrame(np.array(list(data.values())), index=index, columns=['Mean', 'Std', 'Frequency'])
    except:
        import pdb; pdb.set_trace()
    return frame

def collectPilotExperimentData():
    """
    """

    sessionIdentifiersControl = (
        ('dreadd1', '2022-07-12'),
        ('dreadd2', '2022-07-13'),
        ('dreadd4', '2022-07-13')
    )
    sessionIdentifiersCNO = (
        ('dreadd1', '2022-07-14'),
        ('dreadd2', '2022-07-14'),
        ('dreadd4', '2022-07-13')
    )
    sessionIdentifiersMuscimol = (
        ('dreadd1', '2022-07-15'),
        ('dreadd2', '2022-07-15'),
        # ('dreadd4', '2022-07-18')
    )
    sessionIdentifiersCombined = [
        sessionIdentifiersControl,
        sessionIdentifiersCNO,
        sessionIdentifiersMuscimol
    ]

    #
    experimentData = {'Saline': None, 'CNO': None, 'Muscimol': None}
    datasetNames = ('Dreadd', 'Dreadd', 'Muscimol')
    for sessionIdentifiers, experimentLabel, datasetName in zip(sessionIdentifiersCombined, list(experimentData.keys()), datasetNames):
        experimentData[experimentLabel] = compareSaccadeAmplitude(sessionIdentifiers, datasetName)

    return experimentData