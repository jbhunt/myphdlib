import csv
import numpy as np

def createManualSpikeSortingLog(
    sessions,
    filename,
    minimumPresenceRatio=0.9,
    maximumRefractoryPeriodViolationRate=0.5,
    maximumAmplitudeCutoff=0.1,
    maximumProbabilityValue=0.05,
    returnEntries=False,
    targetUnitQuality='high',
    ):
    """
    """

    allEntries = list()
    sessionNumber = 1

    for session in sessions:

        #
        if session.probeTimestamps is None:
            continue
        
        
        #
        sessionEntries = list()
        
        # Load the lowest response probability
        responseProbabilities = np.around(np.vstack([
            session.load('population/zeta/probe/left/p'),
            session.load('population/zeta/probe/right/p'),
            session.load('population/zeta/saccade/nasal/p'),
            session.load('population/zeta/saccade/temporal/p')
        ]).min(axis=0), 3)

        #
        kilosortLabels = session.load('population/metrics/ksl')
        presenceRatio = np.around(session.load('population/metrics/pr'), 2)
        refractoryPeriodViolationRate = np.around(session.load('population/metrics/rpvr'), 2)
        amplitudeCutoff = np.around(session.load('population/metrics/ac'), 2)

        # Filter out units with high spike-sorting quality
        spikeSortingMetricsFilter = np.vstack([
            presenceRatio >= minimumPresenceRatio,
            refractoryPeriodViolationRate <= maximumRefractoryPeriodViolationRate,
            amplitudeCutoff <= maximumAmplitudeCutoff
        ]).all(axis=0)
        if targetUnitQuality == 'high':
            pass
        elif targetUnitQuality == 'low':
            spikeSortingMetricsFilter = np.invert(spikeSortingMetricsFilter)
        else:
            raise Exception(f'{targetUnitQuality} is not a valid spike sorting quality label')

        # Collect entries
        for unit in session.population[spikeSortingMetricsFilter]:
            p = responseProbabilities[unit.index]
            if p > maximumProbabilityValue:
                continue
            if kilosortLabels is not None:
                ksl = 'm' if kilosortLabels[unit.index] == 0 else 'g'
            else:
                ksl = ''
            pr = presenceRatio[unit.index]
            rpvr = refractoryPeriodViolationRate[unit.index]
            ac = amplitudeCutoff[unit.index]
            entry = [
                sessionNumber,
                str(session.date),
                session.animal,
                unit.cluster,
                ksl,
                '',
                p,
                pr,
                rpvr,
                ac,
            ]
            sessionEntries.append(entry)

        # Sort entries by the p-values
        probabilityValues = np.array([entry[6] for entry in sessionEntries])
        for entryIndex in np.argsort(probabilityValues):
            allEntries.append(sessionEntries[entryIndex])

        # Increment the session counter
        sessionNumber += 1

    #
    with open(filename, 'w') as stream:
        columns = (
            'sid',
            'date',
            'animal',
            'cluster',
            'ksl',
            'ul',
            'p',
            'pr',
            'rpvr',
            'ac',
        )
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(allEntries)

    #
    if returnEntries:
        return allEntries