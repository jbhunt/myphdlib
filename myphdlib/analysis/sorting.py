import numpy as np

def createManualCurationLog(
    sessions,
    csv,
    minimumPresenceRatio=0.9,
    maximumRefractoryPeriodViolationRate=0.5,
    maximumAmplitudeCutoff=0.1,
    returnEntries=False
    ):
    """
    """

    lines = list()
    allEntries = list()

    for session in sessions:
        
        #
        sessionEntries = list()
        
        # Load the lowest response probability
        responseProbabilities = np.vstack([
            session.load('population/zeta/probe/left/p'),
            session.load('population/zeta/probe/right/p'),
        ]).min(axis=0)

        # Filter out units with high spike-sorting quality
        spikeSortingMetricsFilter = np.vstack([
            session.load('population/metrics/pr') >= minimumPresenceRatio,
            session.load('population/metrics/rpvr') <= maximumRefractoryPeriodViolationRate,
            session.load('population/metrics/ac') <= maximumAmplitudeCutoff
        ]).all(axis=0)

        # Collect entries
        for unit in session.population[np.invert(spikeSortingMetricsFilter)]:
            p = responseProbabilities[unit.index]
            entry = [
                str(session.date),
                session.animal,
                unit.cluster,
                p
            ]
            sessionEntries.append(entry)

        # Sort entries by the p-values
        probabilitySortedIndices = np.argsort([entry[3] for entry in sessionEntries])
        for entryIndex in probabilitySortedIndices:
            date, animal, cluster, p = sessionEntries[entryIndex]
            line = f'{date},{animal},{cluster},{p:.3f}\n'
            lines.append(line)
            allEntries.append(sessionEntries[entryIndex])

    #
    with open(csv, 'w') as stream:
        for line in lines:
            stream.write(line)

    #
    if returnEntries:
        return allEntries