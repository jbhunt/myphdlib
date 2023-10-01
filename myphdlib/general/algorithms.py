import numpy as np

# TODO: Make the thresholding adaptive such that it searches for a value that results
# in the same number of expected and corrected trials
def detectMissingEvents(
    eventTimestampsObserved,
    eventTimestampsExpected,
    latencyThresholdRange=(0.01, 1.0),
    latencyThresholdStep=0.01,
    padValue=1
    ):
    """
    Identifies the indices for missing events given the observed and expected timestamps
    """

    #
    if eventTimestampsExpected.size == eventTimestampsObserved.size:
        missingEventsMask = np.full(eventTimestampsExpected.size, False)
        insertionIndices = np.array([]).astype(int)
        return True, missingEventsMask, insertionIndices

    #
    for latencyThreshold in np.arange(latencyThresholdRange[0], latencyThresholdRange[1] + latencyThresholdStep, latencyThresholdStep):

        # Pad the timestamps
        # NOTE: This ensures the algorithm can handle missing events at the beginning and end of the observed timestamps array
        eventTimestampsCorrected = np.copy(eventTimestampsObserved).astype(float)
        eventTimestampsCorrected = np.concatenate([
            np.array([eventTimestampsExpected.min() - padValue]),
            eventTimestampsCorrected,
            np.array([eventTimestampsExpected.max() + padValue])
        ])
        eventTimestampsExpectedPadded = np.concatenate([
            np.array([eventTimestampsExpected.min() - padValue]),
            eventTimestampsExpected,
            np.array([eventTimestampsExpected.max() + padValue])
        ])

        # Initialize counter and list of insertion indices
        nMissingEvents = 0
        insertionIndices = list()

        # Main loop
        while True:

            # Algorithm failed
            if nMissingEvents == eventTimestampsObserved.size:
                result = False
                break

            # Algorithm failed
            if eventTimestampsCorrected.size > eventTimestampsExpectedPadded.size:
                result = False
                break

            # Compute the latency between event intervals
            nEventsObserved = eventTimestampsCorrected.size
            mask = np.invert(np.isnan(eventTimestampsCorrected))
            latency = np.subtract(
                np.diff(eventTimestampsExpectedPadded[:nEventsObserved][mask]),
                np.diff(eventTimestampsCorrected[mask])
            )

            # Algorithm succeeded
            thresholdCrossingIndices = np.where(np.abs(latency) >= latencyThreshold)[0]
            if thresholdCrossingIndices.size == 0:
                result = True
                break
            
            # Identify the earliest missing event
            insertionIndex = thresholdCrossingIndices.min() + 1 + np.invert(mask).sum()
            insertionIndices.append(thresholdCrossingIndices.min() + 1)
            eventTimestampsCorrected = np.insert(
                eventTimestampsCorrected,
                insertionIndex,
                np.nan
            )
            nMissingEvents += 1

        # Check result
        if result:
            insertionIndices = np.array(insertionIndices)
            insertionIndices -= 1 # Correct for padding
            missingEventsMask = np.insert(
                np.full(eventTimestampsObserved.size, False),
                insertionIndices,
                True
            )
            break
        else:
            missingEventsMask = np.array([])
            insertionIndices = np.array([])

    return result, missingEventsMask, insertionIndices