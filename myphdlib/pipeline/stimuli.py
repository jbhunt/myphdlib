import numpy as np
from myphdlib.general.labjack import filterPulsesFromPhotologicDevice

class StimuliProcessingMixin():
    """
    """

    def _processMovingBarsProtocol(
        self,
        invertOrientations=True
        ):
        """
        """

        self.log('Processing data from the moving bars stimulus')

        #
        M = self.load('labjack/matrix')
        start, stop = self.load('epochs/mb')
        signal = M[start: stop, self.labjackChannelMapping['stimulus']]

        # Check for data loss
        if np.isnan(signal).sum() > 0:
            self.log(f'Data loss detected during the moving bars stimulus', level='warning')
            return
        
        #
        filtered = filterPulsesFromPhotologicDevice(signal, minimumPulseWidthInSeconds=0.03)

        #
        risingEdgeIndices = np.where(np.diff(filtered) > 0.5)[0]
        barOnsetIndices = risingEdgeIndices[0::2]
        barOffsetIndices = risingEdgeIndices[1::2]
        barCenteredIndices = barOffsetIndices - barOnsetIndices
        barOnsetTimestamps = self.computeTimestamps(
            barOnsetIndices + start
        )
        barOffsetTimestamps = self.computeTimestamps(
            barOffsetIndices + start
        )
        self.save('stimuli/mb/onset/timestamps', barOnsetTimestamps)
        self.save('stimuli/mb/offset/timestamps', barOffsetTimestamps)

        #
        result = list(self.folders.stimuli.rglob('*movingBarsMetadata*'))
        if len(result) != 1:
            raise Exception('Could not locate moving bars stimulus metadata')
        file = result.pop()
        with open(file, 'r') as stream:
            lines = stream.readlines()[5:]
        orientation = list()
        for line in lines:
            event, orientation_, timestamp = line.rstrip('\n').split(', ')
            if int(event) == 1:
                if invertOrientations:
                    orientation_ = round(np.mod(float(orientation_) + 180, 360), 2)
                orientation.append(float(orientation_))
        self.save('stimuli/mb/orientation', np.array(orientation))

        return

    def _correctForCableDisconnectionDuringDriftingGrating(
        self,
        filtered,
        maximumPulseWidthInSeconds=0.6,
        ):
        """
        """

        self.log('Correcting for cable disconnection during the drifting grating stimulus')

        corrected = np.copy(filtered)
        risingEdgeIndices = np.where(np.diff(filtered) > 0.5)[0]
        fallingEdgeIndices = np.where(np.diff(filtered) * -1 > 0.5)[0] + 1
        pulseEpochIndices = np.hstack([
            risingEdgeIndices.reshape(-1, 1),
            fallingEdgeIndices.reshape(-1, 1)
        ])
        pulseWidthsInSeconds = np.diff(pulseEpochIndices, axis=1) / self.labjackSamplingRate
        for flag, epoch in zip(pulseWidthsInSeconds > maximumPulseWidthInSeconds, pulseEpochIndices):
            if flag:
                start, stop = epoch
                corrected[start: stop] = 0

        return corrected

    def _runStimuliModule(self):
        return