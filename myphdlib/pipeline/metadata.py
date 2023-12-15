import numpy as np

class MetadataProcessingMixin():
    """
    """

    def _extractSampleIndicesRange(self):
        """
        """

        if self.experiment == 'Mlati':
            file = self.folders.ephys.joinpath('continuous', 'Neuropix-PXI-100.ProbeA-AP', 'sample_numbers.npy')
        elif self.experiment == 'Dreadds':
            file = self.folders.ephys.joinpath('continuous', 'Neuropix-PXI-100.0', 'timestamps.npy')
        if file.exists == False:
            return
        
        sampleIndices = np.memmap(
            str(file),
            dtype=np.int64,
            shape=(1000000000000,)
        )

        sampleIndexRange = np.array([
            sampleIndices[np.nonzero(sampleIndices)].min(),
            sampleIndices[np.nonzero(sampleIndices)].max()
        ])

        self.save(f'metadata/sir', sampleIndexRange)

        return

    def _runMetadataModule(self):
        """
        """

        self._extractSampleIndicesRange()

        return