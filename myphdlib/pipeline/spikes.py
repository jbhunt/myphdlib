import os
import time
import mat73
import numpy as np
from myphdlib.general.toolkit import psth2
from myphdlib.extensions.matlab import runMatlabScript, locatMatlabAddonsFolder
from simple_spykes.util.ecephys import run_quality_metrics

#
samplingRateNeuropixels = 30000.0

#
matlabScriptTemplate = """
addpath('{0}/npy-matlab-master/npy-matlab')
addpath('{0}/spikes-master/analysis')
spikeTimesFile = '{1}'
spikeClustersFile = '{2}'
gwf.dataDir = '{3}'
gwf.fileName = 'continuous.dat'
gwf.dataType = 'int16'
gwf.nCh = 384
gwf.wfWin = [-31 30]
gwf.nWf = {4}
gwf.spikeTimes = readNPY(spikeTimesFile)
gwf.spikeClusters = readNPY(spikeClustersFile)
result = getWaveForms(gwf)
waveforms = result.waveFormsMean;
fname = '{5}'
writeNPY(waveforms, fname)
exit
"""

class SpikesProcessingMixin(object):
    """
    """

    def _extractSpikeDatasets(
        self,
        sorting='manual',
        ):
        """
        """

        if self.hasDataset('spikes/clusters') and self.hasDataset('spikes/timestamps'):
            return
        self.log(f'Extracting spike clusters and timestamps', level='info')

        spikeTimestamps = np.array([])
        result = list(self.folders.ephys.joinpath('sorting', sorting).glob('spike_times.npy'))
        if len(result) != 1:
            raise Exception('Could not locate the spike times data')
        else:
            spikeTimestamps = np.around(
                np.load(str(result.pop())).flatten() / samplingRateNeuropixels,
                3
            )
        
        #
        spikeClusters = np.array([])
        result = list(self.folders.ephys.joinpath('sorting', sorting).glob('spike_clusters.npy'))
        if len(result) != 1:
            raise Exception('Could not locate the cluster ID data')
        else:
            spikeClusters = np.around(
                np.load(str(result.pop())).flatten(),
                3
            )

        #
        self.save('spikes/timestamps', spikeTimestamps)
        self.save('spikes/clusters', spikeClusters)

        return

    def _extractQualityLabels(
        self,
        ):
        """

        """

        clusterInfoFile = self.home.joinpath('ephys', 'sorting', 'manual', 'cluster_info.tsv')
        if clusterInfoFile.exists() == False:
            self.log('Could not locate cluster info file', level='warning')
            nUnits = len(self.population)
            self.save('metrics/ql', np.full(nUnits, np.nan))
            return

        #
        with open(clusterInfoFile, 'r') as stream:
            lines = stream.readlines()
        columnNames = lines[0].split('\t')
        j = columnNames.index('KSLabel')
        kilosortLabels = np.array([
            row.split('\t')[j] for row in lines[1:]
        ])
        j = columnNames.index('group')
        userLabels = np.array([
            row.split('\t')[j] for row in lines[1:]
        ])
        nUnits = len(kilosortLabels)
        qualityLabels = list()
        for iUnit in range(nUnits):
            if userLabels[iUnit] == '':
                ql = kilosortLabels[iUnit]
            else:
                ql = userLabels[iUnit]
            qualityLabels.append(ql)
        qualityLabels = np.array(qualityLabels)
        qualityLabelsCoded = np.array([
            0 if ql == 'mua' else 1
                for ql in qualityLabels
        ])
        self.save('metrics/ql', qualityLabelsCoded)

        return

    def _extractKilosortLabels(
        self,
        sorting='manual',
        overwrite=True,
        ):
        """
        """

        #
        if self.hasDataset('metrics/ksl') and overwrite == False:
            return

        #
        spikeClusters = np.array([])
        result = list(self.folders.ephys.joinpath('sorting', sorting).glob('spike_clusters.npy'))
        if len(result) != 1:
            raise Exception('Could not locate the cluster ID data')
        else:
            spikeClusters = np.around(
                np.load(str(result.pop())).flatten(),
                3
            )

        #
        clusterNumbers1 = np.unique(spikeClusters)
        nUnits = clusterNumbers1.size

        # Extract the label assigned to each unit by Kilosort
        clusterLabels = list()
        clusterNumbers2 = list()
        result = list(self.folders.ephys.joinpath('sorting', sorting).glob('cluster_KSLabel.tsv'))
        if len(result) != 1:
            self.log(f'Could not locate Kilosort labels', level='warning')
            clusterLabels = np.full(nUnits, np.nan)
        else:
            tsv = result.pop()
            with open(tsv, 'r') as stream:
                lines = stream.readlines()[1:]
            for line in lines:
                cluster, label = line.rstrip('\n').split('\t')
                clusterNumbers2.append(int(cluster))
                clusterLabels.append(0 if label == 'mua' else 1)
            clusterLabels = np.array(clusterLabels)[np.argsort(clusterNumbers2)]

        # Need to delete labels where the cluster number is missing
        missingClusterNumbers = np.setdiff1d(clusterNumbers2, clusterNumbers1)
        missingClusterIndices = np.array([
            np.where(clusterNumbers2 == missingClusterNumber)[0].item()
                for missingClusterNumber in missingClusterNumbers
        ])
        if missingClusterIndices.size != 0:
            clusterLabels = np.delete(clusterLabels, missingClusterIndices)
        
        #
        self.save('metrics/ksl', clusterLabels)

        return

    def _extractSpikeWaveforms(
        self,
        sorting='manual',
        nWaveforms=50,
        nBestChannels=1,
        nogui=True,
        windowsProcessTimeout=60*10, # Ten minutes,
        baselineWindowInSamples=(0, 20),
        ):

        #
        # if self.hasDataset('metrics/bsw'):
        #     return

        self.log(f'Extrcting best average spike waveforms')

        #
        partsFromEphysFolder = (
            'sorting',
             sorting
        )
        spikeWaveformsFile = self.folders.ephys.joinpath(*partsFromEphysFolder, 'spike_waveforms.npy')

        #
        if spikeWaveformsFile.exists() == False:
            matlabAddonsFolder = locatMatlabAddonsFolder()
            matlabScriptLines = matlabScriptTemplate.format(
                matlabAddonsFolder,
                self.folders.ephys.joinpath(*partsFromEphysFolder, 'spike_times.npy'),
                self.folders.ephys.joinpath(*partsFromEphysFolder, 'spike_clusters.npy'),
                self.folders.ephys.joinpath(*partsFromEphysFolder),
                nWaveforms,
                self.folders.ephys.joinpath(*partsFromEphysFolder, 'spike_waveforms.npy'),
            ).strip('\n')
            scriptFilePath = self.folders.ephys.joinpath('sorting', 'extractSpikeWaveforms.m')
            with open(scriptFilePath, 'w') as stream:
                for line in matlabScriptLines:
                    stream.write(line)

            #
            runMatlabScript(
                scriptFilePath,
                nogui=nogui
            )

            #
            if os.name == 'nt':
                t0 = time.time()
                while True:
                    if time.time() - t0 > windowsProcessTimeout:
                        raise Exception(f'Failed to extract spike waveforms')
                    spikeWaveformsFile = self.folders.ephys.joinpath(*partsFromEphysFolder, 'spike_waveforms.npy')
                    if spikeWaveformsFile.exists():
                        break

            #
            elif os.name == 'posix':
                if spikeWaveformsFile.exists() == False:
                    raise Exception(f'Failed to extract spike waveforms')
                
            #
            scriptFilePath.unlink() # Delete script

        #
        spikeWaveformsArray = np.load(spikeWaveformsFile)

        #
        nUnits, nChannels, nSamples = spikeWaveformsArray.shape
        bestSpikeWaveforms = np.full([nUnits, nSamples], np.nan)
        for iUnit in range(nUnits):
            waveformPower = np.array([
                np.sum(np.abs(wf - wf[baselineWindowInSamples[0]:baselineWindowInSamples[1]].mean())) for wf in spikeWaveformsArray[iUnit, :, :]
            ])
            channelIndices = np.argsort(waveformPower)[::-1][:nBestChannels]
            bestSpikeWaveform = spikeWaveformsArray[iUnit, channelIndices, :].mean(0)

            bestSpikeWaveforms[iUnit :] = bestSpikeWaveform - bestSpikeWaveform[baselineWindowInSamples[0]: baselineWindowInSamples[1]].mean()

        #
        self.save(f'metrics/bsw', bestSpikeWaveforms)

        return

    def _extractUnitPositions(
        self,
        sorting='manual',
        ):
        """
        """

        self.log(f'Extracting spatial coordinates for each unit')

        #
        if self.hasDataset('metrics/msp'):
            return

        #
        kilosortResultsFile = self.folders.ephys.joinpath('sorting', sorting, 'rez.mat')
        kilosortResults = mat73.loadmat(kilosortResultsFile)['rez']
        spikeCoordinates = kilosortResults['xy']
        spikeClustersFile = self.folders.ephys.joinpath('sorting', sorting, 'spike_clusters.npy')
        spikeClusters = np.load(spikeClustersFile)
        
        #
        uniqueSpikeClusters = np.unique(spikeClusters)
        nUnits = uniqueSpikeClusters.size
        meanSpikePositions = np.full([nUnits, 2], np.nan)
        for iUnit, uniqueSpikeCluster in enumerate(uniqueSpikeClusters):
            mask = spikeClusters.flatten() == uniqueSpikeCluster
            meanSpikePosition = np.around(spikeCoordinates[mask, :].mean(0), 2)
            meanSpikePositions[iUnit] = meanSpikePosition

        # Need to swap x and y coordinates
        meanSpikePositions = np.fliplr(meanSpikePositions)
        self.save('metrics/msp', meanSpikePositions)


        return

    def _measureSpikeSortingQuality(
        self,
        sorting='manual',
        **kwargs
        ):
        """
        Notes
        -----
        Default threshold values are based on the quality metrics tutorial from the Allen Institute:
        https://allensdk.readthedocs.io/en/latest/_static/examples/nb/ecephys_quality_metrics.html
        """

        self.log(f'Measuring spike-sorting quality')

        # ecephys spike sorting backend
        params_ = {
            "isi_threshold": 0.0015,
            "min_isi": 0.000166,
            "num_channels_to_compare": 7,
            "max_spikes_for_unit": 500,
            "max_spikes_for_nn": 10000,
            "n_neighbors": 4,
            'n_silhouette': 10000,
            "drift_metrics_interval_s": 51,
            "drift_metrics_min_spikes_per_interval": 10,
            "include_pc_metrics": False
        }
        params_.update(kwargs)

        #
        sortingResultsFolder = self.folders.ephys.joinpath('sorting', sorting)
        metrics = run_quality_metrics(
            str(sortingResultsFolder),
            30000.0,
            params_,
            save_to_file=str(sortingResultsFolder.joinpath('quality_metrics.json'))
        )
        presenceRatios = np.array(list(metrics['presence_ratio'].values())).astype(float)
        isiViolationRates = np.array(list(metrics['isi_viol'].values())).astype(float)
        amplitudeCutoffs = np.array(list(metrics['amplitude_cutoff'].values())).astype(float)

        #
        self.save('metrics/pr', presenceRatios)
        self.save('metrics/rpvr', isiViolationRates)
        self.save('metrics/ac', amplitudeCutoffs)

        return

    def _runSpikesModule(
        self,
        sorting='manual',
        ):
        """
        """

        self._extractSpikeDatasets(sorting)
        self._extractQualityLabels()
        self._measureSpikeSortingQuality(sorting)

        return