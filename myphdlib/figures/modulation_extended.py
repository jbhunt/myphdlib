import copy
import numpy as np
from myphdlib.figures.modulation import BasicSaccadicModulationAnalysis
from myphdlib.figures.analysis import Namespace

class ExtendedModulationAnalysis(BasicSaccadicModulationAnalysis):
    """
    """

    def __init__(self, *args, **kwargs):
        """
        """

        self.saccadeFeatures = None
        self.p1 = None
        self.p2 = None
        super().__init__(*args, **kwargs)

        return
    
    def _measureSaccadeFeatures(self):
        """
        """

        #
        saccadeFeatures = {
            'label': list(),
            'velocity': list(),
            'startpoint': list(),
            'endpoint': list(),
            'amplitude': list(),
            'duration': list()
        }

        #
        saccadeLabels = self.session.load(f'saccades/predicted/{self.session.eye}/labels')
        saccadeWaveforms = self.session.load(f'saccades/predicted/{self.session.eye}/waveforms')
        saccadeEpochs = self.session.load(f'saccades/predicted/{self.session.eye}/epochs')
        saccadeTimestamps = self.session.load(f'saccades/predicted/{self.session.eye}/timestamps')

        # Validate session
        poseEstimates = self.session.load(f'pose/filtered')
        if self.session.eye == 'left': 
            eyePosition = poseEstimates[:, 0]
        elif self.session.eye == 'right':
            eyePosition = poseEstimates[:, 2]
        droppedFrames = self.session.load(f'frames/{self.session.eye}/dropped')
        nFramesRecorded = droppedFrames.size
        if nFramesRecorded > eyePosition.size:
            raise Exception()

        # Real saccades
        iterable = list(zip(
            saccadeLabels,
            saccadeWaveforms,
            saccadeEpochs[:, 0],
            saccadeEpochs[:, 1],
            saccadeTimestamps[:, 0],
            saccadeTimestamps[:, 1]
        ))
        for l, wf, f1, f2, t1, t2 in iterable:

            # Label
            saccadeFeatures['label'].append(l)

            # Velocity
            v = np.diff(wf) * self.session.fps
            vmax = np.interp(0.5, np.linspace(0, 1, v.size), v).item()
            saccadeFeatures['velocity'].append(vmax)

            # Startpoint
            p1 = np.interp(
                f1,
                np.arange(nFramesRecorded),
                eyePosition[:nFramesRecorded]
            ).item()
            saccadeFeatures['startpoint'].append(p1)

            # Endpoint
            p2 = np.interp(
                f2,
                np.arange(nFramesRecorded),
                eyePosition[:nFramesRecorded]
            ).item()
            saccadeFeatures['endpoint'].append(p2)

            # Amplitude
            a = abs(p2 - p1)
            saccadeFeatures['amplitude'].append(a)

            # Duration
            dt = t2 - t1
            saccadeFeatures['duration'].append(dt)

        #
        for k in saccadeFeatures.keys():
            saccadeFeatures[k] = np.array(saccadeFeatures[k])

        return saccadeFeatures
    
    def _loadEventDataForProbes(self):
        """
        Recompute probe latency given a subset of saccades
        """

        saccadeTimestamps, saccadeLatencies, saccadeLabels_, gratingMotion = super()._loadEventDataForSaccades()
        saccadeIndicesTarget, saccadeIndicesOther = self._getTargetSaccadeIndices()
        probeLatencies, saccadeLabels = list(), list()
        for probeTimestamp in self.session.probeTimestamps:
            if np.isnan(probeTimestamp):
                probeLatencies.append(np.nan)
                saccadeLabels.append(np.nan)
                continue
            dt = probeTimestamp - saccadeTimestamps[saccadeIndicesTarget]
            closest = np.nanargmin(np.abs(dt))
            probeLatency = dt[closest]
            probeLatencies.append(probeLatency)
            saccadeLabels.append(saccadeLabels_[saccadeIndicesTarget][closest])
        probeLatencies = np.array(probeLatencies)
        saccadeLabels = np.array(saccadeLabels)
        eventsData = (
            self.session.probeTimestamps,
            probeLatencies,
            saccadeLabels,
            self.session.gratingMotionDuringProbes
        )

        return eventsData
    
    def _getTargetSaccadeIndices(self):
        """
        """

        skey = (str(self.session.date), self.session.animal)
        saccadeAmplitudes = self.saccadeFeatures[skey]['amplitude']
        amin, amax = np.nanpercentile(saccadeAmplitudes, self.p1), np.nanpercentile(saccadeAmplitudes, self.p2)
        saccadeIndicesTarget = np.where(np.logical_and(
            saccadeAmplitudes >= amin,
            saccadeAmplitudes <=  amax
        ))[0]
        saccadeIndicesOther = np.array([i for i in range(len(saccadeAmplitudes)) if i not in saccadeIndicesTarget])

        return saccadeIndicesTarget, saccadeIndicesOther

    def _loadEventDataForSaccades(self):
        """
        """

        saccadeIndicesTarget, saccadeIndicesOther = self._getTargetSaccadeIndices()
        saccadeTimestamps, saccadeLatencies, saccadeLabels, gratingMotion = super()._loadEventDataForSaccades()
        eventsData = (
            saccadeTimestamps[saccadeIndicesTarget],
            saccadeLatencies[saccadeIndicesTarget],
            saccadeLabels[saccadeIndicesTarget],
            gratingMotion[saccadeIndicesTarget]
        )

        return eventsData
    
    def _extractSaccadeFeatures(self):
        """
        """

        self.saccadeFeatures = {}
        nSessions = len(self.sessions)
        for i, session in enumerate(self.sessions):
            end = '\r' if (i + 1) != nSessions else '\n'
            print(f'Extracting saccade features for session {i + 1} out of {nSessions}', end=end)
            skey = (str(session.date), session.animal)
            self._session = session
            saccadeFeatures = self._measureSaccadeFeatures()
            self.saccadeFeatures[skey] = saccadeFeatures

        return

    def run(
        self,
        nSplits=3,
        ):
        """
        """

        # Extract saccade features
        if self.saccadeFeatures is None:
            self._extractSaccadeFeatures()

        # Run basic modulation analysis for each split
        percentiles = np.vstack([
            np.linspace(0, 1, nSplits + 1)[:-1],
            np.linspace(0, 1, nSplits + 1)[1:]
        ]).T * 100

        # Save the state of the namespace
        data = copy.deepcopy(self.ns.data)

        #
        nUnits = len(self.ukeys)
        nWindows = len(self.windows)
        nComponents = 5
        result = {
            'pref': np.full([nUnits, nWindows, nComponents, nSplits], np.nan),
            'null': np.full([nUnits, nWindows, nComponents, nSplits], np.nan)
        }
        for i in range(nSplits):
            self.p1, self.p2 = percentiles[i]

            #
            self.computeSaccadeResponseTemplates()
            self.computePerisaccadicPeths()
            self.fitPerisaccadicPeths()

            #
            for probeDirection in ('pref', 'null'):
                mi = self.ns[f'mi/{probeDirection}/real']
                result[probeDirection][:, :, :, i] = mi

        # Recover the initial state of the namespace
        self.ns._data = data
        for probeDirection in ('pref', 'null'):
            self.ns[f'miext/{probeDirection}/real/low'] = result[probeDirection][:, :, :, 0]
            self.ns[f'miext/{probeDirection}/real/medium'] = result[probeDirection][:, :, :, 1]
            self.ns[f'miext/{probeDirection}/real/high'] = result[probeDirection][:, :, :, 2]

        return