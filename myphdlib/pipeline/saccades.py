import time
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.stats import pearsonr
from scipy.signal import find_peaks as findPeaks
from scipy.optimize import curve_fit as fitCurve
from myphdlib.general.curves import relu
from myphdlib.general.toolkit import smooth, resample, interpolate, detectThresholdCrossing, DotDict
from myphdlib.general.session import saveSessionData
from myphdlib.extensions.matplotlib import SaccadeDirectionLabelingGUI, SaccadeEpochLabelingGUI

class SaccadesProcessingMixin(object):
    """
    """

    def _extractEyePosition(
        self,
        likelihoodThreshold=0.99,
        pupilCenterName='pupilCenter',
        ):
        """
        Extract the raw eye position data
        """

        self.log('Extracting eye position data')

        #
        nSamples = 0
        eyePositionLeft = None
        eyePositionRight = None

        #
        for side in ('left', 'right'):
            if side == 'left':
                csv = self.leftEyePose
            elif side == 'right':
                csv = self.rightEyePose
            if csv is not None:
                frame = pd.read_csv(csv, header=list(range(3)), index_col=0).sort_index(level=1, axis=1)
            else:
                continue
            network = frame.columns[0][0]
            x = np.array(frame[network, pupilCenterName, 'x']).flatten()
            y = np.array(frame[network, pupilCenterName, 'y']).flatten()
            l = np.array(frame[network, pupilCenterName, 'likelihood']).flatten()
            x[l < likelihoodThreshold] = np.nan
            y[l < likelihoodThreshold] = np.nan
            coords = np.hstack([
                x.reshape(-1, 1),
                y.reshape(-1, 1)
            ])
            if side == 'left':
                eyePositionLeft = np.copy(coords)
            elif side == 'right':
                eyePositionRight = np.copy(coords)
            if coords.shape[0] > nSamples:
                nSamples = coords.shape[0]

        #
        eyePositionUncorrected = np.full([nSamples, 4], np.nan)
        if eyePositionLeft is not None:
            eyePositionUncorrected[:eyePositionLeft.shape[0], 0:2] = eyePositionLeft
        if eyePositionRight is not None:
            eyePositionUncorrected[:eyePositionRight.shape[0], 2:4] = eyePositionRight
        self.save('pose/uncorrected', eyePositionUncorrected)

        return

    # TODO: Refactor this function such that it uses the dataset which have already
    #       figured out which frames were dropped
    def _correctEyePosition(self, pad=1e6):
        """
        Correct eye position data for missing/dropped frames
        """

        #
        eyePositionUncorrected = self.load('pose/uncorrected')
        nSamples = eyePositionUncorrected.shape[0] + int(pad)
        eyePositionCorrected = np.full([nSamples, 4], np.nan)

        #
        if self.primaryCamera == 'left':
            frameIntervals = np.loadtxt(self.leftCameraTimestamps, dtype=np.int64)
        elif self.primaryCamera == 'right':
            frameIntervals = np.loadtxt(self.rightCameraTimestamps, dtype=np.int64)
        else:
            raise Exception('Could not determine the primary camera')
        factor = np.median(frameIntervals)

        #
        terminationIndex = 0
        for camera in ('left', 'right'):

            #
            if camera == 'left':
                columnIndices = 0, 1
                frameIntervals = np.loadtxt(self.leftCameraTimestamps, dtype=np.int64)
            else:
                columnIndices = 2, 3
                frameIntervals = np.loadtxt(self.rightCameraTimestamps, dtype=np.int64)

            #
            frameIndex = 0
            frameOffset = 0
            missingFrames = 0
            eyePositionUncorrected[0, columnIndices] = eyePositionUncorrected[0, columnIndices]
            for frameInterval in frameIntervals:
                frameOffset += round(frameInterval / factor) - 1
                if frameIndex >= eyePositionUncorrected.shape[0]:
                    missingFrames += 1
                else:
                    eyePositionCorrected[frameIndex + frameOffset, columnIndices] = eyePositionUncorrected[frameIndex, columnIndices]
                frameIndex += 1

            #
            if frameIndex + frameOffset > terminationIndex:
                terminationIndex = frameIndex + frameOffset

        #
        eyePositionCorrected = eyePositionCorrected[:terminationIndex, :]
        self.save('pose/corrected', eyePositionCorrected)

        return

    def _interpolateEyePosition(
        self,
        maximumConsecutiveDroppedFrames=4
        ):
        """
        """

        eyePositionCorrected = self.load('pose/corrected')
        eyePositionInterpolated = np.copy(eyePositionCorrected)
        
        #
        for iColumn in range(eyePositionCorrected.shape[1]):

            #
            pose = eyePositionCorrected[:, iColumn]
            dropped = np.isnan(pose)
            windows = list()

            #
            iRow = 0
            while True:
                if iRow >= dropped.size:
                    break
                flag = dropped[iRow]

                # Figure out how many frames were dropped
                if flag:
                    nDroppedFrames = 0
                    for flag_ in dropped[iRow:]:
                        if flag_ == False:
                            break
                        nDroppedFrames += 1

                    #
                    if nDroppedFrames <= maximumConsecutiveDroppedFrames:
                        if iRow + nDroppedFrames + 1 >= pose.size:
                            iRow += nDroppedFrames
                            continue
                        windows.append([
                            iRow - 1,
                            iRow + nDroppedFrames + 1
                        ])
                    iRow += nDroppedFrames

                # Increment the counter
                else:
                    iRow += 1

            # Interpolate over the windows of dropped frames
            for start, stop in windows:
                x = np.arange(start + 1, stop - 1, 1)
                xp = np.array([start, stop - 1])
                fp = np.array([pose[start], pose[stop - 1]])
                y = np.interp(x, xp, fp)
                eyePositionInterpolated[start + 1: stop - 1, iColumn] = y

        #
        self.save('pose/interpolated', eyePositionInterpolated)

        return

    # TODO: Remove the constant motion of the eye
    def _stabilizeEyePosition(self):
        """
        """

        return

    # TODO: Normlize eye position to some common measurement (e.g., eye width)
    def _normalizeEyePosition(self):
        """
        """

        return

    def _decomposeEyePosition(self, nNeighbors=5, benchmark=False):
        """
        """

        # eyePositionCorrected = self.eyePositionCorrected
        eyePositionInterpolated = self.load('pose/interpolated')
        eyePositionDecomposed = np.full_like(eyePositionInterpolated, np.nan)
        missingDataMask = {
            'left': np.full(eyePositionInterpolated.shape[0], np.nan),
            'right': np.full(eyePositionInterpolated.shape[0], np.nan)
        }

        #
        if benchmark:
            t1 = time.time()

        #
        for columnIndices, X1, side in zip([(0, 2), (2, 4)], np.split(eyePositionInterpolated, 2, axis=1), ('left', 'right')):

            # Check for missing eye position data
            if np.isnan(X1).all(0).all():
                missingDataMask[side] = np.full(eyePositionInterpolated.shape[0], True)
                continue

            # Impute NaN values
            X2 = SimpleImputer(missing_values=np.nan).fit_transform(X1)

            #
            mask = np.invert(np.isnan(X1).any(1))
            missingDataMask[side] = np.invert(mask)

            #
            eyePositionDecomposed[:, columnIndices[0]: columnIndices[1]] = PCA(n_components=2).fit_transform(X2)

        #
        if benchmark:
            t2 = time.time()
            elapsed = round((t2 - t1) / 60, 2)
            self.log(f'Decomposition took {elapsed} minutes')

        #
        eyePositionDecomposed[missingDataMask['left'], 0:2] = np.nan
        eyePositionDecomposed[missingDataMask['right'], 2:4] = np.nan
        # self.write(eyePositionDecomposed, 'eyePositionDecomposed')
        # self.write(missingDataMask, 'missingDataMask')
        self.save('pose/decomposed', eyePositionDecomposed)
        self.save('pose/missing/left', missingDataMask['left'])
        self.save('pose/missing/right', missingDataMask['right'])

        return

    def _reorientEyePosition(self, reflect='left'):
        """
        """

        # eyePositionDecomposed = self.eyePositionDecomposed
        eyePositionDecomposed = self.load('pose/decomposed')
        # eyePositionCorrected = self.eyePositionCorrected
        eyePositionCorrected = self.load('pose/corrected')
        eyePositionReoriented = np.full_like(eyePositionCorrected, np.nan)

        #
        iterable = zip(['left', 'left', 'right', 'right'], np.arange(4))
        for eye, columnIndex in iterable:
            #
            if np.isnan(eyePositionDecomposed[:, columnIndex]).all():
                continue
            signal3 = np.copy(eyePositionDecomposed[:, columnIndex])
            signal1 = eyePositionCorrected[:, columnIndex]
            indices = np.where(np.isnan(signal1))[0]
            signal1 = np.delete(signal1, indices)
            signal2 = eyePositionDecomposed[:, columnIndex]
            signal2 = np.delete(signal2, indices)
            r2, p =  pearsonr(signal1, signal2)
            if r2 > 0 and p < 0.05:
                pass
            elif r2 < 0 and p < 0.05:
                signal3 *= -1
            else:
                raise Exception('Could not determine correlation between raw and decomposed eye position')
            eyePositionReoriented[:, columnIndex] = signal3

        # TODO: Check that left and right eye position is anti-correlated

        #
        # self.write(eyePositionReoriented, 'eyePositionReoriented')
        self.save('pose/reoriented', eyePositionReoriented)

        return

    def _filterEyePosition(self, t=25):
        """
        Filter eye position

        keywords
        --------
        t : int
            Size of time window for smoothing (in ms)
        """

        #
        # missingDataMask = self.missingDataMask
        missingDataMask = {
            'left': self.load('pose/missing/left'),
            'right': self.load('pose/missing/right')
        }
        
        # Determine the nearest odd window size
        smoothingWindowSize = round(t / (1 / self.fps * 1000))
        if smoothingWindowSize % 2 == 0:
            smoothingWindowSize += 1

        # Filter
        # eyePositionReoriented = self.eyePositionReoriented
        eyePositionReoriented = self.load('pose/reoriented')
        eyePositionFiltered = np.full_like(eyePositionReoriented, np.nan)
        for columnIndex in range(eyePositionReoriented.shape[1]):

            #
            if np.isnan(eyePositionReoriented[:, columnIndex]).all():
                continue

            # Interpolate missing values
            interpolated = interpolate(eyePositionReoriented[:, columnIndex])

            #
            smoothed = smooth(interpolated, smoothingWindowSize)

            #
            if columnIndex in (0, 1):
                smoothed[missingDataMask['left']] = np.nan
            else:
                smoothed[missingDataMask['right']] = np.nan

            #
            eyePositionFiltered[:, columnIndex] = smoothed

        # Save filtered eye position data
        self.save('pose/filtered', eyePositionFiltered)

        return

    # TODO: Record saccade onset timesetamps
    def _detectPutativeSaccades(
        self,
        amplitudeThreshold=0.99,
        minimumInterPeakInterval=0.075,
        perisaccadicWindow=(-0.2, 0.2),
        centerSaccadeWaveforms=False,
        smoothingWindowSize=0.025,
        ):
        """
        """

        self.log('Extracting putative saccades')

        # Minimum inter-saccade interval (in seconds)
        distanceThreshold = self.fps * minimumInterPeakInterval

        # Sample offset added to each peak sample index
        peakOffsets = np.array([
            round(perisaccadicWindow[0] * self.fps),
            round(perisaccadicWindow[1] * self.fps)
        ])

        # N samples across saccades waveforms
        nFeatures = peakOffsets[1] - peakOffsets[0]

        # Smoothing window size (in samples)
        wlen = round(smoothingWindowSize * self.fps)
        if wlen % 2 == 0:
            wlen += 1

        #
        eyePositionFiltered = self.load('pose/filtered')
        eyePositionImputed = np.full_like(eyePositionFiltered, np.nan)
        for iColumn, column in enumerate(eyePositionFiltered.T):

            # Skip over columns that are entirely NaNs
            if np.isnan(column).all():
                eyePositionImputed[:, iColumn] = column
                continue

            # Impute missing data
            eyePositionImputed[:, iColumn] = np.interp(
                np.arange(column.size),
                np.arange(column.size)[np.isfinite(column)],
                column[np.isfinite(column)]
            )

        #
        saccadeDetectionResults = {
            'left': {
                'indices': list(),
                'waveforms': list(),
            },
            'right': {
                'indices': list(),
                'waveforms': list(),
            },
        }

        for eye, columnIndex in zip(['left', 'right'], [0, 2]):

            # Check for NaN values
            if np.isnan(eyePositionFiltered[:, columnIndex]).all():
                self.log(f'Eye position data missing for the {eye} eye', level='warning')
                for feature in ('waveforms', 'indices'):
                    saccadeDetectionResults[eye][feature] = None
                continue

            #
            velocity = np.abs(smooth(
                np.diff(eyePositionImputed[:, columnIndex]),
                wlen
            ))
            heightThreshold = np.percentile(velocity, amplitudeThreshold * 100)
            peakIndices, peakProperties = findPeaks(velocity, height=heightThreshold, distance=distanceThreshold)

            #
            for peakIndex in peakIndices:

                # Determine alignment
                if centerSaccadeWaveforms:
                    peakOffsets[1] += 1

                # Extract saccade waveform
                startIndex = peakIndex + peakOffsets[0]
                stopIndex = peakIndex + peakOffsets[1]
                saccadeWaveform = eyePositionFiltered[startIndex: stopIndex, columnIndex]

                # Exclude incomplete saccades
                if saccadeWaveform.size != nFeatures:
                    continue

                #
                saccadeDetectionResults[eye]['waveforms'].append(saccadeWaveform)
                saccadeDetectionResults[eye]['indices'].append(peakIndex)

        # Sort the putative saccades by chronological order
        for eye in ('left', 'right'):
            if saccadeDetectionResults[eye]['waveforms'] is None:
                continue
            else:
                sortedIndex = np.argsort(saccadeDetectionResults[eye]['indices'])
                for feature in ('waveforms', 'indices'):
                    saccadeDetectionResults[eye][feature] = np.array(saccadeDetectionResults[eye][feature])[sortedIndex]

        # Log the results of saccade detection
        for eye in ('left', 'right'):
            if saccadeDetectionResults[eye]['waveforms'] is None:
                continue
            else:
                nSaccades = saccadeDetectionResults[eye]['waveforms'].shape[0]
                self.log(f'{nSaccades} putative saccades detected for the {eye} eye')

        # Save results
        for eye in saccadeDetectionResults.keys():
            for feature in ('indices', 'waveforms'):
                if saccadeDetectionResults[eye][feature] is None:
                    continue
                path = f'saccades/putative/{eye}/{feature}'
                dataset = saccadeDetectionResults[eye][feature]
                if dataset is None:
                    self.log(f'No saccades detected for the {eye} eye', level='warning')
                    continue
                self.save(path, dataset)
        
        return

    def _determineGratingMotionAssociatedWithEachSaccade(
        self,
        interBlockIntervalRange=(2, 9),
        interBlockIntervalStep=0.1,
        bufferPhaseDuration=1,
        ):
        """
        """

        #
        gratingMotionByBlock = self.load('stimuli/dg/grating/motion')
        if gratingMotionByBlock is None:
            self.log(f'Session missing processed data for the drifting grating stimulus', level='warning')
            return
        nBlocks = gratingMotionByBlock.size

        #
        for eye in ('left', 'right'):

            #
            saccadeEpochs = self.load(f'saccades/predicted/{eye}/timestamps')
            if saccadeEpochs is None or saccadeEpochs.shape[0] == 0:
                continue

            #
            saccadeOnsetTimestamps = saccadeEpochs[:, 0]
            gratingMotionBySaccade = list()

            if self.cohort in (1, 2, 3):

                #
                probeOnsetTimestamps = self.load('stimuli/dg/probe/timestamps')
                gratingEpochs = list()
                interProbeIntervals = np.diff(probeOnsetTimestamps)

                #
                interBlockIntervalThresholds = np.arange(interBlockIntervalRange[0], interBlockIntervalRange[1], interBlockIntervalStep)
                nBlocksDetected = np.full(interBlockIntervalThresholds.size, np.nan)

                # Attempt #1: Find the indices of the last probe in each block
                thresholdDetermined = False
                for iRun, interBlockIntervalThreshold in enumerate(interBlockIntervalThresholds):
                    lastProbeIndices = np.concatenate([
                        np.where(interProbeIntervals > interBlockIntervalThreshold)[0],
                        np.array([probeOnsetTimestamps.size - 1])
                    ])
                    nBlocksDetected[iRun] = lastProbeIndices.size
                    if lastProbeIndices.size == nBlocks:
                        thresholdDetermined = True
                        break

                if thresholdDetermined:
                    firstProbeIndices = np.concatenate([
                        np.array([0]),
                        lastProbeIndices[:-1] + 1
                    ])
                    gratingEpochs = np.hstack([
                        probeOnsetTimestamps[firstProbeIndices].reshape(-1, 1),
                        probeOnsetTimestamps[lastProbeIndices].reshape(-1, 1)
                    ])

                # Attempt #2: Interpolate the indices of the last probe in each block
                else:
                    nBlocksDetectedZeroed = nBlocksDetected - nBlocks
                    nBlocksDetectedZeroed[nBlocksDetectedZeroed >= 0] = np.nan
                    index = np.nanargmax(nBlocksDetectedZeroed)
                    interBlockIntervalThreshold = interBlockIntervalThresholds[index]
                    lastProbeIndicesDetected = np.concatenate([
                        np.where(interProbeIntervals > interBlockIntervalThreshold)[0],
                        np.array([probeOnsetTimestamps.size - 1])
                    ])
                    medianProbeInterval = np.median(np.diff(lastProbeIndicesDetected))
                    blocksPerInterval = np.around(
                        np.diff(lastProbeIndicesDetected) / medianProbeInterval,
                        0
                    ).astype(int)
                    lastProbeIndices = list()
                    for iProbe, blocksInInterval in enumerate(blocksPerInterval):
                        lastProbeIndices.append(lastProbeIndicesDetected[iProbe])
                        if blocksInInterval > 1:
                            for iProbeMissing in range(blocksInInterval - 1):
                                lastProbeIndices.append(np.nan)
                    lastProbeIndices.append(lastProbeIndicesDetected[-1])
                    lastProbeIndices = np.array(lastProbeIndices)
                    firstProbeIndices = np.concatenate([
                        np.array([0]),
                        lastProbeIndices[:-1] + 1
                    ])
                    gratingEpochs = list()
                    for a in (firstProbeIndices, lastProbeIndices):
                        mask = np.isnan(a)
                        timestamps = np.full(lastProbeIndices.size, np.nan)
                        timestamps[~mask] = probeOnsetTimestamps[a[~mask].astype(int)]
                        timestamps[mask] = np.interp(
                            np.where(mask)[0],
                            np.arange(timestamps.size)[~mask],
                            timestamps[~mask]
                        )
                        gratingEpochs.append(timestamps)
                    gratingEpochs = np.array(gratingEpochs).T
                    if gratingEpochs.shape[0] == nBlocks:
                        thresholdDetermined = True

                if thresholdDetermined == False:
                    self.log(f'Failed to determine the inter-block interval threshold for the drifting grating protocol', level='warning')
                    return

                gratingEpochs[:, 0] -= bufferPhaseDuration
                gratingEpochs[:, 1] += bufferPhaseDuration

            #
            elif self.cohort in (4, 11, 31):

                #
                motionOnsetTimestamps = self.load('stimuli/dg/grating/timestamps')
                motionOffsetTimestamps = self.load('stimuli/dg/iti/timestamps')
                gratingEpochs = np.hstack([
                    motionOnsetTimestamps.reshape(-1, 1),
                    motionOffsetTimestamps.reshape(-1, 1)
                ])

            #
            else:
                self.log(f'Could not extract grating motion during saccades in the {eye} for self in cohort {self.cohort}')
                self.save(f'saccades/predicted/{eye}/gmds', np.array([]).astype(int))
                return

            #
            nBlocks = gratingEpochs.shape[0]
            for saccadeOnsetTimestamp in saccadeOnsetTimestamps:
                searchResult = False
                for blockIndex in range(nBlocks):
                    gratingOnsetTimestamp, gratingOffsetTimestamp = gratingEpochs[blockIndex]
                    gratingMotion = gratingMotionByBlock[blockIndex]
                    if gratingOnsetTimestamp <= saccadeOnsetTimestamp <= gratingOffsetTimestamp:
                        searchResult = True
                        break
                if searchResult:
                    gratingMotionBySaccade.append(gratingMotion)
                else:
                    gratingMotionBySaccade.append(0)

            #
            gratingMotionBySaccade = np.array(gratingMotionBySaccade)
            self.save(f'saccades/predicted/{eye}/gmds', gratingMotionBySaccade)

    def _runSaccadesModule(self):
        """
        """

        self._extractEyePosition()
        self._correctEyePosition()
        self._interpolateEyePosition()
        self._decomposeEyePosition()
        self._reorientEyePosition()
        self._filterEyePosition()
        self._detectPutativeSaccades()

        return