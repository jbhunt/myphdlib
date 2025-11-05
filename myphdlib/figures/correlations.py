import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import find_contours
from myphdlib.general.toolkit import psth2
from myphdlib.figures.analysis import AnalysisBase
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib.gridspec import GridSpec
import scikit_posthocs
from scipy import stats
from statsmodels.stats.anova import AnovaRM
import polars
try:
    import unit_localizer as uloc
except:
    uloc = None
    print('Warning: Failed to import unit_localizer package')

def measurePolygonArea(coords):
    x = coords[:, 0]
    y = coords[:, 1]
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

class CorrelationAnalysis(AnalysisBase):
    """
    """


    def __init__(
        self,
        **kwargs,

        ):
        """
        """

        super().__init__(**kwargs)

        #
        self.examples = (

        )

        return
    
    def _computeReceptiveFields(
        self,
        responseWindow=(0, 0.2),
        baselineWindow=(-0.1, 0),
        binsize=0.01,
        smoothingWindowSize=3,
        subregionSize=10,
        ):
        """
        Compute receptive fields
        """

        nUnits = len(self.ukeys)
        R = {
            'on': list(),
            'off': list()
        }
        for iUnit in range(nUnits):

            #
            end = '\r' if iUnit + 1 != nUnits else '\n'
            print(f'Computing receptive fields for unit {iUnit + 1} out of {nUnits}', end=end)

            #
            self.ukey = self.ukeys[iUnit]

            # Estimate mu and sigma for z-scoring
            t1 = np.floor(self.unit.timestamps.min())
            t2 = np.ceil(self.unit.timestamps.max())
            bins = np.around(np.concatenate([
                np.arange(t1, t2, binsize),
                np.array([t2,])
            ]), 3)
            counts, edges = np.histogram(
                self.unit.timestamps,
                bins=bins
            )
            sample = counts / binsize
            mu = round(sample.mean(), 3)
            sigma = round(sample.std(), 3)

            # Load the event timestamps
            spotTimestamps = {
                'on': list(),
                'off': list(),
            }
            stimulusFields = {
                'on': list(),
                'off': list()
            }
            for block in ('pre', 'post'):
                if self.session.hasDataset(f'stimuli/sn/{block}'):
                    stimulusFields_ = self.session.load(f'stimuli/sn/{block}/fields')
                    spotTimestamps_ = self.session.load(f'stimuli/sn/{block}/timestamps')
                    for f, t in zip(stimulusFields_, spotTimestamps_[0::2]):
                        if np.isnan(t):
                            continue
                        spotTimestamps['on'].append(t)
                        stimulusFields['on'].append(f)
                    for f, t in zip(stimulusFields_, spotTimestamps_[1::2]):
                        if np.isnan(t):
                            continue
                        spotTimestamps['off'].append(t)
                        stimulusFields['off'].append(f)
            for block in spotTimestamps.keys():
                spotTimestamps[block] = np.array(spotTimestamps[block])
                stimulusFields[block] = np.array(stimulusFields[block])

            #
            if spotTimestamps['on'].size == 0:
                for k in R.keys():
                    R[k].append(np.full([10, 17], np.nan)) # TODO: Un-hardcode this
                continue

            #
            nRows, nCols = stimulusFields['on'][0].shape
            values, counts = np.unique(stimulusFields['on'], axis=0, return_counts=True)
            nTrials = counts[0]
            ri = {
                'on': np.full([nRows, nCols], 0.0),
                'off': np.full([nRows, nCols], 0.0)
            }
            for block in ri.keys():

                # Compute baseline FR
                t, M = psth2(
                    spotTimestamps[block],
                    self.unit.timestamps,
                    window=baselineWindow,
                    binsize=binsize
                )
                bl = M.sum(1) / np.diff(baselineWindow).item()

                # Compute response FR
                for t, f in zip(spotTimestamps[block], stimulusFields[block]):
                    i, j = np.where(f != -1) # Identify subregion
                    t_, M = psth2(
                        np.array([t]),
                        self.unit.timestamps,
                        window=responseWindow,
                        binsize=binsize
                    )
                    fr = M.sum(1).item() / np.diff(responseWindow).item()
                    ri[block][i, j] += fr

                # Average over trials
                ri[block] /= nTrials
                ri[block] -= mu
                ri[block] /= sigma

            #
            for k in ri.keys():
                R[k].append(ri[k])

        #
        for k in R.keys():
            R[k] = np.array(R[k])

        # Smoothing
        for k in R.keys():
            for iUnit in range(R[k].shape[0]):
                R[k][iUnit] = gaussian_filter(R[k][iUnit], smoothingWindowSize / subregionSize)

        # Store receptive fields
        self.ns['rf/on/fr'] = R['on']
        self.ns['rf/off/fr'] = R['off']

        return R

    def _measureReceptiveFieldProperties(
        self,
        responseThreshold=0.5,
        subregionSize=10,
        ):
        """
        """

        nUnits = len(self.ukeys)
        x0 = self.ns['rf/on/fr'].shape[2] / 2 * subregionSize
        y0 = self.ns['rf/on/fr'].shape[1] / 2 * subregionSize
        # xmax = self.ns['rf/on/fr'].shape[2] / 2 * subregionSize
        # fig, ax = plt.subplots()

        #
        for k in ('on', 'off'):
            areas = list()
            complexities = list()
            centroids = list()
            eccentricities = list()
            for iUnit in range(nUnits):
                rf = self.ns[f'rf/{k}/fr'][iUnit, :, :].T
                contours = find_contours(
                    rf,
                    level=responseThreshold
                )
                areaBySubregion = list()
                for contour in contours:
                    a = measurePolygonArea(contour) * subregionSize
                    areaBySubregion.append(a)
                areaBySubregion = np.array(areaBySubregion)
                totalArea = areaBySubregion.sum()
                if totalArea == 0:
                    areas.append(np.nan)
                    complexities.append(np.nan)
                    centroids.append([np.nan, np.nan])
                    eccentricities.append(np.nan)
                    continue
                weights = areaBySubregion / totalArea

                #
                centroidBySubregion = np.array([cnt.mean(0) for cnt in contours])
                centroid = np.average(centroidBySubregion, axis=0, weights=weights) * subregionSize
                centroid[0] -= x0
                centroid[1] -= y0
                eccentricity = centroid[-1]
                sample = np.abs(rf.flatten())
                complexity = stats.entropy(sample / sample.sum(), base=2) / np.log2(sample.size)

                #
                areas.append(totalArea)
                complexities.append(complexity)
                centroids.append(centroid)
                eccentricities.append(eccentricity)

                #
                # ax.imshow(self.ns[f'rf/{k}/fr'][iUnit, :, :], vmax=3, vmin=-3)
                # for contour in contours:
                #     ax.plot(contour[:, 1], contour[:, 0], color='k')
                # fig.tight_layout()
                # date, animal, cluster = self.ukeys[iUnit]
                # fig.savefig(f'/home/jbhunt/Desktop/rfs/{date}_{animal}_{cluster}_{k}.png')
                # ax.cla()

            #
            self.ns[f'rf/{k}/area'] = np.array(areas)
            self.ns[f'rf/{k}/eccentricity'] = np.array(eccentricities)
            self.ns[f'rf/{k}/complexity'] = np.array(complexities)
            self.ns[f'rf/{k}/centroid'] = np.array(centroids)

        # plt.close(fig)


        return
    
    def _estimateUnitCoordinates(
        self,
        insertionParameters,
        skullThickness=0.2
        ):
        """
        """

        df = polars.read_csv(insertionParameters)
        data = {}
        nSessions = len(self.sessions)
        for iSession, session in enumerate(self.sessions):
            end = '\r' if (iSession + 1) != nSessions else '\n'
            print(f'Computing unit depths for session {iSession + 1} out of {nSessions}', end=end)
            
            result = df.filter(
                (polars.col("Date") == str(session.date)) & (polars.col("Animal") == session.animal)
            )
            insertionPoint = np.array([
                result['y'].item() if (result['y'].len() != 0 and result['y'].item() is not None) else np.nan,
                result['x'].item() if (result['x'].len() != 0 and result['x'].item() is not None) else np.nan,
                result['z'].item() if (result['z'].len() != 0 and result['z'].item() is not None) else np.nan
            ])
            insertionOffset = np.array([
                result['b'].item() if (result['b'].len() != 0 and result['b'].item() is not None) else np.nan,
                result['a'].item() if (result['a'].len() != 0 and result['a'].item() is not None) else np.nan
            ])
            insertionAngle = result['Angle'].item() if (result['Angle'].len() != 0 and result['Angle'].item() is not None) != 0 else np.nan
            insertionDepth = result['Depth'].item() if (result['Depth'].len() != 0 and result['Depth'].item() is not None) else np.nan

            #
            if np.isnan(np.concatenate([insertionPoint, insertionOffset, [insertionAngle], [insertionDepth]])).any():
                transformed = None
            else:
                insertionPointCorrected = np.copy(insertionPoint)
                insertionPointCorrected[:2] += insertionOffset
                labels, points, transformed = uloc.localizeUnitsWithInsertionParameters(
                    kilosortOutputFolder=session.home.joinpath('ephys', 'sorting', 'manual'),
                    insertionPoint=insertionPoint,
                    insertionDepth=insertionDepth,
                    insertionAngle=insertionAngle,
                    skullThickness=skullThickness
                )
            data[(str(session.date), session.animal)] = transformed

        #
        nUnits = len(self.ukeys)
        coordinatesByUnit = np.full([nUnits, 3], np.nan)
        for iUnit in range(nUnits):
            self.ukey = self.ukeys[iUnit]
            coordinates = data[(str(self.session.date), self.session.animal)]

            if coordinates is None:
                ap, ml, dv = np.nan, np.nan, np.nan
            else:
                ap, ml, dv = coordinates[self.unit.index]
            coordinatesByUnit[iUnit] = np.array([ap, ml, dv])

        #
        self.ns['coordinates'] = coordinatesByUnit

        return
    
    def _lookupPolarityIndices(self):
        """
        """

        data = {}
        for session in self.sessions:
            skey = (str(session.date), session.animal)
            if session.hasDataset('metrics/lpi') == False:
                data[skey] = None
                continue
            lpi = session.load('metrics/lpi')
            data[skey] = lpi

        nUnits = len(self.ukeys)
        polarityIndices = np.full(nUnits, np.nan)
        for iUnit in range(nUnits):
            date, animal, cluster = self.ukeys[iUnit]
            skey = (str(date), animal)
            lpi = data[skey]
            if lpi is None:
                continue
            self.ukey = self.ukeys[iUnit]
            polarityIndices[iUnit] = lpi[self.unit.index]

        #
        self.ns['lpi'] = polarityIndices

        return
    
    def _lookupResponseComplexity(self):
        """
        """

        nUnits = len(self.ukeys)
        complexity = np.full(nUnits, np.nan)
        for iUnit in range(nUnits):
            params = self.ns['params/pref/real/extra'][iUnit]
            k = (np.invert(np.isnan(params)).sum() - 1) / 3
            complexity[iUnit] = k
        self.ns['complexity'] = complexity

        return
    
    def _measureSaccadeResponseAmplitude(self):
        """
        """

        nUnits = len(self.ukeys)
        responseAmplitudes = np.full(nUnits, np.nan)
        for iUnit in range(nUnits):
            psthTemporalSaccades = self.ns['psths/temporal/real'][iUnit]
            psthNasalSaccades = self.ns['psths/nasal/real'][iUnit]
            psth = np.concatenate([psthTemporalSaccades, psthNasalSaccades])
            iMax = np.argmax(np.abs(psth))
            responseAmplitudes[iUnit] = psth[iMax]
        standardized = (responseAmplitudes - np.nanmean(responseAmplitudes)) / np.nanstd(responseAmplitudes)
        self.ns['amplitude/saccade'] = standardized

        return
    
    def _lookupProbeResponseAmplitude(self):
        """
        """

        nUnits = len(self.ukeys)
        responseAmplitudes = np.full(nUnits, np.nan)
        for iUnit in range(nUnits):
            responseAmplitudes[iUnit] = self.ns['params/pref/real/extra'][iUnit, 0]
        standardized = (responseAmplitudes - np.nanmean(responseAmplitudes)) / np.nanstd(responseAmplitudes)
        self.ns['amplitude/probe'] = standardized

        return
    
    def plotCorrelations(
        self,
        figsize=(8, 15),
        minimumAmplitude=5,
        receptiveFieldLabels=None,
        miRange=(-2, 2),
        ):
        """
        """

        # 
        nUnits = len(self.ukeys)
        nTests = 0
        cmap = plt.get_cmap('coolwarm', 3)
        blue, red = cmap(0), cmap(2)

        # Filter by response amplitude
        hasVisualResponse = list()
        for iUnit in range(len(self.ukeys)):
            a = abs(self.ns['params/pref/real/extra'][iUnit, 0])
            if a >= minimumAmplitude:
                hasVisualResponse.append(True)
            else:
                hasVisualResponse.append(False)
        hasVisualResponse = np.array(hasVisualResponse)

        # Filter by receptive field
        hasReceptiveField = {
            'on': np.full(nUnits, False),
            'off': np.full(nUnits, False)
        }
        if receptiveFieldLabels is not None:
            with open(receptiveFieldLabels, 'r') as stream:
                lines = stream.readlines()[1:]
            for ln in lines:
                date, animal, cluster, on, off = ln.rstrip('\n').split(',')
                ukey = (date, animal, int(cluster))
                iUnit = self._indexUnitKey(ukey)
                hasReceptiveField['on'][iUnit] = bool(on)
                hasReceptiveField['off'][iUnit] = bool(off)
        else:
            for k in ('on', 'off'):
                hasReceptiveField[k] = np.full(nUnits, True)

        # Filter by valid unit depth
        isValidDepth = list()
        zmin = np.nanpercentile(self.ns['coordinates'][:, -1], 5)
        zmax = np.nanpercentile(self.ns['coordinates'][:, -1], 95)
        for iUnit in range(nUnits):
            if (self.ns['coordinates'][iUnit, -1] < zmin) or (self.ns['coordinates'][iUnit, -1] > zmax):
                isValidDepth.append(False)
            else:
                isValidDepth.append(True)
        isValidDepth = np.array(isValidDepth)

        # Masks used to filter units for each IV
        masksByFeature = (
            np.vstack([hasReceptiveField['on'], isValidDepth, hasVisualResponse]).all(0), # ON RF area
            np.vstack([hasReceptiveField['off'], isValidDepth, hasVisualResponse]).all(0), # OFF RF area
            np.vstack([hasReceptiveField['on'], isValidDepth, hasVisualResponse]).all(0), # ON RF complexity
            np.vstack([hasReceptiveField['off'], isValidDepth, hasVisualResponse]).all(0), # OFF RF complexity
            np.vstack([hasReceptiveField['on'], isValidDepth, hasVisualResponse]).all(0), # ON RF x-position
            np.vstack([hasReceptiveField['off'], isValidDepth, hasVisualResponse]).all(0), # OFF RF x-position
            np.vstack([hasReceptiveField['on'], isValidDepth, hasVisualResponse]).all(0), # ON RF y-position
            np.vstack([hasReceptiveField['off'], isValidDepth, hasVisualResponse]).all(0), # OFF RF y-position
            np.vstack([np.logical_or(hasReceptiveField['on'], hasReceptiveField['off']), isValidDepth, hasVisualResponse]).all(0), # ML coord
            np.vstack([np.logical_or(hasReceptiveField['on'], hasReceptiveField['off']), isValidDepth, hasVisualResponse]).all(0), # DV coord
        )

        #
        nFeatures = len(masksByFeature)
        ivs = np.full([nUnits, nFeatures], np.nan)
        ivs[:, 0] = self.ns['rf/on/area']
        ivs[:, 1] = self.ns['rf/off/area']
        ivs[:, 2] = self.ns['rf/on/complexity']
        ivs[:, 3] = self.ns['rf/off/complexity']
        ivs[:, 4] = self.ns['rf/on/centroid'][:, 0]
        ivs[:, 5] = self.ns['rf/off/centroid'][:, 0]
        ivs[:, 6] = self.ns['rf/on/centroid'][:, 1]
        ivs[:, 7] = self.ns['rf/off/centroid'][:, 1]
        ivs[:, 8] = self.ns['coordinates'][:, 1]
        ivs[:, 9] = self.ns['coordinates'][:, 2]

        # Jitter AP coordinate
        # ivs[:, 8] += np.random.normal(loc=0, scale=0.007, size=ivs.shape[0])

        # Reflect ML coordinate
        ivs[:, 8] *= -1

        #
        nRows = len(self.windows) + 2
        fig, axs = plt.subplots(
            nrows=nRows * 2,
            ncols=nFeatures,
            gridspec_kw={'height_ratios': np.tile([2, 1], nRows)}
        )

        #
        statistics = np.full([nRows, nFeatures, 2], np.nan)

        # DV = Probe response complexity
        iTop, iBottom = 0, 1
        dv = self.ns['globals/labels']
        for j in range(nFeatures):
            samples = [
                ivs[np.logical_and(dv == -1, masksByFeature[0]), j],
                ivs[np.logical_and(dv ==  1, masksByFeature[0]), j],
                ivs[np.logical_and(dv ==  2, masksByFeature[0]), j],
                ivs[np.logical_and(dv ==  3, masksByFeature[0]), j]    
            ]
            samplesFiltered = list()
            for sample in samples:
                samplesFiltered.append(np.delete(sample, np.isnan(sample)))
            bp = axs[iTop, j].boxplot(
                samplesFiltered,
                vert=False,
                notch=True,
                patch_artist=True,
                medianprops={'color': 'k'},
                showfliers=False,
                widths=0.5
            )
            for patch, color in zip(bp['boxes'], ['0.8', '0.8', '0.8', '0.8']):
                patch.set_facecolor(color)

            # Stats
            result = stats.kruskal(*samplesFiltered)
            nTests += 1
            statistics[0, j, 0] = result.statistic
            statistics[0, j, 1] = result.pvalue

        # DV = Modulation index
        for iWin in range(self.windows.shape[0]):

            #
            p = self.ns['p/pref/real'][:, iWin, 0]
            mi = np.clip(self.ns['mi/pref/real'][:, iWin, 0], *miRange)
            cats = (
                p >= 0.01,
                np.logical_and(p >= 0.001, p < 0.05),
                np.logical_and(p >= 0, p < 0.001),
            )

            # Scatterplots
            iTop = int(2 * iWin) + 2
            iBottom = iTop + 1
            for j in range(nFeatures):

                #
                mask = masksByFeature[j]

                #
                for color, cat, alpha in zip(['0.75', '0.0', '0.0'], cats, [1, 1, 1]):
                    axs[iTop, j].scatter(
                        ivs[np.logical_and(mask, cat), j],
                        mi[np.logical_and(mask, cat)],
                        marker='.',
                        s=5,
                        edgecolor='none',
                        color=color,
                        alpha=alpha,
                        rasterized=True
                    )

                # Boxplots
                samples = [
                    ivs[np.vstack([mask, mi < 0, p < 0.05]).all(0), j],
                    ivs[np.vstack([mask, p > 0.05]).all(0),         j],
                    ivs[np.vstack([mask, mi > 0, p < 0.05]).all(0), j]
                ]
                samplesFiltered = list()
                for sample in samples:
                    samplesFiltered.append(np.delete(sample, np.isnan(sample)))
                bp = axs[iBottom, j].boxplot(
                    samplesFiltered,
                    showfliers=False,
                    medianprops={'color': 'k'},
                    patch_artist=True,
                    widths=0.5,
                    notch=True,
                    vert=False
                )
                for patch, color in zip(bp['boxes'], [blue, 'w', red]):
                    patch.set_facecolor(color)

                # Stats
                result = stats.kruskal(*samplesFiltered)
                nTests += 1
                statistics[iWin + 1, j, 0] = result.statistic
                statistics[iWin + 1, j, 1] = result.pvalue

        # DV = Fictive saccadic modulation
        iTop, iBottom = -2, -1
        dv = self.ns['mi/pref/real'][:, 5, 0] - self.ns['mi/pref/fictive'][:, 0, 0]
        yrange = (-1 * 2 * np.max(np.abs(miRange)), 2 * np.max(np.abs(miRange)))
        dv = np.clip(dv, *yrange)
        p = self.ns['p/pref/real'][:, 5, 0]
        cats = (
            p >= 0.01,
            np.logical_and(p >= 0.001, p < 0.05),
            np.logical_and(p >= 0, p < 0.001),
        )
        for j in range(nFeatures):
            for color, cat, alpha in zip(['0.75', '0.0', '0.0'], cats, [1, 1, 1]):
                axs[iTop, j].scatter(
                    ivs[np.logical_and(mask, cat), j],
                    np.clip(dv[np.logical_and(mask, cat)], *yrange),
                    color=color,
                    marker='.',
                    s=5,
                    alpha=alpha,
                    edgecolor='none',
                    rasterized=True
                )
            pairsMask = np.logical_not(np.vstack([
                np.isnan(dv[mask]),
                np.isnan(ivs[mask, j])
            ]).any(0))
            samplesFiltered = [
                dv[mask][pairsMask],
                ivs[mask, j][pairsMask]
            ]
            result = stats.spearmanr(*samplesFiltered)
            nTests += 1
            statistics[-1, j, 0] = np.nan
            statistics[-1, j, 1] = result.pvalue

        #
        xlims = (
            None,
            None,
            None,
            None,
            (-90, 90),
            (-90, 90),
            (-55, 55),
            (-55, 55),
            None,
            None,
            None
        )
        for j in range(nFeatures):
            if xlims[j] is not None:
                for ax in axs[:, j]:
                    ax.set_xlim(xlims[j])
            else:
                xlim = [np.inf, -np.inf]
                for ax in axs[:, j]:
                    if ax.has_data() == False:
                        continue
                    x1, x2 = ax.get_xlim()
                    if x1 < xlim[0]:
                        xlim[0] = x1
                    if x2 > xlim[1]:
                        xlim[1] = x2
                for ax in axs[:, j]:
                    ax.set_xlim(xlim)
            for ax in axs[:-1, j]:
                ax.set_xticks([])

        #
        for i in range(axs.shape[0]):
            ylim = [np.inf, -np.inf]
            for ax in axs[i, :]:
                if ax.has_data() == False:
                    continue
                y1, y2 = ax.get_ylim()
                if y1 < ylim[0]:
                    ylim[0] = y1
                if y2 > ylim[1]:
                    ylim[1] = y2
            for ax in axs[i, 1:]:
                ax.set_yticks([])
            if np.isfinite(ylim).sum() != 2:
                continue
            for ax in axs[i, :]:
                ax.set_ylim(ylim)

        #
        for ax in axs.ravel():
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
        for ax in axs[1::2, 0]:
            ax.set_yticklabels([])

        #
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.3, hspace=0.3)

        return fig, axs, statistics, nTests
    
    def plotModulationBySaccadeAmplitude(
        self,
        minimumAmplitude=5,
        perisaccadicWindowIndices=(3, 4, 5, 6),
        miRange=(-3, 3),
        figsize=(8, 4)
        ):
        """
        """

        # Filter by response amplitude
        hasVisualResponse = list()
        for iUnit in range(len(self.ukeys)):
            a = abs(self.ns['params/pref/real/extra'][iUnit, 0])
            if a >= minimumAmplitude:
                hasVisualResponse.append(True)
            else:
                hasVisualResponse.append(False)
        hasVisualResponse = np.array(hasVisualResponse)

        #
        mi = self.ns['mi/pref/real'][:, perisaccadicWindowIndices, 0]
        p = self.ns['p/pref/real'][:, perisaccadicWindowIndices, 0]
        isSuppressed = np.vstack([
            np.logical_and(mi[:, 0] < 0, p[:, 0] < 0.05),
            np.logical_and(mi[:, 1] < 0, p[:, 1] < 0.05),
            np.logical_and(mi[:, 2] < 0, p[:, 2] < 0.05)
        ]).any(0)

        #
        isEnhanced = np.vstack([
            np.logical_and(mi[:, 0] > 0, p[:, 0] < 0.05),
            np.logical_and(mi[:, 1] > 0, p[:, 1] < 0.05),
            np.logical_and(mi[:, 2] > 0, p[:, 2] < 0.05)
        ]).any(0)

        #
        fig, axs = plt.subplots(nrows=2, ncols=10, sharex=True)
        cmap = plt.get_cmap('coolwarm', 3)
        colors = [cmap(0), cmap(2)]

        fig2, axs2 = plt.subplots(nrows=1, ncols=2)

        stats_omnibus = np.full([2, len(self.windows)], np.nan)
        stats_posthoc = np.full([2, len(self.windows), 3, 3], np.nan)
        for j in np.arange(len(self.windows)):

            #
            for i in range(2):
                color = colors[i]
                if i == 0:
                    isModulated = isSuppressed
                else:
                    isModulated = isEnhanced
                unitIndices = np.where(np.logical_and(
                    hasVisualResponse,
                    isModulated
                ))[0]
                # a = np.clip(unitIndices.size / len(self.ukeys) * 2.5, 0, 1)

                #
                samples = [
                    self.ns['miext/pref/real/low'][unitIndices, j, 0],
                    self.ns['miext/pref/real/medium'][unitIndices, j, 0],
                    self.ns['miext/pref/real/high'][unitIndices, j, 0]
                ]
                samplesFiltered = np.delete(
                    samples,
                    np.isnan(samples).any(axis=0),
                    axis=1
                )
                bp = axs[i, j].boxplot(
                    samplesFiltered.T,
                    vert=True,
                    notch=True,
                    patch_artist=True,
                    medianprops={'color': 'k', 'alpha': 1.0},
                    whiskerprops={'alpha': 1.0},
                    capprops={'alpha': 1.0},
                    showfliers=False,
                    widths=0.5,
                )
                for patch in bp['boxes']:
                    patch.set_facecolor(color)
                    patch.set_alpha(1.0)
                
                # Run stats
                res = stats.friedmanchisquare(*samplesFiltered)
                if j == 5 and i == 0:
                    print(res)
                    print(np.median(samplesFiltered, axis=1))
                stats_omnibus[i, j] = res.pvalue
                res = scikit_posthocs.posthoc_nemenyi_friedman(samplesFiltered.T)
                stats_posthoc[i, j, :] = np.array(res)
                # res = stats.mannwhitneyu(samplesFiltered[0], samplesFiltered[2]) # Small vs large
                # stats_posthoc[i, j, 2, 0] = res.pvalue
                # res = stats.mannwhitneyu(samplesFiltered[1], samplesFiltered[2]) # Medium vs large
                # stats_posthoc[i, j, 2, 1] = res.pvalue
                # res = stats.mannwhitneyu(samplesFiltered[0], samplesFiltered[1]) # Small vs medium
                # stats_posthoc[i, j, 1, 0] = res.pvalue

                #
                if (j == 3 and i == 1) or (j == 5 and i == 0):
                    if i == 0:
                        isModulated = np.logical_and(
                            self.ns['mi/pref/real'][:, j, 0] < 0,
                            self.ns['p/pref/real'][:, j, 0] < 0.05
                        )
                    elif i == 1:
                        isModulated = np.logical_and(
                            self.ns['mi/pref/real'][:, j, 0] > 0,
                            self.ns['p/pref/real'][:, j, 0] < 0.05
                        )
                    unitIndices = np.where(np.logical_and(
                        hasVisualResponse,
                        isModulated
                    ))[0]
                    samples = [
                        self.ns['miext/pref/real/low'][unitIndices, j, 0],
                        self.ns['miext/pref/real/medium'][unitIndices, j, 0],
                        self.ns['miext/pref/real/high'][unitIndices, j, 0]
                    ]
                    samplesFiltered = np.delete(
                        samples,
                        np.isnan(samples).any(axis=0),
                        axis=1
                    )
                    # bp = axs2[i].boxplot(
                    #     samplesFiltered.T,
                    #     positions=[1 - 0.2, 2 - 0.2, 3 - 0.2],
                    #     widths=[0.3, 0.3, 0.3],
                    #     vert=True,
                    #     notch=True,
                    #     patch_artist=True,
                    #     medianprops={'color': 'k', 'alpha': 1.0},
                    #     whiskerprops={'alpha': 1.0},
                    #     capprops={'alpha': 1.0},
                    #     whis=[5, 95],
                    #     showfliers=False,
                    # )
                    # for patch in bp['boxes']:
                    #     patch.set_facecolor('none')
                    #     patch.set_alpha(1.0)
                    ys = list()
                    for ii in np.arange(3):
                        # miRange = (
                        #     np.nanpercentile(samplesFiltered[ii], 5),
                        #     np.nanpercentile(samplesFiltered[ii], 95)
                        # )
                        y = np.clip(samplesFiltered[ii], *miRange)
                        x = np.random.normal(loc=0, scale=0.07, size=y.size) + (ii + 1)
                        axs2[i].scatter(x, y, marker='.', s=3, color='0.8')
                        q1, q2, q3 = (
                            np.nanpercentile(samplesFiltered[ii], 25),
                            np.nanpercentile(samplesFiltered[ii], 50),
                            np.nanpercentile(samplesFiltered[ii], 75)
                        )
                        axs2[i].vlines(ii + 1, q1, q3, color='k')
                        # axs2[i].hlines(q2, ii + 1 - 0.15, ii + 1 + 0.15, color='k')
                        axs2[i].scatter(ii + 1, q2, marker='o', edgecolor='none', color='k', s=25)
                        ys.append(q2)
                    axs2[i].plot([1, 2, 3], ys, color='k')

            
            #
            # samplesFiltered = np.array(samplesFiltered)
            # for y in samplesFiltered.T:
            #     ax.plot(
            #         np.arange(3) + 1,
            #         y,
            #         color='k',
            #         alpha=0.25,
            #        lw=0.5,
            #         # zorder=-1
            #     )
            #     # ax.scatter(np.arange(3) + 1 + 0.15, y, marker='.', color='k', alpha=0.15, s=20, edgecolor='none')

        for ax in axs[:, 1:].ravel():
            ax.set_yticklabels([])
            for sp in ('top', 'right', 'left'):
                ax.spines[sp].set_visible(False)
        for ax in axs[:, 0]:
            for sp in ('top', 'right'):
                ax.spines[sp].set_visible(False)
        axs[0, 0].set_xticks([1, 2, 3])
        for ax in axs[1, :]:
            ax.set_xticklabels([])
        ylim = [np.inf, -np.inf]
        for ax in axs.ravel():
            y1, y2 = ax.get_ylim()
            if y1 < ylim[0]:
                ylim[0] = y1
            if y2 > ylim[1]:
                ylim[1] = y2
        for ax in axs.ravel():
            ax.set_ylim(ylim)
        for ax in axs[:, 1:].ravel():
            ax.set_yticks([])
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, axs, stats_omnibus, stats_posthoc