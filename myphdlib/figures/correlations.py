import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.measure import find_contours
from myphdlib.general.toolkit import psth2
from myphdlib.figures.analysis import AnalysisBase
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from scipy import stats
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
                complexity = stats.entropy(sample / sample.sum())

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
    
    def _computeUnitDepths(
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
            try:
                if np.isnan(np.concatenate([insertionPoint, insertionOffset, [insertionAngle], [insertionDepth]])).any():
                    depths = None

                else:
                    insertionPointCorrected = np.copy(insertionPoint)
                    insertionPointCorrected[:2] += insertionOffset
                    try:
                        labels, points, transformed = uloc.localizeUnitsWithInsertionParameters(
                            kilosortOutputFolder=session.home.joinpath('ephys', 'sorting', 'manual'),
                            insertionPoint=insertionPoint,
                            insertionDepth=insertionDepth,
                            insertionAngle=insertionAngle,
                            skullThickness=skullThickness
                        )
                    except:
                        import pdb; pdb.set_trace()
                depths = transformed[:, -1]
            except:
                import pdb; pdb.set_trace()
            data[(str(session.date), session.animal)] = depths

        #
        nUnits = len(self.ukeys)
        depthsByUnit = np.full(nUnits, np.nan)
        for iUnit in range(nUnits):
            self.ukey = self.ukeys[iUnit]
            depths = data[(str(self.session.date), self.session.animal)]
            if depths is None:
                depth = np.nan
            else:
                depth = depths[self.unit.index]
            depthsByUnit[iUnit] = depth

        #
        self.ns['depth'] = depthsByUnit

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
        figsize=(15, 7),
        minimumAmplitude=5,
        receptiveFieldLabels=None
        ):
        """
        """

        # 
        nUnits = len(self.ukeys)

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

        #
        isValidDepth = list()
        zmin = np.nanpercentile(self.ns['depth'], 5)
        zmax = np.nanpercentile(self.ns['depth'], 95)
        for iUnit in range(nUnits):
            if (self.ns['depth'][iUnit] < zmin) or (self.ns['depth'][iUnit] > zmax):
                isValidDepth.append(False)
            else:
                isValidDepth.append(True)
        isValidDepth = np.array(isValidDepth)

        #
        nFeatures = 10
        y = np.full([nUnits, nFeatures], np.nan)
        y[:, 0] = self.ns['rf/on/area']
        y[np.logical_not(hasReceptiveField['on']), 0] = np.nan
        y[:, 1] = self.ns['rf/off/area']
        y[np.logical_not(hasReceptiveField['off']), 1] = np.nan
        y[:, 2] = self.ns['rf/on/complexity']
        y[np.logical_not(hasReceptiveField['on']), 2] = np.nan
        y[:, 3] = self.ns['rf/off/complexity']
        y[np.logical_not(hasReceptiveField['off']), 3] = np.nan
        y[:, 4] = self.ns['rf/on/centroid'][:, 0]
        y[np.logical_not(hasReceptiveField['on']), 4] = np.nan
        y[:, 5] = self.ns['rf/off/centroid'][:, 0]
        y[np.logical_not(hasReceptiveField['off']), 5] = np.nan
        y[:, 6] = self.ns['rf/on/centroid'][:, 1]
        y[np.logical_not(hasReceptiveField['on']), 6] = np.nan
        y[:, 7] = self.ns['rf/off/centroid'][:, 1]
        y[np.logical_not(hasReceptiveField['on']), 7] = np.nan
        y[:, 8] = self.ns['depth']
        y[np.logical_not(isValidDepth), 8] = np.nan

        #
        fig, axs = plt.subplots(
            ncols=len(self.windows) * 2,
            nrows=nFeatures,
            gridspec_kw={'width_ratios': np.tile([2, 1], len(self.windows))},
        )

        # Make subplots
        for iWin in range(self.windows.shape[0]):

            #
            p = np.clip(self.ns['p/pref/real'][:, iWin, 0], 0.001, 1)
            mi = np.clip(self.ns['mi/pref/real'][:, iWin, 0], -1, 1)
            colors = np.full(mi.size, 'k')
            alphas = np.full(p.size, 0.3)
            alphas = np.interp(p, [0, 0.1], [0.5, 0.0])

            # Scatterplots
            j1 = int(2 * iWin)
            for i in range(nFeatures - 1):
                axs[i, j1].scatter(
                    mi,
                    y[:, i],
                    marker='.',
                    s=10,
                    edgecolor='none',
                    alpha=alphas,
                    color=colors
                )

                # Boxplots
                j2 = j1 + 1
                samples = [
                    y[np.logical_and(mi < 0, p < 0.05), i],
                    y[p > 0.05, i],
                    y[np.logical_and(mi > 0, p < 0.05), i]
                ]
                samplesFiltered = list()
                for sample in samples:
                    samplesFiltered.append(np.delete(sample, np.isnan(sample)))
                bp = axs[i, j2].boxplot(
                    samplesFiltered,
                    showfliers=False,
                    medianprops={'color': 'k'},
                    patch_artist=True,
                    widths=0.5,
                    notch=True
                )
                for patch, color in zip(bp['boxes'], ['b', 'w', 'r']):
                    patch.set_facecolor(color)
                    # patch.set_alpha(0.3)
                    # patch.set_edgecolor('none')

        # Special case of modulation x saccade amplitude
        y_ = np.concatenate([
            np.full(nUnits, 0),
            np.full(nUnits, 1),
            np.full(nUnits, 2)
        ]).astype(float)
        yOffset = np.random.normal(loc=0, scale=0.05, size=y_.size)
        y_ += yOffset
        for iWin in range(len(self.windows)):
            # X = np.clip(np.hstack([
            #     self.ns['miext/pref/real/low'][:, iWin, 0],
            #     self.ns['miext/pref/real/medium'][:, iWin, 0],
            #     self.ns['miext/pref/real/high'][:, iWin, 0]
            # ]).ravel(), -1, 1)
            j1 = int(2 * iWin)
            # j2 = j1 + 1
            samples = [
                np.clip(self.ns['miext/pref/real/low'][hasVisualResponse, iWin, 0], -1, 1),
                np.clip(self.ns['miext/pref/real/medium'][hasVisualResponse, iWin, 0], -1, 1),
                np.clip(self.ns['miext/pref/real/high'][hasVisualResponse, iWin, 0], -1, 1)
            ]
            samplesFiltered = list()
            for sample in samples:
                samplesFiltered.append(np.delete(sample, np.isnan(sample)))
            # axs[-1, j1].scatter(X, y_, color='k', marker='.', s=10, alpha=0.15)
            bp = axs[-1, j1].boxplot(
                samplesFiltered,
                vert=False,
                notch=True,
                patch_artist=True,
                medianprops={'color': 'k'},
                showfliers=False,
                widths=0.5
            )
            for patch, color in zip(bp['boxes'], ['w', 'w', 'w']):
                patch.set_facecolor(color)

        #
        for i in range(axs.shape[0]):
            ylim = [np.inf, -np.inf]
            for ax in axs[i, :]:
                y1, y2 = ax.get_ylim()
                if y1 < ylim[0]:
                    ylim[0] = y1
                if y2 > ylim[1]:
                    ylim[1] = y2
            for ax in axs[i, :]:
                ax.set_ylim(ylim)
            for ax in axs[i, 1:]:
                ax.set_yticks([])

        #
        for ax in axs[:-1, :].ravel():
            ax.set_xticks([])
        for ax in axs[-1, 2:]:
            ax.set_xticks([])
        axs[-1, 0].set_xticks([-1, 0, 1])
        axs[-1, 1].set_xticks([1, 2, 3])
        xlim = axs[0, 1].get_xlim()
        axs[-1, 1].set_xticklabels(['Sup.', 'Unm.', 'Enh.'], rotation=90)
        axs[-1, 1].set_xlim(xlim)
        axs[-1, 0].set_xlabel('MI')

        #
        ylabels = (
            r'$RF_{ON}$ area',
            r'$RF_{OFF}$ area',
            r'$RF_{ON}$ comp.',
            r'$RF_{OFF}$ comp.',
            r'$RF_{ON}$ center (x)',
            r'$RF_{OFF}$ center (x)',
            r'$RF_{ON}$ center (y)',
            r'$RF_{OFF}$ center (y)',
            r'Depth (mm)',
            r'Saccade amp.'
        )
        for i, ylabel in enumerate(ylabels):
            axs[i, 0].set_ylabel(ylabel, rotation=0, labelpad=40)

        #
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.3, hspace=0.3)

        return fig, axs
    
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

class RegressionAnalysis(AnalysisBase):
    """
    """

    def _computeWeights(
        self,
        modulation=-1
        ):
        """
        """

        X = np.vstack([
            self.ns['rf/on/area'],
            np.abs(self.ns['rf/on/eccentricity']),
            self.ns['rf/on/complexity'],
            self.ns['rf/off/area'],
            np.abs(self.ns['rf/off/eccentricity']),
            self.ns['rf/off/complexity'],
            self.ns['depth'],
            self.ns['lpi'],
            self.ns['complexity']
        ]).T
        W = np.full([X.shape[1], len(self.windows)], np.nan)
        P = np.full([X.shape[1], len(self.windows)], np.nan)
        reg = LinearRegression()
        scaler = StandardScaler()
        for iWin in range(self.windows.shape[0]):
            mi = self.ns['mi/pref/real'][:, iWin, 0]
            p = self.ns['p/pref/real'][:, iWin, 0]
            if modulation == -1:
                unitIndices = np.where(np.vstack([
                    np.logical_not(np.isnan(X).any(1)),
                    np.logical_not(np.isnan(mi)),
                    np.logical_not(np.isnan(p)),
                    mi < 0,
                    p < 0.05
                ]).all(0))[0]
            elif modulation == 1:
                unitIndices = np.where(np.vstack([
                    np.logical_not(np.isnan(X).any(1)),
                    np.logical_not(np.isnan(mi)),
                    np.logical_not(np.isnan(p)),
                    mi > 0,
                    p < 0.05
                ]).all(0))[0]
            X2 = X[unitIndices]
            X3 = scaler.fit_transform(X2)
            X4 = sm.add_constant(X3)
            y = mi[unitIndices]
            model = sm.OLS(y, X4)
            results = model.fit()
            W[:, iWin] = results.params[1:]
            P[:, iWin] = results.pvalues[1:]

        return W, P
    
    def plotWeights(self, figsize=(7, 5)):
        """
        """

        wSup, pSup = self._computeWeights(modulation=-1)
        wEnh, pEnh = self._computeWeights(modulation=1)
        fig, axs = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True)
        X = self.windows.mean(1)
        Y = np.arange(wSup.shape[0])[::-1]
        mesh = axs[0, 0].pcolor(X, Y, wSup, vmin=-1, vmax=1, cmap='binary_r')
        axs[1, 0].pcolor(X, Y, pSup, vmin=0, vmax=0.05, cmap='binary_r')
        axs[0, 1].pcolor(X, Y, wEnh, vmin=-1, vmax=1, cmap='binary_r')
        axs[1, 1].pcolor(X, Y, pEnh, vmin=0, vmax=0.05, cmap='binary_r')
        # cax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        # fig.colorbar(mesh, cax=cax)
        yticklabels = (
            r'$RF_{ON}$ area',
            r'$RF_{ON}$ ecc.',
            r'$RF_{ON}$ comp.',
            r'$RF_{OFF}$ area',
            r'$RF_{OFF}$ ecc.',
            r'$RF_{OFF}$ comp.',
            r'Depth',
            r'LPI',
            r'# components',
        )
        axs[0, 0].set_yticks(Y)
        axs[0, 0].set_yticklabels(yticklabels)
        axs[1, 0].set_xlabel('Time from saccade (s)')
        axs[0, 0].set_title('Suppression', fontsize=10)
        axs[0, 1].set_title('Enhancement', fontsize=10)
        fig.set_figwidth(figsize[0])
        fig.set_figheight(figsize[1])
        fig.tight_layout()

        return fig, axs