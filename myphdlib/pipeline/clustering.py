import numpy as np
from myphdlib.general.toolkit import psth2
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score

def translatePreference(
    session,
    probeMotion=-1,
    saccadeLabel=-1
    ):
    """
    """

    probeDirection = 'left' if probeMotion == -1 else 'right'
    saccadeDirection = 'temporal' if saccadeLabel == -1 else 'nasal'
    preference = None
    if session.eye == 'left':
        if probeDirection == 'left':
            if saccadeDirection == 'nasal':
                preference = 'preferred'
            else:
                preference = 'nonpreferred'
        else:
            if saccadeDirection == 'nasal':
                preference = 'nonpreferred'
            else:
                preference = 'preferred'
    else:
        pass

    return preference

class ClusterProcessingMixin():
    """
    """

    def _collectSamplesForClustering(
        self,
        responseWindow=(-0.2, 0.5),
        baselineWindows=([-0.2, 0], [-2.2, -2]),
        normalizingWindow=(0, 0.5),
        perisaccadicWindow=(-0.05, 0.1),
        minimumBaselineActivity=0.5,
        minimumResponseAmplitude=0.5,
        binsize=0.02,
        smoothingKernelWidth=0.01,
        normalizationMethod=1,
        overwrite=False
        ):
        """
        """

        #
        peristimulusWindow = (
            perisaccadicWindow[1] * -1,
            perisaccadicWindow[0] * -1
        )

        #
        tBins, nTrials, nBins = psth2(
            np.array([0]),
            np.array([0]),
            window=responseWindow,
            binsize=binsize,
            returnShape=True,
        )
        nUnits = self.population.count()

        #
        binIndicesForNormalizingWindow = np.where(np.logical_and(
            tBins <= normalizingWindow[1],
            tBins >= normalizingWindow[0]
        ))[0]

        #
        xProbe = np.full([nUnits, nBins], np.nan)
        xSaccade = np.full([nUnits, nBins], np.nan)
        include = np.full(nUnits, False)

        #
        if self.probeTimestamps is None:
            self.save(f'peths/rProbe/dg/preferred', xProbe)
            self.save(f'peths/rSaccade/dg/preferred', xSaccade)
            return

        #
        probeIndices = {
            'left': np.where(np.logical_and(
                np.logical_or(
                    self.probeLatencies < perisaccadicWindow[0],
                    self.probeLatencies > perisaccadicWindow[1]
                ),
                self.gratingMotionDuringProbes == -1,
            ))[0],
            'right': np.where(np.logical_and(
                np.logical_or(
                    self.probeLatencies < perisaccadicWindow[0],
                    self.probeLatencies > perisaccadicWindow[1]
                ),
                self.gratingMotionDuringProbes == +1,
            ))[0],
        }

        #
        saccadeIndices = {
            'nasal': np.where(np.logical_and(
                np.logical_or(
                    self.saccadeLatencies < peristimulusWindow[0],
                    self.saccadeLatencies > peristimulusWindow[1]
                ),
                self.saccadeLabels == -1,
            ))[0],
            'temporal': np.where(np.logical_and(
                np.logical_or(
                    self.saccadeLatencies < peristimulusWindow[0],
                    self.saccadeLatencies > peristimulusWindow[1]
                ),
                self.saccadeLabels == +1,
            ))[0],
        }

        #
        for iUnit, unit in enumerate(self.population):

            #
            if iUnit + 1 == nUnits:
                end = None
            else:
                end = '\r'
            self.log(f'Extracting preferred and non-preferred responses to events ({iUnit + 1} / {nUnits} units)', end=end)
            
            #
            peths = {
                ('probe', 'left'): None,
                ('probe', 'right'): None,
                ('saccade', 'nasal'): None,
                ('saccade', 'temporal'): None
            }

            #
            checks = np.array([False, False])

            #
            for probeMotion, probeDirection in zip([-1, 1], ['left', 'right']):
                probeTimestamps = self.probeTimestamps[probeIndices[probeDirection]]
                t, M = psth2(
                    probeTimestamps,
                    unit.timestamps,
                    window=baselineWindows[0],
                    binsize=None
                )
                mu = M.flatten().mean() / np.diff(baselineWindows[0])
                t, fr = unit.peth(
                    probeTimestamps,
                    responseWindow=responseWindow,
                    binsize=binsize,
                )
                a = np.max(np.abs(fr[binIndicesForNormalizingWindow]))
                peths[('probe', probeDirection)] = fr - mu

                # Check for baseline activity and response amplitude
                if minimumResponseAmplitude is None:
                    checks[0] = True
                elif a > minimumResponseAmplitude:
                    checks[0] = True
                if minimumBaselineActivity is None:
                    checks[1] = True
                elif mu > minimumBaselineActivity:
                    checks[1] = True
            
            # Skip to next unit if fails to meet criteria
            if checks.all() == False:
                continue
            include[iUnit] = True

            #
            for saccadeLabel, saccadeDirection in zip([-1, 1], ['temporal', 'nasal']):
                saccadeTimestamps = self.saccadeTimestamps[saccadeIndices[saccadeDirection], 0]
                t, M = psth2(
                    saccadeTimestamps,
                    unit.timestamps,
                    window=baselineWindows[1],
                    binsize=None
                )
                mu = M.flatten().mean() / np.diff(baselineWindows[1])
                t, fr = unit.peth(
                    saccadeTimestamps,
                    responseWindow=responseWindow,
                    binsize=binsize,
                    kde=True,
                    sd=smoothingKernelWidth
                )
                peths[('saccade', saccadeDirection)] = fr - mu

            #
            responseAmplitudes = {
                ('probe', 'left'): np.max(np.abs(peths[('probe', 'left')][binIndicesForNormalizingWindow])),
                ('probe', 'right'): np.max(np.abs(peths[('probe', 'right')][binIndicesForNormalizingWindow])),
                ('saccade', 'nasal'): np.max(np.abs(peths[('saccade', 'nasal')][binIndicesForNormalizingWindow])),
                ('saccade', 'temporal'): np.max(np.abs(peths[('saccade', 'temporal')][binIndicesForNormalizingWindow]))
            }
            if responseAmplitudes[('probe', 'left')] >= responseAmplitudes[('probe', 'right')]:
                probeDirection = 'left'
                saccadeDirection = 'nasal'
            else:
                probeDirection = 'right'
                saccadeDirection = 'temporal'

            #
            xProbe[iUnit] = peths[('probe', probeDirection)] / responseAmplitudes[('probe', probeDirection)]
            if normalizationMethod == 1:
                xSaccade[iUnit] = peths[('saccade', saccadeDirection)] / responseAmplitudes[('saccade', saccadeDirection)]
            elif normalizationMethod == 2:
                xSaccade[iUnit] = peths[('saccade', saccadeDirection)] / responseAmplitudes[('probe', probeDirection)]

        #
        self.save('peths/rProbe/dg/preferred', xProbe)
        self.save('peths/rSaccade/dg/preferred', xSaccade)

        return

    def _runClusterModule(self):
        """
        """

        self._collectSamplesForClustering()

        return

def filterUnits(
    session,
    responseWindow=(0, 0.3),
    baselineWindow=(-0.2, 0),
    minimumBaselineActivity=0.5,
    minimumResponseAmplitude=0.5,
    ):
    """
    """

    #
    rProbe_, metadata = session.load('peths/rProbe/left', returnMetadata=True)
    t = metadata['t']
    binIndicesForBaselineWindow = np.where(np.logical_and(
        t >= baselineWindow[0],
        t <= baselineWindow[1]
    ))[0]
    binIndicesForResponseWindow = np.where(np.logical_and(
        t >= responseWindow[0],
        t <= responseWindow[1]
    ))[0]

    #
    rProbe = {
        'left': session.load(f'peths/rProbe/left'),
        'right': session.load(f'peths/rProbe/right')
    }

    #
    include = np.full(session.population.count(), True)
    for iUnit in range(session.population.count()): 
        baselineActivity = 0
        responseAmplitude = 0
        for probeDirection in ('left', 'right'):
            mu = rProbe[probeDirection][iUnit][binIndicesForBaselineWindow].mean()
            if mu > baselineActivity:
                baselineActivity = mu
            fr = np.abs(rProbe[probeDirection][iUnit][binIndicesForResponseWindow]).max()
            if fr > responseAmplitude:
                responseAmplitude = fr
        if baselineActivity < minimumBaselineActivity or responseAmplitude < minimumResponseAmplitude:
            include[iUnit] = False

    return include

def clusterUnits(
    sessions,
    k=None,
    kmin=1,
    kmax=15,
    model='gmm',
    minimumBaselineActivity=0.5,
    minimumResponseAmplitude=1,
    ):
    """
    """

    #
    for session in sessions:
        session.population.unfilter()

    # Collect samples
    samples = list()
    inclusionMasksBySession = list()
    for session in sessions:
        zProbe = session.load(f'peths/rProbe/preferred')
        zSaccade = session.load(f'peths/rSaccade/preferred')
        inclusionMask = np.vstack([
            filterUnits(
                session,
                minimumBaselineActivity=minimumBaselineActivity,
                minimumResponseAmplitude=minimumResponseAmplitude
            ),
            np.invert(np.isnan(zProbe).all(1)),
            np.invert(np.isnan(zSaccade).all(1))
        ]).all(0)
        inclusionMasksBySession.append(inclusionMask)
        for unit in session.population:
            sample = np.concatenate([
                zProbe[unit.index, :],
                zSaccade[unit.index, :]
            ])
            if np.isnan(sample).all():
                continue
            else:
                samples.append(sample)
    X = np.array(samples)

    #
    if k is None:
        krange = np.arange(kmin, kmax + 1, 1)
    else:
        krange = None

    # Hierarchical clustering
    if model == 'agg':
        if k is None:
            grid = {
                'n_clusters': krange,
            }
            search = GridSearchCV(
                AgglomerativeClustering(),
                param_grid=grid,
                scoring=lambda estimator, X: -1 * silhouette_score(X, estimator.labels_)
            )
            search.fit(X)
            model = search.best_estimator_
            scores = -1 * search.cv_results_['mean_test_score']
        else:
            model = AgglomerativeClustering(n_clusters=k).fit(X)
            scores = None
        labels = model.labels_

    # Gaussian mixture model
    elif model == 'gmm':
        if k is None:
            grid = {
                'n_components': krange,
            }
            search = GridSearchCV(
                GaussianMixture(max_iter=10000, covariance_type='diag'),
                param_grid=grid,
                scoring=lambda estimator, X: -1 * estimator.bic(X)
            )
            search.fit(X)
            model = search.best_estimator_
            scores = -1 * search.cv_results_['mean_test_score']
        else:
            model = GaussianMixture(n_components=k).fit(X)
            scores = None
        labels = model.predict(X)

    # K-means clustering
    elif model == 'kmeans':
        if k is None:
            grid = {
                'n_clusters': krange,
            }
            search = GridSearchCV(
                KMeans(),
                param_grid=grid,
                scoring=lambda estimator, X: -1 * silhouette_score(X, estimator.labels_)
            )
            search.fit(X)
            model = search.best_estimator_
            scores = -1 * search.cv_results_['mean_test_score']
        else:
            model = AgglomerativeClustering(n_clusters=k).fit(X)
            scores = None
        labels = model.labels_

    #
    xDecomposed = PCA(n_components=2).fit_transform(X)

    #
    nClusters = np.unique(labels).size
    print(f'INFO: Event-related activity optimally fit to {nClusters} clusters')

    # NOTE: This needs to be done for estimators that don't implement a 'predict' method
    nUnitsIncludedBySession = np.array([
        m.sum() for m in inclusionMasksBySession
    ])
    splitIndices = np.cumsum(nUnitsIncludedBySession)[:-1]
    iterable = zip(
        np.split(labels, splitIndices),
        np.split(xDecomposed, splitIndices, axis=0)
    )
    for sessionIndex, (clusterLabels, unitCoords) in enumerate(iterable):

        #
        session = sessions[sessionIndex]
        inclusionMask = inclusionMasksBySession[sessionIndex]

        # Save predicted cluster labels
        clusterLabels_= np.full(session.population.count(), np.nan)
        clusterLabels_[inclusionMask] = clusterLabels
        session.save(f'clustering/probe/labels', clusterLabels_)

        # Save subspace coordinates
        unitCoords_ = np.full([session.population.count(), 2], np.nan)
        unitCoords_[inclusionMask, :] = unitCoords
        session.save(f'clustering/probe/coords', unitCoords_)

    # Save the clustering performance
    if scores is not None:
        session.save(f'clustering/krange', krange)
        session.save(f'clustering/scores', scores)

    return krange, scores