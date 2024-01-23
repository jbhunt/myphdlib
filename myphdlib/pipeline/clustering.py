import numpy as np
from myphdlib.general.toolkit import psth2
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import silhouette_score

class ClusterProcessingMixin():
    """
    """

    def _collectTrainingSamples(
        self,
        responseWindow=(-0.2, 0.5),
        baselineWindowProbe=(-0.2, 0),
        baselineWindowSaccade=(-2.2, -2),
        perisaccadicWindow=(-0.05, 0.1),
        minimumBaselineActivity=None,
        minimumResponseAmplitude=None,
        binsize=0.02,
        smoothingKernelWidth=0.01,
        overwrite=False
        ):
        """
        """

        return

    def _extractNormalizedVisulOnlyResponses(
        self,
        responseWindow=(-0.2, 0.5),
        baselineWindow=(-0.2, 0),
        normalizingWindow=(0, 0.5),
        perisaccadicWindow=(-0.05, 0.1),
        minimumBaselineActivity=0.5,
        minimumResponseAmplitude=1,
        binsize=0.02,
        smoothingKernelWidth=0.01,
        overwrite=False,
        ):
        """
        """

        #
        self.log(f'Extracting preferred and non-preferred responses to the probe stimulus')

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
        datasetKeys = (
            ('rProbe', 'dg', 'preferred'),
            ('rProbe', 'dg', 'nonpreferred'),
        )

        # Check if data is alread processed 
        flags = np.full(len(datasetKeys), False)
        for i, (responseType, protocol, eventDirection) in enumerate(datasetKeys):
            datasetPath = f'peths/{responseType}/{protocol}/{eventDirection}'
            if self.hasDataset(datasetPath):
                flags[i] = True
        if flags.all() and overwrite == False:
           return

        # Skip sessions without the drifting grating stimulus
        if self.probeTimestamps is None:
            for eventType, protocol, eventDirection in datasetKeys:
                datasetPath = f'peths/{eventType}/{eventDirection}'
                self.save(datasetPath, np.full([nUnits, nBins], np.nan))
            return

        #
        factors = {
            'left': np.full(nUnits, np.nan),
            'right': np.full(nUnits, np.nan)
        }
        amplitudes = {
            'left': np.full(nUnits, np.nan),
            'right': np.full(nUnits, np.nan)
        }
        peths = {
            'left': np.full([nUnits, nBins], np.nan),
            'right': np.full([nUnits, nBins], np.nan)
        }

        #
        for probeMotion, probeDirection in zip([-1, +1], ['left', 'right']):

            #
            trialIndices = np.where(np.logical_and(
                np.logical_or(
                    self.probeLatencies < perisaccadicWindow[0],
                    self.probeLatencies > perisaccadicWindow[1]
                ),
                self.gratingMotionDuringProbes == probeMotion,
            ))[0]

            #
            for iUnit, unit in enumerate(self.population):
                if iUnit == nUnits - 1:
                    end = None
                else:
                    end = '\r'
                self.log(f'Working on unit {iUnit + 1} out of {nUnits} (motion={probeMotion})', end=end)

                # Estimate baseline activity
                t, M = psth2(
                    self.probeTimestamps[trialIndices],
                    unit.timestamps,
                    window=baselineWindow,
                    binsize=None
                )
                mu = M.flatten().mean() / np.diff(baselineWindow).item()

                # Filter out units with subthreshold baselines
                if mu < minimumBaselineActivity:
                    continue

                # Measure the evoked response
                t, fr = unit.peth(
                    self.probeTimestamps[trialIndices],
                    responseWindow=responseWindow,
                    binsize=binsize,
                    kde=True,
                    sd=smoothingKernelWidth,
                )

                # Filter out units with subthreshold baselines
                if np.max(np.abs(fr - mu)) < minimumResponseAmplitude:
                    continue
                
                #
                amplitude = np.max(np.abs(fr - mu))
                amplitudes[probeDirection][iUnit] = amplitude

                #
                peths[probeDirection][iUnit] = fr - mu

                # Save the baseline subtracted response
                normalizingWindowMask = np.logical_and(
                    t >= normalizingWindow[0],
                    t <= normalizingWindow[1]
                )
                factor = round(np.max(np.abs(fr - mu)[normalizingWindowMask]), 2)
                factors[probeDirection][iUnit] = factor
                
        #
        X = {
            'preferred': np.full([nUnits, nBins], np.nan),
            'nonpreferred': np.full([nUnits, nBins], np.nan)
        }
        for iUnit in range(nUnits):

            #
            mapping = {
                'left': None,
                'right': None
            }
            aL = amplitudes['left'][iUnit]
            aR = amplitudes['right'][iUnit]
            if np.isnan([aL, aR]).all():
                continue
            if aL > aR:
                mapping['left'] = 'preferred'
                mapping['right'] = 'nonpreferred'
                factor = factors['left'][iUnit]
            else:
                mapping['left'] = 'nonpreferred'
                mapping['right'] = 'preferred'
                factor = factors['right'][iUnit]

            #
            for probeDirection, directionPreference in mapping.items():
                X[directionPreference][iUnit] = peths[probeDirection][iUnit] / factor
        
        #
        for directionPreference in ('preferred', 'nonpreferred'):
           self.save(f'peths/rProbe/dg/{directionPreference}', X[directionPreference])

        return
    
    def _extractNormalizedSaccadOnlyResponses(
        self,
        responseWindow=(-0.2, 0.5),
        baselineWindow=(-2, -1.8),
        normalizingWindow=(-0.3, 0.5),
        perisaccadicWindow=(-0.05, 0.1),
        minimumBaselineActivity=0.5,
        minimumResponseAmplitude=1,
        binsize=0.02,
        smoothingKernelWidth=0.01,
        overwrite=False,
        ):
        """
        """

        peristimulusWindow = np.array([
            perisaccadicWindow[1] * -1,
            perisaccadicWindow[0] * -1
        ])

        #
        self.log(f'Extracting preferred and non-preferred responses to saccades')

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
        datasetKeys = (
            ('rSaccade', 'dg', 'preferred'),
            ('rSaccade', 'dg', 'nonpreferred'),
        )

        # Check if data is alread processed 
        flags = np.full(len(datasetKeys), False)
        for i, (responseType, protocol, eventDirection) in enumerate(datasetKeys):
            datasetPath = f'peths/{responseType}/{protocol}/{eventDirection}'
            if self.hasDataset(datasetPath):
                flags[i] = True
        if flags.all() and overwrite == False:
           return

        # Skip sessions without the drifting grating stimulus
        if self.probeTimestamps is None:
            for eventType, protocol, eventDirection in datasetKeys:
                datasetPath = f'peths/{eventType}/{eventDirection}'
                self.save(datasetPath, np.full([nUnits, nBins], np.nan))
            return

        #
        factors = {
            'nasal': np.full(nUnits, np.nan),
            'temporal': np.full(nUnits, np.nan)
        }
        amplitudes = {
            'nasal': np.full(nUnits, np.nan),
            'temporal': np.full(nUnits, np.nan)
        }
        peths = {
            'nasal': np.full([nUnits, nBins], np.nan),
            'temporal': np.full([nUnits, nBins], np.nan)
        }

        #
        for saccadeLabel, saccadeDirection in zip([-1, +1], ['temporal', 'nasal']):

            #
            trialIndices = np.where(np.logical_and(
                np.logical_or(
                    self.saccadeLatencies < peristimulusWindow[0],
                    self.saccadeLatencies > peristimulusWindow[1]
                ),
                self.saccadeLabels == saccadeLabel,
            ))[0]

            #
            for iUnit, unit in enumerate(self.population):
                if iUnit == nUnits - 1:
                    end = None
                else:
                    end = '\r'
                self.log(f'Working on unit {iUnit + 1} out of {nUnits} (label={saccadeLabel})', end=end)

                # Estimate baseline activity
                t, M = psth2(
                    self.saccadeTimestamps[trialIndices, 0],
                    unit.timestamps,
                    window=baselineWindow,
                    binsize=None
                )
                mu = M.flatten().mean() / np.diff(baselineWindow).item()

                # Filter out units with subthreshold baselines
                if mu < minimumBaselineActivity:
                    continue

                # Measure the evoked response
                t, fr = unit.peth(
                    self.saccadeTimestamps[trialIndices, 0],
                    responseWindow=responseWindow,
                    binsize=binsize,
                    kde=True,
                    sd=smoothingKernelWidth,
                )

                # Filter out units with subthreshold baselines
                if np.max(np.abs(fr - mu)) < minimumResponseAmplitude:
                    continue
                
                #
                amplitude = np.max(np.abs(fr - mu))
                amplitudes[saccadeDirection][iUnit] = amplitude

                #
                peths[saccadeDirection][iUnit] = fr - mu

                # Save the baseline subtracted response
                normalizingWindowMask = np.logical_and(
                    t >= normalizingWindow[0],
                    t <= normalizingWindow[1]
                )
                factor = round(np.max(np.abs(fr - mu)[normalizingWindowMask]), 2)
                factors[saccadeDirection][iUnit] = factor

        #
        X = {
            'preferred': np.full([nUnits, nBins], np.nan),
            'nonpreferred': np.full([nUnits, nBins], np.nan)
        }
        for iUnit in range(nUnits):

            #
            mapping = {
                'nasal': None,
                'temporal': None
            }
            aN = amplitudes['nasal'][iUnit]
            aT = amplitudes['temporal'][iUnit]
            if np.isnan([aN, aT]).all():
                continue
            if aN > aT:
                mapping['nasal'] = 'preferred'
                mapping['temporal'] = 'nonpreferred'
                factor = factors['nasal'][iUnit]
            else:
                mapping['nasal'] = 'nonpreferred'
                mapping['temporal'] = 'preferred'
                factor = factors['temporal'][iUnit]

            #
            for probeDirection, directionPreference in mapping.items():
                X[directionPreference][iUnit] = peths[probeDirection][iUnit] / factor
        
        #
        for directionPreference in ('preferred', 'nonpreferred'):
           self.save(f'peths/rSaccade/dg/{directionPreference}', X[directionPreference])

        return

    def _runClusterModule(self, overwrite=False):
        """
        """

        self._extractNormalizedVisulOnlyResponses(overwrite=overwrite)
        self._extractNormalizedSaccadOnlyResponses(overwrite=overwrite)

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