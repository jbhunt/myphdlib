import numpy as np
import h5py
from datetime import date

def _determineSaccadeDirection(
    label=-1,
    eye='left',
    hemisphere='left',
    ):
    """
    """

    #
    if label == -1:
        if hemisphere == 'left':
            direction = 'ipsi'
        elif hemisphere == 'right':
            direction = 'contra'
    elif label == +1:
        if hemisphere == 'left':
            direction = 'contra'
        elif hemisphere == 'right':
            direction = 'ipsi'

    #
    if eye == 'left':
        pass
    elif eye == 'right':
        direction = 'ipsi' if direction == 'contra' else 'contra'

    return direction

def _measureSaccadeAmplitudeForSingleSession(
    session,
    nt = 101,
    ):
    """
    """

    a = {
        'ipsi': list(),
        'contra': list()
    }
    frameTimestamps = session.load(f'frames/{session.primaryCamera}/timestamps')
    poseEstimates = session.load(f'pose/filtered')
    if poseEstimates.shape[0] != frameTimestamps.size:
        return {'ipsi': np.nan, 'contra': np.nan}
    if session.eye == 'left':
        eyePosition = poseEstimates[:, 0]
    elif session.eye == 'right':
        eyePosition = poseEstimates[:, 2]
    for saccadeLabel, (saccadeOnset, saccadeOffset) in zip(session.saccadeLabels, session.saccadeTimestamps):
        saccadeDirection = _determineSaccadeDirection(saccadeLabel, session.eye, session.hemisphere)
        t = np.linspace(saccadeOnset, saccadeOffset, nt)
        saccadeWaveform = np.interp(t, frameTimestamps, eyePosition)
        saccadeAmplitude = round(abs(saccadeWaveform.max() - saccadeWaveform.min()), 2)
        a[saccadeDirection].append(saccadeAmplitude)

    #
    for saccadeDirection in a.keys():
        a[saccadeDirection] = np.array(a[saccadeDirection]).mean()

    return a

def _measureSaccadeFrequencyForSingleSession(
    session,
    ):
    """
    """

    motionOnset = session.load('stimuli/dg/grating/timestamps')
    motionOffset = session.load('stimuli/dg/iti/timestamps')
    gratingMotionByBlock = session.load('stimuli/dg/grating/motion')
    if any([motionOnset is None, motionOffset is None, len(motionOnset) == 0, len(motionOffset) == 0]):
        return {'ipsi': np.nan, 'contra': np.nan}
    f = {
        'ipsi': list(),
        'contra': list()
    }
    for t1, t2, gm in zip(motionOnset, motionOffset, gratingMotionByBlock):

        #
        dt = t2 - t1
        # t += dt

        #
        if gm == -1 and session.eye == 'left':
            saccadeLabel = +1
        elif gm == +1 and session.eye == 'left':
            saccadeLabel = -1
        elif gm == -1 and session.eye == 'right':
            saccadeLabel = -1
        elif gm == +1 and session.eye == 'right':
            saccadeLabel = +1

        #
        saccadeDirection = _determineSaccadeDirection(
            saccadeLabel,
            session.eye,
            session.hemisphere
        )

        #
        saccadeIndices = np.where(np.logical_and(
            np.logical_and(
                session.saccadeTimestamps[:, 0] >= t1,
                session.saccadeTimestamps[:, 0] <  t2,
            ),
            session.saccadeLabels == saccadeLabel
        ))[0]
        f[saccadeDirection].append(saccadeIndices.size / dt)

    #
    for k in f.keys():
        f[k] = round(np.mean(f[k]), 2)

    return f

class SaccadeMetricsAnalysis():
    """
    """

    def __init__(self):
        self.sessionPairs = list()
        self.dts = None
        return

    def extractSaccadeWaveforms(
        self,
        filename=None
        ):
        """
        """

        saccadeWaveforms = {
            ('saline', 'ipsi'): list(),
            ('saline', 'contra'): list(),
            ('muscimol', 'ipsi'): list(),
            ('muscimol', 'contra'): list()
        }

        for a, b in self.sessionPairs:
            for treatment, session in zip(['saline', 'muscimol'], [a, b]):
                saccadeWaveforms_ = session.load(f'saccades/predicted/{session.eye}/waveforms')
                for saccadeLabel in (-1, +1):
                    saccadeDirection = _determineSaccadeDirection(
                        saccadeLabel,
                        session.eye,
                        session.hemisphere
                    )
                    wf = saccadeWaveforms_[session.saccadeLabels == saccadeLabel].mean(0)
                    saccadeWaveforms[(treatment, saccadeDirection)].append(wf)

        #
        for k in saccadeWaveforms.keys():
            saccadeWaveforms[k] = np.array(saccadeWaveforms[k])

        #
        if filename is not None:
            with h5py.File(filename, 'w') as stream:
                for treatment, saccadeDirection in saccadeWaveforms.keys():
                    datasetPath = f'{treatment}/{saccadeDirection}'
                    data = saccadeWaveforms[(treatment, saccadeDirection)]
                    ds = stream.create_dataset(datasetPath, data.shape, data.dtype, data=data)

        return saccadeWaveforms

    def measureSaccadeFrequency(
        self,
        filename=None
        ):
        """
        """

        fab = {
            'ipsi': list(),
            'contra': list()
        }
        for a, b in self.sessionPairs:
            fa = _measureSaccadeFrequencyForSingleSession(a)
            fb = _measureSaccadeFrequencyForSingleSession(b)
            for saccadeDirection in fab.keys():
                try:
                    fab[saccadeDirection].append([
                        fa[saccadeDirection],
                        fb[saccadeDirection]
                    ])
                except:
                    import pdb; pdb.set_trace()
        for saccadeDirection in fab.keys():
            fab[saccadeDirection] = np.array(fab[saccadeDirection])

        #
        if filename is not None:
            with open(filename, 'w') as stream:
                stream.write(f'Animal, Ipsi (saline), Ipsi (muscimol), Contra (saline), Contra (muscimol)\n')
                nPairs = len(self.sessionPairs)
                for iPair in range(nPairs):
                    a, b = self.sessionPairs[iPair]
                    line = f'{a.animal},'
                    fa, fb = fab['ipsi'][iPair]
                    line += f'{fa:.2f},{fb:.2f},'
                    fa, fb = fab['contra'][iPair]
                    line += f'{fa:.2f},{fb:.2f}\n'
                    stream.write(line)

        return fab

    def measureSaccadeAmplitudes(
        self,
        ):
        """
        """

        aab = {
            'ipsi': list(),
            'contra': list()
        }
        for a, b in self.sessionPairs:
            aa = _measureSaccadeAmplitudeForSingleSession(a)
            ab = _measureSaccadeAmplitudeForSingleSession(b)
            for saccadeDirection in aab.keys():
                if np.isnan([aa[saccadeDirection], ab[saccadeDirection]]).any():
                    continue
                aab[saccadeDirection].append([
                    aa[saccadeDirection],
                    ab[saccadeDirection]
                ])

        return aab

    def loadData(
        self,
        sessions
        ):
        """
        """

        # Parse sessions
        sessionsByTreatment = {
            'saline': list(),
            'muscimol': list(),
        }
        for session in sessions:
            sessionsByTreatment[session.treatment].append(session)
        
        # Create A/B pairs
        sessionPairs = list()
        dts = list()
        for a in sessionsByTreatment['saline']:
            dt = list()
            for b in sessionsByTreatment['muscimol']:
                if b.animal != a.animal:
                    dt.append(np.nan)
                else:
                    diff = b.date - a.date
                    dt.append(diff.days)
            dt = np.array(dt, dtype=float)
            dt[dt < 0] = np.nan
            if np.isnan(dt).all():
                continue
            else:
                i = np.nanargmin(dt)
                b = sessionsByTreatment['muscimol'][i]
            dts.append(dt[i])
            sessionPairs.append((a, b))
        self.sessionPairs = sessionPairs
        self.dts = np.array(dts)

        return