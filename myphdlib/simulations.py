import numpy as np
from myphdlib.toolkit import smooth, detectThresholdCrossing
from sklearn.preprocessing import MinMaxScaler

class ContrastSignalGenerator():
    """
    """

    def __init__(self, seed=0, samplingRate=1000, boundaries=(0, 1)):
        self.seed = seed
        self.samplingRate = samplingRate
        self.boundaries = boundaries
        return

    def generate(
        self,
        stimulus='CW',
        duration=100,
        initialPosition=0,
        **kwargs
        ):

        if stimulus == 'CW':
            signal = self._generateContinuousWalk(duration, initialPosition, **kwargs)
        elif stimulus == 'DW':
            signal = self._generateDiscreteWalk(duration, initialPosition, **kwargs)
        elif stimulus == 'DP':
            signal = self._generateDiscretePulses(duration, initialPosition, **kwargs)
        elif stimulus == 'DN':
            signal = self._generateDiscreteNoise(duration, initialPosition, **kwargs)

        return signal

    def _generateContinuousWalk(
        self,
        duration=100,
        initialPosition=0,
        stepSize=0.05,
        probabilities=(0.1, 0.8, 0.1),
        bounce=True,
        bounceAmplitude=0.1,
        bounceTau=0.01,
        bounceTime=0.5,
        smoothingWindowSize=0.02,
        ):
        """
        """

        # 
        N = round(self.samplingRate * duration)
        signal = np.zeros(N)
        directions = np.array([-1, 0, 1])
        initialProbabilities = np.array(probabilities)
        currentProbabilities = np.copy(initialProbabilities)
        currentPosition = initialPosition
        inBounce = False
        probabilityArray = list()

        #
        for sampleIndex in range(N):

            probabilityArray.append(currentProbabilities)

            # Update probabilities
            if inBounce:
                if len(probabilityOffset) == 0:
                    inBounce = False
                else:
                    offset = probabilityOffset.pop()
                    currentProbabilities = np.copy(initialProbabilities)
                    if bounceDirection == -1:
                        currentProbabilities[0] += offset
                        currentProbabilities[2] -= offset
                    elif bounceDirection == +1:
                        currentProbabilities[0] -= offset
                        currentProbabilities[2] += offset

            # Choose a movement and update the current position
            direction = np.random.choice(directions, p=currentProbabilities)
            if direction == 0:
                pass
            elif direction == -1:
                currentPosition -= stepSize
            elif direction == +1:
                currentPosition += stepSize
            currentPosition = round(currentPosition, 2)

            # Check for boundary coincidences
            if bounce:
                if inBounce == False:
                    if currentPosition == self.boundaries[1]:
                        bounceDirection = -1
                        probabilityOffset = bounceAmplitude * np.exp(-1 * np.arange(round(self.samplingRate * bounceTime)) / round(self.samplingRate * bounceTau))
                        probabilityOffset = probabilityOffset[::-1].tolist()
                        inBounce = True
                    if currentPosition == self.boundaries[0]:
                        bounceDirection = +1
                        probabilityOffset = bounceAmplitude * np.exp(-1 * np.arange(round(self.samplingRate * bounceTime)) / round(self.samplingRate * bounceTau))
                        probabilityOffset = probabilityOffset[::-1].tolist()
                        inBounce = True

            # Check for boundary crossings
            if currentPosition > self.boundaries[1]:
                currentPosition -= stepSize
            if currentPosition < self.boundaries[0]:
                currentPosition += stepSize

            #
            signal[sampleIndex] = currentPosition

        #
        signal = smooth(signal, round(self.samplingRate * smoothingWindowSize))
        probabilityArray = np.array(probabilityArray)

        return signal

    def _generateDiscreteWalk(
        self,
        duration=100,
        initialPosition=0,
        probabilities=(0.1, 0.8, 0.1),
        bounce=True,
        bounceAmplitude=0.1,
        bounceTau=0.01,
        bounceTime=0.5,
        stepSize=0.1,
        stepTimeRange=(3/60, 3/60),
        ):
        """
        """

        #
        N = round(self.samplingRate * duration)
        signal = np.zeros(N)
        values = np.array([-1, 0, 1])
        probabilities = np.ones(3) / 3
        initialProbabilities = np.array(probabilities)
        currentProbabilities = np.copy(initialProbabilities)
        currentPosition = initialPosition
        countdown = 0
        inBounce = False

        #
        for sampleIndex in range(N):

            # Update probabilities
            if inBounce:
                if len(probabilityOffset) == 0:
                    inBounce = False
                else:
                    offset = probabilityOffset.pop()
                    currentProbabilities = np.copy(initialProbabilities)
                    if bounceDirection == -1:
                        currentProbabilities[0] += offset
                        currentProbabilities[2] -= offset
                    elif bounceDirection == +1:
                        currentProbabilities[0] -= offset
                        currentProbabilities[2] += offset

            #
            if countdown != 0:
                signal[sampleIndex] = currentPosition
                countdown -= 1
                continue

            #
            value = np.random.choice(values, size=1, p=currentProbabilities).item()
            stepTime = np.random.uniform(*stepTimeRange, size=1).item()
            if value == 0:
                countdown = round(self.samplingRate * stepTime)
            elif value == -1:
                currentPosition -= stepSize
                countdown = round(self.samplingRate * stepTime)
            elif value == +1:
                currentPosition += stepSize
                countdown = round(self.samplingRate * stepTime)

            # Check for boundary coincidences
            if bounce:
                if inBounce == False:
                    if currentPosition == self.boundaries[1]:
                        bounceDirection = -1
                        probabilityOffset = bounceAmplitude * np.exp(-1 * np.arange(round(self.samplingRate * bounceTime)) / round(self.samplingRate * bounceTau))
                        probabilityOffset = probabilityOffset[::-1].tolist()
                        inBounce = True
                    if currentPosition == self.boundaries[0]:
                        bounceDirection = +1
                        probabilityOffset = bounceAmplitude * np.exp(-1 * np.arange(round(self.samplingRate * bounceTime)) / round(self.samplingRate * bounceTau))
                        probabilityOffset = probabilityOffset[::-1].tolist()
                        inBounce = True

            #
            if currentPosition > self.boundaries[1]:
                currentPosition -= stepSize
            if currentPosition < self.boundaries[0]:
                currentPosition += stepSize

            #
            signal[sampleIndex] = currentPosition

        return signal

    def _generateDiscreteNoise(self, duration, initialPosition=0, stepTime=0.1, stepCount=10):
        """
        """

        N = round(self.samplingRate * duration)
        signal = np.zeros(N)
        options = np.around(np.linspace(self.boundaries[0], self.boundaries[1], stepCount), 2)
        options = np.clip(options, 0, 1)
        currentPosition = initialPosition
        countdown = 0

        for sampleIndex in range(N):

            #
            if countdown != 0:
                signal[sampleIndex] = currentPosition
                countdown -= 1
                continue

            currentPosition = np.random.choice(options, size=1).item()
            countdown = round(self.samplingRate * stepTime)
            signal[sampleIndex] = currentPosition

        return signal

    def _generateDiscretePulses(self, duration=100, initialPosition=0, levels=np.linspace(0.2, 1, 5), pulseTime=0.1, interPulseIntervalRange=(1, 3)):
        """
        """

        N = round(self.samplingRate * duration)
        signal = np.zeros(N)
        countdown = 0
        currentPosition = initialPosition
        countdown = round(self.samplingRate * np.random.uniform(*interPulseIntervalRange, size=1).item())
        inPulse = False
        
        #
        for sampleIndex in range(N):

            # This prevents a pulse from coinciding with the end of the signal
            if signal.size - sampleIndex < round(self.samplingRate * pulseTime) + 1:
                continue
            
            #
            if countdown != 0:
                signal[sampleIndex] = currentPosition
                countdown -= 1
                continue

            #
            if inPulse:
                currentPosition = initialPosition
                countdown = round(self.samplingRate * np.random.uniform(*interPulseIntervalRange, size=1).item())
                inPulse = False

            #
            else:
                currentPosition = np.random.choice(levels, size=1).item()
                countdown = round(self.samplingRate * pulseTime)
                inPulse = True

        return signal

def createRandomSteps(

    ):
    """
    """

    return

def createRandomWalk(
    time=300,
    stepSize=0.05,
    startPosition=0.75,
    boundaries=(0.5, 1),
    fps=60,
    smoothSignal=True,
    smoothingWindowSize=51,
    adaptionStepSize=0.01
    ):
    """
    """

    signal = list()
    values = np.array([-1, 0, 1])
    probabilities = np.array([0.1, 0.8, 0.1])
    currentPosition = startPosition
    for frameIndex in range(round(time * fps)):
        value = np.random.choice(values, size=1, p=probabilities).item()
        if value == 0:
            pass
        elif value == -1:
            currentPosition -= stepSize
        elif value == +1:
            currentPosition += stepSize
        if currentPosition < boundaries[0]:
            currentPosition += stepSize
            # probabilities[0] -= adaptionStepSize
            # probabilities[2] += adaptionStepSize
        if currentPosition > boundaries[1]:
            currentPosition -= stepSize
            # probabilities[2] -= adaptionStepSize
            # probabilities[0] += adaptionStepSize
        signal.append(currentPosition)

    signal = np.array(signal)
    if smoothSignal:
        signal = smooth(signal, smoothingWindowSize)

    return signal

def generateSpikeTimestamps(
    signal,
    baselineFiringRate=10,
    samplingRate=10000,
    contrastAveragingWindow=0.1,
    inputLatency=0.05,
    ignoreSignal=False,
    refractoryPeriod=0.001,
    contrastRatioRange=np.array([[0.9], [1.1]]),
    percentProbabilityIncrease=1.5,
    fps=60):
    """
    """

    spikeTimestamps = list()
    countdown = 0
    baselineProbability = 1 / samplingRate * baselineFiringRate
    featureRange = (
        0,
        +1 * baselineProbability * percentProbabilityIncrease
    )
    # scaler = MinMaxScaler(feature_range=featureRange).fit(contrastRatioRange)
    scaler = MinMaxScaler(feature_range=featureRange).fit([[0.5],[1]])

    #
    for sampleIndex in range(int(signal.size * fps * samplingRate)):

        #
        if countdown != 0:
            countdown -= 1
            continue

        #
        i1 = round(sampleIndex / samplingRate * fps)
        if i1 >= signal.size:
            break
        c1 = signal[i1]
        i2 = round((sampleIndex - (samplingRate * contrastAveragingWindow)) / samplingRate * fps)
        if i2 < 0:
            c2 = np.nan
        else:
            c2 = signal[i2: i1].mean()

        # TODO: Correct probability for refractory period
        p = 1 / samplingRate * baselineFiringRate
        if np.invert(np.isnan(c2)):
            if ignoreSignal == False:
                increment = scaler.transform([[c1]]).item()
                # import pdb; pdb.set_trace()
                p += increment
                # contrastRatio = c2 / c1
                # probabilityIncrement = np.clip(
                #     scaler.transform(np.atleast_2d(contrastRatio)).item(),
                #     *featureRange
                # )
                # p += probabilityIncrement

        #
        fireActionPotential = np.random.choice([0, 1], p=(1 - p, p))
        if fireActionPotential:
            spikeTimestamps.append(round(sampleIndex / samplingRate + inputLatency, 3))
            countdown = round(refractoryPeriod * samplingRate)

    return np.array(spikeTimestamps)

def computeSpikeTriggeredAverage(walk, spikes, fps=60, window=(-0.5, 0.5)):
    """
    """

    windowSize = round(np.diff(window).item()  * fps)
    samples = list()
    for timestamp in spikes:
        frameIndex = round(timestamp * fps)
        sample = walk[frameIndex + round(window[0] * fps): frameIndex + round(window[1] * fps)]
        if sample.size != windowSize:
            continue
        samples.append(sample)

    kernel = np.array(samples).mean(0)
    return kernel

class SquareFilter():
    """
    """

    def __init__(self, filterSize=0.5, squareOnset=0.1, squareWidth=0.1, squareAmplitude=1, samplingRate=1000):
        self._filter = np.zeros(round(samplingRate * filterSize))
        risingEdgeIndex = round(samplingRate * squareOnset)
        fallingEdgeIndex = risingEdgeIndex + round(samplingRate * squareWidth)
        self._filter[risingEdgeIndex: fallingEdgeIndex] = squareAmplitude
        return

    def apply(self, stimulus):
        """
        """

        if stimulus.size != self.filter.size:
            raise Exception()

        return np.convolve(stimulus, self.filter, mode='valid').item()

    @property
    def filter(self):
        return self._filter

def simulateVisualNeuron(theta=0.8, feedbackTau=0.01, feedbackAmplitude=0.1, feedbackDuration=0.1):
    """
    """

    #
    f = SquareFilter()
    s = createRandomWalk(stepSize=0.2, fps=1000, smoothingWindowSize=15)
    g = list()
    for i in range(500, s.size, 1):
        g.append(f.apply(s[i - 500: i]))
    g = np.array(g)
    g = (g - np.min(g)) / (np.max(g) - np.min(g))

    # TODO: Apply feedback to g
    g2 = np.copy(g)
    while True:
        firstCrossing = detectThresholdCrossing(g2, threshold=theta)[counter]
        feedback = feedbackAmplitude * np.exp(-1 * np.linspace(0, feedbackDuration, round(feedbackDuration * 1000)) / feedbackTau)
        g2[firstCrossing + 1: firstCrossing + 1 + round(feedbackDuration * 1000)] -= feedback
        counter += 1
        if detectThresholdCrossing(g2, threshold=theta).size == 0:
            break

    return s, g2

def simulateVisualNeuron2(theta=0.8, time=100, filterParams=(0.5, 0.1, 0.1), feedbackParams=(0.01, 0.1, 0.1), samplingRate=1000):
    """
    """

    s1 = createRandomWalk(stepSize=0.2, fps=samplingRate, smoothingWindowSize=15, time=time)
    s2 = MinMaxScaler().fit_transform(s1.reshape(-1, 1)).flatten()

    # Create square filter
    size, onset, width = filterParams
    f = np.zeros(round(size * samplingRate))
    f[round(onset * samplingRate): round(onset * samplingRate) + round(samplingRate * width)] = 1

    #
    g = list()
    for ti in range(f.size, s2.size, 1):
        si = s2[ti - f.size: ti]
        gi = np.convolve(f, si, mode='valid').item()
        # TODO: rescale gi
        if len(g) > 1:
            # TODO: look for a threshold crossing
            pass
        # TODO: integrate feedback
        g.append(gi)

    return np.array(g), s2, f
