from myphdlib.toolkit.custom import DotDict

samplingRateLabjack         = 1000
samplingRateNeuropixels     = 3000
labjackChannelIndexSimulus  = 7
labjackChannelIndexCameras  = 6
labjackChannelIndexBarcode  = 5
trialCountSparseNoise       = 540 # Number of trials PER BLOCK (i.e., not total)
trialCountMovingBars        = 24
trialCountDriftingGrating   = 30
trialCountNoisyGrating      = None

# Column indices for each signal being sampled by the labjack device
labjackChannelMapping = DotDict({
    'stimulus': 7,
    'cameras': 6,
    'barcode': 5
})

# Number of rising and falling edges per stimulus block
stateTransitionCounts = {
    0: 2160,
    1: 96,
    2: None,
    3: 2160, # Subtract from here???
    4: 96,
    5: 24040
}