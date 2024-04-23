# myphdlib

This is a joke of scientific software and analysis documentation that was written during my graduate research.

## Datasets ##
The processed data for each session is stored in an H5 file named `output.hdf`. Within this file, there are datasets grouped by the associated data processing step.

### Stimulus ###
These datasets contain data associated with the visual stimuli presented during the recordings.

#### Binary noise ####
- `stimuli/bn/hr/hf/fields`
- `stimuli/bn/hr/hf/grids`
- `stimuli/bn/hr/hf/length`
- `stimuli/bn/hr/hf/missing`
- `stimuli/bn/hr/hf/timestamps`
- `stimuli/bn/hr/lf/fields`
- `stimuli/bn/hr/lf/grids`
- `stimuli/bn/hr/lf/length`
- `stimuli/bn/hr/lf/missing`
- `stimuli/bn/hr/lf/timestamps`
- `stimuli/bn/lr/hf/fields`
- `stimuli/bn/lr/hf/grids`
- `stimuli/bn/lr/hf/length`
- `stimuli/bn/lr/hf/missing`
- `stimuli/bn/lr/hf/timestamps`
- `stimuli/bn/lr/lf/fields`
- `stimuli/bn/lr/lf/grids`
- `stimuli/bn/lr/lf/length`
- `stimuli/bn/lr/lf/missing`
- `stimuli/bn/lr/lf/timestamps`

#### Drifting grating ####
- `stimuli/dg/grating/motion` - (1 x N blocks, int) Direction of grating motion for each block (-1 = CCW, +1 = CW)
- `stimuli/dg/grating/timestamps` - (1 x N blocks, float) Timestamp for the beginning of each block
- `stimuli/dg/iti/timestamps` - (1 x N blocks, float) Timestamp for the end of each block
- `stimuli/dg/motion/timestamps` - (1 x N blocks, float) Timestamp for the onset of grating motion for each block
- `stimuli/dg/probe/contrast` - (1 x N trials, float) Probe contrast
- `stimuli/dg/probe/direction` - (1 x N trials, int) Direction of the closest saccade for each probe stimulus
- `stimuli/dg/probe/tts` - (1 x N trials, float) Latency from closest saccade to each probe (Time to saccade)
- `stimuli/dg/probe/motion` - (1 x N trials, int) Direction of grating motion during each probe stimulus
- `stimuli/dg/probe/phase` - (1 x N trials, float) Phase of the grating on the first frame of the probe stimulus
- `stimuli/dg/probe/timestamps` - (1 x N trials, float) Timestamp for each probe stimulus

#### Fictive saccades ####
- `stimuli/fs/coincident`
- `stimuli/fs/probes/timestamps`
- `stimuli/fs/saccades/timestamps`

#### Moving bars ####
- `stimuli/mb/offset/timestamps` - (1 x N trials, float) - Timestamp for the offset of the moving bars
- `stimuli/mb/onset/timestamps` - (1 x N trials, float) - Timestamp for the onset of the moving bars
- `stimuli/mb/orientation` - (1 x N trials, float) - Orientation of the moving bars

#### Sparse noise ####
- `stimuli/sn/pre/coords` - (N trials x 2, float) - Coordinates for the active subregion on each trial (in degrees of visual angle)
- `stimuli/sn/pre/fields` - (N trials x H x W, int) - Stimulus presented for each trial
- `stimuli/sn/pre/missing` - (1 x N trials, bool) - Boolean mask which identifies misssing trials
- `stimuli/sn/pre/signs` - (1 x N trials, bool) - Sign of the stimulus (ON or OFF) for each trial
- `stimuli/sn/pre/timestamps` - (1 x N trials) - Timestamp for each trials
- `stimuli/sn/post/coords` - (N trials x 2, float) - Coordinates for the active subregion on each trial (in degrees of visual angle)
- `stimuli/sn/post/fields` - (N trials x H x W, int) - Stimulus presented for each trial
- `stimuli/sn/post/missing` - (1 x N trials, bool) - Boolean mask which identifies misssing trials
- `stimuli/sn/post/signs` - (1 x N trials, bool) - Sign of the stimulus (ON or OFF) for each trial
- `stimuli/sn/post/timestamps` - (1 x N trials) - Timestamp for each trials

### Population ###
These datasets map directly on to each unit in the extracellular recording.

#### Masks ####
- `population/masks/hq` - (1 x N units, bool) Units that meet or exceed spike sorting quality metric thresholds
- `population/masks/sr` - (1 x N units, bool) Units classified as saccade-related
- `population/masks/vr` - (1 x N units, bool) Units classified as visually responsive

#### Metrics ####
For spike-sorting quality metrics, the following thresholds were used to delimit low- and high-quality units: Presence ration > 0.9, Refractory period violation rate < 0.5, and Amplitude cutoff < 0.1. Units must meet or exceed all three thresholds to be considered "high-quality."
- `population/metrics/ac` - (1 x N units, float) Amplitude cutoff
- `population/metrics/vra/left` - (1 x N units, float) Visual response amplitude for probes during leftward motion (in standard deviations)
- `population/metrics/vra/right` - (1 x N units, float) Visual response amplitude for probes during rightward motion (in standard deviations)
- `population/metrics/pr` - (1 x N units, float) Presence ratio
- `population/metrics/rpvr` - (1 x N units, float) Refractory period violation rate
- `population/metrics/ksl` - (1 x N units, int) Unit classification assigned by Kilosort (0=mult-unit, 1=single-unit)

#### ZETA test ####
ZETA-test p-values and latency to peak response computed for probe stimuli (left and right) and saccades (nasal and temporal).
- `population/zeta/probe/left/latency` - (1 x N units, float) Latency from probe onset to peak firing rate (grating moving CCW)
- `population/zeta/probe/left/p` - (1 x N units, float) ZETA test p-values looking at activity related to visual probes (grating moving CCW)
- `population/zeta/probe/right/latency` - (1 x N units, float) Latency from probe onset to peak firing rate (grating moving CW)
- `population/zeta/probe/right/p` - (1 x N units, float) ZETA test p-values looking at activity related to visual probes (grating moving CW)
- `population/zeta/saccade/nasal/latency` - (1 x N units, float) Latency from saccade onset to peak firing rate (Nasal saccades)
- `population/zeta/saccade/nasal/p` - (1 x N units, float) ZETA test p-values looking at activity related to saccades (Nasal saccades)
- `population/zeta/saccade/temporal/latency` - (1 x N units, float) Latency from saccade onset to peak firing rate (Temporal saccades)
- `population/zeta/saccade/temporal/p` - (1 x N units, float) ZETA test p-values looking at activity related to saccades (Temporal saccades)