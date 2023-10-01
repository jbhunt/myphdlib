# myphdlib

This is a collection of scientific software and analysis documentation that was written during my graduate research.

## H5 file datasets ##
The processed data for each session is stored in an H5 file named `output.hdf`. Within this file, there are datasets grouped by the associated data processing step.

### Population ###
These datasets map directly on to each unit in the extracellular recording.
- `population/masks/hq` - (1 x N units, Boolean) Units that meet or exceed spike sorting quality metric thresholds
- `population/masks/sr` - (1 x N units, Boolean) Units classified as saccade-related
- `population/masks/vr` - (1 x N units, Boolean) Units classified as visually responsive
- `population/metrics/ac` - (1 x N units, float) Amplitude cutoff
- `population/metrics/gvr` - (1 x N units, float) Greatest visual response (z-scored)
- `population/metrics/pr` - (1 x N units, float) Presence ratio
- `population/metrics/rpvr` - (1 x N units, float) Refractory period violation rate
- `population/zeta/probe/left/latency` - (1 x N units, float) Latency from probe onset to peak firing rate (grating moving CCW)
- `population/zeta/probe/left/p` - (1 x N units, float) ZETA test p-values looking at activity related to visual probes (grating moving CCW)
- `population/zeta/probe/right/latency` - (1 x N units, float) Latency from probe onset to peak firing rate (grating moving CW)
- `population/zeta/probe/right/p` - (1 x N units, float) ZETA test p-values looking at activity related to visual probes (grating moving CW)
- `population/zeta/saccade/nasal/latency` - (1 x N units, float) Latency from saccade onset to peak firing rate (Nasal saccades)
- `population/zeta/saccade/nasal/p` - (1 x N units, float) ZETA test p-values looking at activity related to saccades (Nasal saccades)
- `population/zeta/saccade/temporal/latency` - (1 x N units, float) Latency from saccade onset to peak firing rate (Temporal saccades)
- `population/zeta/saccade/temporal/p` - (1 x N units, float) ZETA test p-values looking at activity related to saccades (Temporal saccades)