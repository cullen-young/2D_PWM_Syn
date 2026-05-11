Plane Wave Reverse Time Migration -- Reproducible Research Repository
======================================================================

This repository contains three Madagascar (RSF) SCons workflows that together
implement 2D plane wave reverse time migration (RTM) on the Overthrust velocity
model. The methodology follows a three-stage pipeline: forward modeling of shot
records, plane wave phase encoding of source and receiver wavefields, and plane
wave RTM via crosscorrelation imaging.


REPOSITORY STRUCTURE
--------------------
SConstruct_data_repo     Stage 1: 2D acoustic forward modeling
SConstruct_encode_repo   Stage 2: Plane wave phase encoding (source & receiver wavefields)
SConstruct_pwrtm_repo    Stage 3: Plane wave reverse time migration
README.txt               This file


SCIENTIFIC BACKGROUND
---------------------

Plane Wave Migration

Conventional shot-profile RTM applies a separate imaging condition for each
individual source. Plane wave migration instead encodes a suite of shots with a
linear time delay that simulates a plane wave incident at a particular ray
parameter (slowness). By summing encoded shots prior to migration, the number of
independent propagations is reduced from one per shot to one per plane wave
angle, substantially lowering computational cost while preserving the
full-aperture illumination needed for wide-angle imaging.

The encoding delay applied to source i at position x_i for plane wave slowness p is:

    t_i = p * x_i

In this implementation, the delay is introduced by zero-padding each shot's
wavelet prior to propagation. After phase encoding, source and receiver
wavefields are individually back-propagated and combined through a zero-lag
crosscorrelation imaging condition.

Velocity Model

The 3D SEG/EAGE Overthrust model is used as the test case. A single 2D inline
slice is extracted and conditioned:
  - A spatially smoothed version serves as the RTM migration velocity (background model).
  - The model is padded on all four sides with edge-trace continuation to suppress
    absorbing boundary artifacts during propagation.
  - Density is set to a constant (1 g/cm^3) throughout.

VSP Receiver Geometry

Receivers are arranged in a vertical seismic profile (VSP) geometry: a fixed
horizontal position with receivers distributed along the depth axis. This geometry
is used to record the downgoing and upgoing wavefields separately, enabling the
construction of both source-side and receiver-side wavefields for the imaging
condition.


SOFTWARE DEPENDENCIES
---------------------
Madagascar (RSF) 4.2          Core processing framework (all sf* commands)
sfwfld2d_gpu                  GPU-accelerated 2D acoustic finite-difference
                              wavefield propagator (CWP build)
sformsby                      Ormsby bandpass wavelet generator (RSF user utilities)
rsf.recipes (wplot, geom, awe) Madagascar Python recipe modules
Python 2/3                    SCons and RSF scripting
CUDA-capable GPU              Required for sfwfld2d_gpu

The RSF installation assumed throughout is located at:
    /beegfs/sets/cwp/code/RSF/RSF4.2_CU12.9.1/

Update the sformsby and sfwfld2d_gpu paths in each SConstruct if your
installation differs.


STAGE 1: FORWARD MODELING (SConstruct_data_repo)
-------------------------------------------------

Purpose:
Generates synthetic 2D acoustic shot records for all 601 sources in the survey
using the conditioned Overthrust velocity model. Each shot is forward modeled
independently with a GPU finite-difference propagator. The per-shot wavelet
carries a linear time shift that encodes the plane wave moveout across the survey.

Survey Parameters:
  Model size          801 x 187 samples    (Horizontal x Vertical)
  Spatial sampling    5 m (0.005 km)       Both axes
  Time samples        3000                 Total record length
  Time sampling       0.2 ms (0.0002 s)
  Record length       0.6 s
  Number of sources   601
  Source spacing      5 m
  Source x-range      0.5 to 3.5 km
  Source depth        10 m                 Near-surface
  Number of receivers 187                  VSP array
  Receiver x-position 2.005 km             Fixed horizontal
  Receiver depth range 0 to 0.93 km        Vertical array
  Wavelet             Ormsby bandpass      8-20-100-150 Hz
  Wavelet delay       200 samples
  Snapshot interval   Every 10 time steps

Key Processing Steps:

  Velocity model conditioning:
    The Overthrust 3D volume is sliced to produce a 2D model, smoothed with a
    3x3 spatial filter (2 passes) to produce a migration background velocity,
    and padded by extending edge traces 2 km on the left and right and to depth
    on the bottom. The padded model is resampled to a 2.5 m grid via sinc
    interpolation before being passed to the propagator.

  Per-shot wavelet construction:
    Each source receives its own wavelet with a source-index-dependent zero-pad
    applied at the start of the time axis:
        pad_exp = (iexp + 500) * 2  samples
    This introduces the progressive time delay that, when all shots are
    superimposed, approximates a plane wave at a fixed slowness. The +500 offset
    ensures the entire padded wavelet fits within the extended time axis without
    wrap-around.

  Acoustic forward modeling:
    sfwfld2d_gpu solves the 2D acoustic wave equation with absorbing boundary
    conditions (dabc=y), free surface reflection at z=0 (free=y), and full
    wavefield snapshots saved at every 10th time step (snap=y).

  VSP wavefield extraction and muting:
    A single depth slice of the wavefield at the VSP column is extracted from
    the snapshot cube. The direct wave is suppressed in the F-K domain using a
    smoothed binary mask that zeros the cone of apparent velocities corresponding
    to the direct arrival.

Primary Outputs:
  alld                     nt x nrx x nsx    All shot receiver records
  allwavd_vsp_transp       nrx x nsx x nsnap VSP wavefield snapshots, transposed
  allwavd_vsp_mute_transp  nrx x nsx x nsnap Muted VSP wavefields, transposed


STAGE 2: PHASE ENCODING (SConstruct_encode_repo)
-------------------------------------------------

Purpose:
Transforms the shot-domain VSP receiver wavefields and reference source wavelet
from Stage 1 into the plane wave (slowness) domain using the Receiver-to-Plane-wave
(R2P) and Source-to-Plane-wave (S2P) operators. The resulting encoded wavefields
are returned to the time domain via inverse FFT for input to the RTM.

Encoding Parameters:
  Number of angles    501
  Angular range       -25.0 to +25.0 degrees
  Angular step        0.1 degrees
  Reference velocity  2.0 km/s
  VSP column index    400 (x = 2.0 km in padded model)

Key Processing Steps:

  Velocity slice extraction:
    A depth profile of the padded velocity model at the VSP receiver column
    is extracted and resampled to match the VSP wavefield depth axis (192 levels
    at 2x subsampling).

  Receiver-side encoding (R2P):
    The muted VSP receiver wavefields are forward FFT'd along the time axis and
    phase-shifted across the receiver array to project onto the plane wave basis.

  Source-side encoding (S2P):
    A source wavefield is constructed at each VSP receiver depth by applying a
    progressive time shift (zero-pad) to the reference wavelet. The per-depth
    wavefields are assembled and encoded into the plane wave domain symmetrically
    to the receiver encoding.

Primary Outputs:
  pwmrr_ifft   Time-domain plane wave encoded receiver wavefield
  pwmrs_ifft   Time-domain plane wave encoded source wavefield


STAGE 3: PLANE WAVE RTM (SConstruct_pwrtm_repo)
------------------------------------------------

Purpose:
Performs 2D plane wave RTM by forward propagating the encoded source wavefield
and back-propagating the encoded receiver wavefield for each plane wave angle.
A zero-lag crosscorrelation imaging condition is applied at each depth to
accumulate the subsurface reflectivity image. Individual angle images are
stacked and a Laplacian filter is applied to produce the final RTM image.

RTM Parameters:
  Propagation grid    1000 x 296 samples (5.0 x 1.48 km at 5 m spacing)
  Angle subset        321 angles (indices 99 to 419, approx. -15 to +17 degrees)
  Imaging condition   Zero-lag crosscorrelation (ictype=1)
  Boundary condition  Absorbing + free surface
  Illumination        Output per angle for amplitude normalization

Key Processing Steps:

  Wavefield resampling:
    Encoded source and receiver wavefields are sinc-interpolated from the
    Stage 2 grid onto the finer RTM propagation grid (2.5 m spatial,
    0.2 ms temporal). A half-integration operator is applied as part of
    the imaging condition normalization.

  Per-angle RTM:
    For each plane wave angle the GPU propagator forward-propagates the
    source wavefield and back-propagates the receiver wavefield, applying
    the crosscorrelation imaging condition at each depth step. An
    illumination map is saved for amplitude correction.

  Image stacking and display:
    All per-angle images are concatenated and stacked. A 30-angle sub-stack
    (every 8th angle) is produced for QC. The Laplacian filter enhances
    reflector sharpness. The final image is windowed to the zone of interest
    (x: 1.5 to 2.5 km, z: 0 to 0.8 km) for display.

Primary Outputs:
  pwm_overthrust_30q_halfint_ic1        Final stacked RTM image (Laplacian IC)
  pwm_overthrust_30q_halfint_ic1_illum  Illumination-corrected stacked image


RUNNING THE WORKFLOWS
---------------------
Stages must be run in order. Each SConstruct reads output from the previous stage.

    scons -f SConstruct_data_repo      # Stage 1: Forward modeling
    scons -f SConstruct_encode_repo    # Stage 2: Phase encoding
    scons -f SConstruct_pwrtm_repo     # Stage 3: Plane wave RTM

SCons tracks file dependencies automatically. To rebuild only updated targets,
run the same command again and SCons will skip already-built files.

To run a specific target without building downstream dependents:
    scons -f SConstruct_data_repo alld


NOTES FOR REPRODUCIBILITY
--------------------------
- The Overthrust velocity model must be accessible at the path hardcoded in
  SConstruct_data_repo. Update this path if redeploying to a different system.
- sfwfld2d_gpu and sfrtm2dGPU_DASVSP_CWM require a CUDA-capable GPU. Job
  submission settings (number of GPUs, memory) may need to be adjusted for
  your cluster environment.
- The rsf.cluster import at the top of SConstruct_encode_repo can be enabled
  for distributed cluster execution.
- All intermediate RSF files are written to the working directory. Disk usage
  for the full 601-shot survey is substantial; ensure sufficient scratch space
  before running Stage 1.
