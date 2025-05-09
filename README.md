# 2D Plane Wave Migration Using Synthetic Data from the SEG/EAGE Overthrust Model

This repository contains code and scripts for generating and migrating 2D synthetic plane-wave data using the SEG/EAGE Overthrust model. The framework enables testing and evaluation of **plane wave reverse time migration (RTM)** using a reciprocal **DAS VSP geometry**, with customizable parameters for acquisition and wavefield propagation.

## Directory Structure

- `data/`  
  Contains `SConstruct` and related source files to generate 2D synthetic seismic data using a finite-difference modeling scheme.  
  - Customize number of shots, receivers, model dimensions, and wavelet properties in this script.  
  - Run using the `scons` command.

- `data_prep/`  
  Applies **linear time delays** to modeled data.  
  - `SConstruct` uses compiled C routines:  
    - `MS2P.c` for source time delay application  
    - `MR2P.c` for receiver time delay application  
    - Run using the `scons` command.
  - Outputs are time-shifted source and receiver wavefields for use in migration.
  
- `rtm/`  
  Contains the RTM migration implementation.  
  - `SConstruct` calls `Mrtm2dGPU_DASVSP_CWM.cu`, a CUDA-based 2D RTM kernel tailored for **2D plane-wave migration**.  
  - The number of plane waves and their corresponding incidence angles can be configured here.
  - Run using the `scons` command.

Paper Abstract:
Reverse-time migration (RTM) of 3-D vertical seismic profile (VSP) data can be a computationally expensive task -- especially when using distributed acoustic sensing (DAS) measurements with frequencies commonly exceeding 100 Hz to generate high-resolution images. To address this challenge, we develop a conical-wave migration (CWM) framework tailored to 3-D VSP geometries. We adapt previously reported surface-based phase-encoding theory to reciprocal downhole geometries to generate downgoing conical source wavefronts that both suppress upgoing source wavefield energy and lower the computation cost by the ratio of the number of reciprocal conical-wave to standard shot-profile migrations. Synthetic tests using the SEG/EAGE Overthrust model demonstrate the effectiveness of the developed CWM approach by evaluating image quality and computational efficiency when testing different velocity-dependent phase-encoding functions, the number of conical waves, and an illumination compensation strategy. For the Overthrust model, we find that using 25 conical waves with illumination compensation results in a VSP-RTM image comparable in quality to a shot-profile RTM image in reciprocal source-receiver geometry while removing near-well artifacts associated with first-order internal multiples generated during source wavefield propagation. The most significant benefit of CWM is the reduction in computational cost, with an observed $24\times$ efficiency increase over non-reciprocal shot-profile RTM in the Overthrust example. Finally, the noted success in performing seismic imaging at frequencies up to 100 Hz suggests that CWM is well-suited for imaging DAS 3-D and time-lapse (4-D) VSP data sets for a range of reservoir monitoring, carbon sequestration, and geothermal applications.
