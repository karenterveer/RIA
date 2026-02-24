# RIA — Reconstruction with Information Field Theory for Air Showers

Python modules for reconstruction of extensive
air-shower radio footprints using Information Field Theory.

## Attribution

Several components in this package are JAX-differentiable re-implementations
of established tools:

- **Atmosphere model and coordinate transforms** are based on the
  [radiotools](https://github.com/nu-radio/radiotools) package, re-written
  in JAX for automatic differentiation and JIT compilation.
- **Lateral distribution function (LDF)** is based on the
  [geoceLDF](https://github.com/cg-laser/geoceLDF) package, likewise
  made differentiable via JAX.

## Features

- **Atmosphere model** — five-layer parametric atmosphere with GDAS support
- **Coordinate transforms** — JAX-differentiable ground ↔ shower-plane transforms
- **Lateral distribution function** — B-spline parametrisations (geomagnetic + charge-excess)
- **Forward model** — combined LDF + hyperbolic wavefront timing, integrated with NIFTy8 correlated fields
- **Timing quality control** — per-station outlier detection, iterative pruning, local uncertainty estimation
- **Bayesian inference** — high-level `reconstruct()` wrapper around `jft.optimize_kl`

## Dependencies

- Python ≥ 3.11
- NumPy
- JAX
- NIFTy8 (`nifty8.re`)
- SciPy

## Usage

```python
import numpy as np
from ria import reconstruct, config

# Adjust configuration before reconstruction
config.N_VI_ITERATIONS = 8
config.N_SAMPLES = 60

# Prepare data
positions = np.stack([x_antennas, y_antennas])  # shape (2, N)
fluences = ...      # shape (N,)  [eV/m^2]
times = ...         # shape (N,)  [s]
noise_std = ...     # shape (N,)  [eV/m^2]

results = reconstruct(
    positions, fluences, times, noise_std,
    mean_zenith=zenith_rad,
    mean_azimuth=azimuth_rad,
    mean_core_x=core_x_m,
    mean_core_y=core_y_m,
    noise_floor_mean=1.0,  # optional prior mean for the noise floor
    atmosphere_path="/path/to/gdas_file.dat",  # optional - if not provided, tests/example_data/ATMOSPHERE_EXAMPLE.DAT is used
)

print(f"X_max = {results['xmax'][0]:.1f} ± {results['xmax'][1]:.1f} g/cm²")
print(f"E_rad = {results['erad'][0]:.2e} ± {results['erad'][1]:.2e} eV")
```

## Package structure

```
ria/
├── __init__.py         Public API
├── config.py           Config parameters
├── atmosphere.py       Atmosphere code (radiotools based)
├── coordinates.py      Coordinate transforms (radiotools based)
├── ldf.py              Lateral distribution function (geoceLDF based)
├── forward_model.py    FootprintModel (LDF + wavefront + (optional) particle)
├── timing.py           Timing quality control
├── optimize.py         reconstruct() wrapper
└── data/               Pre-fitted spline data (pickle) (geoceLDF based)
```

## Testing

```bash
cd RIA
python -m pytest tests/
```
