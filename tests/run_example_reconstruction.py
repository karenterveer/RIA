#!/usr/bin/env python3
"""
Example-data reconstruction test for the RIA package.
=====================================================

Loads pre-extracted example data (positions, fluences, timings, noise)
from ``tests/example_data/`` and reconstructs using ``ria.reconstruct()``.
The example data was generated once from a real CoREAS simulation, with
measured noise applied at a voltage level.

Both fluence and noise was calculated over a 300ns window around the 
signal peak. The noise level is the mean fluence from multiple 
non-overlapping noise windows, and the noise std is the standard deviation 
of those noise fluences.

Can be run standalone::

    python test_example_reconstruction.py

or via pytest::

    python -m pytest tests/test_example_reconstruction.py -v -s
"""

import sys
import os
import logging

import numpy as np
import pytest
import matplotlib
matplotlib.use("Agg")

# Ensure RIA is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from ria import reconstruct, config, FootprintModel
try:
    from ria import plotting as fancy_reco_plot
    _HAS_FANCY_PLOT = True
except ImportError:
    _HAS_FANCY_PLOT = False

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ======================================================================
#  Paths to example data
# ======================================================================
EXAMPLE_DATA_DIR = os.path.join(os.path.dirname(__file__), "example_data")
EXAMPLE_DATA_FILE = os.path.join(EXAMPLE_DATA_DIR, "example_data.npz")
EXAMPLE_TRUTH_FILE = os.path.join(EXAMPLE_DATA_DIR, "example_truth.npz")
EXAMPLE_ATM_FILE = os.path.join(EXAMPLE_DATA_DIR, "ATMOSPHERE_EXAMPLE.DAT")

# Reconstruction speed settings (fast for testing)
FAST_N_ITERATIONS = 5
FAST_N_SAMPLES = 60

# Check whether example data exists
_HAS_EXAMPLE_DATA = (
    os.path.isfile(EXAMPLE_DATA_FILE)
    and os.path.isfile(EXAMPLE_TRUTH_FILE)
)

def generate_diagnostic_plots(results, data, truth, output_path):
    """Generate fancy reconstruction plots using fancy_reco_plot module."""
    if not _HAS_FANCY_PLOT:
        logger.warning("fancy_reco_plot not found, skipping plot generation.")
        return

    logger.info(f"Generating diagnostic plot: {output_path}")
    model = results["model"]
    samples = results["samples"]
    
    # 1. Grid setup
    min_x, min_y = float(model.min_x), float(model.min_y)
    extent = float(model.extent)
    dims = model.dims
    
    if truth["positions_regular"] is not None:
        # Use the dense truth grid (e.g., 120x120) to evaluate the model
        positions_regular = jnp.asarray(truth["positions_regular"])
    else:
        # Fallback to the model's internal CF grid resolution
        gx = jnp.linspace(min_x, min_x + extent, dims[0])
        gy = jnp.linspace(min_y, min_y + extent, dims[1])
        positions_regular = jnp.stack(jnp.meshgrid(gx, gy, indexing='ij'), axis=0).reshape(2, -1)
    
    # 2. Regular grid model for mapping
    # We use a trick: FootprintModel.get_signal_fluence_without_cf is an operator
    # that we can call with latent parameters. We can re-bind it to regular grid.
    from ria import config as ria_config
    
    # We must match the prior settings that were used for the original `model`, 
    # otherwise the operators will have mismatched domains and the spatial maps will be wrong.
    STD_ANGLE_PRIOR_DEG = 3.0
    DEFAULT_CORE_PRIOR_M = 30.0
    
    common_kwargs = {
        "magnetic_field_vector": model.magnetic_field_vector,
        "atmosphere_path": model.atmosphere_path,
        "enable_syst_cf": model.enable_syst_cf,
        "enable_timing_cf": model.enable_timing_cf,
        "grid_settings": {"min_x": min_x, "min_y": min_y, "extent": extent, "dims": dims},
        "timing_std_s": ria_config.TIMING_UNCERTAINTY_S,
        # Priors must exactly match what was fitted
        "params_phi": {"a_min": float(truth["azimuth"]) - STD_ANGLE_PRIOR_DEG * np.pi/180, 
                       "a_max": float(truth["azimuth"]) + STD_ANGLE_PRIOR_DEG * np.pi/180},
        "params_theta": {"a_min": float(truth["zenith"]) - STD_ANGLE_PRIOR_DEG * np.pi/180, 
                         "a_max": float(truth["zenith"]) + STD_ANGLE_PRIOR_DEG * np.pi/180},
        "params_X": {"a_min": float(data["mean_core_x"]) - float(data["std_core_m"]), 
                     "a_max": float(data["mean_core_x"]) + float(data["std_core_m"])},
        "params_Y": {"a_min": float(data["mean_core_y"]) - float(data["std_core_m"]), 
                     "a_max": float(data["mean_core_y"]) + float(data["std_core_m"])},
        "params_noise_floor_mean": {"mean": float(data["noise_floor_mean"]) if data.get("noise_floor_mean") is not None else 1.0, 
                                    "std": float(data["noise_floor_mean"]) / 20.0 if data.get("noise_floor_mean") is not None else 0.1},
        "params_t0": {"mean": float(data["t0_initial_guess"]) if data.get("t0_initial_guess") is not None else float(np.median(data["times"])), 
                      "std": float(data["t0_prior_std"]) if data.get("t0_prior_std") is not None else 200e-9},
    }
    model_reg = FootprintModel(positions_regular[0], positions_regular[1], **common_kwargs)
    
    # Batch evaluate samples for maps
    # Use jft.mean_and_std with correct_bias=True for proper map estimation,
    # just like in reconstruction_with_RIA.py
    import nifty8.re as jft
    
    no_cf_fluence_samples_tuple = tuple(model_reg.get_signal_fluence_without_cf(s) for s in samples)
    post_fluence_no_cf_mean, post_fluence_std = jft.mean_and_std(no_cf_fluence_samples_tuple, correct_bias=True)
    
    no_cf_timing_samples_tuple = tuple(model_reg.get_signal_timing_without_cf(s) for s in samples)
    post_timing_no_cf_mean, post_timing_std = jft.mean_and_std(no_cf_timing_samples_tuple, correct_bias=True)
    
    
    # We still need CF maps for the plot
    scf_maps = jnp.stack([np.exp(model_reg.syst_cf_op(s)) for s in samples]) # Note exponentiation for syst cf!
    timing_clip_ns = 10
    timing_fields_raw = [model_reg.timing_cf_op_2(s) for s in samples]
    tcf_maps = jnp.stack([np.clip(field, -timing_clip_ns*1e-9, timing_clip_ns*1e-9) for field in timing_fields_raw])
    
    # Also need signal response for 'reco_fluence_signal_only'
    sr_samples_tuple = tuple(model_reg(s) for s in samples)
    post_sr_mean_stacked, _ = jft.mean_and_std(sr_samples_tuple, correct_bias=True)
    post_fluence_mean = post_sr_mean_stacked[0]

    noise_floor_mean_val = results.get("noise_floor_mean", (0,0))[0]
    if hasattr(noise_floor_mean_val, 'item'):
        noise_floor_mean_val = noise_floor_mean_val.item()
        
    noise_floor_mean = np.asarray([noise_floor_mean_val] * data["positions"].shape[1])
    reco_fluence_signal_only = post_fluence_mean.squeeze() - noise_floor_mean_val

    # --- Parameter extraction (using local param_methods for corrections) ---
    param_methods = {
        "zenith": lambda s: model.zen_and_az(s)[0],
        "azimuth": lambda s: model.zen_and_az(s)[1],
        "core_x": lambda s: model.core(s)[0],
        "core_y": lambda s: model.core(s)[1],
        "erad": lambda s: model.Erad(s),
        "xmax": lambda s: model.X_max(s),
        "xmax_timing_offset": lambda s: model.xmax_timing_offset(s) if hasattr(model, 'xmax_timing_offset') else 0.0,
    }
    
    # operate on the list of samples to apply param_methods, then stack.
    samp_xmax = np.array([param_methods["xmax"](s) for s in samples])
    samp_xmax_offset = np.array([param_methods["xmax_timing_offset"](s) for s in samples])
    samp_xmax_timing = samp_xmax + samp_xmax_offset
    
    # Correct E_rad samples using the energy correction factor, just like reconstruction_with_RIA.py
    ecf_samples = np.array([model.get_energy_correction_factor(s) for s in samples])
    raw_samp_erad = np.array([param_methods["erad"](s) for s in samples])
    samp_erad = raw_samp_erad / (ecf_samples + 1e-15)
    
    samp_core_x = np.array([param_methods["core_x"](s) for s in samples])
    samp_core_y = np.array([param_methods["core_y"](s) for s in samples])
    samp_zen = np.rad2deg(np.array([param_methods["zenith"](s) for s in samples]))
    samp_az = np.rad2deg(np.array([param_methods["azimuth"](s) for s in samples]))

    # --- 3. Aggregate results ---
    plot_data = {
        'positions_lofar': data["positions"],
        'fluences_lofar': data["fluences"],
        'times_lofar': data["times"],
        'is_signal_lofar': np.ones(data["fluences"].shape, dtype=bool),
        'use_timing': results["timing_mask"],
        
        'positions_regular': np.asarray(positions_regular),
        'fluences_reg': np.asarray(truth["fluences_reg"]) if truth["fluences_reg"] is not None else np.zeros(positions_regular.shape[1]),
        'times_reg': np.asarray(truth["times_reg"]) if truth["times_reg"] is not None else np.zeros(positions_regular.shape[1]),
        'fluences_lofar_truth': np.asarray(truth["fluences_lofar_truth"]) if truth["fluences_lofar_truth"] is not None else None,
        'times_lofar_truth': np.asarray(truth["times_lofar_truth"]) if truth["times_lofar_truth"] is not None else None,
        
        'post_fluence_no_cf_mean': np.asarray(post_fluence_no_cf_mean),
        'post_fluence_std': np.asarray(post_fluence_std),
        'post_timing_no_cf_mean': np.asarray(post_timing_no_cf_mean),
        'post_timing_std': np.asarray(post_timing_std),
        
        'reco_fluence_signal_only': np.asarray(reco_fluence_signal_only),
        'noise_floor_mean': noise_floor_mean,
        
        'mean_syst_cf': np.asarray(jnp.mean(scf_maps, axis=0)),
        'std_syst_cf': np.asarray(jnp.std(scf_maps, axis=0)),
        'mean_timing_cf': np.asarray(jnp.mean(tcf_maps, axis=0)) * 1e9, # ns
        'std_timing_cf': np.asarray(jnp.std(tcf_maps, axis=0)) * 1e9,   # ns
        
        'model_min_x': min_x,
        'model_min_y': min_y,
        'model_extent': extent,
        
        'truth_dict': {
            'zenith_rad': truth["zenith"],
            'azimuth_rad': truth["azimuth"],
            'core_x_m': truth["core_x"],
            'core_y_m': truth["core_y"],
            'xmax_gpcm2': truth["xmax"],
            'erad_ev': truth["erad"],
        },
        'reco_dict': results,
        
        'samp_xmax': samp_xmax,
        'samp_xmax_timing': samp_xmax_timing,
        'samp_erad': samp_erad,
        'raw_samp_erad': raw_samp_erad,
        'samp_core_x': samp_core_x,
        'samp_core_y': samp_core_y,
        'samp_zen': samp_zen,
        'samp_az': samp_az,
    }
    
    # Handle optional corrections if they exist
    if "energy_correction_factor" in results:
        plot_data['reco_dict']['energy_correction_factor'] = results["energy_correction_factor"]

    try:
        fancy_reco_plot.generate_reco_plot(plot_data, output_path, use_latex=False)
        logger.info(f"Plot saved to: {output_path}")
    except Exception as e:
        logger.error(f"Failed to generate fancy plot: {e}")

def load_example_data():
    """Load pre-saved example data and truth values."""
    d = np.load(EXAMPLE_DATA_FILE)
    data = {
        "positions": d["positions"].astype(np.float64),
        "fluences": d["fluences"].astype(np.float64),
        "times": d["times"].astype(np.float64),
        "noise_std": float(d.get("noise_std", d.get("noise_level", 1.0))),
        "mean_zenith": float(d["mean_zenith"]),
        "mean_azimuth": float(d["mean_azimuth"]),
        "mean_core_x": float(d["mean_core_x"]),
        "mean_core_y": float(d["mean_core_y"]),
        "std_core_m": float(d["std_core_m"]),
        "t0_initial_guess": float(d["t0_initial_guess"]) if "t0_initial_guess" in d else None,
        "t0_prior_std": float(d["t0_prior_std"]) if "t0_prior_std" in d else None,
        "noise_floor_mean": float(d.get("noise_floor_mean", d.get("noise_mean", 1.0))),
        "timing_mask": d["timing_mask"].astype(bool) if "timing_mask" in d else None,
        "timing_uncertainties": d["timing_uncertainties"].astype(np.float64) if "timing_uncertainties" in d else None,
    }
    
    # Gracefully fix pre-existing array length mismatches in the example data
    n_ant = data["positions"].shape[1]
    if data["timing_mask"] is not None and len(data["timing_mask"]) != n_ant:
        data["timing_mask"] = data["timing_mask"][:n_ant]
    if data["timing_uncertainties"] is not None and len(data["timing_uncertainties"]) != n_ant:
        data["timing_uncertainties"] = data["timing_uncertainties"][:n_ant]

    t = np.load(EXAMPLE_TRUTH_FILE)
    truth = {
        "zenith": float(t["zenith"]),
        "azimuth": float(t["azimuth"]),
        "core_x": float(t["core_x"]),
        "core_y": float(t["core_y"]),
        "erad": float(t["erad"]),
        "xmax": float(t["xmax"]),
        "positions_regular": t["positions_regular"] if "positions_regular" in t else None,
        "fluences_reg": t["fluences_reg"] if "fluences_reg" in t else None,
        "times_reg": t["times_reg"] if "times_reg" in t else None,
        "fluences_lofar_truth": t["fluences_lofar_truth"] if "fluences_lofar_truth" in t else None,
        "times_lofar_truth": t["times_lofar_truth"] if "times_lofar_truth" in t else None,
    }
    return data, truth

def run_example_reconstruction():
    """Load real example data and reconstruct."""
    logger.info("=" * 60)
    logger.info("  RIA Example-Data Reconstruction Test (Radio Only)")
    logger.info("=" * 60)

    if not _HAS_EXAMPLE_DATA:
        logger.error("Example data not found in tests/example_data/.")
        sys.exit(1)

    data, truth = load_example_data()
    n_ant = data["positions"].shape[1]
    logger.info(f"Loaded example data: {n_ant} antennas")

    # Use fast settings for testing
    config.N_VI_ITERATIONS = FAST_N_ITERATIONS
    config.N_SAMPLES = FAST_N_SAMPLES
    config.FLUENCE_CF = False # set to True to test the fluence correlated field!
    config.RESAMPLING_MODE = "linear_resample"
    config.SAMPLING_MODE = "nonlinear_sample"

    # Use atmosphere file if available
    atm_path = EXAMPLE_ATM_FILE if os.path.isfile(EXAMPLE_ATM_FILE) else None

    logger.info(f"Running reconstruction ({FAST_N_ITERATIONS} iterations, "
                f"{FAST_N_SAMPLES} samples) ...")

    
    # Pass t0 prior via model_kwargs if available
    model_kwargs = {}
    if data.get("t0_initial_guess") is not None and data.get("t0_prior_std") is not None:
        model_kwargs["params_t0"] = {
            "mean": data["t0_initial_guess"],
            "std": data["t0_prior_std"]
        }
        
    results = reconstruct(
        data["positions"],
        data["fluences"],
        data["times"],
        data["noise_std"],
        mean_zenith=data["mean_zenith"],
        mean_azimuth=data["mean_azimuth"],
        mean_core_x=data["mean_core_x"],
        mean_core_y=data["mean_core_y"],
        std_core_m=data["std_core_m"],
        noise_floor_mean=data["noise_floor_mean"],
        t0_initial_guess=data.get("t0_initial_guess"),
        timing_mask=data["timing_mask"],
        timing_uncertainties=data["timing_uncertainties"],
        atmosphere_path=atm_path,
        enable_timing_qc=False,
        seed=42,
        model_kwargs=model_kwargs if model_kwargs else None,
    )

    # --- Plotting ---
    plot_path = os.path.join(os.path.dirname(__file__), "plots", "example_reco.png")
    generate_diagnostic_plots(results, data, truth, plot_path)

    logger.info("=" * 60)
    return results, truth

# ======================================================================
#  Pytest interface
# ======================================================================

@pytest.mark.skipif(not _HAS_EXAMPLE_DATA, reason="Example data not found.")
def test_example_reconstruction():
    """End-to-end reconstruction on real example simulation data."""
    results, truth = run_example_reconstruction()


if __name__ == "__main__":
    run_example_reconstruction()
