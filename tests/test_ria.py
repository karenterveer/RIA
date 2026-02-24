"""
Unit tests for the RIA package.

Tests cover module imports, atmosphere calculations, coordinate
transforms, LDF evaluation, forward-model instantiation, and timing
quality control.
"""

import sys
import os
import numpy as np
import pytest

# Ensure the RIA package root is on the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ===================================================================
#  Import tests
# ===================================================================

class TestImports:
    """Verify that all public modules and symbols can be imported."""

    def test_top_level_imports(self):
        from ria import FootprintModel, reconstruct, Atmosphere, CoordinateTransform, config
        assert FootprintModel is not None
        assert reconstruct is not None
        assert Atmosphere is not None

    def test_submodule_imports(self):
        from ria import ldf, timing
        assert hasattr(ldf, "LDF")
        assert hasattr(timing, "detect_timing_outliers")

    def test_config_attributes(self):
        from ria import config
        assert hasattr(config, "C_LIGHT")
        assert hasattr(config, "N_VI_ITERATIONS")
        assert hasattr(config, "MAGNETIC_FIELD_VECTORS")


# ===================================================================
#  Atmosphere tests
# ===================================================================

class TestAtmosphere:
    """Test the Atmosphere class with built-in models."""

    @pytest.fixture
    def atm(self):
        from ria.atmosphere import Atmosphere
        return Atmosphere(model=17)

    def test_grammage_at_sea_level(self, atm):
        """Grammage at sea level should be ~1033 g/cm^2."""
        import jax.numpy as jnp
        gram = atm.get_atmosphere(jnp.array(0.0))
        # In g/m^2 internally — multiply by 1e-4 to get g/cm^2
        gram_gcm2 = float(gram) * 1e-4
        assert 900 < gram_gcm2 < 1100, f"Sea-level grammage {gram_gcm2} g/cm^2 out of range"

    def test_grammage_monotonically_decreases(self, atm):
        """Grammage should decrease with increasing height."""
        import jax.numpy as jnp
        heights = jnp.linspace(0, 50000, 100)
        grammages = atm.get_atmosphere(heights)
        diffs = jnp.diff(grammages)
        assert jnp.all(diffs <= 0), "Grammage should decrease with height"

    def test_density_positive(self, atm):
        """Density should be positive at all reasonable heights."""
        import jax.numpy as jnp
        heights = jnp.linspace(0, 50000, 100)
        densities = atm._get_density_from_height(heights)
        assert jnp.all(densities >= 0), "Negative density encountered"

    def test_refractive_index_above_one(self, atm):
        """Refractive index should be >= 1 at moderate depths."""
        import jax.numpy as jnp
        grammage = jnp.array(700.0)
        zenith = jnp.array(0.3)
        n = atm.get_refractive_index(grammage, zenith)
        assert float(n) >= 1.0, f"Refractive index {n} < 1.0"

    def test_cherenkov_angle_positive(self, atm):
        """Cherenkov angle should be positive at moderate depths."""
        import jax.numpy as jnp
        grammage = jnp.array(700.0)
        zenith = jnp.array(0.3)
        theta_c = atm.get_cherenkov_angle(grammage, zenith)
        assert float(theta_c) > 0, f"Cherenkov angle {theta_c} <= 0"

    def test_height_grammage_roundtrip(self, atm):
        """get_vertical_height should invert get_atmosphere."""
        import jax.numpy as jnp
        h_in = jnp.array(5000.0)
        gram = atm.get_atmosphere(h_in)
        h_out = atm.get_vertical_height(gram)
        np.testing.assert_allclose(float(h_out), float(h_in), atol=1.0)

    def test_different_models_give_different_values(self):
        from ria.atmosphere import Atmosphere
        import jax.numpy as jnp
        atm1 = Atmosphere(model=1)
        atm17 = Atmosphere(model=17)
        g1 = float(atm1.get_atmosphere(jnp.array(0.0)))
        g17 = float(atm17.get_atmosphere(jnp.array(0.0)))
        assert g1 != g17, "Different models should give different grammages"


# ===================================================================
#  Coordinate transform tests
# ===================================================================

class TestCoordinateTransform:
    """Test the CoordinateTransform class."""

    @pytest.fixture
    def ct(self):
        from ria.coordinates import CoordinateTransform
        import jax.numpy as jnp
        return CoordinateTransform(
            zenith=jnp.array(0.3),
            azimuth=jnp.array(1.0),
        )

    def test_roundtrip_vxB(self, ct):
        """Transforming to vxB and back should recover the original."""
        import jax.numpy as jnp
        pos = jnp.array([100.0, 200.0, 0.0])
        sp = ct.transform_to_vxB_vxvxB(pos)
        recovered = ct.transform_from_vxB_vxvxB(sp)
        np.testing.assert_allclose(
            np.array(recovered), np.array(pos), atol=1e-6,
        )

    def test_roundtrip_onsky(self, ct):
        """Transforming to on-sky and back should recover the original."""
        import jax.numpy as jnp
        pos = jnp.array([100.0, 200.0, 50.0])
        sky = ct.transform_from_ground_to_onsky(pos)
        recovered = ct.transform_from_onsky_to_ground(sky)
        np.testing.assert_allclose(
            np.array(recovered), np.array(pos), atol=1e-6,
        )

    def test_batch_transform(self, ct):
        """Batch transform of (N, 3) positions should work."""
        import jax.numpy as jnp
        positions = jnp.array([
            [100.0, 200.0, 0.0],
            [-50.0, 150.0, 0.0],
        ])
        sp = ct.transform_to_vxB_vxvxB(positions)
        assert sp.shape == (2, 3)

    def test_spherical_to_cartesian(self):
        from ria.coordinates import spherical_to_cartesian
        import jax.numpy as jnp
        # Zenith=0 should give (0, 0, 1)
        v = spherical_to_cartesian(jnp.array(0.0), jnp.array(0.0))
        np.testing.assert_allclose(np.array(v), [0.0, 0.0, 1.0], atol=1e-10)

    def test_get_angle_perpendicular(self):
        from ria.coordinates import get_angle
        import jax.numpy as jnp
        v1 = jnp.array([1.0, 0.0, 0.0])
        v2 = jnp.array([0.0, 1.0, 0.0])
        angle = get_angle(v1, v2)
        np.testing.assert_allclose(float(angle), np.pi / 2, atol=1e-6)


# ===================================================================
#  LDF tests
# ===================================================================

class TestLDF:
    """Test the LDF functions."""

    def test_ldf_returns_tuple(self):
        """LDF should return a 5-element tuple."""
        from ria.ldf import LDF
        import jax.numpy as jnp
        result = LDF(
            x=jnp.array(100.0), y=jnp.array(0.0),
            Erad=jnp.array(1e7), xmax=jnp.array(700.0),
            zenith=jnp.array(0.3), azimuth=jnp.array(1.0),
        )
        assert len(result) == 5

    def test_ldf_positive_fluence(self):
        """Total fluence should be >= 0."""
        from ria.ldf import LDF
        import jax.numpy as jnp
        f_total, *_ = LDF(
            x=jnp.array(100.0), y=jnp.array(0.0),
            Erad=jnp.array(1e7), xmax=jnp.array(700.0),
            zenith=jnp.array(0.3), azimuth=jnp.array(1.0),
        )
        assert float(f_total) >= 0, "Negative fluence"

    def test_ldf_decreases_with_distance(self):
        """Fluence should decrease with distance from core."""
        from ria.ldf import LDF
        import jax.numpy as jnp
        f_near, *_ = LDF(
            x=jnp.array(50.0), y=jnp.array(0.0),
            Erad=jnp.array(1e7), xmax=jnp.array(700.0),
            zenith=jnp.array(0.3), azimuth=jnp.array(1.0),
        )
        f_far, *_ = LDF(
            x=jnp.array(300.0), y=jnp.array(0.0),
            Erad=jnp.array(1e7), xmax=jnp.array(700.0),
            zenith=jnp.array(0.3), azimuth=jnp.array(1.0),
        )
        assert float(f_near) > float(f_far), "Fluence should decrease with distance"

    def test_ldf_scales_with_energy(self):
        """Doubling the energy should roughly double the fluence."""
        from ria.ldf import LDF
        import jax.numpy as jnp
        kwargs = dict(
            x=jnp.array(100.0), y=jnp.array(0.0),
            xmax=jnp.array(700.0),
            zenith=jnp.array(0.3), azimuth=jnp.array(1.0),
        )
        f1, *_ = LDF(Erad=jnp.array(1e7), **kwargs)
        f2, *_ = LDF(Erad=jnp.array(2e7), **kwargs)
        ratio = float(f2) / max(float(f1), 1e-30)
        assert 1.5 < ratio < 2.5, f"Energy scaling ratio {ratio} out of expected range"

    def test_bspline_evaluation(self):
        """B-spline evaluation should return finite values."""
        from ria.ldf import evaluate_bspline, _spline_data
        import jax.numpy as jnp
        t, c, k = _spline_data["rcut_geo"]
        result = evaluate_bspline(jnp.array([500.0, 600.0, 700.0]), t, c, k)
        assert jnp.all(jnp.isfinite(result)), "B-spline returned non-finite values"


# ===================================================================
#  Forward model tests
# ===================================================================

class TestFootprintModel:
    """Test the FootprintModel class."""

    @pytest.fixture
    def model(self):
        from ria.forward_model import FootprintModel
        import jax

        # Enable 64-bit
        jax.config.update("jax_enable_x64", True)

        x = np.linspace(-200, 200, 20)
        y = np.linspace(-200, 200, 20)
        return FootprintModel(x, y)

    def test_instantiation(self, model):
        """Model should instantiate without errors."""
        assert model is not None

    def test_init_dict(self, model):
        """Model init should return a non-empty dict."""
        import jax
        key = jax.random.PRNGKey(0)
        init = model.init(key)
        assert len(init) > 0

    def test_output_shape(self, model):
        """__call__ should return shape (2, N_antennas)."""
        import jax
        key = jax.random.PRNGKey(0)
        x = model.init(key)
        result = model(x)
        assert result.shape == (2, 20), f"Expected (2, 20), got {result.shape}"

    def test_fluence_row_positive(self, model):
        """Fluence predictions should be non-negative."""
        import jax
        import jax.numpy as jnp
        key = jax.random.PRNGKey(42)
        x = model.init(key)
        result = model(x)
        fluences = result[0]
        assert jnp.all(fluences >= 0), "Negative fluence in model output"


# ===================================================================
#  Timing QC tests
# ===================================================================

class TestTimingQC:
    """Test the timing quality control functions."""

    @pytest.fixture
    def timing_data(self):
        """Synthetic timing data with one clear outlier."""
        rng = np.random.RandomState(42)
        n = 50
        x = rng.uniform(-200, 200, n)
        y = rng.uniform(-200, 200, n)
        positions = np.stack([x, y])
        # Plane wave-like timing + noise
        times = 1e-9 * (0.01 * x + 0.02 * y + rng.normal(0, 2.0, n))
        station_ids = np.zeros(n, dtype=int)
        # Insert outlier
        times[0] += 100e-9
        return positions, times, station_ids

    def test_outlier_detection(self, timing_data):
        """The outlier at index 0 should be flagged."""
        from ria.timing import detect_timing_outliers
        pos, t, sids = timing_data
        mask = detect_timing_outliers(pos, t, sids)
        # The inserted outlier should be flagged (not kept)
        assert not mask[0], "Outlier at index 0 was not detected"
        assert np.sum(mask) >= 40, f"Too many points removed: {np.sum(mask)}"

    def test_iterative_pruning(self, timing_data):
        """Iterative pruning should converge."""
        from ria.timing import iterative_timing_pruning
        pos, t, _ = timing_data
        initial_mask = np.ones(len(t), dtype=bool)
        mask, std, *_ = iterative_timing_pruning(pos, t, initial_mask)
        assert std < 50e-9, f"Pruning std {std*1e9:.1f} ns is too large"
        assert int(np.sum(mask)) > 20, "Too few points after pruning"

    def test_local_uncertainties(self, timing_data):
        """Per-antenna uncertainties should be positive and finite."""
        from ria.timing import get_local_timing_uncertainties
        pos, t, _ = timing_data
        unc = get_local_timing_uncertainties(pos, t)
        assert np.all(unc > 0), "Zero or negative uncertainty"
        assert np.all(np.isfinite(unc)), "Non-finite uncertainty"


# ===================================================================
#  Config tests
# ===================================================================

class TestConfig:
    """Test the configuration module."""

    def test_physical_constants(self):
        from ria import config
        assert config.C_LIGHT > 2.9e8
        assert config.C_LIGHT < 3.1e8

    def test_lofar_field_vector(self):
        from ria import config
        vec = config.MAGNETIC_FIELD_VECTORS["lofar"]
        assert len(vec) == 3
        assert np.linalg.norm(vec) > 0

    def test_config_mutability(self):
        """Config values should be overridable."""
        from ria import config
        original = config.N_VI_ITERATIONS
        config.N_VI_ITERATIONS = 99
        assert config.N_VI_ITERATIONS == 99
        config.N_VI_ITERATIONS = original
