#!/usr/bin/env python3
"""
Reconstruction Plotter Module

# After reconstruction, data is auto-exported with the plot
generate_reco_plot(plot_data, "output/plot.png")  # Creates plot.png and plot_data.json

"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LogNorm, LinearSegmentedColormap
from scipy.stats import gaussian_kde
import os
import json

# =============================================================================
# PLOT STYLE CONFIGURATION
# =============================================================================

PLOT_STYLE = {
    "text.usetex": True,
    "font.family": "serif",
    # "text.latex.preamble": r"\usepackage[T1]{fontenc}",
    "font.size": 26,
    "axes.labelsize": 36,
    "axes.titlesize": 36,
    "axes.titleweight": "bold",  # Added global bold title setting
    "xtick.labelsize": 29,
    "ytick.labelsize": 29,
    "legend.fontsize": 32,
    "figure.dpi": 300,
}

PLOT_GREEN = '#405d3a'
MARKER_SIZE = 75
MARKER_STYLE = 's'
C_LIGHT = 299792458.0

# Define custom colormap from White to Green
PLOT_CMAP = LinearSegmentedColormap.from_list("WhiteGreen", ["white", PLOT_GREEN])

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def setup_ax_style(ax, xlabel, ylabel, show_grid=False):
    """Apply consistent paper-ready style to axis."""
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if show_grid:
        ax.grid(True, linestyle=':', linewidth=0.5, alpha=0.6, color="gray")
    else:
        ax.grid(False)
    ax.tick_params(axis='both', which='major', direction='in', length=6, width=1.0)


def safe_kde(data, ax, color='black', truth=None, label=None):
    """Safely plot a KDE distribution with 1-sigma band, falling back to histogram if KDE fails."""
    data = np.asarray(data).flatten()
    data = data[np.isfinite(data)]
    if len(data) < 3:
        ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', transform=ax.transAxes)
        return
    try:
        kde = gaussian_kde(data)
        
        # Padding to prevent cutoff
        d_min, d_max = np.min(data), np.max(data)
        d_range = d_max - d_min
        padding = d_range * 0.3 if d_range > 0 else 1.0
        
        x_range = np.linspace(d_min - padding, d_max + padding, 200)
        y_kde = kde(x_range)
        
        # 1. Plot full distribution (lighter)
        ax.fill_between(x_range, y_kde, color=color, alpha=0.2, linewidth=0)
        
        # 2. Plot 1-sigma band (darker) - 16th to 84th percentile
        p16, p84 = np.percentile(data, [16, 84])
        # Create a mask for the range to fill
        sigma_mask = (x_range >= p16) & (x_range <= p84)
        ax.fill_between(x_range, y_kde, where=sigma_mask, color=color, alpha=0.4, linewidth=0)
        
        # 3. Plot the line
        ax.plot(x_range, y_kde, color=color, lw=2.0, label=label)
        
    except Exception:
        ax.hist(data, bins=30, color=color, alpha=0.5, density=True, histtype='stepfilled', label=label)
    
    if truth is not None:
        ax.axvline(truth, color='black', linestyle='--', lw=1.5, label='Truth' if label is None else None)


def plot_2d_contour(x_data, y_data, ax, truth_point=None, x_label='', y_label='', title='', use_latex=True):

    """Plot 2D contour/scatter with optional truth marker."""
    x_data = np.asarray(x_data).flatten()
    y_data = np.asarray(y_data).flatten()
    valid = np.isfinite(x_data) & np.isfinite(y_data)
    x_data, y_data = x_data[valid], y_data[valid]
    
    if len(x_data) < 5:
        ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', transform=ax.transAxes)
        return
    
    try:
        xy = np.vstack([x_data, y_data])
        kde = gaussian_kde(xy)
        
        # Calculate padding so the grid covers the 'hills' of the KDE outside the samples
        x_min, x_max = np.min(x_data), np.max(x_data)
        y_min, y_max = np.min(y_data), np.max(y_data)
        
        x_span = x_max - x_min
        y_span = y_max - y_min
        
        x_pad = 0.3 * x_span if x_span > 0 else 1.0
        y_pad = 0.3 * y_span if y_span > 0 else 1.0
        
        x_grid = np.linspace(x_min - x_pad, x_max + x_pad, 100)
        y_grid = np.linspace(y_min - y_pad, y_max + y_pad, 100)
        
        X, Y = np.meshgrid(x_grid, y_grid)
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)
        
        # Calculate 1, 2, and 3 sigma (68%, 95%, 99.7%) containment levels
        z_values = Z.ravel()
        z_sorted = np.sort(z_values)[::-1]
        z_cum = np.cumsum(z_sorted)
        z_cum /= z_cum[-1] # Normalize to 1
        
        level_997 = z_sorted[np.argmin(np.abs(z_cum - 0.997))]
        level_95  = z_sorted[np.argmin(np.abs(z_cum - 0.95))]
        level_68  = z_sorted[np.argmin(np.abs(z_cum - 0.68))]
        
        # Plot filled contours for the containment regions
        # Use a smooth gradient within the levels starting from 3-sigma
        fill_levels = np.linspace(level_997, np.max(Z), 50)
        ax.contourf(X, Y, Z, levels=fill_levels, cmap=PLOT_CMAP, extend='max', alpha=0.8)
        
        # Draw explicit contour lines for 1, 2, and 3 sigma
        # Linewidths: 3-sigma (thinnest), 2-sigma, 1-sigma (thickest)
        cs = ax.contour(X, Y, Z, levels=[level_997, level_95, level_68], colors=[PLOT_GREEN]*3, 
                   linewidths=[0.8, 1.4, 2.2], alpha=0.8)
        
        # Add sigma labels to the contour lines
        fmt = {level_997: r'3$\sigma$', level_95: r'2$\sigma$', level_68: r'1$\sigma$'}
        clbls = ax.clabel(cs, fmt=fmt, inline=True, fontsize=18)
        
        # Add white outline and bold weight for better legibility
        from matplotlib.patheffects import withStroke
        plt.setp(clbls, path_effects=[withStroke(linewidth=3, foreground='white')], fontweight='bold')
    except Exception as e:
        print(f"WARNING: Contour failed for {title}: {e}")
    
    if truth_point is not None:
        ax.scatter(*truth_point, marker='*', s=150, c='black', zorder=10, label='Truth')
    
    ax.scatter(np.mean(x_data), np.mean(y_data), marker='+', s=120, c=PLOT_GREEN, linewidth=2.0, zorder=9, label='Mean')
    
    
    setup_ax_style(ax, x_label, y_label)
    set_ax_title(ax, title, use_latex)

    # Expand limits to make room for legend in top right (using the padded grid limits)
    cur_xlim = ax.get_xlim()
    cur_ylim = ax.get_ylim()
    width = cur_xlim[1] - cur_xlim[0]
    height = cur_ylim[1] - cur_ylim[0]
    ax.set_xlim(cur_xlim[0], cur_xlim[1] + 0.2 * width)   # +20% width to the right
    ax.set_ylim(cur_ylim[0], cur_ylim[1] + 0.2 * height)  # +20% height to the top

    # Legend with frameless white background
    ax.legend(loc='upper right', fontsize=24, frameon=True, facecolor='None', framealpha=0.3, edgecolor=PLOT_GREEN)


# =============================================================================
# DATA EXPORT/IMPORT FUNCTIONS
# =============================================================================

def _numpy_to_list(obj):
    """Recursively convert numpy/JAX arrays to lists for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif hasattr(obj, '__jax_array__') or type(obj).__name__ == 'ArrayImpl':
        # Handle JAX arrays by converting to numpy first
        return np.asarray(obj).tolist()
    elif type(obj).__name__ == 'Vector' and hasattr(obj, 'tree'):
        # Handle jft.Vector objects
        return _numpy_to_list(obj.tree)
    elif isinstance(obj, dict):
        return {k: _numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_numpy_to_list(item) for item in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif hasattr(obj, 'item'):
        # Handle scalar JAX/numpy types with .item() method
        return obj.item()
    elif obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    else:
        # Fallback for un-serializable objects (like FootprintModel, functions, etc.)
        return str(obj)


def _list_to_numpy(obj):
    """Recursively convert lists back to numpy arrays after JSON deserialization."""
    if isinstance(obj, dict):
        return {k: _list_to_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        # Try to convert to numpy array if it looks like numeric data
        try:
            arr = np.array(obj)
            if arr.dtype.kind in ('f', 'i', 'b'):  # float, int, or bool
                return arr
        except (ValueError, TypeError):
            pass
        # Otherwise recursively process
        return [_list_to_numpy(item) for item in obj]
    return obj


def export_plot_data(plot_data, output_path):
    """
    Export plot data to a JSON file for later re-plotting.
    
    Parameters
    ----------
    plot_data : dict
        The plot data dictionary as passed to generate_reco_plot.
    output_path : str
        Path for the output JSON file (typically same base name as plot, with .json extension).
    
    Returns
    -------
    str
        The path to the exported JSON file.
    """
    serializable_data = _numpy_to_list(plot_data)
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(serializable_data, f, indent=2)
    
    return output_path


def import_plot_data(input_path):
    """
    Import plot data from a JSON file for re-plotting.
    
    Parameters
    ----------
    input_path : str
        Path to the JSON file containing exported plot data.
    
    Returns
    -------
    dict
        The plot data dictionary, with arrays restored as numpy arrays.
    """
    with open(input_path, 'r') as f:
        loaded_data = json.load(f)
    
    plot_data = _list_to_numpy(loaded_data)
    
    print(f"[reco_plot] Imported plot data from: {input_path}")
    return plot_data



def set_ax_title(ax, title, use_latex):
    """Set axis title with appropriate formatting based on LaTeX usage."""
    if use_latex:
        ax.set_title(r'\textbf{' + title.replace('%', r'\%') + '}')
    else:
        ax.set_title(title, fontweight='bold')


# =============================================================================
# MAIN PLOTTING FUNCTION
# =============================================================================

def generate_reco_plot(plot_data, output_path, export_data=True, use_latex=True):
    """
    Generate a publication-quality reconstruction summary plot in a 5x4 grid.
    Columns: Fluence, Timing (PW sub), Correlated Fields, Posteriors
    
    Parameters
    ----------
    plot_data : dict
        Dictionary containing all data needed for plotting.
    output_path : str
        Path for the output plot image.
    export_data : bool, optional
        If True (default), exports plot_data to a JSON file alongside the plot.
        This allows re-plotting later using import_plot_data() without running
        the full reconstruction.
    use_latex : bool, optional
        If True, use LaTeX for text rendering. Default is False.
    
    Returns
    -------
    str
        The path to the saved plot.
    """
    # specific style updates
    current_style = PLOT_STYLE.copy()
    current_style["text.usetex"] = use_latex
    if use_latex:
        current_style["text.latex.preamble"] = r"\usepackage[T1]{fontenc}"
    
    plt.rcParams.update(current_style)
    
    # Extract data from plot_data dict
    positions_lofar = plot_data['positions_lofar']
    fluences_lofar = plot_data['fluences_lofar']
    times_lofar = plot_data['times_lofar']
    is_signal_lofar = plot_data['is_signal_lofar']
    use_timing = plot_data['use_timing']
    
    positions_regular = plot_data['positions_regular']
    fluences_reg = plot_data['fluences_reg']
    times_reg = plot_data['times_reg']
    
    post_fluence_no_cf_mean = plot_data['post_fluence_no_cf_mean']
    post_fluence_std = plot_data['post_fluence_std']
    post_timing_no_cf_mean = plot_data['post_timing_no_cf_mean']
    post_timing_std = plot_data['post_timing_std']
    
    reco_fluence_signal_only = plot_data['reco_fluence_signal_only'] # This is with CF
    
    mean_syst_cf = plot_data['mean_syst_cf']
    std_syst_cf = plot_data['std_syst_cf']
    mean_timing_cf = plot_data['mean_timing_cf']
    std_timing_cf = plot_data['std_timing_cf']
    
    model_min_x = plot_data['model_min_x']
    model_min_y = plot_data['model_min_y']
    model_extent = plot_data['model_extent']
    
    truth_dict = plot_data['truth_dict']
    reco_dict = plot_data['reco_dict']
    
    samp_xmax = plot_data['samp_xmax']
    samp_xmax_timing = plot_data['samp_xmax_timing']
    samp_erad = plot_data['samp_erad']
    raw_samp_erad = plot_data.get('raw_samp_erad', samp_erad) # Fallback if missing
    samp_core_x = plot_data['samp_core_x']
    samp_core_y = plot_data['samp_core_y']
    samp_zen = plot_data['samp_zen']
    samp_az = plot_data['samp_az']
    
    # --- Create Figure ---
    fig = plt.figure(figsize=(30, 30), constrained_layout=True)
    gs = gridspec.GridSpec(5, 4, figure=fig)
    
    def add_cbar(mappable, ax, label=''):
        """Colorbar anchored to axes frame — always matches plot height."""
        cax = ax.inset_axes([1.1, 0, 0.05, 1])
        return fig.colorbar(mappable, cax=cax, label=label)
    
    # --- Limits and Masks ---
    # CF grid spans full model extent; imshow_extent must match the actual grid coordinates
    imshow_extent = [model_min_x, model_min_x + model_extent, model_min_y, model_min_y + model_extent]
    # Cropped view limits for all spatial plots (100m padding from grid edges)
    plot_xlims = (model_min_x+100, model_min_x + model_extent-100)
    plot_ylims = (model_min_y+100, model_min_y + model_extent-100)

    
    good_timing_mask = use_timing
    demoted_signal_mask = is_signal_lofar & ~use_timing
    no_signal_mask = ~is_signal_lofar
    
    all_signal_fluences = fluences_lofar[is_signal_lofar]
    vmin_noisy = np.min(all_signal_fluences) if all_signal_fluences.size > 0 else None
    vmax_noisy = np.max(all_signal_fluences) if all_signal_fluences.size > 0 else None
    
    max_fluence_truth = np.max(fluences_reg) if fluences_reg.size > 0 else 1.0
    plot_mask_reg = fluences_reg >= (0.05 * max_fluence_truth)
    
    # 2D Mask for CF plots (Column 3)
    try:
        mask_2d = plot_mask_reg.reshape(mean_syst_cf.shape)
    except Exception:
        mask_2d = np.ones_like(mean_syst_cf, dtype=bool)
    
    vmin_fluence = np.percentile(fluences_reg[plot_mask_reg], 1) if np.any(plot_mask_reg) else None
    vmax_fluence = np.percentile(fluences_reg[plot_mask_reg], 99) if np.any(plot_mask_reg) else None
    
    # --- NO Plane Wave Subtraction ---
    time_truth_pw_sub_ns = times_reg * 1e9
    time_data_pw_sub_ns = times_lofar * 1e9
    time_reco_pw_sub_ns = post_timing_no_cf_mean * 1e9
    
    # Center on median of plotted points
    if np.any(plot_mask_reg):
        # 1. Center Truth on Truth Median
        med_tr = np.median(time_truth_pw_sub_ns[plot_mask_reg])
        time_truth_pw_sub_ns -= med_tr
        
        # 2. Align Reco to Reco Median
        med_re = np.median(time_reco_pw_sub_ns[plot_mask_reg])
        time_reco_pw_sub_ns -= med_re
        
        # 3. Align Data to Median (Assuming 0-mean reference)
        if np.any(good_timing_mask):
            med_data = np.median(time_data_pw_sub_ns[good_timing_mask])
            time_data_pw_sub_ns -= med_data
            
    # Define common color scale based on combined Truth and Reco range
    if np.any(plot_mask_reg):
        t_min_true = np.min(time_truth_pw_sub_ns[plot_mask_reg])
        t_max_true = np.max(time_truth_pw_sub_ns[plot_mask_reg])
        t_min_reco = np.min(time_reco_pw_sub_ns[plot_mask_reg])
        t_max_reco = np.max(time_reco_pw_sub_ns[plot_mask_reg])
        vmin_t = min(t_min_true, t_min_reco)
        vmax_t = max(t_max_true, t_max_reco)
    else:
        vmin_t, vmax_t = None, None
    
    # =========================================================================
    # COLUMN 1: Fluence
    # =========================================================================
    # 1. Fluence Data
    ax_f_data = fig.add_subplot(gs[0, 0])
    sc_fd = ax_f_data.scatter(positions_lofar[0, good_timing_mask], positions_lofar[1, good_timing_mask],
                              c=fluences_lofar[good_timing_mask], cmap='plasma_r', s=40, vmin=vmin_noisy, vmax=vmax_noisy, rasterized=True)
    ax_f_data.scatter(positions_lofar[0, demoted_signal_mask], positions_lofar[1, demoted_signal_mask],
                      c=fluences_lofar[demoted_signal_mask], cmap='plasma_r', marker='x', s=25, vmin=vmin_noisy, vmax=vmax_noisy, rasterized=True)
    ax_f_data.scatter(positions_lofar[0, no_signal_mask], positions_lofar[1, no_signal_mask], marker='.', color='gray', s=20, alpha=0.7, rasterized=True)
    setup_ax_style(ax_f_data, 'x (m)', 'y (m)')
    set_ax_title(ax_f_data, 'Fluence Data', use_latex)
    ax_f_data.set_xlim(plot_xlims); ax_f_data.set_ylim(plot_ylims); ax_f_data.set_aspect('equal')
    add_cbar(sc_fd, ax_f_data, label=r'Fluence (eV/m$^2$)')
    
    # 2. Fluence Truth
    ax_f_truth = fig.add_subplot(gs[1, 0], sharex=ax_f_data, sharey=ax_f_data)
    sc_ft = ax_f_truth.scatter(positions_regular[0, plot_mask_reg], positions_regular[1, plot_mask_reg],
                               c=fluences_reg[plot_mask_reg], cmap='plasma_r', s=MARKER_SIZE, marker=MARKER_STYLE, vmin=vmin_fluence, vmax=vmax_fluence, rasterized=True)
    setup_ax_style(ax_f_truth, 'x (m)', 'y (m)')
    set_ax_title(ax_f_truth, 'Fluence (Truth)', use_latex)
    ax_f_truth.set_xlim(plot_xlims); ax_f_truth.set_ylim(plot_ylims); ax_f_truth.set_aspect('equal')
    add_cbar(sc_ft, ax_f_truth, label=r'Fluence (eV/m$^2$)')
    
    # 3. Fluence Reco (no CF)
    ax_f_reco = fig.add_subplot(gs[2, 0], sharex=ax_f_data, sharey=ax_f_data)
    sc_fr = ax_f_reco.scatter(positions_regular[0, plot_mask_reg], positions_regular[1, plot_mask_reg],
                              c=post_fluence_no_cf_mean[plot_mask_reg], cmap='plasma_r', s=MARKER_SIZE, marker=MARKER_STYLE, vmin=vmin_fluence, vmax=vmax_fluence, rasterized=True)
    setup_ax_style(ax_f_reco, 'x (m)', 'y (m)')
    set_ax_title(ax_f_reco, 'Posterior Mean Fluence', use_latex)
    ax_f_reco.set_xlim(plot_xlims); ax_f_reco.set_ylim(plot_ylims); ax_f_reco.set_aspect('equal')
    add_cbar(sc_fr, ax_f_reco, label=r'Fluence (eV/m$^2$)')
    
    # 4. Fluence Uncertainty (%)
    ax_f_std = fig.add_subplot(gs[3, 0], sharex=ax_f_data, sharey=ax_f_data)
    percent_unc = 100 * (np.squeeze(post_fluence_std)[plot_mask_reg] / np.abs(post_fluence_no_cf_mean[plot_mask_reg] + 1e-15))
    sc_fs = ax_f_std.scatter(positions_regular[0, plot_mask_reg], positions_regular[1, plot_mask_reg],
                             c=percent_unc, cmap='inferno_r', s=MARKER_SIZE, marker=MARKER_STYLE, norm=LogNorm(vmin=np.percentile(percent_unc, 5), vmax=np.percentile(percent_unc, 95)), rasterized=True)
    setup_ax_style(ax_f_std, 'x (m)', 'y (m)')
    set_ax_title(ax_f_std, 'Fluence Uncertainty (%)', use_latex)
    ax_f_std.set_xlim(plot_xlims); ax_f_std.set_ylim(plot_ylims); ax_f_std.set_aspect('equal')
    add_cbar(sc_fs, ax_f_std, label='% Unc')
    
    # 5. Fluence Diff (%)
    ax_f_diff = fig.add_subplot(gs[4, 0], sharex=ax_f_data, sharey=ax_f_data)
    f_diff_pct = 100 * (post_fluence_no_cf_mean[plot_mask_reg] - fluences_reg[plot_mask_reg]) / np.maximum(fluences_reg[plot_mask_reg], 1e-15)
    vlim_f_diff = np.percentile(np.abs(f_diff_pct), 95)
    sc_f_diff = ax_f_diff.scatter(positions_regular[0, plot_mask_reg], positions_regular[1, plot_mask_reg],
                                  c=f_diff_pct, cmap='PuOr', s=MARKER_SIZE, marker=MARKER_STYLE, vmin=-vlim_f_diff, vmax=vlim_f_diff, rasterized=True)
    setup_ax_style(ax_f_diff, 'x (m)', 'y (m)')
    set_ax_title(ax_f_diff, 'Fluence Diff (%)', use_latex)
    ax_f_diff.set_xlim(plot_xlims); ax_f_diff.set_ylim(plot_ylims); ax_f_diff.set_aspect('equal')
    add_cbar(sc_f_diff, ax_f_diff, label='% Diff')
    
    # =========================================================================
    # COLUMN 2: Timing (PW sub)
    # =========================================================================
    # 1. Timing Data
    ax_t_data = fig.add_subplot(gs[0, 1], sharex=ax_f_data, sharey=ax_f_data)
    if np.any(good_timing_mask):
        sc_td = ax_t_data.scatter(positions_lofar[0, good_timing_mask], positions_lofar[1, good_timing_mask],
                                  c=time_data_pw_sub_ns[good_timing_mask], cmap='viridis', s=40, vmin=vmin_t, vmax=vmax_t, rasterized=True)
        add_cbar(sc_td, ax_t_data, label='Rel. Time (ns)')
    ax_t_data.scatter(positions_lofar[0, demoted_signal_mask], positions_lofar[1, demoted_signal_mask], color='orange', marker='x', s=25, rasterized=True)
    ax_t_data.scatter(positions_lofar[0, no_signal_mask], positions_lofar[1, no_signal_mask], marker='.', color='gray', s=20, alpha=0.7, rasterized=True)
    setup_ax_style(ax_t_data, 'x (m)', '')
    set_ax_title(ax_t_data, 'Timing Data', use_latex)
    ax_t_data.set_xlim(plot_xlims); ax_t_data.set_ylim(plot_ylims); ax_t_data.set_aspect('equal')
    plt.setp(ax_t_data.get_yticklabels(), visible=False)
    
    # 2. Timing Truth
    ax_t_truth = fig.add_subplot(gs[1, 1], sharex=ax_f_data, sharey=ax_f_data)
    sc_tt = ax_t_truth.scatter(positions_regular[0, plot_mask_reg], positions_regular[1, plot_mask_reg],
                               c=time_truth_pw_sub_ns[plot_mask_reg], cmap='viridis', s=MARKER_SIZE, marker=MARKER_STYLE, vmin=vmin_t, vmax=vmax_t, rasterized=True)
    setup_ax_style(ax_t_truth, 'x (m)', '')
    set_ax_title(ax_t_truth, 'Timing (Truth)', use_latex)
    ax_t_truth.set_xlim(plot_xlims); ax_t_truth.set_ylim(plot_ylims); ax_t_truth.set_aspect('equal')
    plt.setp(ax_t_truth.get_yticklabels(), visible=False)
    add_cbar(sc_tt, ax_t_truth, label='Rel. Time (ns)')
    
    # 3. Timing Reco (no CF)
    ax_t_reco = fig.add_subplot(gs[2, 1], sharex=ax_f_data, sharey=ax_f_data)
    sc_tr = ax_t_reco.scatter(positions_regular[0, plot_mask_reg], positions_regular[1, plot_mask_reg],
                              c=time_reco_pw_sub_ns[plot_mask_reg], cmap='viridis', s=MARKER_SIZE, marker=MARKER_STYLE, vmin=vmin_t, vmax=vmax_t, rasterized=True)
    setup_ax_style(ax_t_reco, 'x (m)', '')
    set_ax_title(ax_t_reco, 'Posterior Mean Timing', use_latex)
    ax_t_reco.set_xlim(plot_xlims); ax_t_reco.set_ylim(plot_ylims); ax_t_reco.set_aspect('equal')
    plt.setp(ax_t_reco.get_yticklabels(), visible=False)
    add_cbar(sc_tr, ax_t_reco, label='Rel. Time (ns)')
    
    # 4. Timing Uncertainty (ns)
    ax_t_std = fig.add_subplot(gs[3, 1], sharex=ax_f_data, sharey=ax_f_data)
    time_unc_ns = post_timing_std.squeeze()[plot_mask_reg] * 1e9
    sc_ts = ax_t_std.scatter(positions_regular[0, plot_mask_reg], positions_regular[1, plot_mask_reg],
                             c=time_unc_ns, cmap='YlGn', s=MARKER_SIZE, marker=MARKER_STYLE, vmin=0, vmax=np.percentile(time_unc_ns, 95), rasterized=True)
    setup_ax_style(ax_t_std, 'x (m)', '')
    set_ax_title(ax_t_std, 'Timing Uncertainty (ns)', use_latex)
    ax_t_std.set_xlim(plot_xlims); ax_t_std.set_ylim(plot_ylims); ax_t_std.set_aspect('equal')
    plt.setp(ax_t_std.get_yticklabels(), visible=False)
    add_cbar(sc_ts, ax_t_std, label='Std (ns)')
    
    # 5. Timing Abs Diff
    ax_t_diff = fig.add_subplot(gs[4, 1], sharex=ax_f_data, sharey=ax_f_data)
    t_diff = time_reco_pw_sub_ns[plot_mask_reg] - time_truth_pw_sub_ns[plot_mask_reg]
    vlim_t_diff = np.percentile(np.abs(t_diff), 95)
    sc_t_diff = ax_t_diff.scatter(positions_regular[0, plot_mask_reg], positions_regular[1, plot_mask_reg],
                                  c=t_diff, cmap='BrBG', s=MARKER_SIZE, marker=MARKER_STYLE, vmin=-vlim_t_diff, vmax=vlim_t_diff, rasterized=True)
    setup_ax_style(ax_t_diff, 'x (m)', '')
    set_ax_title(ax_t_diff, 'Timing Diff. (ns)', use_latex)
    ax_t_diff.set_xlim(plot_xlims); ax_t_diff.set_ylim(plot_ylims); ax_t_diff.set_aspect('equal')
    plt.setp(ax_t_diff.get_yticklabels(), visible=False)
    add_cbar(sc_t_diff, ax_t_diff, label='Diff. (ns)')
    
    # =========================================================================
    # COLUMN 3: Correlated Fields
    # =========================================================================
    # 1. Fluence Syst Mean
    ax_fs_mean = fig.add_subplot(gs[0, 2], sharex=ax_f_data, sharey=ax_f_data)
    # Mask data where pulse is not identified
    fsm_plot = np.where(mask_2d, mean_syst_cf, np.nan)
    im_fsm = ax_fs_mean.imshow(fsm_plot.T, extent=imshow_extent, cmap='copper_r', origin='lower', rasterized=True)
    setup_ax_style(ax_fs_mean, 'x (m)', '')
    set_ax_title(ax_fs_mean, 'Mean Fluence CF', use_latex)
    ax_fs_mean.set_xlim(plot_xlims); ax_fs_mean.set_ylim(plot_ylims); ax_fs_mean.set_aspect('equal')
    plt.setp(ax_fs_mean.get_yticklabels(), visible=False)
    add_cbar(im_fsm, ax_fs_mean)
    
    # 2. Fluence Syst Std
    ax_fs_std = fig.add_subplot(gs[1, 2], sharex=ax_f_data, sharey=ax_f_data)
    fss_plot = np.where(mask_2d, std_syst_cf, np.nan)
    im_fss = ax_fs_std.imshow(fss_plot.T, extent=imshow_extent, cmap='pink_r', origin='lower', rasterized=True)
    setup_ax_style(ax_fs_std, 'x (m)', '')
    set_ax_title(ax_fs_std, r'$\sigma$ Fluence CF', use_latex)
    ax_fs_std.set_xlim(plot_xlims); ax_fs_std.set_ylim(plot_ylims); ax_fs_std.set_aspect('equal')
    plt.setp(ax_fs_std.get_yticklabels(), visible=False)
    add_cbar(im_fss, ax_fs_std)
    
    # 3. Timing Syst Mean
    ax_ts_mean = fig.add_subplot(gs[2, 2], sharex=ax_f_data, sharey=ax_f_data)
    tsm_plot = np.where(mask_2d, mean_timing_cf, np.nan)
    timing_cf_std_val = np.nanstd(tsm_plot)
    vlim_tcf = 2 * timing_cf_std_val if timing_cf_std_val > 0 else 1
    im_tsm = ax_ts_mean.imshow(tsm_plot.T, extent=imshow_extent, cmap='GnBu_r', origin='lower', vmin=-vlim_tcf, vmax=vlim_tcf, rasterized=True)
    setup_ax_style(ax_ts_mean, 'x (m)', '')
    set_ax_title(ax_ts_mean, 'Mean Timing CF (ns)', use_latex)
    ax_ts_mean.set_xlim(plot_xlims); ax_ts_mean.set_ylim(plot_ylims)
    plt.setp(ax_ts_mean.get_yticklabels(), visible=False)
    add_cbar(im_tsm, ax_ts_mean, label='Rel. Time (ns)')
    
    # 4. Timing Syst Std
    ax_ts_std = fig.add_subplot(gs[3, 2], sharex=ax_f_data, sharey=ax_f_data)
    tss_plot = np.where(mask_2d, std_timing_cf, np.nan)
    im_tss = ax_ts_std.imshow(tss_plot.T, extent=imshow_extent, cmap='bone', origin='lower', rasterized=True)
    setup_ax_style(ax_ts_std, 'x (m)', '')
    set_ax_title(ax_ts_std, r'$\sigma$ Timing CF (ns)', use_latex)
    ax_ts_std.set_xlim(plot_xlims); ax_ts_std.set_ylim(plot_ylims)
    plt.setp(ax_ts_std.get_yticklabels(), visible=False)
    add_cbar(im_tss, ax_ts_std, label='Std (ns)')
    
    # 5. Fluence Pull (sigma)
    ax_f_pull = fig.add_subplot(gs[4, 2], sharex=ax_f_data, sharey=ax_f_data)
    f_std_safe = np.copy(post_fluence_std).squeeze()
    f_std_safe[f_std_safe == 0] = 1.0 # Safety
    
    fluence_pull = (post_fluence_no_cf_mean[plot_mask_reg] - fluences_reg[plot_mask_reg]) / f_std_safe[plot_mask_reg]
    vlim_fp = 4.0
    
    sc_fp = ax_f_pull.scatter(positions_regular[0, plot_mask_reg], positions_regular[1, plot_mask_reg],
                              c=fluence_pull, cmap='PiYG_r', s=MARKER_SIZE, marker=MARKER_STYLE, 
                              vmin=-vlim_fp, vmax=vlim_fp, rasterized=True)
                              
    setup_ax_style(ax_f_pull, 'x (m)', '')
    set_ax_title(ax_f_pull, r'Fluence Pull ($\sigma$)', use_latex)
    ax_f_pull.set_xlim(plot_xlims); ax_f_pull.set_ylim(plot_ylims); ax_f_pull.set_aspect('equal')
    plt.setp(ax_f_pull.get_yticklabels(), visible=False)
    add_cbar(sc_fp, ax_f_pull, label=r'Pull ($\sigma$)')
    
    # =========================================================================
    # COLUMN 4: Posteriors
    # =========================================================================
    # 1. Xmax posterior
    ax_xmax = fig.add_subplot(gs[0, 3])
    ax_xmax.inset_axes([1.1, 0, 0.05, 1]).set_visible(False) # Spacer to match colorbar gap
    xmax_truth = truth_dict.get('xmax_gpcm2', None)
    reco_xmax_t_mean = np.mean(samp_xmax_timing)
    samples_flat = samp_xmax_timing.squeeze()
    reco_xmax_t_std = np.std(samp_xmax_timing)
    xmax_t_label = fr'Rec: ${reco_xmax_t_mean:.1f} \pm {reco_xmax_t_std:.1f}$'
    safe_kde(samp_xmax_timing, ax_xmax, color=PLOT_GREEN, label=xmax_t_label, truth=xmax_truth)
    if xmax_truth is not None:
        ax_xmax.plot([], [], color='black', linestyle='--', lw=1.5, label=fr'Truth: ${xmax_truth:.1f}$')
    set_ax_title(ax_xmax, r'$X_{\rm max}$ Posterior', use_latex)
    ax_xmax.set_xlabel(r'$X_{\rm max}$ (g/cm$^2$)')
    ax_xmax.set_yticks([])
    
    # Expand limits to ensure legend doesn't overlap
    current_ymin, current_ymax = ax_xmax.get_ylim()
    current_xmin, current_xmax = ax_xmax.get_xlim()
    ax_xmax.set_ylim(0, current_ymax * 1.6) # 1.6x height
    ax_xmax.set_xlim(current_xmin, current_xmax + (current_xmax - current_xmin) * 0.15) # +15% width right
    
    # Legend with frameless white background
    ax_xmax.legend(loc='upper right', fontsize=22, frameon=True, facecolor="white", framealpha=1.0, edgecolor="none")
    ax_xmax.set_box_aspect(1)
    
    # 2. Erad posterior
    ax_erad = fig.add_subplot(gs[1, 3])
    ax_erad.inset_axes([1.1, 0, 0.05, 1]).set_visible(False) # Spacer to match colorbar gap
    erad_truth = truth_dict.get('erad_ev', None)

    # Check if CF is actually doing anything. If the CF grid is basically 0 everywhere, 
    # then CF is disabled and we should use raw Erad to avoid float32 precision noise from the correction factor calculation
    # which can create fake width/spikes in the posterior.
    if np.max(np.abs(mean_syst_cf)) < 1e-6:
        valid_mask = ~np.isnan(raw_samp_erad) & ~np.isinf(raw_samp_erad)
        samp_erad_v = raw_samp_erad[valid_mask]
        plot_erad = raw_samp_erad
    else:
        valid_mask = ~np.isnan(samp_erad) & ~np.isinf(samp_erad)
        samp_erad_v = samp_erad[valid_mask]
        plot_erad = samp_erad
    
    # Ensure the label matches the exact distribution being plotted
    reco_erad_mean = np.mean(samp_erad_v)
    reco_erad_std = np.std(samp_erad_v)
    def to_sci(val): return fr'{val:.1e}'.replace('e+0', 'e').replace('e+', 'e')
    erad_label = fr'Rec: ${to_sci(reco_erad_mean.item())} \pm {to_sci(reco_erad_std.item())}$'
    safe_kde(plot_erad, ax_erad, color=PLOT_GREEN, label=erad_label, truth=erad_truth)
    if erad_truth is not None:
        ax_erad.plot([], [], color='black', linestyle='--', lw=1.5, label=fr'Truth: ${to_sci(erad_truth)}$')
    set_ax_title(ax_erad, r'$E_{\rm rad}$ Posterior', use_latex)
    ax_erad.set_xlabel(r'$E_{\rm rad}$ (eV)')
    ax_erad.set_yticks([])

    # Force scientific notation on x-axis
    ax_erad.ticklabel_format(style='sci', axis='x', scilimits=(0,0))
    
    # Expand limits to ensure legend doesn't overlap
    current_ymin, current_ymax = ax_erad.get_ylim()
    current_xmin, current_xmax = ax_erad.get_xlim()
    ax_erad.set_ylim(0, current_ymax * 1.6) # 1.6x height
    ax_erad.set_xlim(current_xmin, current_xmax + (current_xmax - current_xmin) * 0.15) # +15% width right
    
    # Legend with frameless white background
    ax_erad.legend(loc='upper right', fontsize=22, frameon=True, facecolor="white", framealpha=1.0, edgecolor="none")
    ax_erad.set_box_aspect(1)
    
    # 3. Core position 2D
    ax_core = fig.add_subplot(gs[2, 3])
    ax_core.inset_axes([1.1, 0, 0.05, 1]).set_visible(False) # Spacer to match colorbar gap
    core_truth = (truth_dict.get('core_x_m', None), truth_dict.get('core_y_m', None))
    plot_2d_contour(samp_core_x, samp_core_y, ax_core, truth_point=(core_truth[0], core_truth[1]),
                    x_label='x (m)', y_label='y (m)', title='Core Position', use_latex=use_latex)
    ax_core.set_box_aspect(1)
    
    # 4. Direction 2D
    ax_dir = fig.add_subplot(gs[3, 3])
    ax_dir.inset_axes([1.1, 0, 0.05, 1]).set_visible(False) # Spacer to match colorbar gap
    az_truth = np.rad2deg(truth_dict.get('azimuth_rad', 0))
    zen_truth = np.rad2deg(truth_dict.get('zenith_rad', 0))
    plot_2d_contour(samp_az, samp_zen, ax_dir, truth_point=(az_truth, zen_truth),
                    x_label='Azimuth (°)', y_label='Zenith (°)', title='Arrival Direction', use_latex=use_latex)
    ax_dir.set_box_aspect(1)
    
    # 5. Timing Pull (sigma)
    ax_t_pull = fig.add_subplot(gs[4, 3], sharex=ax_f_data, sharey=ax_f_data)
    
    t_std_safe_ns = post_timing_std.squeeze()[plot_mask_reg] * 1e9
    t_std_safe_ns[t_std_safe_ns == 0] = 1.0 # Safety
    
    timing_pull = (time_reco_pw_sub_ns[plot_mask_reg] - time_truth_pw_sub_ns[plot_mask_reg]) / t_std_safe_ns
    vlim_tp = 3.0
    
    sc_tp = ax_t_pull.scatter(positions_regular[0, plot_mask_reg], positions_regular[1, plot_mask_reg],
                              c=timing_pull, cmap='PRGn_r', s=MARKER_SIZE, marker=MARKER_STYLE, 
                              vmin=-vlim_tp, vmax=vlim_tp, rasterized=True)
                              
    setup_ax_style(ax_t_pull, 'x (m)', '')
    set_ax_title(ax_t_pull, r'Timing Pull ($\sigma$)', use_latex)
    ax_t_pull.set_xlim(plot_xlims); ax_t_pull.set_ylim(plot_ylims); ax_t_pull.set_aspect('equal')
    plt.setp(ax_t_pull.get_yticklabels(), visible=False)
    add_cbar(sc_tp, ax_t_pull, label=r'Pull ($\sigma$)')

    # --- Save ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    # --- Export data for re-plotting ---
    if export_data:
        # Save JSON alongside the plot with same base name
        base_path = os.path.splitext(output_path)[0]
        json_path = base_path + '_data.json'
        export_plot_data(plot_data, json_path)
    
    return output_path