import os
import argparse
import itertools
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from astropy.io import fits
import warnings
from multiprocessing import Pool, cpu_count
from functools import partial

warnings.filterwarnings('ignore')

EMISSION_LINES = {
    'Lyα': 1215.67, 'N V': 1240.81, 'C IV': 1549.06, 'C III]': 1908.73,
    'Mg II': 2798.75, '[O II]': 3728.48, 'Hβ': 4861.33, '[O III]': 5006.84, 'Hα': 6562.81
}

def parse_args():
    parser = argparse.ArgumentParser(description="Generate spectrum plots for jump candidates.")
    parser.add_argument("--cands", type=str, nargs='+', default=[
        "post_clustering.csv", "latent_targets_with_nonzero_p95_counts.csv", 
        "lya_all_results.csv", "ciii_all_results.csv", "civ_all_results.csv", 
        "nv_all_results.csv", "mgii_all_results.csv"
    ])
    parser.add_argument("--cat", type=str, default="post_chi2_calc.csv")
    parser.add_argument("--recon-dir", type=str, default="reconstructions")
    parser.add_argument("--out-dir", type=str, default="plots")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--rebin", type=int, default=15)
    parser.add_argument("--cores", type=int, default=0, help="Num cores to use (0 = auto)")
    return parser.parse_args()

def rebin_spectrum_raw(wave, flux, factor):
    n = len(flux)
    n_keep = n - (n % factor)
    if n_keep == 0: return wave, flux
    wave_binned = np.mean(wave[:n_keep].reshape(-1, factor), axis=1)
    flux_binned = np.mean(flux[:n_keep].reshape(-1, factor), axis=1)
    return wave_binned, flux_binned

def process_target(tid, args, df_cat, df_cluster):
    target_dir = os.path.join(args.out_dir, str(tid))
    
    # Resumeability check
    if os.path.exists(target_dir) and len(os.listdir(target_dir)) > 0:
        return f"Skipped {tid} (exists)"
        
    target_data = df_cat[df_cat['TARGETID'] == tid]
    if target_data.empty:
        return f"Skipped {tid} (no cat data)"
        
    os.makedirs(target_dir, exist_ok=True)
        
    z = target_data['Z'].iloc[0]
    nights = sorted(target_data['LASTNIGHT'].unique())
    night_cache = {}
    
    for night in nights:
        obs_row = target_data[target_data['LASTNIGHT'] == night].iloc[0]
        cluster_id_str = ""
        
        if df_cluster is not None:
            cand_row = df_cluster[(df_cluster['TARGETID'] == tid) & (df_cluster['LASTNIGHT'] == night)]
            if not cand_row.empty and 'CLUSTER' in cand_row.columns and not pd.isna(cand_row['CLUSTER'].iloc[0]):
                cluster_id_str = f" (Cls {int(cand_row['CLUSTER'].iloc[0])})"

        chi2 = obs_row.get('REDUCED_CHI2', np.nan)
        snr = obs_row.get('SNR', obs_row.get('MEDIAN_SNR', np.nan))
        chi2_str = f"{chi2:.2f}" if not pd.isna(chi2) else "N/A"
        snr_str = f"{snr:.1f}" if not pd.isna(snr) else "N/A"
        
        night_clean = str(night).replace('-', '').split('T')[0]
        recon_fits = os.path.join(args.recon_dir, str(tid), night_clean, f"recon_{tid}_{night_clean}.fits")
        
        if os.path.exists(recon_fits):
            try:
                with fits.open(recon_fits) as hdul:
                    header = hdul[1].header
                    data = hdul[1].data
                    
                    norm = header.get('NORM', 1.0)
                    z_val = header.get('Z', z) 
                    wave_obs = data['REST_WAVE'] * (1 + z_val) 
                    
                    orig = data['ORIG_FLUX'] * norm
                    recon = data['RECON_FLUX'] * norm
                    weights = data['UPDATED_WEIGHTS']
                    valid_mask = weights > 0
                    
                    orig_masked = orig.copy()
                    orig_masked[~valid_mask] = np.nan
                    
                    wave_coarse, coarse = rebin_spectrum_raw(wave_obs, orig, factor=args.rebin)
                    
                    night_cache[night] = {
                        'wave_obs': wave_obs, 'orig': orig, 'orig_masked': orig_masked,
                        'recon': recon, 'wave_coarse': wave_coarse, 'coarse': coarse,
                        'weights': weights, 
                        'label': f"{night}{cluster_id_str} | $\\chi^2_\\nu$={chi2_str}, SNR={snr_str}"
                    }
            except Exception:
                pass
    
    if not night_cache:
        if len(os.listdir(target_dir)) == 0:
            os.rmdir(target_dir)
        return f"Failed {tid} (no FITS loaded)"

    combos = list(itertools.combinations(night_cache.keys(), 2))
    if len(night_cache.keys()) > 2:
        combos.append(tuple(night_cache.keys()))
    elif len(night_cache.keys()) == 1:
        combos = [tuple(night_cache.keys())] 

    cmap = plt.colormaps.get_cmap('tab10')

    for combo in combos:
        fig, axes = plt.subplots(4, 1, figsize=(14, 19), sharex=True)
        combo_name = "all_obs" if (len(combo) == len(night_cache.keys()) and len(combo) > 2) else "_".join([str(n).replace('-','').split('T')[0] for n in combo])
        
        combo_nights_count = len(combo)
        raw_bg_alpha = max(0.02, 0.08 / combo_nights_count)
        raw_mask_alpha = max(0.08, 0.25 / combo_nights_count)
        
        all_orig = []
        all_recon = []
        all_coarse = []

        for i, night in enumerate(combo):
            nd = night_cache[night]
            color = cmap(i % 10)
            
            w_obs, orig, orig_masked = nd['wave_obs'], nd['orig'], nd['orig_masked']
            recon, w_coarse, coarse = nd['recon'], nd['wave_coarse'], nd['coarse']
            weights, lbl = nd['weights'], nd['label']
            
            for ax in axes[:3]:
                ax.plot(w_obs, orig, color=color, alpha=raw_bg_alpha, linewidth=0.5, zorder=1)
                ax.plot(w_obs, orig_masked, color=color, alpha=raw_mask_alpha, linewidth=0.9, zorder=2)
            
            axes[0].plot(w_obs, recon, color=color, alpha=0.82, linewidth=2.0, zorder=10, label=f"Recon: {lbl}")
            axes[1].plot(w_coarse, coarse, color=color, linestyle='-', alpha=0.9, linewidth=1.8, zorder=10, label=f"Coarse: {lbl}")
            axes[2].plot(w_obs, recon, color=color, alpha=0.6, linewidth=1.5, zorder=10, label=f"Recon: {lbl}")
            axes[2].plot(w_coarse, coarse, color=color, linestyle='--', alpha=0.9, linewidth=1.5, zorder=11, label=f"Coarse: {lbl}")
            axes[3].plot(w_obs, weights, color=color, alpha=0.8, linewidth=1.2, zorder=10, label=f"Weights: {lbl}")

            plot_region = (w_obs >= 3500) & (w_obs <= 10000)
            if np.any(plot_region):
                all_recon.append(recon[plot_region])
                all_orig.append(orig_masked[plot_region & ~np.isnan(orig_masked)])
            
            coarse_plot_region = (w_coarse >= 3500) & (w_coarse <= 10000)
            if np.any(coarse_plot_region):
                all_coarse.append(coarse[coarse_plot_region & ~np.isnan(coarse)])

        for ax_idx, ax in enumerate(axes):
            ax.set_xlim(3500, 10000)
            if ax_idx < 3:
                if all_orig and all_recon and all_coarse:
                    c_orig = np.concatenate(all_orig)
                    if len(c_orig) > 0:
                        p1 = np.percentile(c_orig, 1.5)
                        peak_val = max(np.nanmax(np.concatenate(all_recon)), np.nanmax(np.concatenate(all_coarse)), np.percentile(c_orig, 99.5))
                        yrange = peak_val - p1
                        ax.set_ylim(p1 - 0.15 * yrange, peak_val + 0.25 * yrange)
                ax.set_ylabel("Unnormalized Flux", fontsize=11)
            else:
                ax.set_ylim(bottom=0)
                ax.set_ylabel("Weight (Inv. Var)", fontsize=11)

            for name, w_rest in EMISSION_LINES.items():
                w_obs_line = w_rest * (1 + z)
                if 3500 <= w_obs_line <= 10000:
                    ax.axvline(w_obs_line, color='gray', linestyle='--', alpha=0.4, zorder=0)
                    ax.text(w_obs_line + 40, ax.get_ylim()[1] - (ax.get_ylim()[1]-ax.get_ylim()[0])*0.04, name, 
                            color='black', rotation=90, va='top', ha='left', alpha=0.7, fontsize=9, fontweight='bold')

            handles, labels_list = ax.get_legend_handles_labels()
            if labels_list: 
                by_label = dict(zip(labels_list, handles))
                ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=8, 
                          facecolor='white', framealpha=0.9, edgecolor='gray').set_zorder(20)

        axes[0].set_title(f"Reconstruction over Raw (Target: {tid} | z={z:.4f})", fontsize=12, fontweight='bold')
        axes[1].set_title(f"Coarse Rebinned over Raw (Target: {tid} | z={z:.4f})", fontsize=12, fontweight='bold')
        axes[2].set_title(f"Both Recon and Coarse over Raw (Target: {tid} | z={z:.4f})", fontsize=12, fontweight='bold')
        axes[3].set_title(f"Weights / Inverse Variance (Target: {tid} | z={z:.4f})", fontsize=12, fontweight='bold')
        axes[3].set_xlabel("Observed Wavelength [Å]", fontsize=11)

        plt.tight_layout()
        plt.savefig(os.path.join(target_dir, f"{tid}_obs_{combo_name}.png"), dpi=100) 
        plt.close('all') 
    return f"Completed {tid}"

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    df_cat = pd.read_csv(args.cat)
    all_targets = set()
    df_cluster = None
    
    for cand_file in args.cands:
        if os.path.exists(cand_file):
            df = pd.read_csv(cand_file)
            if 'TARGETID' in df.columns:
                all_targets.update(df['TARGETID'].unique())
            if 'post_clustering.csv' in cand_file:
                df_cluster = df
                
    targets = sorted(list(all_targets))
    if args.limit > 0:
        targets = targets[:args.limit]
        
    print(f"Total unique targets: {len(targets)}")

    # Set up Multiprocessing
    num_cores = args.cores if args.cores > 0 else max(1, cpu_count() - 2)
    print(f"Starting multiprocessing pool with {num_cores} cores.")
    
    process_func = partial(process_target, args=args, df_cat=df_cat, df_cluster=df_cluster)
    
    with Pool(num_cores) as pool:
        for i, result in enumerate(pool.imap_unordered(process_func, targets)):
            if (i + 1) % 50 == 0:
                print(f"Progress: {i + 1}/{len(targets)} targets processed.")

    print("All targets processed successfully.")

if __name__ == "__main__":
    main()