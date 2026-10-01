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
warnings.filterwarnings('ignore')

EMISSION_LINES = {
    'Lyα': 1215.67,
    'N V': 1240.81,
    'C IV': 1549.06,
    'C III]': 1908.73,
    'Mg II': 2798.75,
    '[O II]': 3728.48,
    'Hβ': 4861.33,
    '[O III]': 5006.84,
    'Hα': 6562.81
}

def parse_args():
    parser = argparse.ArgumentParser(description="Generate spectrum plots for jump candidates.")
    parser.add_argument("--cands", type=str, default="post_clustering.csv")
    parser.add_argument("--cat", type=str, default="post_chi2_calc.csv")
    parser.add_argument("--recon-dir", type=str, default="reconstructions")
    parser.add_argument("--out-dir", type=str, default="plots")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--rebin", type=int, default=15, help="Number of pixels to average for the coarse rebinning")
    return parser.parse_args()

def rebin_spectrum_raw(wave, flux, factor):
    """binning of raw flux array."""
    n = len(flux)
    n_keep = n - (n % factor)
    
    if n_keep == 0:
        return wave, flux
        
    wave_binned = np.mean(wave[:n_keep].reshape(-1, factor), axis=1)
    flux_binned = np.mean(flux[:n_keep].reshape(-1, factor), axis=1)
    
    return wave_binned, flux_binned
    
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    
    df_cands = pd.read_csv(args.cands)
    df_cat = pd.read_csv(args.cat)
    targets = df_cands['TARGETID'].unique()
    
    if args.limit > 0:
        targets = targets[:args.limit]

    for idx, tid in enumerate(targets):
        target_data = df_cat[df_cat['TARGETID'] == tid]
        if target_data.empty:
            continue
            
        z = target_data['Z'].iloc[0]
        nights = sorted(target_data['LASTNIGHT'].unique())
        
        target_dir = os.path.join(args.out_dir, str(tid))
        os.makedirs(target_dir, exist_ok=True)
        
        night_cache = {}
        for night in nights:
            obs_row = target_data[target_data['LASTNIGHT'] == night].iloc[0]
            cand_row = df_cands[(df_cands['TARGETID'] == tid) & (df_cands['LASTNIGHT'] == night)]
            
            if not cand_row.empty and 'CLUSTER' in cand_row.columns and not pd.isna(cand_row['CLUSTER'].iloc[0]):
                cluster_id = f"Cls {int(cand_row['CLUSTER'].iloc[0])}"
            else:
                cluster_id = "Cls ?"

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
                        
                        # Generate the coarse rebinned array 
                        wave_coarse, coarse = rebin_spectrum_raw(wave_obs, orig, factor=args.rebin)
                        
                        night_cache[night] = {
                            'wave_obs': wave_obs,
                            'orig': orig,
                            'orig_masked': orig_masked,
                            'recon': recon,
                            'wave_coarse': wave_coarse,
                            'coarse': coarse,
                            'weights': weights,
                            'label': f"{night} ({cluster_id}) | $\\chi^2_\\nu$={chi2_str}, SNR={snr_str}"
                        }
                except Exception as e:
                    print(f"Warning: Failed reading {tid} night {night}: {e}")
        
        if not night_cache:
            continue

        combos = list(itertools.combinations(night_cache.keys(), 2))
        if len(night_cache.keys()) > 2:
            combos.append(tuple(night_cache.keys()))
        elif len(night_cache.keys()) == 1:
            combos = [tuple(night_cache.keys())] 

        cmap = plt.colormaps.get_cmap('tab10')

        for combo in combos:
            fig, axes = plt.subplots(4, 1, figsize=(14, 19), sharex=True)
            
            if len(combo) == len(night_cache.keys()) and len(combo) > 2:
                combo_name = "all_obs"
            else:
                combo_name = "_".join([str(n).replace('-','').split('T')[0] for n in combo])
            
            combo_nights_count = len(combo)
            raw_bg_alpha = max(0.02, 0.08 / combo_nights_count)
            raw_mask_alpha = max(0.08, 0.25 / combo_nights_count)
            
            all_orig_fluxes = []
            all_recon_fluxes = []
            all_coarse_fluxes = []

            for i, night in enumerate(combo):
                nd = night_cache[night]
                color = cmap(i % 10)
                
                wave_obs = nd['wave_obs']
                orig = nd['orig']
                orig_masked = nd['orig_masked']
                recon = nd['recon']
                
                wave_coarse = nd['wave_coarse']
                coarse = nd['coarse']
                weights = nd['weights']
                lbl = nd['label']
                
                # Only plot the raw flux backgrounds on the top 3 plots
                for ax in axes[:3]:
                    ax.plot(wave_obs, orig, color=color, alpha=raw_bg_alpha, linewidth=0.5, zorder=1)
                    ax.plot(wave_obs, orig_masked, color=color, alpha=raw_mask_alpha, linewidth=0.9, zorder=2)
                
                axes[0].plot(wave_obs, recon, color=color, alpha=0.82, linewidth=2.0, zorder=10, label=f"Recon: {lbl}")
                axes[1].plot(wave_coarse, coarse, color=color, linestyle='-', alpha=0.9, linewidth=1.8, zorder=10, label=f"Coarse: {lbl}")
                
                axes[2].plot(wave_obs, recon, color=color, alpha=0.6, linewidth=1.5, zorder=10, label=f"Recon: {lbl}")
                axes[2].plot(wave_coarse, coarse, color=color, linestyle='--', alpha=0.9, linewidth=1.5, zorder=11, label=f"Coarse: {lbl}")

                # 4th subplot for weights
                axes[3].plot(wave_obs, weights, color=color, alpha=0.8, linewidth=1.2, zorder=10, label=f"Weights: {lbl}")

                plot_region = (wave_obs >= 3500) & (wave_obs <= 10000)
                if np.any(plot_region):
                    all_recon_fluxes.append(recon[plot_region])
                    all_orig_fluxes.append(orig_masked[plot_region & ~np.isnan(orig_masked)])
                
                coarse_plot_region = (wave_coarse >= 3500) & (wave_coarse <= 10000)
                if np.any(coarse_plot_region):
                    all_coarse_fluxes.append(coarse[coarse_plot_region & ~np.isnan(coarse)])

            for ax_idx, ax in enumerate(axes):
                ax.set_xlim(3500, 10000)
                
                # Setup Y limits depending on if it's a flux plot or a weight plot
                if ax_idx < 3:
                    if len(all_orig_fluxes) > 0 and len(all_recon_fluxes) > 0 and len(all_coarse_fluxes) > 0:
                        concat_orig = np.concatenate(all_orig_fluxes)
                        concat_recon = np.concatenate(all_recon_fluxes)
                        concat_coarse = np.concatenate(all_coarse_fluxes)
                        
                        if len(concat_orig) > 0:
                            p1 = np.percentile(concat_orig, 1.5)
                            peak_val = max(np.nanmax(concat_recon), np.nanmax(concat_coarse), np.percentile(concat_orig, 99.5))
                            yrange = peak_val - p1
                            ymin, ymax = p1 - 0.15 * yrange, peak_val + 0.25 * yrange
                            ax.set_ylim(ymin, ymax)
                    ax.set_ylabel("Unnormalized Flux", fontsize=11)
                else:
                    ax.set_ylim(bottom=0)
                    ax.set_ylabel("Weight (Inv. Var)", fontsize=11)

                for name, w_rest in EMISSION_LINES.items():
                    w_obs = w_rest * (1 + z)
                    if 3500 <= w_obs <= 10000:
                        ax.axvline(w_obs, color='gray', linestyle='--', alpha=0.4, zorder=0)
                        ax.text(w_obs + 40, ax.get_ylim()[1] - (ax.get_ylim()[1]-ax.get_ylim()[0])*0.04, name, 
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
            plt.savefig(os.path.join(target_dir, f"{tid}_obs_{combo_name}.png"), dpi=150)
            plt.close(fig)

if __name__ == "__main__":
    main()