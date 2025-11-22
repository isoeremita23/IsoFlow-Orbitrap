import os
import re
import numpy as np
import pandas as pd
from pymsfilereader import MSFileReader
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FixedLocator, FixedFormatter, LogFormatter

# =============================================================================
# === USER CONFIGURATION SECTION ==============================================
# =============================================================================

# --- 1. PATHS & SETTINGS ---
FOLDER_PATH = r"H:\My Drive\Orbitrap\Test_19\test"
MAX_WORKERS = 4  # Number of CPU cores for parallel processing
TOLERANCE = 0.001 # m/z tolerance for peak matching

# --- 2. ANALYSIS PARAMETERS ---
# Define the specific m/z values to track.
# Format: [Base_Peak_Mass, Isotopologue_1, Isotopologue_2, ...]
TARGET_MZS = np.array([61.9884, 62.9854, 62.99258, 63.99261])

# Mapping m/z values to human-readable names
MZ_TO_ISOTOPOLOG = {
    61.9884: 'M0',
    62.9854: '15N',  # Note: Ensure these match TARGET_MZS exactly or within tolerance
    62.99258: '17O',
    63.99261: '18O'
}

# Time window (in minutes) for averaging ratios
RT_START = 1.1
RT_END = 9.1

# --- 3. STANDARD BRACKETING SETUP ---
REFERENCE_NAME = 'USGS34'  # The name string found in the filename for the Reference
SAMPLE_NAME_TAG = 'USGS32' # The name string found in the filename for the Sample

# --- 4. ANALYSIS MODE ---
# Options: 'VALIDATION' (We know the sample deltas) or 'UNKNOWN' (Real sample analysis)
ANALYSIS_MODE = 'UNKNOWN' 

# --- 5. REFERENCE & SAMPLE DATA (Certified Values) ---
# Ratios of the standard relative to which deltas are calculated (e.g., Air-N2, VSMOW)
KNOWN_RATIOS = pd.DataFrame({
    'isotopolog': ['15N', '17O', '18O'],
    'reference': ['Air-N2', 'VSMOW', 'VSMOW'],
    'ratio_known': [0.003676, 0.001140, 0.006016]
})

# Certified Delta values for the Reference Standard (Bracketing Standard)
DELTA_REFERENCE = pd.DataFrame({
    'isotopolog': ['15N', '17O', '18O'],
    'delta_known': [-1.8, -14.8, -27.78] # e.g., per mille vs standards above
})

# Certified Delta values for the Sample (ONLY USED IF ANALYSIS_MODE == 'VALIDATION')
# If Mode is 'UNKNOWN', these values are ignored.
DELTA_SAMPLE_KNOWN = pd.DataFrame({
    'isotopolog': ['15N', '17O', '18O'],
    'delta_known': [180.0, 13.2, 25.7]
})

# =============================================================================
# === CORE PROCESSING LOGIC ===================================================
# =============================================================================

def process_raw_file(raw_path):
    """
    Opens a single .raw file, extracts scan headers and specific m/z intensities.
    Returns a pandas DataFrame containing scan-by-scan data.
    """
    raw_file = os.path.basename(raw_path)
    sample_name = raw_file.split('_')[0]
    
    # Extract injection number from filename (expects format Name_InjNumber.raw)
    match = re.search(r'(\d+)\.raw$', raw_file, re.IGNORECASE)
    injection = int(match.group(1)) if match else 999

    try:
        raw = MSFileReader(raw_path)
    except Exception as e:
        print(f"Error reading {raw_file}: {e}")
        return pd.DataFrame()

    num_scans = raw.NumSpectra
    n_targets = len(TARGET_MZS)
    n_total = num_scans * n_targets

    # Pre-allocate arrays for performance
    data_dict = {k: np.zeros(n_total, dtype=float) for k in [
        "rt_min", "mz", "intensity", "noise", "resolution_peak", "baseline",
        "TIC", "BasePeakMass", "BasePeakIntensity", "LowMass", "HighMass",
        "AGC_Target", "MicroScanCount", "InjectionTime_ms", "FT_Resolution"
    ]}
    
    # Fill constant/repetitive columns
    data_dict["Sample_name"] = np.full(n_total, sample_name, dtype=object)
    data_dict["Injection"] = np.full(n_total, injection, dtype=int)
    data_dict["scan"] = np.repeat(np.arange(1, num_scans + 1), n_targets)
    data_dict["mz"] = np.tile(TARGET_MZS, num_scans)

    last_trailer = None
    last_scan_num = None

    # Iterate through scans
    pbar = tqdm(total=num_scans, desc=f"{raw_file}", leave=False, dynamic_ncols=True)
    
    for scan_num in range(1, num_scans + 1):
        idx_start = (scan_num - 1) * n_targets
        idx_end = idx_start + n_targets

        # 1. Get Scan Header Info
        hdr = raw.GetScanHeaderInfoForScanNum(scan_num)
        
        # 2. Get Trailer Extra (Optimized to avoid re-fetching if sequential)
        if last_trailer and scan_num == last_scan_num + 1:
            trailer = last_trailer # Sometimes trailer doesn't change, optimization logic if applicable
        else:
            trailer = raw.GetTrailerExtraForScanNum(scan_num)
        
        # (Force refresh of trailer for safety in this implementation)
        trailer = raw.GetTrailerExtraForScanNum(scan_num)
        last_trailer = trailer
        last_scan_num = scan_num

        # 3. Get Label Data (Centroided peaks)
        label_data = raw.GetLabelData(scan_num)
        
        if label_data is not None:
            labels, _ = label_data
            mz_arr = np.array(labels.mass)
            int_arr = np.array(labels.intensity)
            noise_arr = np.array(labels.noise)
            res_arr = np.array(labels.resolution)
            base_arr = np.array(labels.baseline)

            # Match extracted peaks to TARGET_MZS within TOLERANCE
            diff = np.abs(mz_arr[:, None] - TARGET_MZS)
            for j in range(n_targets):
                # Find indices where difference is within tolerance
                idx_match = np.where(diff[:, j] <= TOLERANCE)[0]
                if idx_match.size > 0:
                    i = idx_match[0] # Take the first match
                    data_dict["intensity"][idx_start + j] = int_arr[i]
                    data_dict["noise"][idx_start + j] = noise_arr[i]
                    data_dict["resolution_peak"][idx_start + j] = res_arr[i]
                    data_dict["baseline"][idx_start + j] = base_arr[i]

        # 4. Fill Header/Trailer Data
        # Note: Using .get with defaults to handle missing keys safely
        data_dict["rt_min"][idx_start:idx_end] = hdr.get('StartTime', np.nan)
        data_dict["TIC"][idx_start:idx_end] = hdr.get('TIC', np.nan)
        data_dict["BasePeakMass"][idx_start:idx_end] = hdr.get('BasePeakMass', np.nan)
        data_dict["BasePeakIntensity"][idx_start:idx_end] = hdr.get('BasePeakIntensity', np.nan)
        data_dict["LowMass"][idx_start:idx_end] = hdr.get('LowMass', np.nan)
        data_dict["HighMass"][idx_start:idx_end] = hdr.get('HighMass', np.nan)
        
        data_dict["AGC_Target"][idx_start:idx_end] = trailer.get('AGC Target', np.nan)
        data_dict["MicroScanCount"][idx_start:idx_end] = trailer.get('Micro Scan Count', np.nan)
        data_dict["InjectionTime_ms"][idx_start:idx_end] = trailer.get('Ion Injection Time (ms)', np.nan)
        data_dict["FT_Resolution"][idx_start:idx_end] = trailer.get('FT Resolution', np.nan)

        if scan_num % 100 == 0:
            pbar.update(100)
            
    pbar.update(num_scans % 100)
    pbar.close()
    
    return pd.DataFrame(data_dict)

def compute_deltas(ratios_summary, suffix):
    """
    Calculates delta values using the Reference-Sample-Reference bracketing method.
    Interpolates the reference ratio between the injection before and after the sample.
    """
    # 1. Identify Reference Injections
    ref_before = ratios_summary[ratios_summary['is_ref']].rename(
        columns={'Injection':'ref_before', 'ratio':'ref_before_ratio', 'ratio_sem':'ref_before_ratio_sem'}
    ).drop(columns='is_ref')
    
    # 2. Merge Sample with Reference (Before)
    deltas = ratios_summary[~ratios_summary['is_ref']].merge(ref_before, on=['isotopolog','basepeak'], how='left')
    deltas = deltas[deltas['Injection'] == deltas['ref_before'] + 1] # Strict sequence: Ref -> Sample

    # 3. Merge Sample with Reference (After)
    ref_after = ratios_summary[ratios_summary['is_ref']].rename(
        columns={'Injection':'ref_after', 'ratio':'ref_after_ratio', 'ratio_sem':'ref_after_ratio_sem'}
    ).drop(columns='is_ref')
    
    deltas = deltas.merge(ref_after, on=['isotopolog','basepeak'], how='left')
    deltas = deltas[deltas['Injection'] == deltas['ref_after'] - 1] # Strict sequence: Sample -> Ref

    # 4. Perform Delta Calculation
    deltas['sample_name'] = SAMPLE_NAME_TAG
    delta_ref = DELTA_REFERENCE.rename(columns={'delta_known':'delta_ref'})
    deltas = deltas.merge(delta_ref, on='isotopolog', how='left')

    # Average the bracketing references
    deltas['ref_ratio'] = 0.5 * (deltas['ref_before_ratio'] + deltas['ref_after_ratio'])
    deltas['ref_ratio_sem'] = 0.5 * np.sqrt(deltas['ref_before_ratio_sem']**2 + deltas['ref_after_ratio_sem']**2)

    # Standard Delta Equation: delta = (R_sam / R_ref) * (delta_ref + 1000) - 1000
    deltas['delta'] = deltas['ratio'] / deltas['ref_ratio'] * (deltas['delta_ref'] + 1000) - 1000
    
    # Error Propagation
    deltas['delta_sem'] = (deltas['delta'] + 1000) * np.sqrt(
        (deltas['ratio_sem'] / deltas['ratio'])**2 + (deltas['ref_ratio_sem'] / deltas['ref_ratio'])**2
    )

    # Merge with global known ratios (Absolute ratios)
    deltas = deltas.merge(KNOWN_RATIOS, on='isotopolog', how='left')
    deltas['ratio_corr'] = (deltas['delta'] / 1000 + 1) * deltas['ratio_known']
    deltas['inclusion_mode'] = suffix

    # 5. Summarize Results
    deltas_summary = deltas.groupby(['sample_name','isotopolog']).agg(
        n=('ratio','count'),
        ratio_raw_mean=('ratio','mean'),
        ratio_raw_sdev=('ratio','std'),
        ratio_corrected_mean=('ratio_corr','mean'),
        ratio_corrected_sdev=('ratio_corr','std'),
        delta_mean=('delta','mean'),
        delta_sdev=('delta','std')
    ).reset_index()

    # 6. Handle "Validation" vs "Unknown" logic
    if ANALYSIS_MODE == 'VALIDATION':
        deltas_summary = deltas_summary.merge(
            DELTA_SAMPLE_KNOWN.rename(columns={'delta_known':'delta_expected'}), 
            on='isotopolog', how='left'
        )
        deltas_summary = deltas_summary.merge(
            KNOWN_RATIOS.rename(columns={'ratio_known':'reference_ratio'}), 
            on='isotopolog', how='left'
        )
        deltas_summary['ratio_expected'] = (deltas_summary['delta_expected'] / 1000 + 1) * deltas_summary['reference_ratio']
        deltas_summary['err_%'] = (deltas_summary['delta_mean'] - deltas_summary['delta_expected']) / deltas_summary['delta_expected'] * 100
        deltas_summary['cv_%'] = deltas_summary['delta_sdev'] / deltas_summary['delta_mean'] * 100
        deltas_summary['diff_permil'] = deltas_summary['delta_mean'] - deltas_summary['delta_expected']
    else:
        # For unknown samples, we cannot calculate error against expected
        deltas_summary['delta_expected'] = np.nan
        deltas_summary['err_%'] = np.nan
        deltas_summary['diff_permil'] = np.nan
        deltas_summary['cv_%'] = deltas_summary['delta_sdev'] / deltas_summary['delta_mean'] * 100

    deltas_summary['inclusion_mode'] = suffix
    return deltas, deltas_summary

def generate_isotopologue_plots(df, output_dir):
    """Generates Time vs Ratio plots for visual inspection."""
    df_plot = df[(df['basepeak_ions'] > 0) & (df['isotopolog'].isin(['15N', '17O', '18O']))]
    
    for (sample, inj), subdf in df_plot.groupby(['Sample_name', 'Injection']):
        fig, ax = plt.subplots(figsize=(10, 6))
        for iso in ['15N', '17O', '18O']:
            sub_iso = subdf[subdf['isotopolog'] == iso]
            if not sub_iso.empty:
                ax.plot(sub_iso['rt_min'], sub_iso['ratio'], marker='.', linestyle='-', linewidth=0.8, label=iso, rasterized=True)
        
        ax.set_title(f"{sample} - Injection {inj}\nIsotopologue ratios over full run")
        ax.set_xlabel("Retention time (min)")
        ax.set_ylabel("Ratio (ions_incremental / M0)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{sample}_inj{inj}_isotopolog_ratios.png"), dpi=300)
        plt.close(fig)

def generate_shot_noise_plots(df, output_dir):
    """
    Generates log-log plots comparing experimental SE vs theoretical Shot Noise limit.
    """
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    
    # Calculate cumulative stats
    target_isos = ["15N", "17O", "18O"]
    for (sample, inj, iso), g in df[df["isotopolog"].isin(target_isos)].groupby(["Sample_name", "Injection", "isotopolog"], sort=False):
        g = g.sort_values("scan").reset_index(drop=True)
        
        Na = g["ions_incremental"].to_numpy(dtype=float) # Isotopologue
        Nb = g["basepeak_ions"].to_numpy(dtype=float)    # Base Peak (M0)
        
        # Shot Noise Calculation (Poisson Statistics)
        # SE_shot_noise = 1 / sqrt(N_effective)
        with np.errstate(divide="ignore", invalid="ignore"):
            Na_cum = np.cumsum(Na)
            Nb_cum = np.cumsum(Nb)
            n_eff_cum = np.where((Na_cum + Nb_cum) > 0, (Na_cum * Nb_cum) / (Na_cum + Nb_cum), np.nan)
            shot_noise_cum = np.where(n_eff_cum > 0, 1.0 / np.sqrt(n_eff_cum), np.nan)
            shot_noise_cum_permil = shot_noise_cum * 1000.0
            
            # Experimental Standard Error calculation
            ratio = g["ratio"]
            exp_std = ratio.expanding().std(ddof=1)
            exp_count = ratio.expanding().count()
            exp_mean = ratio.expanding().mean()
            exp_sem = exp_std / np.sqrt(exp_count)
            ratio_rel_se_permil = (exp_sem / exp_mean) * 1000.0
            ratio_rel_se_permil = ratio_rel_se_permil.replace([np.inf, -np.inf], np.nan)

        g = g.assign(
            shot_noise_cum_permil=shot_noise_cum_permil,
            ratio_rel_se_permil=ratio_rel_se_permil
        )
        rows.append(g)

    if not rows:
        return

    df_shotnoise = pd.concat(rows, ignore_index=True)

    # Plotting
    for (sample, inj), subdf in df_shotnoise.groupby(["Sample_name", "Injection"]):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = {"15N": "tab:red", "17O": "tab:purple", "18O": "tab:green"}
        
        for iso in target_isos:
            if iso not in colors: continue 
            s = subdf[subdf["isotopolog"] == iso]
            if s.empty: continue
            
            ax.plot(s["rt_min"], s["shot_noise_cum_permil"], linestyle="--", color=colors[iso], label=f"{iso} Theoretical Limit")
            ax.plot(s["rt_min"], s["ratio_rel_se_permil"], linestyle="-", marker=".", color=colors[iso], label=f"{iso} Experimental SE")

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1, 10)
        ax.set_ylim(0.1, 10)
        ax.set_xlabel("Retention time (min)")
        ax.set_ylabel("Relative standard error (‰)")
        
        # Format log axes
        ax.xaxis.set_major_locator(FixedLocator([1, 2, 5, 10]))
        ax.xaxis.set_major_formatter(FixedFormatter(['1', '2', '5', '10']))
        ax.yaxis.set_major_locator(FixedLocator([0.1, 1, 10]))
        ax.yaxis.set_major_formatter(FixedFormatter(['0.1', '1', '10']))
        
        ax.legend()
        ax.set_title(f"{sample} - Injection {inj}\nCumulative Shot Noise Analysis")
        plt.tight_layout()
        fig.savefig(os.path.join(output_dir, f"{sample}_inj{inj}_shotnoise.png"), dpi=300)
        plt.close(fig)

# =============================================================================
# === MAIN EXECUTION ==========================================================
# =============================================================================

if __name__ == "__main__":
    # 1. Locate Files
    if not os.path.exists(FOLDER_PATH):
        raise FileNotFoundError(f"The path {FOLDER_PATH} does not exist.")
        
    raw_files = [os.path.join(FOLDER_PATH, f) for f in os.listdir(FOLDER_PATH) if f.lower().endswith(".raw")]
    if not raw_files:
        raise FileNotFoundError("No .raw files found in the directory.")

    print(f"--- IsoFlow-Orbitrap ---")
    print(f"Mode: {ANALYSIS_MODE}")
    print(f"Files found: {len(raw_files)}")
    print(f"Processing with {MAX_WORKERS} workers...")

    # 2. Process Files in Parallel
    all_results = []
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for df_file in tqdm(executor.map(process_raw_file, raw_files), total=len(raw_files), desc="Extracting Data", dynamic_ncols=True):
            if not df_file.empty:
                all_results.append(df_file)
    
    if not all_results:
        print("No data extracted.")
        exit()
        
    df = pd.concat(all_results, ignore_index=True)
    print(f"Extraction completed. Total scans: {len(df)}")

    # 3. Prepare Output Directories
    output_plots = os.path.join(FOLDER_PATH, "Output_Plots_Ratios")
    output_table = os.path.join(FOLDER_PATH, "Output_Tables")
    output_shotnoise = os.path.join(FOLDER_PATH, "Output_Plots_ShotNoise")
    
    for d in [output_plots, output_table, output_shotnoise]:
        os.makedirs(d, exist_ok=True)

    # 4. Calculate Derived Metrics
    # Calculate 'ions_incremental' (Correcting for Transient math)
    # Formula: Intensity / Noise * 3 * sqrt(240000 / Res) * sqrt(MicroScans)
    df['ions_incremental'] = np.where(
        df['noise'] > 0,
        df['intensity'] / df['noise'] * 3 * np.sqrt(240000 / df['FT_Resolution']) * np.sqrt(df['MicroScanCount']),
        0
    )

    # Map M/Z to Isotopologue names
    # Note: We use the tolerance config to match
    def get_iso_name(mz_val):
        for target, name in MZ_TO_ISOTOPOLOG.items():
            if abs(mz_val - target) <= TOLERANCE:
                return name
        return 'unknown'
    
    df['isotopolog'] = df['mz'].apply(get_iso_name)

    # 5. Flag Outliers (AGC Check)
    agc_fold_cutoff = 2
    df['ions_injected'] = df['TIC'] * df['InjectionTime_ms']
    
    def flag_outliers(group):
        mean_ions = group['ions_injected'].mean()
        lower = mean_ions / agc_fold_cutoff
        upper = mean_ions * agc_fold_cutoff
        group['is_outlier'] = (group['ions_injected'] < lower) | (group['ions_injected'] > upper)
        return group
        
    df = df.groupby(['Sample_name', 'Injection']).apply(flag_outliers).reset_index(drop=True)

    # 6. Calculate Ratios vs M0 (Base Peak)
    m0_df = df[df['isotopolog'] == 'M0'][['Sample_name', 'Injection', 'scan', 'ions_incremental']].rename(columns={'ions_incremental': 'basepeak_ions'})
    df = df[df['isotopolog'] != 'M0'].merge(m0_df, on=['Sample_name', 'Injection', 'scan'], how='left')
    df['basepeak'] = 'M0'
    df['ratio'] = df['ions_incremental'] / df['basepeak_ions']

    # 7. Summarize Ratios (Averaging Window)
    df_summary_inclusive = df[(df['rt_min'] >= RT_START) & (df['rt_min'] <= RT_END) & (~df['is_outlier'])]
    df_summary_exclusive = df_summary_inclusive[df_summary_inclusive['ions_incremental'] > 0]

    def summarize(group):
        # Ratio of Sums (Statistical standard for shot noise limited data)
        ratio_sum = group['ions_incremental'].sum() / group['basepeak_ions'].sum()
        sem = group['ratio'].sem()
        return pd.Series({'ratio': ratio_sum, 'ratio_sem': sem})

    def compute_ratios_summary(df_in):
        ratios_summary = df_in.groupby(['Sample_name', 'Injection', 'isotopolog', 'basepeak']).apply(summarize).reset_index()
        ratios_summary['is_ref'] = ratios_summary['Sample_name'] == REFERENCE_NAME
        return ratios_summary

    ratios_summary_inclusive = compute_ratios_summary(df_summary_inclusive)
    ratios_summary_exclusive = compute_ratios_summary(df_summary_exclusive)

    # 8. Compute Deltas
    deltas_inclusive, deltas_summary_inclusive = compute_deltas(ratios_summary_inclusive, "inclusive (0s kept)")
    deltas_exclusive, deltas_summary_exclusive = compute_deltas(ratios_summary_exclusive, "exclusive (0s removed)")
    
    deltas_both = pd.concat([deltas_inclusive, deltas_exclusive], ignore_index=True)
    deltas_summary_both = pd.concat([deltas_summary_inclusive, deltas_summary_exclusive], ignore_index=True)

    # 9. Statistics on Removed Scans
    def compute_excluded_stats(df_incl, df_excl):
        g_incl = df_incl.groupby(['Sample_name', 'Injection', 'isotopolog']).size().reset_index(name='n_inclusive')
        g_excl = df_excl.groupby(['Sample_name', 'Injection', 'isotopolog']).size().reset_index(name='n_exclusive')
        merged = g_incl.merge(g_excl, on=['Sample_name', 'Injection', 'isotopolog'], how='left').fillna(0)
        merged['perc_scans_removed'] = (merged['n_inclusive'] - merged['n_exclusive']) / merged['n_inclusive'] * 100
        return merged

    excluded_stats = compute_excluded_stats(df_summary_inclusive, df_summary_exclusive)

    # 10. Save Results
    print("Saving Excel tables...")
    deltas_inclusive.to_excel(os.path.join(output_table, "data_all_scans_inclusive.xlsx"), index=False)
    deltas_exclusive.to_excel(os.path.join(output_table, "data_all_scans_exclusive.xlsx"), index=False)
    
    with pd.ExcelWriter(os.path.join(output_table, "summary_deltas.xlsx")) as writer:
        deltas_summary_both.to_excel(writer, sheet_name='Deltas Summary', index=False)
        excluded_stats.to_excel(writer, sheet_name='Excluded Scans Stats', index=False)

    # 11. Generate Plots
    print("Generating plots...")
    sns.set(style="whitegrid")
    plt.ioff()
    
    generate_isotopologue_plots(df, output_plots)
    generate_shot_noise_plots(df, output_shotnoise)

    print(f"Done! Results saved to: {FOLDER_PATH}")