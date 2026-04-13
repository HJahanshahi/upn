import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns

# Set plotting style
plt.style.use('default')
plt.rcParams.update({
    'font.size': 16,
    'axes.labelsize': 18,
    'axes.titlesize': 20,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
    'legend.fontsize': 15,
    'figure.titlesize': 24,
    'lines.linewidth': 3
})

# Load results
results = torch.load('upn_markovian_corrected_results.pt')

print("="*80)
print("VISUALIZATION: UPN ON MARKOVIAN DYNAMICAL SYSTEMS")
print("="*80)

# System names
systems = list(results.keys())
methods = ['UPN', 'Ensemble', 'Deterministic']

# Colors for methods
colors = {
    'UPN': '#2E86AB',        # Blue
    'Ensemble': '#A23B72',   # Purple
    'Deterministic': '#F18F01'  # Orange
}

#######################
# Figure 1: Performance Comparison Bar Charts
#######################

fig, axes = plt.subplots(2, 2, figsize=(18, 14))
axes = axes.flatten()

for idx, system in enumerate(systems):
    ax = axes[idx]
    
    # Extract MSE for each method
    mse_upn = results[system]['UPN']['MSE']
    mse_ens = results[system]['Ensemble']['MSE']
    mse_det = results[system]['Deterministic']['MSE']
    
    # Bar positions
    x = np.arange(3)
    mse_values = [mse_upn, mse_ens, mse_det]
    bar_colors = [colors['UPN'], colors['Ensemble'], colors['Deterministic']]
    
    bars = ax.bar(x, mse_values, color=bar_colors, edgecolor='black', linewidth=2, alpha=0.8)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, mse_values)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height * 1.02,
                f'{val:.4f}',
                ha='center', va='bottom', fontweight='bold', fontsize=14)
    
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=16)
    ax.set_ylabel('Mean Squared Error', fontsize=18)
    ax.set_title(f'{system}', fontsize=20, fontweight='bold')
    ax.grid(True, axis='y', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

plt.suptitle('MSE Comparison Across Systems', fontsize=24, fontweight='bold', y=0.995)
plt.tight_layout()
plt.savefig('comparison_mse_bars.png', dpi=300, bbox_inches='tight')
plt.show()

#######################
# Figure 2: Coverage Comparison
#######################

fig, ax = plt.subplots(figsize=(14, 8))

x = np.arange(len(systems))
width = 0.35

# Extract coverage for UPN and Ensemble
upn_coverage = [results[sys]['UPN']['Coverage_95'] for sys in systems]
ens_coverage = [results[sys]['Ensemble']['Coverage_95'] for sys in systems]

bars1 = ax.bar(x - width/2, upn_coverage, width, label='UPN', 
               color=colors['UPN'], edgecolor='black', linewidth=2, alpha=0.8)
bars2 = ax.bar(x + width/2, ens_coverage, width, label='Ensemble', 
               color=colors['Ensemble'], edgecolor='black', linewidth=2, alpha=0.8)

# Add target line at 0.95
ax.axhline(y=0.95, color='red', linestyle='--', linewidth=3, label='Target (95%)', alpha=0.7)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height * 1.01,
                f'{height:.2f}',
                ha='center', va='bottom', fontweight='bold', fontsize=14)

ax.set_xlabel('System', fontsize=18)
ax.set_ylabel('95% Confidence Interval Coverage', fontsize=18)
ax.set_title('Uncertainty Calibration: 95% CI Coverage', fontsize=24, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(systems, fontsize=16)
ax.legend(fontsize=16, loc='lower right', frameon=True, fancybox=True, shadow=True)
ax.grid(True, axis='y', alpha=0.3, linestyle='--')
ax.set_ylim([0, 1.05])
ax.set_axisbelow(True)

plt.tight_layout()
plt.savefig('comparison_coverage.png', dpi=300, bbox_inches='tight')
plt.show()

#######################
# Figure 3: Summary Table
#######################

fig, ax = plt.subplots(figsize=(16, 10))
ax.axis('tight')
ax.axis('off')

# Prepare table data
table_data = []
table_data.append(['System', 'Method', 'MSE', 'NLL', 'Coverage 95%'])

for system in systems:
    for method in methods:
        if method in results[system]:
            mse = results[system][method]['MSE']
            nll = results[system][method].get('NLL', float('nan'))
            cov = results[system][method].get('Coverage_95', float('nan'))
            
            mse_str = f'{mse:.6f}' if not np.isnan(mse) else 'N/A'
            nll_str = f'{nll:.4f}' if not np.isnan(nll) else 'N/A'
            cov_str = f'{cov:.3f}' if not np.isnan(cov) else 'N/A'
            
            table_data.append([system, method, mse_str, nll_str, cov_str])

# Create table
table = ax.table(cellText=table_data, cellLoc='center', loc='center',
                colWidths=[0.25, 0.2, 0.15, 0.15, 0.15])

table.auto_set_font_size(False)
table.set_fontsize(14)
table.scale(1, 2.5)

# Style header row
for i in range(5):
    cell = table[(0, i)]
    cell.set_facecolor('#4472C4')
    cell.set_text_props(weight='bold', color='white', fontsize=16)

# Alternate row colors
for i in range(1, len(table_data)):
    for j in range(5):
        cell = table[(i, j)]
        if i % 2 == 0:
            cell.set_facecolor('#E7E6E6')
        else:
            cell.set_facecolor('#FFFFFF')
        
        # Bold system names
        if j == 0:
            cell.set_text_props(weight='bold')
        
        # Highlight best MSE in each system
        if j == 2 and i > 1:
            current_system = table_data[i][0]
            system_rows = [r for r in table_data if r[0] == current_system]
            mse_values = [float(r[2]) if r[2] != 'N/A' else float('inf') for r in system_rows]
            if float(table_data[i][2]) == min(mse_values) and table_data[i][2] != 'N/A':
                cell.set_facecolor('#90EE90')
                cell.set_text_props(weight='bold')

plt.title('Complete Results Summary', fontsize=24, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig('results_table.png', dpi=300, bbox_inches='tight')
plt.show()

#######################
# Figure 4: Method Comparison Matrix
#######################

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# MSE Heatmap
mse_matrix = np.zeros((len(systems), len(methods)))
for i, system in enumerate(systems):
    for j, method in enumerate(methods):
        mse_matrix[i, j] = results[system][method]['MSE']

im1 = axes[0].imshow(mse_matrix, cmap='YlOrRd', aspect='auto')
axes[0].set_xticks(np.arange(len(methods)))
axes[0].set_yticks(np.arange(len(systems)))
axes[0].set_xticklabels(methods, fontsize=14)
axes[0].set_yticklabels(systems, fontsize=14)
axes[0].set_title('MSE Heatmap', fontsize=20, fontweight='bold')

# Add text annotations
for i in range(len(systems)):
    for j in range(len(methods)):
        text = axes[0].text(j, i, f'{mse_matrix[i, j]:.4f}',
                           ha="center", va="center", color="black", fontsize=12, fontweight='bold')

plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

# NLL Heatmap (UPN only)
nll_values = [results[sys]['UPN']['NLL'] for sys in systems]
nll_matrix = np.array(nll_values).reshape(-1, 1)

im2 = axes[1].imshow(nll_matrix, cmap='Blues', aspect='auto')
axes[1].set_xticks([0])
axes[1].set_yticks(np.arange(len(systems)))
axes[1].set_xticklabels(['UPN'], fontsize=14)
axes[1].set_yticklabels(systems, fontsize=14)
axes[1].set_title('NLL (UPN Only)', fontsize=20, fontweight='bold')

for i in range(len(systems)):
    text = axes[1].text(0, i, f'{nll_values[i]:.4f}',
                       ha="center", va="center", color="black", fontsize=12, fontweight='bold')

plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

# Coverage Heatmap
cov_matrix = np.zeros((len(systems), 2))
for i, system in enumerate(systems):
    cov_matrix[i, 0] = results[system]['UPN']['Coverage_95']
    cov_matrix[i, 1] = results[system]['Ensemble']['Coverage_95']

im3 = axes[2].imshow(cov_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
axes[2].set_xticks([0, 1])
axes[2].set_yticks(np.arange(len(systems)))
axes[2].set_xticklabels(['UPN', 'Ensemble'], fontsize=14)
axes[2].set_yticklabels(systems, fontsize=14)
axes[2].set_title('95% Coverage', fontsize=20, fontweight='bold')

for i in range(len(systems)):
    for j in range(2):
        text = axes[2].text(j, i, f'{cov_matrix[i, j]:.3f}',
                           ha="center", va="center", color="black", fontsize=12, fontweight='bold')

plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

plt.suptitle('Performance Matrix: All Metrics', fontsize=24, fontweight='bold')
plt.tight_layout()
plt.savefig('performance_heatmaps.png', dpi=300, bbox_inches='tight')
plt.show()

#######################
# Figure 5: Calibration Analysis
#######################

fig, ax = plt.subplots(figsize=(12, 8))

# Plot calibration for each system
markers = ['o', 's', '^', 'd']
for idx, system in enumerate(systems):
    upn_cov = results[system]['UPN']['Coverage_95']
    ens_cov = results[system]['Ensemble']['Coverage_95']
    
    ax.scatter(0.95, upn_cov, s=300, marker=markers[idx], 
              color=colors['UPN'], edgecolors='black', linewidth=2,
              label=f'{system} (UPN)', alpha=0.8, zorder=3)
    ax.scatter(0.95, ens_cov, s=300, marker=markers[idx], 
              color=colors['Ensemble'], edgecolors='black', linewidth=2,
              label=f'{system} (Ensemble)', alpha=0.8, zorder=3)

# Perfect calibration line
ax.plot([0, 1], [0, 1], 'k--', linewidth=3, label='Perfect Calibration', zorder=1)

# Acceptable region (±5%)
ax.axhspan(0.90, 1.00, alpha=0.1, color='green', zorder=0)
ax.text(0.5, 0.92, 'Acceptable Range', ha='center', fontsize=14, 
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

ax.set_xlabel('Expected Coverage', fontsize=18)
ax.set_ylabel('Actual Coverage', fontsize=18)
ax.set_title('Calibration Analysis: Expected vs Actual Coverage', fontsize=24, fontweight='bold')
ax.set_xlim([0.85, 1.0])
ax.set_ylim([0.0, 1.05])
ax.legend(loc='lower right', fontsize=12, ncol=2, frameon=True, fancybox=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('calibration_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

#######################
# Print Summary Statistics
#######################

print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

for system in systems:
    print(f"\n{system}:")
    print("-" * 60)
    
    for method in methods:
        if method in results[system]:
            mse = results[system][method]['MSE']
            nll = results[system][method].get('NLL', None)
            cov = results[system][method].get('Coverage_95', None)
            
            print(f"\n  {method}:")
            print(f"    MSE:        {mse:.6f}")
            if nll is not None and not np.isnan(nll):
                print(f"    NLL:        {nll:.4f}")
            if cov is not None and not np.isnan(cov):
                print(f"    Coverage:   {cov:.3f} (target: 0.950)")
                cal_error = abs(cov - 0.95)
                print(f"    Cal. Error: {cal_error:.3f}")

# Aggregate statistics
print("\n" + "="*80)
print("AGGREGATE PERFORMANCE")
print("="*80)

upn_mse_avg = np.mean([results[sys]['UPN']['MSE'] for sys in systems])
ens_mse_avg = np.mean([results[sys]['Ensemble']['MSE'] for sys in systems])
det_mse_avg = np.mean([results[sys]['Deterministic']['MSE'] for sys in systems])

upn_cov_avg = np.mean([results[sys]['UPN']['Coverage_95'] for sys in systems])
ens_cov_avg = np.mean([results[sys]['Ensemble']['Coverage_95'] for sys in systems])

print(f"\nAverage MSE:")
print(f"  UPN:           {upn_mse_avg:.6f}")
print(f"  Ensemble:      {ens_mse_avg:.6f}")
print(f"  Deterministic: {det_mse_avg:.6f}")

print(f"\nAverage 95% Coverage:")
print(f"  UPN:      {upn_cov_avg:.3f} (error: {abs(upn_cov_avg-0.95):.3f})")
print(f"  Ensemble: {ens_cov_avg:.3f} (error: {abs(ens_cov_avg-0.95):.3f})")

print("\n" + "="*80)
print("VISUALIZATION COMPLETE")
print("="*80)
print("\nGenerated files:")
print("  - comparison_mse_bars.png")
print("  - comparison_coverage.png")
print("  - results_table.png")
print("  - performance_heatmaps.png")
print("  - calibration_analysis.png")