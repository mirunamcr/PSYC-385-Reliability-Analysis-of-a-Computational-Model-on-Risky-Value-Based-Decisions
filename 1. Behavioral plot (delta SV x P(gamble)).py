# Behavioral plot (delta SV x P(gamble))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Load the data
df = pd.read_csv("Phase2_data (Raw Otto).csv")

# Create the chose_certain column
df['chose_certain'] = (df['Gamble'] == 0)


print(f"Total trials: {len(df)}")
print(f"P(Gamble) values: {sorted(df['P_Gamble'].unique())}")
print(f"Chose certain: {df['chose_certain'].mean():.3f}")
print()


# Figure with Four combinations of choice type/outcome probability
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle('P(Choose Risky) vs ΔSV by Choice Type and Outcome Probability', fontsize=16, fontweight='bold')

# Define the four conditions
conditions_4panel = {
    'P=0.5, Risky': (df['P_Gamble'] == 0.5),
    'P=0.9, Risky': (df['P_Gamble'] == 0.9),
    # FIGURE 1 PLOTS
    'P=0.5, Certain': (df['P_Gamble'] == 0.5),  # plot P(risky)
    'P=0.9, Certain': (df['P_Gamble'] == 0.9), 
}

# Panel positions
panel_positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
panel_titles = ['P=0.5 (Low Risk)', 'P=0.9 (High Risk)', 'P=0.5 (Low Risk)', 'P=0.9 (High Risk)']
colors_panel = ['blue', 'red', 'blue', 'red']

# Make bins for all panels
bins_panel = np.linspace(df['deltaSV'].min(), df['deltaSV'].max(), 20)

for i, ((label, condition), (row, col), title, color) in enumerate(zip(conditions_4panel.items(), panel_positions, panel_titles, colors_panel)):
    ax = axes[row, col]
    
    # Get data for this condition
    cond_data = df[condition & df['deltaSV'].notna()].copy()
    
    if len(cond_data) > 10:
        # Create bins and calculate P(risky)
        cond_data['bin'] = pd.cut(cond_data['deltaSV'], bins_panel, include_lowest=True) 
        cond_p_risky = cond_data.groupby('bin', observed=True)['Gamble'].mean()
        cond_counts = cond_data.groupby('bin', observed=True)['Gamble'].count()
        valid_bins = cond_counts[cond_counts > 3].index
        bin_centers = [b.mid for b in valid_bins]
        p_risky_values = cond_p_risky.loc[valid_bins]
        
        # Plot the line
        ax.plot(bin_centers, p_risky_values, 'o-', color=color, markersize=5, linewidth=2, label=label)
        
        # Add reference lines
        ax.axhline(0.5, color='gray', linestyle='--', alpha=0.5)
        ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
        
        # Formatting the plot
        ax.set_xlabel('ΔSV (SVG - SVC)')
        ax.set_ylabel('P(Choose Risky)')
        ax.set_title(f'{title}\n(n={len(cond_data)} trials)')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        ax.legend()
        
        # Put prob. and mean in title 
        mean_p_risky = cond_data['Gamble'].mean()
        ax.text(0.05, 0.95, f'Mean P(Risky) = {mean_p_risky:.3f}',  # 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.savefig('Four_Panel_Choice_Probability_Analysis.png', dpi=300, bbox_inches='tight')
plt.show()
