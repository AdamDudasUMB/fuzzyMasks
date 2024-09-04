import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

# INPUT
df = pd.read_csv('0-sensor.csv', sep=',')
alpha = 0.3

# CORRELATION ANALYSIS
corr_matrix = df.corr(method='spearman').abs().round(2)
np.fill_diagonal(corr_matrix.values, 0)

corr_values = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)).stack()

corr_values = corr_values.values


# BORDER COMPUTATION
sigma = (np.mean(corr_values) + np.max(corr_values)) / 2 + alpha
sigma = round(sigma, 2)
theta = round(sigma - alpha, 2)

# VISUALIZATION 1: frequency graph and fuzzy membership function

fig, ax1 = plt.subplots()

hist_values, bins, patches = ax1.hist(corr_values, bins=20, edgecolor='black', color='steelblue')
ax1.set_xlim(0, 1)
ax1.set_xlabel('Correlation coefficient value')
ax1.set_ylabel('Correlation coefficient value frequency')

ax2 = ax1.twinx()

def custom_line(x, theta, sigma):
    y = np.zeros_like(x)  
    y[x >= sigma] = 1  
    in_between = (x > theta) & (x < sigma)
    y[in_between] = (x[in_between] - theta) / (sigma - theta)  
    return y

x_values = np.linspace(0, 1, 500)
y_values = custom_line(x_values, theta, sigma)

ax2.plot(x_values, y_values, color='red', lw=1)
ax2.set_ylabel('Fuzzy membership value')

line = Line2D([sigma, sigma], [0, 1], color='red', linestyle=':', linewidth=1)
ax2.add_line(line)

ax1.annotate(r'$\sigma$', xy=(sigma, 0), xytext=(sigma, -5), textcoords='offset points',
             ha='center', va='top', fontsize=12, color='red')
ax1.annotate(r'$\theta$', xy=(theta, 0), xytext=(theta, -5), textcoords='offset points',
             ha='center', va='top', fontsize=12, color='red')

padding = max(hist_values) * 0.1

ax1.set_ylim(0, max(hist_values) + padding)
ax2.set_ylim(0, 1 + (padding / max(hist_values)))

plt.show()
#plt.savefig("sensor.png", dpi = 1200)

corr_matrix = df.corr(method='spearman').round(2)

pairs = np.where((corr_matrix.abs() >= theta) & (corr_matrix.abs() <= sigma))
selected_pairs = [(corr_matrix.index[i], corr_matrix.columns[j]) 
                  for i, j in zip(pairs[0], pairs[1]) if i < j]

n = len(selected_pairs)
grid_size = int(np.ceil(np.sqrt(n)))

#VISUALIZATION 2: regression and correlation context

fig, axes = plt.subplots(grid_size, grid_size, figsize=(15, 15))
axes = axes.flatten()

for idx, pair in enumerate(selected_pairs):
    feature1, feature2 = pair
    
    ax1 = axes[idx]
    sns.regplot(x=df[feature1], y=df[feature2], lowess=True, 
                scatter_kws={'color': 'steelblue', 's': 10, 'alpha': 1}, 
                line_kws={'color': 'red', 'linewidth': 1}, ax=ax1)
    ax1.set_xlabel(feature1, fontsize=8)
    ax1.set_ylabel(feature2, fontsize=8)
    
    ax2 = ax1.inset_axes([0, -0.8, 1, 0.5])
    partial_corr_matrix = corr_matrix.loc[[feature1, feature2]]
    sns.heatmap(partial_corr_matrix, annot=True, cmap='YlGnBu', vmin=-1, vmax=1, ax=ax2, cbar=False)
    ax2.set_xticklabels(partial_corr_matrix.columns, fontsize=9, rotation=45)
    ax2.set_yticklabels(partial_corr_matrix.index, fontsize=9, rotation=45)
    
for idx in range(n, len(axes)):
    axes[idx].axis('off')

plt.tight_layout()
plt.show()
#plt.savefig("file.png", dpi = 600)