import matplotlib.pyplot as plt
import numpy as np
import os

# Create plots directory if it doesn't exist
os.makedirs('plots', exist_ok=True)

# Data from the report
algorithms = ['Bron-Kerbosch', 'Tomita', 'Chiba-Nishizeki']
datasets = ['Email-Enron', 'Wiki-Vote', 'as-Skitter']

# Execution times in seconds for each algorithm on each dataset
execution_times = {
    'Email-Enron': [2.78, 3.22, 1.39],
    'Wiki-Vote': [1.47, 3.39, 1.07],
    'as-Skitter': [5154.58, 5981.81, 929.47]
}

# Set the style to be similar to the provided image
plt.style.use('ggplot')
plt.rcParams['axes.facecolor'] = '#f8f9fa'
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.color'] = '#e5e5e5'
plt.rcParams['grid.linestyle'] = '-'

# Define the blue color similar to the histogram
bar_color = '#4169E1'  # Royal blue

# Plot for Email-Enron
plt.figure(figsize=(10, 6))
plt.bar(algorithms, execution_times['Email-Enron'], color=bar_color, edgecolor='black')
plt.title('Execution Time for Email-Enron Dataset', fontsize=16)
plt.ylabel('Execution Time (seconds)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('plots/email_enron_execution_time.png', dpi=300)
plt.close()

# Plot for Wiki-Vote
plt.figure(figsize=(10, 6))
plt.bar(algorithms, execution_times['Wiki-Vote'], color=bar_color, edgecolor='black')
plt.title('Execution Time for Wiki-Vote Dataset', fontsize=16)
plt.ylabel('Execution Time (seconds)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('plots/wiki_vote_execution_time.png', dpi=300)
plt.close()

# Plot for as-Skitter
plt.figure(figsize=(10, 6))
plt.bar(algorithms, execution_times['as-Skitter'], color=bar_color, edgecolor='black')
plt.title('Execution Time for as-Skitter Dataset', fontsize=16)
plt.ylabel('Execution Time (seconds)', fontsize=14)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.savefig('plots/as_skitter_execution_time.png', dpi=300)
plt.close()

print("Plots have been saved to the 'plots' directory.")