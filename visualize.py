import os
import re
import matplotlib.pyplot as plt
import numpy as np

# Paths to results and plots directories
results_dir = "results"
plots_dir = "plots"

# Create the plots directory if it doesn't exist
os.makedirs(plots_dir, exist_ok=True)

# Only process Bron-Kerbosch results
files = [f for f in os.listdir(results_dir) if f.startswith('bron_kerbosch_results_')]

# Process each file and create a separate histogram
for filename in files:
    filepath = os.path.join(results_dir, filename)
    
    # Extract dataset name from filename
    dataset_name = filename.replace('bron_kerbosch_results_', '').replace('.txt', '')
    
    # Read the file
    with open(filepath, 'r') as file:
        content = file.read()
    
    # Extract clique distribution using regex
    sizes = []
    counts = []
    
    for line in content.splitlines():
        match = re.match(r'Size (\d+): (\d+) cliques', line)
        if match:
            size = int(match.group(1))
            count = int(match.group(2))
            sizes.append(size)
            counts.append(count)
    
    # Create a new figure for each dataset
    plt.figure(figsize=(10, 6))
    
    # Find the maximum clique size
    max_clique_size = max(sizes)
    
    # Create colors list, red for max clique size, blue for others
    colors = ['red' if size == max_clique_size else 'royalblue' for size in sizes]
    
    # Use plt.bar with edge alignment to simulate histogram without gaps
    bars = plt.bar([x+0.5 for x in sizes], counts, width=1.0, align='center', 
                  edgecolor='black', color=colors)
    
    plt.xlim(min(sizes)-0.5, max(sizes)+0.5)
    
    # Set x-axis to integer ticks only
    plt.xticks(np.arange(min(sizes), max(sizes)+1, step=1 if max(sizes) <= 20 else 5))
    
    plt.xlabel('Clique Size')
    plt.ylabel('Number of Cliques')
    plt.title(f'Clique Size Distribution for {dataset_name}')
    plt.grid(True, axis='y', alpha=0.3)
    
    # Add annotation for the maximal clique size
    max_size_index = sizes.index(max_clique_size)
    max_count = counts[max_size_index]
    plt.annotate(f'Maximal: {max_clique_size}',
                xy=(max_clique_size, max_count),
                xytext=(max_clique_size, max_count + max(counts)*0.1),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                ha='center')
    
    plt.axvline(max_clique_size, color='#ff0044', linestyle='--', linewidth=2, label=f'Max Clique Size: {max_clique_size}')
    plt.legend(facecolor='white', edgecolor='cyan', fontsize=12)
    
    # Save the plot in the plots directory
    output_file = os.path.join(plots_dir, f'histogram_{dataset_name}.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created histogram for {dataset_name} at {output_file}")
