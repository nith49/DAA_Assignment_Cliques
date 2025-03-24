# CS F364 DAA Assignment 1 (Group-26)

## Contributors

1. **Pavan Sai Pasala**
   - Contributed to the implementation of the CLIQUES algorithm based on the Tomita et al. (2006) paper
   - contributed to the implementation Arboricity based algorithm based on the Chiba et al. (1985) paper
   - Contributed to the implementation of the Bron-Kerbosch algorithm based on the Eppstein et al. (2006) paper
   - Developed the project website

2. **Naga siva Nithin Kota**
   - Contributed to the implementation of the CLIQUES algorithm based on the Tomita et al. (2006) paper
   - contributed to the implementation Arboricity based algorithm based on the Chiba et al. (1985) paper
   - Contributed to the implementation of the Bron-Kerbosch algorithm based on the Eppstein et al. (2006) paper
   - Developed the project website

3. **Kanishk sharma**
   - Co-authored the project report 
4. **Jeremy karra**
   - Co-authored the project report 


## Website
The project website is available at: https://nith49.github.io/DAA_Assignment/

## Hardware and OS used
Hardware: intel i7 13th gen, 32GB RAM
Operating System: windows 11

# Maximal Clique Enumeration

This project implements and compares three algorithms for maximal clique enumeration in undirected graphs:

- **Bron-Kerbosch algorithm with degeneracy ordering** (from "Listing All Maximal Cliques in Sparse Graphs in Near-Optimal Time")
- **Tomita algorithm** (from "The Worst-Case Time Complexity for Generating All Maximal Cliques")
- **Chiba-Nishizeki algorithm** (from "Arboricity and Subgraph Listing Algorithms")

## Requirements

- C++ compiler with C++17 support
- Make or CMake (optional)
- 16GB+ RAM recommended for large datasets (as-Skitter)
- Multithreading support

## Dataset Preparation

Download the datasets from Stanford SNAP:

- **Email-Enron**
- **Wiki-Vote**
- **as-Skitter**

Extract the files to your project directory. The datasets should have the following file names:

```
Email-Enron.txt
Wiki-Vote.txt
as-skitter.txt
```

Ensure the files are in the expected format (tab or space-separated edge list).

## Building from Source

### Using g++ directly:
```bash

g++ -std=c++17 -O3 bron.cpp -o bron
g++ -std=c++17 -O3 tomita-clique.cpp -o tomita-clique
g++ -std=c++17 -O3 Chiba-Nishizeki.cpp -o Chiba-Nishizeki
```



## Execution Instructions

### Basic Usage:
```bash
./bron_kerbosch <graph_file> [max_cliques] [time_limit_seconds] [num_threads]
./tomita <graph_file> [max_cliques] [time_limit_seconds] [num_threads]
./chiba_nishizeki <graph_file> [max_cliques] [time_limit_seconds] [num_threads]
or you  can also just give something like ./tomita <graph_file> without [max_cliques] [time_limit_seconds] [num_threads]
```

### Parameters:

- `<graph_file>`: Path to the input graph file (**required**)
- `[max_cliques]`: Maximum number of cliques to find before stopping (default: **100,000,000**)
- `[time_limit_seconds]`: Time limit in seconds before stopping (default: **8000**)
- `[num_threads]`: Number of threads to use (default: **auto-detect**)

### Example Commands:

```bash
./bron_kerbosch Email-Enron.txt
./tomita Wiki-Vote.txt
./chiba_nishizeki Email-Enron.txt
./chiba_nishizeki as-skitter.txt
```


## Output

Results will be saved in the `results/` directory. This directory contains text files with detailed statistics about:

- Total number of maximal cliques found
- Size of the largest clique
- Execution time
- Distribution of clique sizes

The `plots/` directory contains visual representations of the results, including:

- Execution time comparisons across different datasets
- Histogram distributions of clique sizes
- Other performance metrics
- Size of the largest clique
- Execution time
- Distribution of clique sizes

## Visualization Scripts

To generate visualizations of the results:
```bash
python3 generate_plots.py
```

This will create charts comparing:

- Execution times of all three algorithms
- Clique size distributions for each dataset

## Troubleshooting

- If you encounter memory issues with the **as-Skitter** dataset, try reducing the number of threads.
- For very large graphs, execution may take several hours.
- Check the log output for any errors or warnings during execution.

## Repository Structure
```
C:.
|   as-skitter.txt
|   bron.cpp
|   Chiba-Nishizeki.cpp
|   compare.py
|   Email-Enron.txt
|   index.html
|   Readme.md
|   tomita-clique.cpp
|   visualize.py
|   Wiki-Vote.txt
|   report.pdf
|
+---plots
|       as_skitter_execution_time.png
|       email_enron_execution_time.png
|       histogram_as-skitter.png
|       histogram_Email-Enron.png
|       histogram_Wiki-Vote.png
|       wiki_vote_execution_time.png
|
\---results
        bron_kerbosch_results_as-skitter.txt
        bron_kerbosch_results_Email-Enron.txt
        bron_kerbosch_results_Wiki-Vote.txt
        chiba_results_as-skitter.txt
        chiba_results_Email-Enron.txt
        chiba_results_Wiki-Vote.txt
        tomita_results_as-skitter.txt
        tomita_results_Email-Enron.txt
        tomita_results_Wiki-Vote.txt
```

## Performance Summary

| Dataset       | Bron-Kerbosch (s) | Tomita (s) | Chiba-Nishizeki (s) |
|--------------|------------------|------------|------------------|
| Email-Enron  | 2.78             | 3.22       | 1.39             |
| Wiki-Vote    | 1.47             | 3.39       | 1.07             |
| as-Skitter   | 5,154.58         | 5,981.81   | 929.47           |

