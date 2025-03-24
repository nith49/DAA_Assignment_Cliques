#include <iostream>
#include <vector>
#include <set>
#include <algorithm>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>
#include <unordered_map>
#include <unordered_set>
#include <thread>
#include <mutex>
#include <atomic>

using namespace std;

class Graph {
private:
    int n; // Number of vertices
    vector<unordered_set<int>> adjacencyList;

public:
    Graph(int vertices) : n(vertices) {
        adjacencyList.resize(n);
    }

    void addEdge(int u, int v) {
        if (u == v) return; // Skip self-loops
        adjacencyList[u].insert(v);
        adjacencyList[v].insert(u);
    }

    bool isAdjacent(int u, int v) const {
        if (u >= n || v >= n) return false;
        return adjacencyList[u].find(v) != adjacencyList[u].end();
    }

    const unordered_set<int>& getNeighbors(int vertex) const {
        static const unordered_set<int> emptySet;
        if (vertex >= n) return emptySet;
        return adjacencyList[vertex];
    }

    int getNumVertices() const {
        return n;
    }

    // Function to return vertices that are adjacent to all vertices in the set
    unordered_set<int> getCommonNeighbors(const unordered_set<int>& vertices) const {
        unordered_set<int> result;
        if (vertices.empty()) {
            for (int i = 0; i < n; i++) {
                result.insert(i);
            }
            return result;
        }

        // Start with neighbors of the first vertex
        auto it = vertices.begin();
        result = adjacencyList[*it];
        ++it;

        // Intersect with neighbors of remaining vertices
        for (; it != vertices.end(); ++it) {
            unordered_set<int> temp;
            for (int v : result) {
                if (adjacencyList[*it].find(v) != adjacencyList[*it].end()) {
                    temp.insert(v);
                }
            }
            result = temp;
        }

        return result;
    }
};

// Implementation of the Tomita algorithm (CLIQUES procedure)
class TomitaAlgorithm {
private:
    const Graph& graph;
    vector<set<int>> maximalCliques;
    int maxCliqueSize;
    vector<int> cliqueDistribution;
    atomic<size_t> cliqueCount;
    size_t maxCliques;
    chrono::seconds timeLimit;
    chrono::time_point<chrono::high_resolution_clock> startTime;
    mutex resultsMutex;
    atomic<bool> timeLimitExceeded;
    atomic<bool> maxCliquesReached;
    int numThreads;

    // Optimized pivot selection - only sampling a subset of vertices
    int fastPivotSelection(const unordered_set<int>& SUBG, const unordered_set<int>& CAND) const {
        int bestPivot = -1;
        int maxIntersectionSize = -1;
        
        // For very large sets, use sampling to find a good pivot faster
        const int MAX_SAMPLE_SIZE = 30;  // This can be adjusted
        
        // Function to process a single candidate pivot
        auto processPivot = [&](int u) {
            int intersectionSize = 0;
            for (int v : graph.getNeighbors(u)) {
                if (CAND.find(v) != CAND.end()) {
                    intersectionSize++;
                }
            }
            
            if (intersectionSize > maxIntersectionSize) {
                maxIntersectionSize = intersectionSize;
                bestPivot = u;
            }
        };
        
        // Sample from SUBG
        if (SUBG.size() > MAX_SAMPLE_SIZE) {
            // Sample MAX_SAMPLE_SIZE random elements from SUBG
            vector<int> sampleP(SUBG.begin(), SUBG.end());
            random_shuffle(sampleP.begin(), sampleP.end());
            for (int i = 0; i < min(MAX_SAMPLE_SIZE, (int)sampleP.size()); i++) {
                processPivot(sampleP[i]);
            }
        } else {
            // Process all elements in SUBG
            for (int u : SUBG) {
                processPivot(u);
            }
        }
        
        // Sample from CAND
        if (CAND.size() > MAX_SAMPLE_SIZE) {
            // Sample MAX_SAMPLE_SIZE random elements from CAND
            vector<int> sampleX(CAND.begin(), CAND.end());
            random_shuffle(sampleX.begin(), sampleX.end());
            for (int i = 0; i < min(MAX_SAMPLE_SIZE, (int)sampleX.size()); i++) {
                processPivot(sampleX[i]);
            }
        } else {
            // Process all elements in CAND
            for (int u : CAND) {
                processPivot(u);
            }
        }
        
        return bestPivot;
    }

    // Helper function for the EXPAND procedure
    void EXPAND(unordered_set<int>& SUBG, unordered_set<int>& CAND, unordered_set<int>& Q) {
        // Check for time limit and max cliques
        auto now = chrono::high_resolution_clock::now();
        if (chrono::duration_cast<chrono::seconds>(now - startTime) > timeLimit) {
            timeLimitExceeded = true;
            return;
        }
        
        if (cliqueCount >= maxCliques) {
            maxCliquesReached = true;
            return;
        }

        if (SUBG.empty() && CAND.empty()) {
            // Found a maximal clique (Q)
            reportClique(Q);
            return;
        }

        if (CAND.empty()) {
            return;
        }

        // Select pivot u to maximize |CAND ∩ Γ(u)| using optimized selection
        int pivot = fastPivotSelection(SUBG, CAND);

        // Create a vector of candidates for extension to avoid modifying CAND during iteration
        vector<int> ext_candidates;
        if (pivot == -1) {
            ext_candidates = vector<int>(CAND.begin(), CAND.end());
        } else {
            for (int v : CAND) {
                if (!graph.isAdjacent(pivot, v)) {
                    ext_candidates.push_back(v);
                }
            }
        }

        // Process candidates without creating full copies for each iteration
        for (int q : ext_candidates) {
            if (timeLimitExceeded || maxCliquesReached) return;
            
            // Add q to current clique
            Q.insert(q);
            
            // Create new SUBG and CAND for recursive call
            unordered_set<int> newSUBG, newCAND;
            const auto& q_neighbors = graph.getNeighbors(q);
            
            for (int v : SUBG) {
                if (q_neighbors.find(v) != q_neighbors.end()) {
                    newSUBG.insert(v);
                }
            }
            
            for (int v : CAND) {
                if (q_neighbors.find(v) != q_neighbors.end()) {
                    newCAND.insert(v);
                }
            }
            
            // Recursive call with the new sets
            EXPAND(newSUBG, newCAND, Q);
            
            // Remove q from current clique
            Q.erase(q);
            
            // Move q from CAND to SUBG for next iteration
            CAND.erase(q);
            SUBG.insert(q);
        }
    }

    // Function to report a maximal clique
    void reportClique(const unordered_set<int>& Q) {
        set<int> clique(Q.begin(), Q.end());
        if (!clique.empty()) {
            lock_guard<mutex> lock(resultsMutex);
            
            maximalCliques.push_back(clique);
            size_t newCount = ++cliqueCount;
            
            // Update statistics
            int size = clique.size();
            maxCliqueSize = max(maxCliqueSize, size);
            
            // Update clique size distribution
            if (size >= (int)cliqueDistribution.size()) {
                cliqueDistribution.resize(size + 1, 0);
            }
            cliqueDistribution[size]++;
            
            // Progress reporting
            if (newCount % 50000 == 0) {
                auto now = chrono::high_resolution_clock::now();
                auto elapsed = chrono::duration_cast<chrono::seconds>(now - startTime).count();
                cout << "Found " << newCount << " maximal cliques so far. Elapsed time: " 
                    << elapsed << " seconds." << endl;
            }
        }
    }
    
    // Partition the initial vertices for parallel processing
    vector<vector<int>> partitionVertices(int numPartitions) {
        vector<vector<int>> partitions(numPartitions);
        int n = graph.getNumVertices();
        
        // Simple partitioning - just distribute vertices evenly
        for (int i = 0; i < n; i++) {
            partitions[i % numPartitions].push_back(i);
        }
        
        return partitions;
    }
    
    // Worker function for parallel processing
    void workerFunction(const vector<int>& vertices) {
        for (int startVertex : vertices) {
            if (timeLimitExceeded || maxCliquesReached) break;
            
            unordered_set<int> Q = {startVertex};
            unordered_set<int> SUBG;
            unordered_set<int> CAND;
            
            // Initialize CAND with neighbors of startVertex
            const auto& neighbors = graph.getNeighbors(startVertex);
            for (int v : neighbors) {
                if (v > startVertex) { // Only consider vertices with higher IDs to avoid duplicates
                    CAND.insert(v);
                } else {
                    SUBG.insert(v);
                }
            }
            
            // Start expansion from this vertex
            EXPAND(SUBG, CAND, Q);
            
            // Also try with just the vertex itself (for isolated vertices)
            if (neighbors.empty()) {
                reportClique(Q);
            }
        }
    }

public:
    TomitaAlgorithm(const Graph& g, size_t maxCliquesLimit = 100000000, int timeLimitSecs = 3600, int threads = 0) 
        : graph(g), maxCliqueSize(0), cliqueCount(0), maxCliques(maxCliquesLimit), 
          timeLimit(timeLimitSecs), timeLimitExceeded(false), maxCliquesReached(false) {
        
        // Determine number of threads (0 = auto)
        if (threads <= 0) {
            threads = thread::hardware_concurrency();
            if (threads == 0) threads = 4;  // Default if hardware_concurrency not available
        }
        numThreads = threads;
        
        cliqueDistribution.resize(1, 0); // Initialize with space for size 0
        startTime = chrono::high_resolution_clock::now();
    }

    // Main findAllMaximalCliques procedure with parallelization
    void findAllMaximalCliques() {
        cout << "Starting Tomita algorithm..." << endl;
        cout << "Max cliques limit: " << maxCliques << ", Time limit: " << timeLimit.count() << " seconds" << endl;
        cout << "Using " << numThreads << " threads" << endl;
        
        maximalCliques.clear();
        maxCliqueSize = 0;
        cliqueDistribution.clear();
        cliqueDistribution.resize(1, 0);
        cliqueCount = 0;
        startTime = chrono::high_resolution_clock::now();
        
        try {
            // Partition vertices for parallel processing
            vector<vector<int>> partitions = partitionVertices(numThreads);
            
            // Create and start worker threads
            vector<thread> threads;
            for (int t = 0; t < numThreads; t++) {
                threads.emplace_back(&TomitaAlgorithm::workerFunction, this, partitions[t]);
            }
            
            // Wait for all threads to finish
            for (auto& t : threads) {
                t.join();
            }
            
            if (timeLimitExceeded) {
                cout << "Enumeration stopped early: Time limit exceeded" << endl;
                cout << "Results are partial but still valid for analysis." << endl;
            } else if (maxCliquesReached) {
                cout << "Enumeration stopped early: Maximum number of cliques reached" << endl;
                cout << "Results are partial but still valid for analysis." << endl;
            } else {
                cout << "Complete enumeration finished normally." << endl;
            }
        } catch (const exception& e) {
            cerr << "Exception in clique enumeration: " << e.what() << endl;
        }
    }

    // Getter for results
    const vector<set<int>>& getMaximalCliques() const {
        return maximalCliques;
    }

    int getMaxCliqueSize() const {
        return maxCliqueSize;
    }

    int getTotalMaximalCliques() const {
        return maximalCliques.size();
    }

    const vector<int>& getCliqueDistribution() const {
        return cliqueDistribution;
    }
};

// Function to read graph data from SNAP format
Graph readSnapGraph(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        cerr << "Error: Could not open file " << filename << endl;
        exit(1);
    }

    string line;
    unordered_map<int, int> nodeMapping;
    vector<pair<int, int>> edges;
    int mappedIndex = 0;

    // Read the edges
    while (getline(file, line)) {
        // Skip comment lines
        if (line[0] == '#') continue;

        istringstream iss(line);
        int u, v;
        if (!(iss >> u >> v)) continue; // Skip malformed lines

        // Map node IDs to sequential indices (zero-based)
        if (nodeMapping.find(u) == nodeMapping.end()) {
            nodeMapping[u] = mappedIndex++;
        }
        if (nodeMapping.find(v) == nodeMapping.end()) {
            nodeMapping[v] = mappedIndex++;
        }

        edges.push_back({nodeMapping[u], nodeMapping[v]});
    }

    // Create graph (automatically converts to undirected)
    Graph graph(mappedIndex);
    for (const auto& edge : edges) {
        graph.addEdge(edge.first, edge.second);
    }

    cout << "Graph loaded: " << mappedIndex << " vertices, " << edges.size() << " edges" << endl;
    cout << "Converted to undirected graph with zero-based vertex indices (0 to " << (mappedIndex-1) << ")" << endl;
    return graph;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        cerr << "Usage: " << argv[0] << " <graph_file> [max_cliques] [time_limit_seconds] [num_threads]" << endl;
        return 1;
    }

    string filename = argv[1];
    
    // Optional parameters
    size_t maxCliques = 100000000;
    int timeLimitSeconds = 8000;
    int numThreads = 0;  // 0 means auto-detect
    
    if (argc >= 3) {
        maxCliques = stoul(argv[2]);
    }
    
    if (argc >= 4) {
        timeLimitSeconds = stoi(argv[3]);
    }
    
    if (argc >= 5) {
        numThreads = stoi(argv[4]);
    }
    
    try {
        // Read the graph
        cout << "Reading graph from " << filename << "..." << endl;
        Graph graph = readSnapGraph(filename);
        
        // Find all maximal cliques
        TomitaAlgorithm algorithm(graph, maxCliques, timeLimitSeconds, numThreads);
        
        // Create output directory if it doesn't exist
        #ifdef _WIN32
            system("if not exist results mkdir results");
        #else
            system("mkdir -p results");
        #endif
        
        // Measure execution time
        cout << "Starting clique enumeration algorithm..." << endl;
        auto start = chrono::high_resolution_clock::now();
        
        algorithm.findAllMaximalCliques();
        
        auto end = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = end - start;
        
        // Print results
        cout << "\n----------------------------------------" << endl;
        cout << "Tomita Algorithm Results:" << endl;
        cout << "Total maximal cliques found: " << algorithm.getTotalMaximalCliques() << endl;
        cout << "Size of largest clique: " << algorithm.getMaxCliqueSize() << endl;
        cout << "Execution time: " << elapsed.count() << " seconds" << endl;
        
        // Print the distribution of clique sizes
        cout << "\nClique size distribution:" << endl;
        const vector<int>& distribution = algorithm.getCliqueDistribution();
        for (size_t i = 1; i < distribution.size(); i++) {
            if (distribution[i] > 0) {
                cout << "Size " << i << ": " << distribution[i] << " cliques" << endl;
            }
        }
        cout << "----------------------------------------" << endl;
        
        // Save results to file
        string outputFilename = "results/tomita_results_" + filename;
        ofstream resultsFile(outputFilename);
        
        if (resultsFile.is_open()) {
            resultsFile << "Tomita Algorithm Results:" << endl;
            resultsFile << "Dataset: " << filename << endl;
            resultsFile << "Total maximal cliques found: " << algorithm.getTotalMaximalCliques() << endl;
            resultsFile << "Size of largest clique: " << algorithm.getMaxCliqueSize() << endl;
            resultsFile << "Execution time: " << elapsed.count() << " seconds" << endl;
            
            resultsFile << "\nClique size distribution:" << endl;
            for (size_t i = 1; i < distribution.size(); i++) {
                if (distribution[i] > 0) {
                    resultsFile << "Size " << i << ": " << distribution[i] << " cliques" << endl;
                }
            }
            
            resultsFile.close();
            cout << "Results saved to " << outputFilename << endl;
        } else {
            cerr << "Warning: Could not open results file for writing." << endl;
        }
    }
    catch (const exception& e) {
        cerr << "ERROR: " << e.what() << endl;
        return 1;
    }
    catch (...) {
        cerr << "ERROR: Unknown exception occurred" << endl;
        return 1;
    }
    
    return 0;
}

//golden code