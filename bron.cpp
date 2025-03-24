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
#include <queue>
#include <bitset>

using namespace std;


constexpr int MAX_VERTICES = 500000; 

class Graph {
private:
    int n;
    vector<vector<int>> adjacencyList; // Changed to vector<int> for better cache locality
    // Removed adjacency matrix for memory efficiency with large graphs

public:
    Graph(int vertices) : n(vertices) {
        adjacencyList.resize(n);
        // No adjacency matrix for memory efficiency
    }

    void addEdge(int u, int v) {
        if (u == v) return; // Skip self-loops
        
        // Add to adjacency list if it doesn't exist
        if (find(adjacencyList[u].begin(), adjacencyList[u].end(), v) == adjacencyList[u].end()) {
            adjacencyList[u].push_back(v);
        }
        if (find(adjacencyList[v].begin(), adjacencyList[v].end(), u) == adjacencyList[v].end()) {
            adjacencyList[v].push_back(u);
        }
    }

    bool isAdjacent(int u, int v) const {
        if (u >= n || v >= n) return false;
        
        // Check adjacency list
        return find(adjacencyList[u].begin(), adjacencyList[u].end(), v) != adjacencyList[u].end();
    }

    const vector<int>& getNeighbors(int vertex) const {
        static const vector<int> emptyList;
        if (vertex >= n) return emptyList;
        return adjacencyList[vertex];
    }

    int getNumVertices() const {
        return n;
    }

    // Optimized degeneracy ordering
    vector<int> degeneracyOrdering() const {
        vector<int> ordering(n);
        vector<int> degrees(n);
        vector<vector<int>> buckets(n);  // Bucket sort by degree
        
        // Initialize degrees and buckets
        for (int i = 0; i < n; i++) {
            degrees[i] = adjacencyList[i].size();
            buckets[degrees[i]].push_back(i);
        }
        
        // Process vertices in order of increasing degree
        int nextPos = 0;
        for (int d = 0; d < n; d++) {  // Optimized: only need to go up to n-1
            while (!buckets[d].empty()) {
                // Take a vertex from current bucket
                int v = buckets[d].back();
                buckets[d].pop_back();
                
                // Skip if the vertex has been processed or its degree has changed
                if (degrees[v] != d) continue;
                
                // Add to ordering
                ordering[nextPos++] = v;
                
                // Update degrees of neighbors
                for (int u : adjacencyList[v]) {
                    if (degrees[u] > d) {  // Only process unprocessed neighbors
                        // Find and remove from current bucket - optimized with direct value check
                        auto& bucket = buckets[degrees[u]];
                        for (size_t i = 0; i < bucket.size(); i++) {
                            if (bucket[i] == u) {
                                bucket[i] = bucket.back();  // Replace with last element
                                bucket.pop_back();          // Remove last element
                                break;
                            }
                        }
                        
                        // Decrement degree and add to new bucket
                        degrees[u]--;
                        buckets[degrees[u]].push_back(u);
                    }
                }
            }
        }
        
        return ordering;
    }
};

// Optimized Bron-Kerbosch algorithm implementation
class BronKerboschAlgorithm {
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
    
    // Bitset arrays for faster set operations - only used for smaller graphs
    vector<bitset<MAX_VERTICES>> vertexNeighbors;
    
    // Memory-efficient alternative using vector<bool> for large graphs
    vector<vector<bool>> sparseNeighbors;

    // Pre-compute the neighborhood information for faster intersection operations
    void precomputeNeighborhoods() {
        int n = graph.getNumVertices();
        
        if (n <= MAX_VERTICES) {
            // Use bitsets for smaller graphs (fast but memory intensive)
            vertexNeighbors.resize(n);
            for (int i = 0; i < n; i++) {
                for (int neighbor : graph.getNeighbors(i)) {
                    vertexNeighbors[i].set(neighbor);
                }
            }
        } else {
            // For larger graphs, use a hybrid approach with sparse representation
            // Only precompute for high-degree vertices that will be used as pivots frequently
            
            // First, identify high-degree vertices (top 10% by degree)
            vector<pair<int, int>> vertexDegrees;
            for (int i = 0; i < n; i++) {
                vertexDegrees.push_back({i, (int)graph.getNeighbors(i).size()});
            }
            
            // Sort by degree (descending)
            sort(vertexDegrees.begin(), vertexDegrees.end(), 
                 [](const pair<int, int>& a, const pair<int, int>& b) {
                     return a.second > b.second;
                 });
            
            // Only create sparse neighbor matrices for top vertices (memory-saving approach)
            int numHighDegreeVertices = min(10000, n / 10); // At most 10k vertices
            sparseNeighbors.resize(n); // But allocate space for all vertices
            
            cout << "Precomputing neighborhood information for " << numHighDegreeVertices 
                 << " high-degree vertices..." << endl;
                 
            for (int i = 0; i < numHighDegreeVertices; i++) {
                int vertex = vertexDegrees[i].first;
                sparseNeighbors[vertex].resize(n, false);
                
                for (int neighbor : graph.getNeighbors(vertex)) {
                    sparseNeighbors[vertex][neighbor] = true;
                }
            }
        }
    }

    // Optimized pivot selection using the available neighborhood information
    int selectPivot(const vector<int>& P, const vector<int>& X) const {
        int pivot = -1;
        size_t maxConnections = 0;
        
        // First try to find pivot from X (optimization from Tomita et al.)
        for (int u : X) {
            size_t connections = countConnections(u, P);
            if (connections > maxConnections) {
                maxConnections = connections;
                pivot = u;
            }
        }
        
        // Then try P if we didn't find a good pivot in X
        for (int u : P) {
            size_t connections = countConnections(u, P);
            if (connections > maxConnections) {
                maxConnections = connections;
                pivot = u;
            }
        }
        
        return pivot;
    }
    
    // Helper function to count connections between a vertex and a set
    size_t countConnections(int u, const vector<int>& vertices) const {
        size_t connections = 0;
        
        // Use the most efficient method available based on precomputation
        if (u < (int)vertexNeighbors.size() && !vertexNeighbors.empty()) {
            // Use bitsets for small graphs
            for (int v : vertices) {
                if (vertexNeighbors[u].test(v)) {
                    connections++;
                }
            }
        } else if (u < (int)sparseNeighbors.size() && !sparseNeighbors[u].empty()) {
            // Use sparse precomputed neighbors for large graphs with high-degree vertices
            for (int v : vertices) {
                if (sparseNeighbors[u][v]) {
                    connections++;
                }
            }
        } else {
            // Fall back to adjacency list lookup
            const auto& neighbors = graph.getNeighbors(u);
            for (int v : vertices) {
                if (find(neighbors.begin(), neighbors.end(), v) != neighbors.end()) {
                    connections++;
                }
            }
        }
        
        return connections;
    }

    // Core Bron-Kerbosch function with pivot (heavily optimized)
    void bronKerboschPivot(const vector<int>& P, const vector<int>& R, const vector<int>& X) {
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

        // If P and X are both empty, R is a maximal clique
        if (P.empty() && X.empty()) {
            reportClique(R);
            return;
        }
        
        // Select a pivot vertex
        int pivot = selectPivot(P, X);
        
        // Get vertices to process (P \ Γ(pivot))
        vector<int> verticesToProcess;
        if (pivot != -1) {
            if (!vertexNeighbors.empty() && pivot < (int)vertexNeighbors.size()) {
                // Use bitset for faster neighbor check
                for (int v : P) {
                    if (!vertexNeighbors[pivot].test(v)) {
                        verticesToProcess.push_back(v);
                    }
                }
            } else {
                const auto& pivotNeighbors = graph.getNeighbors(pivot);
                for (int v : P) {
                    if (find(pivotNeighbors.begin(), pivotNeighbors.end(), v) == pivotNeighbors.end()) {
                        verticesToProcess.push_back(v);
                    }
                }
            }
        } else {
            verticesToProcess = P;
        }
        
        // Process each vertex
        for (int v : verticesToProcess) {
            if (timeLimitExceeded || maxCliquesReached) return;
            
            // Create new R for recursive call (R ∪ {v})
            vector<int> newR(R);
            newR.push_back(v);
            
            // Create new P and X sets for recursive call (P ∩ Γ(v) and X ∩ Γ(v))
            vector<int> newP, newX;
            newP.reserve(P.size());  // Pre-allocate for best performance
            newX.reserve(X.size());
            
            // Use the most efficient intersection method available
            if (!vertexNeighbors.empty() && v < (int)vertexNeighbors.size()) {
                // Use bitsets for small graphs
                for (int u : P) {
                    if (vertexNeighbors[v].test(u)) {
                        newP.push_back(u);
                    }
                }
                
                for (int u : X) {
                    if (vertexNeighbors[v].test(u)) {
                        newX.push_back(u);
                    }
                }
            } else if (v < (int)sparseNeighbors.size() && !sparseNeighbors[v].empty()) {
                // Use sparse precomputed neighbors for high-degree vertices in large graphs
                for (int u : P) {
                    if (sparseNeighbors[v][u]) {
                        newP.push_back(u);
                    }
                }
                
                for (int u : X) {
                    if (sparseNeighbors[v][u]) {
                        newX.push_back(u);
                    }
                }
            } else {
                // Fall back to adjacency list
                const auto& vNeighbors = graph.getNeighbors(v);
                
                // For larger P and X sets, build a hash set of neighbors for O(1) lookups
                if (P.size() + X.size() > 100) {
                    unordered_set<int> neighborSet(vNeighbors.begin(), vNeighbors.end());
                    
                    for (int u : P) {
                        if (neighborSet.find(u) != neighborSet.end()) {
                            newP.push_back(u);
                        }
                    }
                    
                    for (int u : X) {
                        if (neighborSet.find(u) != neighborSet.end()) {
                            newX.push_back(u);
                        }
                    }
                } else {
                    // For smaller sets, linear search is faster due to cache locality
                    for (int u : P) {
                        if (find(vNeighbors.begin(), vNeighbors.end(), u) != vNeighbors.end()) {
                            newP.push_back(u);
                        }
                    }
                    
                    for (int u : X) {
                        if (find(vNeighbors.begin(), vNeighbors.end(), u) != vNeighbors.end()) {
                            newX.push_back(u);
                        }
                    }
                }
            }
            
            // Recursive call
            bronKerboschPivot(newP, newR, newX);
            
            // Move v from P to X (in the caller's context)
            auto pWithoutV = P;
            auto it = find(pWithoutV.begin(), pWithoutV.end(), v);
            if (it != pWithoutV.end()) {
                pWithoutV.erase(it);
            }
            
            auto xWithV = X;
            xWithV.push_back(v);
            
            // In-place update P and X for next iteration
            bronKerboschPivot(pWithoutV, R, xWithV);
            return; // Tail recursion optimization
        }
    }

    // Function to report a maximal clique
    void reportClique(const vector<int>& R) {
        if (!R.empty()) {
            lock_guard<mutex> lock(resultsMutex);
            
            set<int> clique(R.begin(), R.end());
            maximalCliques.push_back(clique);
            size_t newCount = ++cliqueCount;
            
            // Update statistics
            int size = R.size();
            maxCliqueSize = max(maxCliqueSize, size);
            
            // Update clique size distribution
            if (size >= (int)cliqueDistribution.size()) {
                cliqueDistribution.resize(size + 1, 0);
            }
            cliqueDistribution[size]++;
            
            // Progress reporting (less frequent updates for better performance)
            if (newCount % 50000 == 0) {
                auto now = chrono::high_resolution_clock::now();
                auto elapsed = chrono::duration_cast<chrono::seconds>(now - startTime).count();
                cout << "Found " << newCount << " maximal cliques so far. Elapsed time: " 
                    << elapsed << " seconds." << endl;
            }
        }
    }
    
    // Worker function for parallel degeneracy ordering (memory-optimized)
    void workerFunction(const vector<int>& vertices, const vector<int>& degeneracyOrder) {
        int n = degeneracyOrder.size();
        
        // Process each vertex assigned to this thread
        for (int i : vertices) {
            // Check early termination conditions
            if (timeLimitExceeded || maxCliquesReached) break;
            
            int vi = degeneracyOrder[i];
            
            // Initialize sets
            vector<int> P, R = {vi}, X;
            P.reserve(graph.getNeighbors(vi).size()); // Pre-allocate memory
            X.reserve(graph.getNeighbors(vi).size());
            
            // Build efficient lookup for degeneracy ordering positions
            // This avoids repeated linear searches in the degeneracy order
            static vector<int> orderPosition;
            if (orderPosition.size() != n) {
                orderPosition.resize(n);
                for (int pos = 0; pos < n; pos++) {
                    orderPosition[degeneracyOrder[pos]] = pos;
                }
            }
            
            // Get neighbors of current vertex
            const auto& neighbors = graph.getNeighbors(vi);
            
            // Determine if using a hash-based lookup would be beneficial
            bool useHashLookup = neighbors.size() > 100;
            unordered_set<int> neighborSet;
            if (useHashLookup) {
                neighborSet.insert(neighbors.begin(), neighbors.end());
            }
            
            // Build P and X sets in a single pass through neighbors
            for (int neighbor : neighbors) {
                int pos = orderPosition[neighbor];
                
                if (pos > i) {
                    // Neighbor comes after vi in degeneracy ordering, add to P
                    P.push_back(neighbor);
                } else if (pos < i) {
                    // Neighbor comes before vi in degeneracy ordering, add to X
                    X.push_back(neighbor);
                }
            }
            
            // Run Bron-Kerbosch algorithm on this subproblem
            bronKerboschPivot(P, R, X);
            
            // Optional: report progress periodically
            if (i % 10000 == 0) {
                lock_guard<mutex> lock(resultsMutex);
                auto now = chrono::high_resolution_clock::now();
                auto elapsed = chrono::duration_cast<chrono::seconds>(now - startTime).count();
                //cout << "Thread processed vertex " << i << "/" << vertices.size() 
                  //   << " (position " << i << "/" << n << "). Elapsed: " 
                    // << elapsed << "s, Cliques: " << cliqueCount << endl;
            }
        }
    }
    
    // Improved vertex partition strategy for better load balancing
    vector<vector<int>> partitionVertices(int numPartitions, const vector<int>& degeneracyOrder) {
        vector<vector<int>> partitions(numPartitions);
        
        // Estimate workload based on vertex degree (higher degree = more work)
        vector<pair<int, int>> vertexWorkload;  // (index, estimated workload)
        for (int i = 0; i < (int)degeneracyOrder.size(); i++) {
            int vertex = degeneracyOrder[i];
            int degree = graph.getNeighbors(vertex).size();
            // Estimate workload as degree^2 (empirical approximation)
            vertexWorkload.push_back({i, degree * degree});
        }
        
        // Sort by estimated workload (descending)
        sort(vertexWorkload.begin(), vertexWorkload.end(), 
             [](const pair<int, int>& a, const pair<int, int>& b) {
                 return a.second > b.second;
             });
        
        // Distribute vertices using a greedy approach (assign heaviest jobs first)
        vector<int> partitionLoads(numPartitions, 0);
        for (const auto& [idx, load] : vertexWorkload) {
            // Find partition with minimum current load
            int minLoadPartition = 0;
            for (int p = 1; p < numPartitions; p++) {
                if (partitionLoads[p] < partitionLoads[minLoadPartition]) {
                    minLoadPartition = p;
                }
            }
            
            // Assign vertex to this partition
            partitions[minLoadPartition].push_back(idx);
            partitionLoads[minLoadPartition] += load;
        }
        
        return partitions;
    }

public:
    BronKerboschAlgorithm(const Graph& g, size_t maxCliquesLimit = 100000000, int timeLimitSecs = 3600, int threads = 0) 
        : graph(g), maxCliqueSize(0), cliqueCount(0), maxCliques(maxCliquesLimit), 
          timeLimit(timeLimitSecs), timeLimitExceeded(false), maxCliquesReached(false) {
        
        // Determine number of threads (0 = auto)
        if (threads <= 0) {
            threads = thread::hardware_concurrency();
            if (threads == 0) threads = 4;  // Default if hardware_concurrency not available
            
            // Very conservative thread count for very large graphs to prevent memory issues
            if (g.getNumVertices() > 1000000) {
                threads = min(threads, 4);
            } else if (g.getNumVertices() > 500000) {
                threads = min(threads, 8);
            }
        }
        numThreads = threads;
        
        cliqueDistribution.resize(1, 0); // Initialize with space for size 0
        startTime = chrono::high_resolution_clock::now();
    }

    // Main entry point - find all maximal cliques using the Bron-Kerbosch algorithm with degeneracy ordering
    void findAllMaximalCliques() {
        cout << "Starting Bron-Kerbosch algorithm with degeneracy ordering..." << endl;
        cout << "Max cliques limit: " << maxCliques << ", Time limit: " << timeLimit.count() << " seconds" << endl;
        cout << "Using " << numThreads << " threads" << endl;
        
        maximalCliques.clear();
        maxCliqueSize = 0;
        cliqueDistribution.clear();
        cliqueDistribution.resize(1, 0);
        cliqueCount = 0;
        timeLimitExceeded = false;
        maxCliquesReached = false;
        startTime = chrono::high_resolution_clock::now();
        
        try {
            // Precompute neighborhood bitsets only for small to medium-sized graphs
            if (graph.getNumVertices() <= MAX_VERTICES) {
                cout << "Precomputing vertex neighborhoods for faster set operations..." << endl;
                precomputeNeighborhoods();
            } else {
                cout << "Graph is too large for bitset optimization, using standard operations instead." << endl;
                // Keep vertexNeighbors empty to signal we're not using bitsets
            }
            
            // Compute degeneracy ordering
            cout << "Computing degeneracy ordering..." << endl;
            vector<int> degeneracyOrder = graph.degeneracyOrdering();
            cout << "Degeneracy ordering computed." << endl;
            
            // For very large graphs, reduce number of threads to save memory
            if (graph.getNumVertices() > 1000000 && numThreads > 4) {
                cout << "Large graph detected, reducing thread count to 4 to conserve memory" << endl;
                numThreads = 4;
            }
            
            // Partition vertices for parallel execution with improved load balancing
            vector<vector<int>> partitions = partitionVertices(numThreads, degeneracyOrder);
            
            // Create and start worker threads
            vector<thread> threads;
            for (int t = 0; t < numThreads; t++) {
                threads.emplace_back(&BronKerboschAlgorithm::workerFunction, this, 
                                     ref(partitions[t]), ref(degeneracyOrder));
            }
            
            // Wait for all threads to finish
            for (auto& t : threads) {
                if (t.joinable()) {
                    t.join();
                }
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

    // Getters for results
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

// Optimized function to read graph data from SNAP format
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
    
    // Reserve space for edges (reduce reallocations)
    edges.reserve(10000000);  // Pre-allocate for large graphs

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
    cout << "Creating graph with " << mappedIndex << " vertices..." << endl;
    Graph graph(mappedIndex);
    
    cout << "Adding " << edges.size() << " edges..." << endl;
    
    // Batch processing for better performance
    const int BATCH_SIZE = 100000;
    for (size_t i = 0; i < edges.size(); i += BATCH_SIZE) {
        size_t end = min(i + BATCH_SIZE, edges.size());
        for (size_t j = i; j < end; j++) {
            graph.addEdge(edges[j].first, edges[j].second);
        }
        
        // Progress reporting for large graphs
        if (edges.size() > 1000000 && (i % 1000000 == 0)) {
            cout << "Processed " << i << " edges..." << endl;
        }
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
        BronKerboschAlgorithm algorithm(graph, maxCliques, timeLimitSeconds, numThreads);
        
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
        cout << "Bron-Kerbosch Algorithm Results:" << endl;
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
        string outputFilename = "results/bron_kerbosch_results_" + filename;
        ofstream resultsFile(outputFilename);
        
        if (resultsFile.is_open()) {
            resultsFile << "Bron-Kerbosch Algorithm with Degeneracy Ordering Results:" << endl;
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