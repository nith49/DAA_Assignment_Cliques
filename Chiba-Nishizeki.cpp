#include <iostream>
#include <fstream>
#include <vector>
#include <unordered_set>
#include <set>
#include <algorithm>
#include <chrono>
#include <string>
#include <cstdlib>
#include <stdexcept>
#include <thread>
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <future>
#include <sstream>

using namespace std;

class Graph {
private:
    size_t numVertices;
    vector<unordered_set<int>> adjacencyList;
    bool hasComputedDegeneracy;
    vector<int> degeneracyOrdering;

public:
    Graph(size_t n) : numVertices(n), adjacencyList(n), hasComputedDegeneracy(false) {}

    void addEdge(int u, int v) {
        if (u >= numVertices || v >= numVertices) {
            throw runtime_error("Vertex index out of bounds");
        }
        
        // Avoid self-loops
        if (u == v) return;
        
        // Avoid duplicate edges
        if (adjacencyList[u].count(v) > 0) return;
        
        adjacencyList[u].insert(v);
        adjacencyList[v].insert(u);
        
        // Invalidate degeneracy ordering if it exists
        hasComputedDegeneracy = false;
    }

    const unordered_set<int>& getNeighbors(int v) const {
        if (v >= numVertices) {
            throw runtime_error("Vertex index out of bounds");
        }
        return adjacencyList[v];
    }

    size_t getNumVertices() const {
        return numVertices;
    }
    
    // Get degree of a vertex
    size_t getDegree(int v) const {
        if (v >= numVertices) {
            throw runtime_error("Vertex index out of bounds");
        }
        return adjacencyList[v].size();
    }

    // Get degeneracy ordering of the graph
    vector<int> getDegeneracyOrdering() {
        if (hasComputedDegeneracy) {
            return degeneracyOrdering;
        }
        
        // Initialize the ordering
        degeneracyOrdering.clear();
        degeneracyOrdering.reserve(numVertices);
        
        // Create a temporary copy of degrees
        vector<int> degree(numVertices);
        vector<vector<int>> buckets(numVertices);
        vector<bool> removed(numVertices, false);
        
        // Initialize degrees and buckets
        for (size_t v = 0; v < numVertices; ++v) {
            degree[v] = adjacencyList[v].size();
            buckets[degree[v]].push_back(v);
        }
        
        // Process vertices in order of minimum degree
        for (size_t i = 0; i < numVertices; ++i) {
            // Find the vertex with minimum degree
            int minDegree = 0;
            while (minDegree < numVertices && buckets[minDegree].empty()) {
                minDegree++;
            }
            
            if (minDegree >= numVertices) {
                break;  // Should not happen in a connected graph
            }
            
            // Get a vertex with minimum degree
            int v = buckets[minDegree].back();
            buckets[minDegree].pop_back();
            removed[v] = true;
            
            // Add to ordering
            degeneracyOrdering.push_back(v);
            
            // Update degrees of neighbors
            for (int u : adjacencyList[v]) {
                if (removed[u]) continue;
                
                // Find current position of u in its bucket
                auto& bucket = buckets[degree[u]];
                auto pos = find(bucket.begin(), bucket.end(), u);
                
                if (pos != bucket.end()) {
                    // Remove from current bucket
                    bucket.erase(pos);
                    
                    // Update degree
                    degree[u]--;
                    
                    // Add to new bucket
                    buckets[degree[u]].push_back(u);
                }
            }
        }
        
        // Reverse the ordering to get vertices in order of decreasing degree
        reverse(degeneracyOrdering.begin(), degeneracyOrdering.end());
        
        hasComputedDegeneracy = true;
        return degeneracyOrdering;
    }
    
    // Get the graph density (edges/max possible edges)
    double getDensity() const {
        size_t edgeCount = 0;
        for (const auto& neighbors : adjacencyList) {
            edgeCount += neighbors.size();
        }
        edgeCount /= 2;  // Each edge is counted twice
        
        size_t maxEdges = numVertices * (numVertices - 1) / 2;
        return static_cast<double>(edgeCount) / maxEdges;
    }
    
    // Print graph statistics
    void printStats() const {
        size_t edgeCount = 0;
        size_t maxDegree = 0;
        size_t totalDegree = 0;
        
        for (const auto& neighbors : adjacencyList) {
            size_t degree = neighbors.size();
            edgeCount += degree;
            maxDegree = max(maxDegree, degree);
            totalDegree += degree;
        }
        
        edgeCount /= 2;  // Each edge is counted twice
        double avgDegree = static_cast<double>(totalDegree) / numVertices;
        
        cout << "Graph Statistics:" << endl;
        cout << "  Vertices: " << numVertices << endl;
        cout << "  Edges: " << edgeCount << endl;
        cout << "  Max Degree: " << maxDegree << endl;
        cout << "  Average Degree: " << avgDegree << endl;
        cout << "  Density: " << getDensity() << endl;
    }
};

// Read a graph in SNAP format (edge list)
Graph readSnapGraph(const string& filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Could not open file: " + filename);
    }
    
    string line;
    int maxVertex = -1;
    vector<pair<int, int>> edges;
    
    // First pass: determine the number of vertices
    while (getline(file, line)) {
        if (line.empty() || line[0] == '#') continue;
        
        int u, v;
        if (sscanf(line.c_str(), "%d %d", &u, &v) == 2) {
            // Skip self-loops
            if (u == v) continue;
            
            maxVertex = max({maxVertex, u, v});
            edges.push_back({u, v});
        }
    }
    
    if (maxVertex == -1) {
        throw runtime_error("No valid edges found in the file");
    }
    
    // Create the graph (vertices are 0-indexed in our implementation)
    Graph graph(maxVertex + 1);
    
    // Add all edges
    for (const auto& edge : edges) {
        graph.addEdge(edge.first, edge.second);
    }
    
    return graph;
}

class ChibaAlgorithm {
private:
    Graph graph;
    size_t maxCliques;
    int timeLimit;
    int numThreads;
    
    atomic<size_t> totalMaximalCliques{0};
    vector<int> cliqueDistribution;
    int maxCliqueSize{0};
    
    mutex distributionMutex;
    mutex debugMutex;
    
    // Timer variables
    chrono::time_point<chrono::high_resolution_clock> startTime;
    bool timeLimitReached{false};
    
    // Thread pool variables
    vector<thread> threadPool;
    queue<function<void()>> tasks;
    mutex queueMutex;
    condition_variable condition;
    bool stop{false};
    
    // Debug variables
    const size_t DEBUG_INTERVAL = 50000; // Print debug info every 50k cliques
    atomic<size_t> lastReportedCount{0};

    void updateCliqueDistribution(const vector<int>& clique) {
        int size = clique.size();
        
        lock_guard<mutex> lock(distributionMutex);
        if (size >= cliqueDistribution.size()) {
            cliqueDistribution.resize(size + 1, 0);
        }
        cliqueDistribution[size]++;
        maxCliqueSize = max(maxCliqueSize, size);
        
        // Check if we need to print debug information
        size_t currentCount = totalMaximalCliques.load();
        size_t lastReported = lastReportedCount.load();
        
        if (currentCount >= lastReported + DEBUG_INTERVAL) {
            // Try to update the last reported count atomically
            // This ensures only one thread prints debug info at a time
            if (lastReportedCount.compare_exchange_strong(lastReported, currentCount)) {
                auto now = chrono::high_resolution_clock::now();
                chrono::duration<double> elapsed = now - startTime;
                
                lock_guard<mutex> debugLock(debugMutex);
                cout << "Found " << currentCount << " cliques, elapsed time: " 
                     << elapsed.count() << " seconds, largest clique: " << maxCliqueSize << endl;
            }
        }
    }
    
    bool checkTimeLimit() {
        if (timeLimit <= 0) return false;
        
        auto now = chrono::high_resolution_clock::now();
        chrono::duration<double> elapsed = now - startTime;
        
        if (elapsed.count() > timeLimit) {
            timeLimitReached = true;
            return true;
        }
        
        return false;
    }
    
    void chibaNishizeki(vector<int>& P, vector<int>& X, vector<int>& R) {
        if (P.empty() && X.empty()) {
            // Found a maximal clique
            totalMaximalCliques++;
            updateCliqueDistribution(R);
            return;
        }
        
        if (totalMaximalCliques >= maxCliques || checkTimeLimit()) {
            return;
        }
        
        // Choose pivot vertex to maximize |P ∩ N(u)|
        int pivot = -1;
        int maxIntersectionSize = -1;
        
        // Consider vertices from both P and X for pivot selection
        vector<int> pivotCandidates;
        pivotCandidates.insert(pivotCandidates.end(), P.begin(), P.end());
        pivotCandidates.insert(pivotCandidates.end(), X.begin(), X.end());
        
        for (int u : pivotCandidates) {
            const auto& neighbors = graph.getNeighbors(u);
            int intersectionSize = 0;
            
            for (int v : P) {
                if (neighbors.count(v) > 0) {
                    intersectionSize++;
                }
            }
            
            if (intersectionSize > maxIntersectionSize) {
                maxIntersectionSize = intersectionSize;
                pivot = u;
            }
        }
        
        // P \ N(pivot) - vertices that will extend the current clique
        vector<int> candidates;
        if (pivot != -1) {
            const auto& pivotNeighbors = graph.getNeighbors(pivot);
            for (int v : P) {
                if (pivotNeighbors.count(v) == 0) {
                    candidates.push_back(v);
                }
            }
        } else {
            candidates = P;  // If no pivot, try all vertices in P
        }
        
        // Try each candidate
        for (int v : candidates) {
            // Create intersection of P and N(v)
            vector<int> newP;
            const auto& vNeighbors = graph.getNeighbors(v);
            
            for (int u : P) {
                if (u != v && vNeighbors.count(u) > 0) {
                    newP.push_back(u);
                }
            }
            
            // Create intersection of X and N(v)
            vector<int> newX;
            for (int u : X) {
                if (vNeighbors.count(u) > 0) {
                    newX.push_back(u);
                }
            }
            
            // Add v to the current clique R
            R.push_back(v);
            
            // Recursive call
            chibaNishizeki(newP, newX, R);
            
            // Backtrack
            R.pop_back();
            
            // Move v from P to X for future iterations
            P.erase(remove(P.begin(), P.end(), v), P.end());
            X.push_back(v);
            
            if (totalMaximalCliques >= maxCliques || timeLimitReached) {
                break;
            }
        }
    }
    
    // Worker thread function
    void workerThread() {
        while (true) {
            function<void()> task;
            {
                unique_lock<mutex> lock(queueMutex);
                condition.wait(lock, [this] { return stop || !tasks.empty(); });
                if (stop && tasks.empty()) {
                    return;
                }
                task = move(tasks.front());
                tasks.pop();
            }
            task();
        }
    }
    
    // Add task to thread pool
    template<class F>
    void enqueue(F&& f) {
        {
            unique_lock<mutex> lock(queueMutex);
            tasks.emplace(forward<F>(f));
        }
        condition.notify_one();
    }

public:
    ChibaAlgorithm(const Graph& g, size_t maxCl = 100000000, int timeLim = 8000, int threads = 0)
        : graph(g), maxCliques(maxCl), timeLimit(timeLim), numThreads(threads),
          cliqueDistribution(1, 0) {
        
        if (numThreads <= 0) {
            numThreads = thread::hardware_concurrency();
            if (numThreads == 0) numThreads = 1;  // Fallback if detection fails
        }
        
        // Start thread pool
        for (int i = 0; i < numThreads; ++i) {
            threadPool.emplace_back(&ChibaAlgorithm::workerThread, this);
        }
    }
    
    ~ChibaAlgorithm() {
        {
            unique_lock<mutex> lock(queueMutex);
            stop = true;
        }
        condition.notify_all();
        for (auto& thread : threadPool) {
            if (thread.joinable()) {
                thread.join();
            }
        }
    }
    
    void findAllMaximalCliques() {
        startTime = chrono::high_resolution_clock::now();
        
        // Print initial debug info
        cout << "Starting clique enumeration with " << numThreads << " threads" << endl;
        
        // Get degeneracy ordering for better performance
        vector<int> ordering = graph.getDegeneracyOrdering();
        
        // Vertex to position in ordering mapping for faster lookup
        vector<int> position(graph.getNumVertices(), -1);
        for (int i = 0; i < ordering.size(); ++i) {
            position[ordering[i]] = i;
        }
        
        // Process vertices in degeneracy ordering
        vector<future<void>> futures;
        
        for (int i = 0; i < ordering.size() && !timeLimitReached; ++i) {
            int v = ordering[i];
            
            // Create P as neighbors of v that come later in the ordering
            vector<int> P;
            const auto& neighbors = graph.getNeighbors(v);
            
            for (int u : neighbors) {
                // Check if u comes after v in the ordering (more efficient lookup)
                if (position[u] > i) {
                    P.push_back(u);
                }
            }
            
            // X contains neighbors of v that come before v in the ordering
            vector<int> X;
            for (int u : neighbors) {
                if (position[u] < i) {
                    X.push_back(u);
                }
            }
            
            // R contains just the vertex v initially
            vector<int> R = {v};
            
            // Run algorithm for this subproblem
            if (i < numThreads && numThreads > 1) {
                // For initial vertices, process in parallel
                enqueue([=, P = P, X = X, R = R]() mutable {
                    chibaNishizeki(P, X, R);
                });
            } else {
                // Process sequentially
                chibaNishizeki(P, X, R);
            }
            
            if (totalMaximalCliques >= maxCliques) {
                cout << "Maximum number of cliques reached (" << maxCliques << "). Stopping." << endl;
                break;
            }
            
            // Print progress for initial vertices
            
        }
        
        // Wait for all tasks to complete
        {
            unique_lock<mutex> lock(queueMutex);
            while (!tasks.empty()) {
                lock.unlock();
                this_thread::sleep_for(chrono::milliseconds(10));
                lock.lock();
            }
        }
        
        // Update statistics
        if (timeLimitReached) {
            cout << "Time limit reached (" << timeLimit << " seconds). Stopping." << endl;
        }
    }
    
    size_t getTotalMaximalCliques() const {
        return totalMaximalCliques;
    }
    
    int getMaxCliqueSize() const {
        return maxCliqueSize;
    }
    
    const vector<int>& getCliqueDistribution() const {
        return cliqueDistribution;
    }
};

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
        
        // Print graph statistics
        graph.printStats();
        
        // Find all maximal cliques
        ChibaAlgorithm algorithm(graph, maxCliques, timeLimitSeconds, numThreads);
        
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
        cout << "Chiba Algorithm Results:" << endl;
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
        string outputFilename = "results/chiba_results_" + filename;
        ofstream resultsFile(outputFilename);
        
        if (resultsFile.is_open()) {
            resultsFile << "Chiba Algorithm Results:" << endl;
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