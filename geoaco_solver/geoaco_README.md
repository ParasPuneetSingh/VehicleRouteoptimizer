# Geographic ACO Implementation

## Overview
`geoaco.jl` implements a hybrid approach combining K-means geographic clustering with Ant Colony Optimization (ACO) for solving the Capacitated Vehicle Routing Problem (CVRP). This approach partitions the problem geographically before applying optimization algorithms.

## Dependencies
```julia
using CSV, DataFrames, StatsBase, Clustering, Plots, LinearAlgebra, Combinatorics, Base.Threads
```

## Implementation 1: Basic Geographic Clustering with TSP

### Features
- K-means clustering based on geographic coordinates
- Nearest neighbor TSP for each cluster
- Simple route optimization per geographic region
- Visualization of clusters

### Key Components
- **Geographic Clustering**: Uses K-means to partition stops by location
- **TSP Solver**: Nearest neighbor algorithm for each cluster
- **Route Construction**: Builds routes within each geographic cluster
- **Visualization**: Plots cluster assignments on map

### Parameters
- `num_clusters = 5`: Number of geographic clusters
- `bus_capacity = 20`: Vehicle capacity
- `avg_speed = 50.0`: Average speed for time calculations

## Implementation 2: Advanced Clustering with ACO

### Features
- Enhanced K-means clustering
- ACO optimization within each cluster
- Comparison between TSP and ACO approaches
- Parallel processing capabilities

### Key Components
- **Enhanced Clustering**: Improved K-means with better initialization
- **ACO per Cluster**: Ant colony optimization within each geographic region
- **Performance Comparison**: Compares TSP vs ACO results
- **Parallel Processing**: Thread-based parallelization for efficiency

### Parameters
- `k = 5`: Number of clusters
- `num_ants = 50`: Ants per cluster
- `num_iter = 300`: Iterations per cluster
- `alpha = 1.0`: Pheromone importance
- `beta = 5.0`: Heuristic importance
- `rho = 0.1`: Evaporation rate
- `Q = 100.0`: Pheromone deposit factor

## Implementation 3: Balanced Clustering with ACO

### Features
- Smart cluster merging for balanced workloads
- Advanced ACO with improved constraints
- Better solution validation
- Enhanced error handling

### Key Components
- **Cluster Balancing**: Merges small clusters for better workload distribution
- **Centroid Calculation**: Computes cluster centroids for merging decisions
- **Improved ACO**: Enhanced ant colony optimization with better constraints
- **Solution Validation**: Comprehensive solution checking

### Parameters
- `k = 5`: Initial number of clusters
- `min_size = 4`: Minimum stops per cluster (excluding depot)
- `num_ants = 50`: Ants per cluster
- `num_iter = 300`: Iterations per cluster

## Algorithm Comparison

| Feature | Basic Clustering | Advanced Clustering | Balanced Clustering |
|---------|------------------|-------------------|-------------------|
| Clustering Method | Simple K-means | Enhanced K-means | Smart K-means with merging |
| Optimization | Nearest Neighbor TSP | ACO per cluster | Enhanced ACO |
| Load Balancing | No | No | Yes |
| Parallel Processing | No | Yes | Yes |
| Solution Quality | Basic | Good | Excellent |

## Key Functions

### `tsp_nearest_neighbor(cluster_nodes, dist_matrix)`
- Solves TSP using nearest neighbor heuristic
- Returns optimized route for a cluster
- Handles depot inclusion automatically

### `ant_colony_tsp(cluster, dist_matrix)`
- ACO implementation for TSP within a cluster
- Uses pheromone and visibility matrices
- Returns optimized route and cost

### `merge_small_clusters!(clusters)`
- Merges clusters that are too small
- Uses geographic proximity for merging decisions
- Ensures balanced workload distribution

### `cluster_centroid(cluster_idx)`
- Calculates centroid of a cluster
- Used for geographic merging decisions
- Handles edge cases (single-node clusters)

## Usage

### Running All Implementations
```julia
# The script runs all three implementations sequentially
# Results show optimized routes for each approach
```

### Output Format
```
📦 Route Comparison per Bus (TSP vs ACO)

🚌 Bus 1:
  - TSP Route  : Depot → Stop A → Stop B → Depot
    Distance   : 45.2 km
  - ACO Route  : Depot → Stop B → Stop A → Depot
    Distance   : 42.1 km

📊 Total TSP Distance: 234.7 km
📊 Total ACO Distance: 218.3 km
```

## Visualization
The script includes plotting capabilities:
- Geographic scatter plot of clusters
- Color-coded cluster assignments
- Longitude vs Latitude visualization

## Performance Notes
- **Basic Clustering**: Fastest but basic solution quality
- **Advanced Clustering**: Better quality with ACO optimization
- **Balanced Clustering**: Best solution quality with load balancing

## Data Requirements
- `bus_stops_metadata.csv`: Bus stop data with coordinates
- `distance_matrix.csv`: Distance matrix between all stops

## Notes
- All implementations automatically handle depot inclusion in all clusters
- Geographic clustering reduces problem complexity
- ACO within clusters provides better optimization than simple TSP
- Cluster merging ensures balanced workload distribution
- The approach scales well for larger datasets 