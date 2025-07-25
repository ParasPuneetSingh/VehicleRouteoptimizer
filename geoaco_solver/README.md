# Geographic ACO Implementation

## Overview
This folder contains sophisticated hybrid implementations that combine K-means geographic clustering with Ant Colony Optimization (ACO) for solving the Capacitated Vehicle Routing Problem (CVRP).

## Purpose
**Phase 3: Sophistication - Problem Decomposition + Optimization**
- Combine multiple optimization strategies
- Leverage geographic intelligence for problem decomposition
- Achieve superior scalability and performance through hybrid approaches
- Demonstrate the power of combining domain knowledge with optimization

## Contents

### Files
- `geoaco.jl` - Main implementation file containing three hybrid approaches
- `geoaco_README.md` - Detailed documentation for the implementations
- `route_optimization_2opt.jl` - **NEW**: Advanced route optimization with 2-Opt local search

### Implementations

#### 1. Basic Geographic Clustering with TSP
- **Approach**: K-means clustering with nearest neighbor TSP
- **Features**: Geographic partitioning, simple TSP optimization, visualization
- **Use Case**: Understanding geographic decomposition

#### 2. Advanced Clustering with ACO
- **Approach**: Enhanced clustering with ACO optimization per cluster
- **Features**: ACO per cluster, performance comparison, parallel processing
- **Use Case**: Improved optimization within geographic regions

#### 3. Balanced Clustering with ACO
- **Approach**: Smart cluster merging with advanced ACO
- **Features**: Load balancing, enhanced ACO, better solution validation
- **Use Case**: Production-ready hybrid optimization

#### 4. **Route Optimization with 2-Opt Local Search** ⭐ **NEW**
- **Approach**: ACO + 2-Opt local search for geographically sensible routes
- **Features**: 
  - Ant Colony Optimization for initial route construction
  - 2-Opt local search for route smoothing and zigzag reduction
  - Intelligent route segmentation and merging
  - Geographic clustering for balanced workload distribution
- **Use Case**: **Production-ready routes with minimal zigzag patterns**
- **Key Innovation**: Combines metaheuristic optimization with local search for practical, implementable routes

## Key Characteristics
- **Complexity**: High
- **Innovation**: Advanced (hybrid approach)
- **Solution Quality**: Excellent
- **Scalability**: Excellent
- **Problem Decomposition**: Geographic clustering
- **Parallel Processing**: Extensive
- **Geographic Implementability**: **Superior** (especially with 2-Opt optimization)

## Usage

### Running All Implementations
```bash
cd geoaco_solver

# Run basic geographic clustering
julia geoaco.jl

# Run advanced route optimization with 2-Opt
julia route_optimization_2opt.jl
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

### New 2-Opt Implementation Output
```
🚍 Final 5-Bus Network (Geographically Optimized):
Bus 1: depot → Stop A → Stop B → depot | Distance: 45.2 km (2-Opt optimized)
Bus 2: depot → Stop C → Stop D → depot | Distance: 38.7 km
...

📏 Total fleet distance: 234.7 km
🎯 Geographic optimization reduces zigzag patterns for more practical routes
```

## Algorithm Features

### Traditional Approaches
- **Geographic Intelligence**: K-means clustering based on coordinates
- **Problem Decomposition**: Divide large problem into smaller subproblems
- **Parallel Optimization**: ACO applied to each geographic cluster
- **Load Balancing**: Smart cluster merging for balanced workloads
- **Visualization**: Geographic scatter plots of clusters

### **NEW: 2-Opt Route Optimization**
- **Zigzag Reduction**: Eliminates inefficient crossovers and backtracks
- **Local Search**: Iteratively improves route segments
- **Geographic Sensibility**: Routes follow natural geographic patterns
- **Practical Implementation**: Routes that drivers can actually follow
- **Performance Enhancement**: Reduces total distance through local optimization

## Hybrid Approach Benefits
- **Scalability**: Handles larger problems efficiently
- **Quality**: Combines best of clustering and optimization
- **Intelligence**: Leverages geographic knowledge
- **Flexibility**: Adapts to different problem sizes
- **Implementability**: **Routes are practical for real-world use**

## Geographic Implementability Advantages

### **Why 2-Opt Makes Routes More Implementable**

1. **Reduces Zigzag Patterns**
   - Traditional algorithms can create routes that cross back and forth
   - 2-Opt eliminates inefficient crossovers
   - Routes follow more natural geographic flows

2. **Improves Driver Experience**
   - Routes are easier to follow and understand
   - Reduces confusion and navigation errors
   - More efficient for real-world driving conditions

3. **Cost Savings**
   - Shorter total distances
   - Reduced fuel consumption
   - Lower operational costs

4. **Better Customer Service**
   - More predictable arrival times
   - Improved route efficiency
   - Better resource utilization

## Performance Notes
- **Basic Clustering**: Fastest but basic solution quality
- **Advanced Clustering**: Better quality with ACO optimization
- **Balanced Clustering**: Best solution quality with load balancing
- **2-Opt Optimization**: **Superior geographic implementability** with practical routes

## Data Requirements
- `bus_stops_metadata.csv`: Bus stop data with coordinates
- `distance_matrix.csv`: Distance matrix between all stops
- `BuswithRoute(APIgen).csv`: Raw data for 2-Opt implementation

## Dependencies
```julia
using CSV, DataFrames, StatsBase, Clustering, Plots, LinearAlgebra, Combinatorics, Base.Threads, Random
```

## Notes
- All implementations automatically handle depot inclusion in all clusters
- Geographic clustering reduces problem complexity
- ACO within clusters provides better optimization than simple TSP
- Cluster merging ensures balanced workload distribution
- **2-Opt optimization ensures routes are practically implementable**
- The approach scales well for larger datasets
- **New 2-Opt implementation provides the most geographically sensible routes**

## Next Steps
After exploring sophisticated hybrid approaches, compare with:
- `../vrp_solver/` - Professional baseline methods
- `../aco_solver/` - Nature-inspired optimization approaches 