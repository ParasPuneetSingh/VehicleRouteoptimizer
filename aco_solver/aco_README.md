# Ant Colony Optimization (ACO) Implementation

## Overview
`aco.jl` contains three different implementations of Ant Colony Optimization for solving the Capacitated Vehicle Routing Problem (CVRP). Each implementation offers different strategies and improvements over the previous version.

## Dependencies
```julia
using CSV, DataFrames, Random, StatsBase
```

## Implementation 1: Basic ACO with Haversine Distance

### Features
- Basic ACO algorithm implementation
- Haversine distance calculation for geographic coordinates
- Simple pheromone and heuristic management
- Fixed number of vehicles (5 buses)

### Key Components
- **Haversine Distance**: Calculates real geographic distances between coordinates
- **Pheromone Matrix**: Tracks pheromone levels between all node pairs
- **Heuristic Matrix**: Inverse distance heuristic for ant movement
- **Route Construction**: Builds routes using pheromone and heuristic information
- **Solution Splitting**: Splits complete route into individual bus routes

### Parameters
- `ALPHA = 1.0`: Pheromone importance factor
- `BETA = 5.0`: Heuristic importance factor
- `RHO = 0.1`: Evaporation rate
- `Q = 500.0`: Pheromone deposit factor
- `N_ANTS = 500`: Number of ants per iteration
- `N_ITERS = 500`: Total iterations
- `MAX_BUSES = 5`: Maximum vehicles
- `CAPACITY = 20.0`: Bus capacity

## Implementation 2: Enhanced ACO with Time Constraints

### Features
- Improved ACO with time-based constraints
- Better pheromone management
- Time matrix for route duration calculations
- Enhanced solution construction

### Key Components
- **Time Matrix**: Converts distance to time using average speed
- **Visibility Matrix**: Inverse distance for ant movement guidance
- **Enhanced Route Construction**: Considers both capacity and time constraints
- **Improved Pheromone Update**: Better pheromone trail management
- **Solution Validation**: Ensures all constraints are satisfied

### Parameters
- `num_ants = 2000`: Number of ants
- `num_iterations = 10000`: Total iterations
- `alpha = 1.0`: Pheromone importance
- `beta = 5.0`: Heuristic importance
- `evaporation_rate = 0.2`: Pheromone evaporation
- `bus_capacity = 20`: Vehicle capacity
- `max_time = 120.0`: Maximum route time in minutes
- `avg_speed = 50.0`: Average speed in km/h

## Implementation 3: Constraint-Aware ACO

### Features
- Advanced constraint handling
- Better solution validation
- Improved pheromone management
- Enhanced error handling

### Key Components
- **Constraint Validation**: Ensures solutions meet all constraints
- **Improved Route Construction**: Better handling of capacity and time limits
- **Enhanced Pheromone Update**: More sophisticated pheromone trail management
- **Solution Quality**: Better solution validation and reporting
- **Error Handling**: Robust error handling for edge cases

### Parameters
- `num_ants = 200`: Reduced number of ants for efficiency
- `num_iterations = 1000`: Reduced iterations
- `evaporation_rate = 0.5`: Higher evaporation rate
- `max_time = 150.0`: Increased time limit
- `avg_speed = 60.0`: Higher average speed

## Algorithm Comparison

| Feature | Basic ACO | Enhanced ACO | Constraint-Aware ACO |
|---------|-----------|--------------|---------------------|
| Distance Calculation | Haversine | Matrix-based | Matrix-based |
| Time Constraints | No | Yes | Yes |
| Pheromone Management | Basic | Enhanced | Advanced |
| Solution Validation | Basic | Improved | Advanced |
| Error Handling | Minimal | Moderate | Comprehensive |
| Performance | Moderate | High | High |

## Usage

### Running All Implementations
```julia
# The script runs all three implementations sequentially
# Results show optimized routes for each approach
```

### Output Format
```
=== Optimized 5-Bus Routes ===
Bus 1: depot → Stop A → Stop B → depot
Bus 2: depot → Stop C → Stop D → depot
...
Total distance: 234.7 km
Number of buses used: 5
```

## Key Functions

### `haversine(lat1, lon1, lat2, lon2)`
- Calculates geographic distance between two points
- Uses Earth's radius for accurate calculations
- Returns distance in kilometers

### `build_route(τ, η, demand, capacity, max_buses)`
- Constructs a single ant's solution
- Considers capacity and bus constraints
- Returns route and number of buses used

### `aco_cvrp(D, η, demand, capacity)`
- Main ACO algorithm
- Manages pheromone updates
- Tracks best solution found

## Performance Notes
- **Basic ACO**: Good for understanding ACO principles
- **Enhanced ACO**: Better solution quality with time constraints
- **Constraint-Aware ACO**: Most robust with comprehensive constraint handling

## Data Requirements
- `bus_stops_metadata.csv`: Bus stop data with coordinates and demand
- `distance_matrix.csv`: Distance matrix between all stops

## Notes
- All implementations use depot-first ordering
- Solutions are validated for capacity and time constraints
- Pheromone trails are symmetric (τ[i,j] = τ[j,i])
- The algorithm automatically handles depot identification and reordering 