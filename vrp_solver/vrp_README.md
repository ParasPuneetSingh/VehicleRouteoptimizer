# VRP Solver Implementation

## Overview
`vrp.jl` contains two professional-grade implementations for solving the Capacitated Vehicle Routing Problem (CVRP):

1. **Hygese Solver**: Uses the CVRPLIB and Hygese libraries
2. **OR-Tools Solver**: Uses Google's OR-Tools constraint programming

## Dependencies
```julia
using CSV, DataFrames, CVRPLIB, Hygese, PyCall
```

## Implementation 1: Hygese Solver

### Features
- Professional-grade CVRP solver
- Time limit constraints (200 seconds)
- Capacity constraints (20 passengers per vehicle)
- Distance optimization
- Explicit distance matrix support

### Key Components
- **Data Loading**: Loads bus stop metadata and distance matrix
- **Depot Identification**: Finds and reorders depot to first position
- **CVRP Instance Creation**: Creates CVRPLIB.CVRP instance with constraints
- **Solver Execution**: Uses Hygese with algorithm parameters
- **Result Display**: Shows optimized routes with distances and demand

### Parameters
- `capacity = 20`: Maximum passengers per vehicle
- `max_speed = 40`: Maximum speed in km/h
- `max_time = 2.5`: Maximum route time in hours
- `timeLimit = 200.0`: Solver time limit in seconds

## Implementation 2: OR-Tools Solver

### Features
- Google's OR-Tools constraint programming
- Vehicle minimization approach
- Guided local search metaheuristic
- Fixed cost penalties for vehicle usage

### Key Components
- **PyCall Integration**: Uses PyCall to interface with OR-Tools
- **Distance Callback**: Custom distance evaluation function
- **Demand Callback**: Capacity constraint implementation
- **Vehicle Minimization**: Iteratively reduces number of vehicles
- **Solution Validation**: Ensures all constraints are satisfied

### Parameters
- `capacity = 20`: Vehicle capacity
- `fixed_cost = 30000`: Penalty for using each vehicle
- `time_limit = 60`: Solver time limit per attempt
- `max_buses = 5`: Maximum number of vehicles to try

## Usage

### Running Hygese Solver
```julia
# The script automatically runs the Hygese solver
# Results show optimized routes with distances and demand
```

### Running OR-Tools Solver
```julia
# The script includes two OR-Tools runs:
solve_cvrp(5)  # Try with 5 buses
solve_cvrp(4)  # Try with 4 buses
```

## Output Format

### Hygese Output
```
🚐 Optimized Minibus Routes (ending at depot: 'Mondelez International, Pune'):
Bus 1: Mondelez International → Stop A → Stop B → Mondelez International (Demand: 15, Distance: 45.2 km)
...
🚌 Buses used: 5
🛣️  Total road distance: 234.7 km
⏱️  Solver time: 12.3 seconds
```

### OR-Tools Output
```
🚐 OR-Tools CVRP Result:
🚌 Bus 1: Mondelez International → Stop A → Stop B → Mondelez International | Load: 15 | Distance: 45.2
...
✅ Total buses used: 5
```

## Algorithm Comparison

| Feature | Hygese | OR-Tools |
|---------|--------|----------|
| Approach | Professional solver | Constraint programming |
| Vehicle minimization | No | Yes |
| Time constraints | Yes | Yes |
| Capacity constraints | Yes | Yes |
| Solution quality | High | High |
| Speed | Fast | Moderate |

## Data Requirements
- `bus_stops_metadata.csv`: Must contain columns: stop_name, latitude, longitude, num_passengers, is_destination
- `distance_matrix.csv`: Square matrix of distances between all stops

## Notes
- Both implementations reorder data so depot is at index 1
- OR-Tools implementation includes fallback error handling
- Hygese provides more detailed solution statistics
- OR-Tools focuses on minimizing the number of vehicles used 