# ACO Solver - Nature-Inspired Optimization Approaches

## Overview
This folder contains Ant Colony Optimization (ACO) implementations for solving the Capacitated Vehicle Routing Problem (CVRP) using nature-inspired metaheuristic algorithms.

## Purpose
**Phase 2: Innovation - Nature-Inspired Algorithms**
- Explore nature-inspired optimization techniques for better solution quality
- Implement adaptive learning mechanisms through pheromone trails
- Improve solution quality through metaheuristics
- Demonstrate the power of bio-inspired algorithms

## Contents

### Files
- `aco.jl` - Main implementation file containing three ACO variants
- `aco_README.md` - Detailed documentation for the implementations

### Implementations

#### 1. Basic ACO with Haversine Distance
- **Approach**: Basic ACO with geographic distance calculation
- **Features**: Simple pheromone management, haversine distance, fixed vehicle count
- **Use Case**: Understanding ACO principles

#### 2. Enhanced ACO with Time Constraints
- **Approach**: Improved ACO with time-based constraints
- **Features**: Time matrix, better pheromone management, enhanced solution construction
- **Use Case**: Real-world constraint handling

#### 3. Constraint-Aware ACO
- **Approach**: Advanced ACO with comprehensive constraint handling
- **Features**: Better solution validation, enhanced error handling, robust pheromone management
- **Use Case**: Production-ready optimization

## Key Characteristics
- **Complexity**: Medium
- **Innovation**: High (nature-inspired)
- **Solution Quality**: Very High
- **Computation Time**: Moderate
- **Adaptability**: Excellent

## Usage
```bash
cd aco_solver
julia aco.jl
```

## Dependencies
```julia
using CSV, DataFrames, Random, StatsBase
```

## Expected Output
- Optimized routes for each vehicle
- Total distance traveled
- Number of vehicles used
- ACO performance metrics and convergence

## Algorithm Features
- **Pheromone Trails**: Chemical communication simulation
- **Heuristic Guidance**: Distance-based decision making
- **Adaptive Learning**: Solution quality influences future decisions
- **Constraint Satisfaction**: Capacity and time constraints

## Next Steps
After exploring nature-inspired optimization, explore:
- `../geoaco_solver/` - Hybrid geographic optimization methods
- `../vrp_solver/` - Professional baseline methods 