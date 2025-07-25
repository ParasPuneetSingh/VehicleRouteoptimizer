# VRP Solver - Professional Optimization Approaches

## Overview
This folder contains professional-grade implementations for solving the Capacitated Vehicle Routing Problem (CVRP) using established optimization libraries and methods.

## Purpose
**Phase 1: Foundation - Established Methods**
- Establish baseline performance using proven, reliable methods
- Understand problem constraints and requirements
- Provide reference implementations for comparison

## Contents

### Files
- `vrp.jl` - Main implementation file containing two professional solvers
- `vrp_README.md` - Detailed documentation for the implementations

### Implementations

#### 1. Hygese Solver
- **Library**: CVRPLIB and Hygese
- **Approach**: Professional-grade CVRP solver
- **Features**: Time limits, capacity constraints, distance optimization
- **Use Case**: Reliable baseline performance

#### 2. OR-Tools Solver
- **Library**: Google's OR-Tools
- **Approach**: Constraint programming with vehicle minimization
- **Features**: Vehicle minimization, guided local search, fixed cost penalties
- **Use Case**: Advanced constraint handling

## Key Characteristics
- **Complexity**: Low
- **Reliability**: High
- **Speed**: Fast
- **Innovation Level**: Established methods
- **Scalability**: Good

## Usage
```bash
cd vrp_solver
julia vrp.jl
```

## Dependencies
```julia
using CSV, DataFrames, CVRPLIB, Hygese, PyCall
```

## Expected Output
- Optimized routes for each vehicle
- Total distance traveled
- Number of vehicles used
- Solver performance metrics

## Next Steps
After understanding the problem with these professional solvers, explore:
- `../aco_solver/` - Nature-inspired optimization approaches
- `../geoaco_solver/` - Hybrid geographic optimization methods 