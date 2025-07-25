# =============================================================================
# VRP Solver Implementation
# =============================================================================
# This file contains two professional-grade implementations for solving the
# Capacitated Vehicle Routing Problem (CVRP):
# 
# 1. Hygese Solver: Uses CVRPLIB and Hygese libraries
# 2. OR-Tools Solver: Uses Google's OR-Tools constraint programming
# 
# Sections:
# 1. Imports and Dependencies
# 2. Hygese Solver Implementation
# 3. OR-Tools Solver Implementation
# =============================================================================

# =============================================================================
# Section 1: Imports and Dependencies
# =============================================================================
using CSV, DataFrames
using CVRPLIB, Hygese

# =============================================================================
# Section 2: Hygese Solver Implementation
# =============================================================================
# Professional-grade CVRP solver using CVRPLIB and Hygese libraries
# Features: Time limits, capacity constraints, distance optimization

# --- 1. Load Cleaned Data ---
meta = CSV.read("bus_stops_metadata.csv", DataFrame)
D = Matrix(CSV.read("distance_matrix.csv", DataFrame)) |> x -> round.(Int, x)  # convert to Int matrix

lat = meta.latitude
lon = meta.longitude
demand = round.(Int, meta.num_passengers)  # convert demand to integers

n = length(lat)
ids = meta.stop_name

# --- 2. Find Depot ---
depot_idx = findfirst(meta.is_destination .== true)
depot_idx === nothing && error("No depot found in metadata!")

# --- 3. Reorder so depot is first ---
perm = [depot_idx; collect(1:depot_idx-1); collect(depot_idx+1:n)]
lat = lat[perm]
lon = lon[perm]
demand = demand[perm]
ids = ids[perm]
D = D[perm, perm]  # permute distance matrix

# --- 4. Check depot demand ---
demand[1] == 0 || error("Depot must have zero demand!")

# --- 5. Create CVRP instance ---
capacity = 20  # minibus capacity
max_speed = 40
max_time = 2.5
max_dist = max_speed * max_time  # max distance in km (or consistent with D units)
cvrp = CVRPLIB.CVRP(
    "minibus_routes",
    n,
    "EXPLICIT",         # explicitly provided distance matrix
    D,
    capacity,
    max_dist,
    0.0,
    hcat(lat, lon),
    demand,
    1,
    0,
    collect(2:n)
)

# --- 6. Solve using Hygese ---
params = AlgorithmParameters(timeLimit=200.0, seed=42)
result = solve_cvrp(cvrp, params; verbose=true)

# --- 7. Show Results ---
println("\n🚐 Optimized Minibus Routes (ending at depot: '$(ids[1])'):")
for (k, route) in enumerate(result.routes)
    names = ids[route]
    d = sum(demand[route])
    dist = 0.0
    for i in 1:length(route)-1
        dist += D[route[i], route[i+1]]
    end
    dist += D[route[end], route[1]]  # return to depot
    println("Bus $(k): ", join(names, " → "), " (Demand: $(d), Distance: $(round(dist, digits=2)) km)")
end

println("\n🚌 Buses used: ", length(result.routes))
println("🛣️  Total road distance: $(round(result.cost, digits=2)) km")
println("⏱️  Solver time: $(round(result.time, digits=2)) seconds")

# =============================================================================
# Section 3: OR-Tools Solver Implementation
# =============================================================================
# Google's OR-Tools constraint programming approach
# Features: Vehicle minimization, guided local search, fixed cost penalties

# Solution for 5 buses no time constraint
using PyCall

# Import OR-Tools via PyCall with conda-forge channel
pyimport_conda("ortools", "ortools", "conda-forge")
ortools = pyimport("ortools.constraint_solver")
routing_module = pyimport("ortools.constraint_solver.pywrapcp")
routing_enums = pyimport("ortools.constraint_solver.routing_enums_pb2")

strategy_enum = routing_enums.FirstSolutionStrategy
local_enum = routing_enums.LocalSearchMetaheuristic

# --- Load Data ---
meta = CSV.read("bus_stops_metadata.csv", DataFrame)
D = Matrix(CSV.read("distance_matrix.csv", DataFrame))

# Extract demand and depot
demand = Int.(round.(meta.num_passengers))
depot_idx = findfirst(meta.is_destination .== true)

# --- Reorder depot to be first node ---
perm = vcat(depot_idx, setdiff(1:size(D, 1), [depot_idx]))
D = D[perm, perm]
demand = demand[perm]

# --- Constants ---
n = length(demand)
capacity = 20

# --- Main Solver Loop (Minimize # of Buses) ---
function solve_cvrp(max_buses::Int)
    """
    Solve CVRP using OR-Tools with vehicle minimization.
    
    Args:
        max_buses: Maximum number of vehicles to try
    """
    for num_vehicles in max_buses:-1:1
        println("\n🔍 Trying with $num_vehicles buses...")

        manager = routing_module.RoutingIndexManager(n, num_vehicles, 0)
        routing = routing_module.RoutingModel(manager)

        # Distance Callback
        distance_callback(from_index::Int, to_index::Int) = begin
            from_node = manager[:IndexToNode](from_index)
            to_node = manager[:IndexToNode](to_index)
            return Int(round(D[from_node + 1, to_node + 1]))
        end

        dist_idx = routing[:RegisterTransitCallback](distance_callback)
        routing[:SetArcCostEvaluatorOfAllVehicles](dist_idx)

        # Demand Callback
        demand_callback(from_index::Int) = begin
            from_node = manager[:IndexToNode](from_index)
            return demand[from_node + 1]
        end

        demand_idx = routing[:RegisterUnaryTransitCallback](demand_callback)

        routing[:AddDimensionWithVehicleCapacity](
            demand_idx, 0, fill(capacity, num_vehicles), true, "Capacity"
        )

        # Penalize vehicle usage
        for v in 0:num_vehicles-1
            routing[:SetFixedCostOfVehicle](30000, v)
        end

        # Solver Parameters
        search_params = routing_module[:DefaultRoutingSearchParameters]()
        search_params[:first_solution_strategy] = strategy_enum[:PATH_CHEAPEST_ARC]
        search_params[:local_search_metaheuristic] = local_enum[:GUIDED_LOCAL_SEARCH]
        search_params[:time_limit][:seconds] = 60

        solution = routing[:SolveWithParameters](search_params)
        if solution !== nothing
            print_solution(routing, manager, solution, num_vehicles)
            return
        else
            println("❌ No solution found with $num_vehicles buses.")
        end
    end
    println("⚠️ Could not solve with fewer than $max_buses buses.")
end

# --- Output Function ---
function print_solution(routing, manager, solution, num_vehicles)
    """
    Print the optimized solution with route details.
    
    Args:
        routing: OR-Tools routing model
        manager: Routing index manager
        solution: Found solution
        num_vehicles: Number of vehicles used
    """
    println("\n🚐 OR-Tools CVRP Result:")
    buses_used = 0
    for vehicle_id in 0:num_vehicles-1
        index = routing[:Start](vehicle_id)
        route = []
        total_load = 0
        total_dist = 0

        while !routing[:IsEnd](index)
            node_index = manager[:IndexToNode](index)
            stop_name = meta.stop_name[perm[node_index + 1]]
            push!(route, stop_name)
            total_load += demand[node_index + 1]

            prev_index = index
            index = solution[:Value](routing[:NextVar](prev_index))

            # Only get arc cost if next is not end
            if !routing[:IsEnd](index)
                try
                    total_dist += routing[:GetArcCostForVehicle](prev_index, index, vehicle_id)
                catch err
                    @warn "Error getting arc cost: $err (vehicle $vehicle_id, from $prev_index to $index)"
                    total_dist += 0  # Fallback to 0
                end
            end
        end
        push!(route, meta.stop_name[perm[1]])  # depot name

        if total_load > 0
            buses_used += 1
            println("🚌 Bus $(vehicle_id + 1): ", join(route, " → "),
                    " | Load: $total_load | Distance: $total_dist")
        end
    end
    println("\n✅ Total buses used: $buses_used")
end

# --- Run Solver ---
solve_cvrp(5)  # Start trying from 5 buses and go downward
solve_cvrp(4)  # Try with 4 buses
