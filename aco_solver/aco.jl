# =============================================================================
# Ant Colony Optimization (ACO) Implementation
# =============================================================================
# This file contains three different implementations of Ant Colony Optimization
# for solving the Capacitated Vehicle Routing Problem (CVRP):
# 
# 1. Basic ACO with Haversine Distance
# 2. Enhanced ACO with Time Constraints
# 3. Constraint-Aware ACO with Advanced Features
# 
# Sections:
# 1. Imports and Dependencies
# 2. Implementation 1: Basic ACO with Haversine Distance
# 3. Implementation 2: Enhanced ACO with Time Constraints
# 4. Implementation 3: Constraint-Aware ACO
# =============================================================================

# =============================================================================
# Section 1: Imports and Dependencies
# =============================================================================
using CSV, DataFrames, Random, StatsBase

# =============================================================================
# Section 2: Implementation 1 - Basic ACO with Haversine Distance
# =============================================================================
# Basic ACO implementation with geographic distance calculation
# Features: Simple pheromone management, haversine distance, fixed vehicle count

# --- 1. Load raw data ---
df = CSV.read("bus_stops_metadata.csv", DataFrame)
raw_lat = df.latitude
raw_lon = df.longitude
raw_demand = Float64.(df.num_passengers)
raw_ids = df.stop_name

# --- 2. Identify depot (zero demand) and permute ---
dest_idx = findfirst(df.is_destination .== true)
dest_idx === nothing && error("No destination stop found!")
n = length(raw_lat)
perm = [dest_idx; setdiff(1:n, dest_idx)]
lat = raw_lat[perm]
lon = raw_lon[perm]
demand = raw_demand[perm]
ids = raw_ids[perm]

# --- 3. Haversine distance matrix (Float64 km) ---
deg2rad(θ) = θ * π / 180
function haversine(lat1, lon1, lat2, lon2)
    """
    Calculate geographic distance between two points using Haversine formula.
    
    Args:
        lat1, lon1: Coordinates of first point
        lat2, lon2: Coordinates of second point
    
    Returns:
        float: Distance in kilometers
    """
    r = 6371.0  # Earth's radius in km
    φ1, φ2 = deg2rad(lat1), deg2rad(lat2)
    Δφ, Δλ = deg2rad(lat2 - lat1), deg2rad(lon2 - lon1)
    a = sin(Δφ/2)^2 + cos(φ1)*cos(φ2)*sin(Δλ/2)^2
    c = 2 * atan(sqrt(a), sqrt(1-a))
    return r * c
end

D = [haversine(lat[i], lon[i], lat[j], lon[j]) for i in 1:n, j in 1:n]
η = 1.0 ./ (D .+ eps()) # heuristic (inverse distance)

# --- 4. ACO parameters ---
const ALPHA = 1.0 # pheromone importance
const BETA = 5.0 # heuristic importance
const RHO = 0.1 # evaporation rate
const Q = 500.0 # pheromone deposit factor
const N_ANTS = 500 # number of ants per iteration
const N_ITERS = 500 # total iterations
const MAX_BUSES = 5 # maximum vehicles
const CAPACITY = 20.0 # bus capacity

# initialize pheromone matrix
τ = fill(1.0, n, n)

# --- 5. Build a single ant's solution with bus-limit check ---
function build_route(τ, η, demand, capacity, max_buses)
    """
    Build a single ant's route considering capacity and bus constraints.
    
    Args:
        τ: Pheromone matrix
        η: Heuristic matrix
        demand: Demand at each stop
        capacity: Vehicle capacity
        max_buses: Maximum number of vehicles
    
    Returns:
        tuple: (route, number_of_buses_used)
    """
    unvisited = Set(2:n)
    route = [1]
    load = 0.0
    buses = 1
    while !isempty(unvisited)
        i = route[end]
        feas = [j for j in unvisited if load + demand[j] ≤ capacity]
        if isempty(feas)
            push!(route, 1)
            buses += 1
            if buses > max_buses
                return route, Inf
            end
            load = 0.0
            continue
        end
        weights = [(τ[i,j]^ALPHA) * (η[i,j]^BETA) for j in feas]
        weights ./= sum(weights)
        j = sample(feas, Weights(weights))
        push!(route, j)
        delete!(unvisited, j)
        load += demand[j]
    end
    push!(route, 1)
    return route, buses
end

# --- 6. Main ACO routine ---
function aco_cvrp(D, η, demand, capacity; n_ants=N_ANTS, n_iters=N_ITERS,
max_buses=MAX_BUSES)
    """
    Main Ant Colony Optimization algorithm for CVRP.
    
    Args:
        D: Distance matrix
        η: Heuristic matrix
        demand: Demand at each stop
        capacity: Vehicle capacity
        n_ants: Number of ants per iteration
        n_iters: Number of iterations
        max_buses: Maximum number of vehicles
    
    Returns:
        tuple: (best_route, best_distance)
    """
    best_route = nothing
    best_len = Inf
    for iter in 1:n_iters
        all_routes = Vector{Vector{Int}}(undef, n_ants)
        lengths = fill(0.0, n_ants)
        for k in 1:n_ants
            rk, buses = build_route(τ, η, demand, capacity, max_buses)
            all_routes[k] = rk
            if buses ≤ max_buses && rk !== nothing
                lengths[k] = sum(D[rk[i], rk[i+1]] for i in 1:length(rk)-1)
            else
                lengths[k] = Inf
            end
            if lengths[k] < best_len
                best_len = lengths[k]
                best_route = rk
            end
        end
        τ .*= (1 - RHO)
        for k in 1:n_ants
            if lengths[k] < Inf
                Δτ = Q / lengths[k]
                rt = all_routes[k]
                for (i,j) in zip(rt[1:end-1], rt[2:end])
                    τ[i,j] += Δτ
                    τ[j,i] += Δτ
                end
            end
        end
        @info "Iter $iter → best dist: $(round(best_len; digits=1)) km"
    end
    return best_route, best_len
end

# --- 7. Run ACO and split into per-bus routes ---
best_route, best_distance = aco_cvrp(D, η, demand, CAPACITY)
# split by depot (1) markers
bus_routes = [[]]
for stop in best_route[2:end] # skip initial depot
    if stop == 1
        push!(bus_routes, [])
    else
        push!(bus_routes[end], stop)
    end
end

# --- 8. Print each bus's route ---
println("\n=== Optimized 5-Bus Routes ===")
for (k, route_stops) in enumerate(bus_routes)
    names = [ids[s] for s in route_stops]
    println("Bus $(k): depot → ", join(names, " → "), " → depot")
end
println("\nTotal distance: ", round(best_distance; digits=1), " km")
println("Number of buses used: ", length(bus_routes))

# =============================================================================
# Section 3: Implementation 2 - Enhanced ACO with Time Constraints
# =============================================================================
# Enhanced ACO with time-based constraints and improved pheromone management
# Features: Time matrix, better pheromone management, enhanced solution construction

##################################

using CSV, DataFrames, StatsBase

# ===============================
# Load Data
# ===============================
df_metadata = CSV.read("bus_stops_metadata.csv", DataFrame)
df_raw = CSV.read("distance_matrix.csv", DataFrame; header=false)
dist_matrix = Matrix{Float64}(df_raw)

# Ensure square matrix
if size(dist_matrix, 1) != size(dist_matrix, 2)
    @views last_col = dist_matrix[:, end]
    dist_matrix = hcat(dist_matrix, last_col)
end

@assert size(dist_matrix, 1) == size(dist_matrix, 2) "Distance matrix must be square"
n_nodes = size(dist_matrix, 1)

# Load and align demands
demands = Int64.(df_metadata.num_passengers)
if length(demands) < n_nodes
    demands = vcat(0, demands)  # depot has 0 demand
end

stop_names = vcat("Depot", String.(df_metadata.stop_name))

function preprocess_distance_matrix!(dist_mat::Matrix{Float64})
    @inbounds for i in 1:size(dist_mat, 1)
        dist_mat[i, i] = Inf
    end
    return dist_mat
end

dist_matrix = preprocess_distance_matrix!(dist_matrix)

# ===============================
# Parameters
# ===============================
num_ants = 2000
num_iterations = 10000
alpha = 1.0
beta = 5.0
evaporation_rate = 0.2
Q = 500.0

bus_capacity = 20
max_time = 120.0  # minutes
avg_speed = 50.0  # km/h
time_matrix = dist_matrix ./ avg_speed .* 60.0  # minutes

# ===============================
# ACO Functions
# ===============================
function calc_visibility(dist_mat::Matrix{Float64})
    visibility = 1.0 ./ dist_mat
    visibility[isinf.(visibility)] .= 0.0
    return visibility
end

function select_next_city(current, unvisited, pheromone, visibility)
    @inbounds begin
        weights = similar(unvisited, Float64)
        for (i, j) in enumerate(unvisited)
            weights[i] = (pheromone[current, j]^alpha) * (visibility[current, j]^beta)
        end
        total = sum(weights)
        if total == 0.0
            return rand(unvisited)
        end
        weights ./= total
        return unvisited[sample(1:length(unvisited), Weights(weights))]
    end
end

function construct_solution()
    max_buses = 5
    routes = [ [1] for _ in 1:max_buses ]  # depot at start
    capacities = fill(bus_capacity, max_buses)
    times = fill(max_time, max_buses)
    current_pos = fill(1, max_buses)  # depot index
    visited = falses(n_nodes)
    visited[1] = true  # depot

    unassigned = Set(2:n_nodes)

    while !isempty(unassigned)
        progress = false
        for stop in unassigned
            assigned = false
            for i in 1:max_buses
                cap_ok = demands[stop] <= capacities[i]
                time_needed = time_matrix[current_pos[i], stop] + time_matrix[stop, 1]  # include return to depot
                time_ok = time_needed <= times[i]

                if cap_ok && time_ok
                    push!(routes[i], stop)
                    capacities[i] -= demands[stop]
                    times[i] -= time_matrix[current_pos[i], stop]
                    current_pos[i] = stop
                    visited[stop] = true
                    assigned = true
                    progress = true
                    break
                end
            end
            if assigned
                delete!(unassigned, stop)
            end
        end

        if !progress
            # Not able to assign remaining customers with current constraints
            break
        end
    end

    # Close all routes by returning to depot
    for i in 1:max_buses
        push!(routes[i], 1)
    end

    return routes
end


function total_distance(routes)
    visited = Set{Int}()
    total = 0.0
    for route in routes
        for i in 1:length(route)-1
            total += dist_matrix[route[i], route[i+1]]
            push!(visited, route[i])
        end
        push!(visited, route[end])
    end
    # Penalize unvisited stops heavily
    if length(visited) < n_nodes
        total += 1e5 * (n_nodes - length(visited))
    end
    return total
end


function update_pheromones!(pheromone, routes_list, dists)
    pheromone .*= (1 - evaporation_rate)
    @inbounds for (routes, dist) in zip(routes_list, dists)
        contrib = Q / dist
        for route in routes
            @inbounds for i in 1:length(route)-1
                a, b = route[i], route[i+1]
                pheromone[a, b] += contrib
                pheromone[b, a] += contrib
            end
        end
    end
end

# ===============================
# Main ACO Loop
# ===============================
pheromone = fill(1.0, n_nodes, n_nodes)
visibility = calc_visibility(dist_matrix)

best_routes = nothing
best_cost = Inf

for iter in 1:num_iterations
    routes_list = Vector{Vector{Vector{Int}}}(undef, num_ants)
    dists = Vector{Float64}(undef, num_ants)

    @inbounds for ant in 1:num_ants
        routes = construct_solution()
        routes_list[ant] = routes
        dists[ant] = total_distance(routes)
    end

    update_pheromones!(pheromone, routes_list, dists)

    min_idx = argmin(dists)
    if dists[min_idx] < best_cost
        best_cost = dists[min_idx]
        best_routes = routes_list[min_idx]
    end
    println("Iteration $iter: best cost so far = $(round(best_cost, digits=2)) with $(length(best_routes)) buses")
end

# ===============================
# Print Final Routes
# ===============================
println("\nFinal Best Routes (Total Distance = $(round(best_cost, digits=2)) km):")
for (i, route) in enumerate(best_routes)
    if length(route) > 2  # only print non-empty routes (more than depot → depot)
        names = join([stop_names[node] for node in route], " → ")
        println("Bus $(i): $names")
    end
end

# =============================================================================
# Section 4: Implementation 3 - Constraint-Aware ACO
# =============================================================================
# Advanced ACO with comprehensive constraint handling and improved validation
# Features: Better solution validation, enhanced error handling, robust pheromone management

#updated constraints code
#########################
using CSV, DataFrames, StatsBase

# ===============================
# Load Data
# ===============================
df_metadata = CSV.read("bus_stops_metadata.csv", DataFrame)
df_raw = CSV.read("distance_matrix.csv", DataFrame; header=false)
dist_matrix = Matrix{Float64}(df_raw)

# Ensure square matrix
if size(dist_matrix, 1) != size(dist_matrix, 2)
    @views last_col = dist_matrix[:, end]
    dist_matrix = hcat(dist_matrix, last_col)
end

@assert size(dist_matrix, 1) == size(dist_matrix, 2) "Distance matrix must be square"
n_nodes = size(dist_matrix, 1)

# Load and align demands
demands = Int64.(df_metadata.num_passengers)
if length(demands) < n_nodes
    demands = vcat(0, demands)  # depot has 0 demand
end

stop_names = vcat("Depot", String.(df_metadata.stop_name))

function preprocess_distance_matrix!(dist_mat::Matrix{Float64})
    @inbounds for i in 1:size(dist_mat, 1)
        dist_mat[i, i] = Inf
    end
    return dist_mat
end

dist_matrix = preprocess_distance_matrix!(dist_matrix)

# ===============================
# Parameters
# ===============================
num_ants = 200
num_iterations = 1000
alpha = 1.0
beta = 5.0
evaporation_rate = 0.5
Q = 500.0

bus_capacity = 20
max_time = 150.0  # minutes
avg_speed = 60.0  # km/h
time_matrix = dist_matrix ./ avg_speed .* 60.0  # minutes

# ===============================
# ACO Functions
# ===============================
function calc_visibility(dist_mat::Matrix{Float64})
    visibility = 1.0 ./ dist_mat
    visibility[isinf.(visibility)] .= 0.0
    return visibility
end

function select_next_city(current, unvisited, pheromone, visibility)
    @inbounds begin
        weights = similar(unvisited, Float64)
        for (i, j) in enumerate(unvisited)
            weights[i] = (pheromone[current, j]^alpha) * (visibility[current, j]^beta)
        end
        total = sum(weights)
        if total == 0.0
            return rand(unvisited)
        end
        weights ./= total
        return unvisited[sample(1:length(unvisited), Weights(weights))]
    end
end

function construct_solution()
    routes = Vector{Vector{Int}}()
    visited = falses(n_nodes)
    visited[1] = true  # depot
    bus_count = 0

    while count(!, visited) > 1
        if bus_count >= 5
            return nothing  # invalid solution
        end

        route = [1]
        cap_left = bus_capacity
        time_left = max_time
        current = 1

        while true
            candidates = Int[]
            @inbounds for j in 2:n_nodes
                if !visited[j] && demands[j] <= cap_left &&
                   (time_matrix[current, j] + time_matrix[j, 1] <= time_left)
                    push!(candidates, j)
                end
            end

            if isempty(candidates)
                break
            end

            next = select_next_city(current, candidates, pheromone, visibility)
            push!(route, next)
            visited[next] = true
            cap_left -= demands[next]
            time_left -= time_matrix[current, next]
            current = next
        end

        push!(route, 1)
        push!(routes, route)
        bus_count += 1
    end

    return routes
end

function total_distance(routes)
    if routes === nothing
        return Inf
    end
    total = 0.0
    @inbounds for route in routes
        @inbounds for i in 1:length(route)-1
            total += dist_matrix[route[i], route[i+1]]
        end
    end
    return total
end

function update_pheromones!(pheromone, routes_list, dists)
    pheromone .*= (1 - evaporation_rate)
    @inbounds for (routes, dist) in zip(routes_list, dists)
        if routes === nothing || isinf(dist)
            continue
        end
        contrib = Q / dist
        for route in routes
            @inbounds for i in 1:length(route)-1
                a, b = route[i], route[i+1]
                pheromone[a, b] += contrib
                pheromone[b, a] += contrib
            end
        end
    end
end

# ===============================
# Main ACO Loop
# ===============================
pheromone = fill(1.0, n_nodes, n_nodes)
visibility = calc_visibility(dist_matrix)

best_routes = nothing
best_cost = Inf

for iter in 1:num_iterations
    routes_list = Vector{Union{Nothing, Vector{Vector{Int}}}}(undef, num_ants)
    dists = fill(Inf, num_ants)

    @inbounds for ant in 1:num_ants
        routes = construct_solution()
        routes_list[ant] = routes
        dists[ant] = total_distance(routes)
    end

    update_pheromones!(pheromone, routes_list, dists)

    min_idx = argmin(dists)
    if dists[min_idx] < best_cost
        best_cost = dists[min_idx]
        best_routes = routes_list[min_idx]
    end
    if best_routes === nothing
        println("\n❌ No valid route found using only 5 buses. Try relaxing constraints or increasing iterations.")
    else
        println("\n✅ Final Best Routes (Total Distance = $(round(best_cost, digits=2)) km):")
        for (i, route) in enumerate(best_routes)
            names = join([stop_names[node] for node in route], " → ")
            println("Bus $(i): $names")
        end
    end

end

# ===============================
# Print Final Routes
# ===============================
println("\nFinal Best Routes (Total Distance = $(round(best_cost, digits=2)) km):")
for (i, route) in enumerate(best_routes)
    names = join([stop_names[node] for node in route], " → ")
    println("Bus $(i): $names")
end
