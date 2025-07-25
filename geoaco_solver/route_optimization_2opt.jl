# =============================================================================
# Route Optimization with 2-Opt Local Search
# =============================================================================
# This implementation combines Ant Colony Optimization (ACO) with 2-Opt local search
# to create geographically sensible routes that reduce zigzag patterns.
# 
# Key Features:
# - ACO for initial route construction
# - 2-Opt local search for route smoothing
# - Geographic clustering for balanced workload
# - Intelligent route merging for exactly 5 buses
# 
# Sections:
# 1. Imports and Dependencies
# 2. Data Loading and Preprocessing
# 3. Distance and Heuristic Matrix Construction
# 4. ACO Parameters and Initialization
# 5. Ant Colony Route Construction
# 6. ACO Main Optimization Loop
# 7. 2-Opt Local Search Implementation
# 8. Route Segmentation and Optimization
# =============================================================================

# =============================================================================
# Section 1: Imports and Dependencies
# =============================================================================
using CSV, DataFrames, Random, StatsBase

# =============================================================================
# Section 2: Data Loading and Preprocessing
# =============================================================================

# Load and clean CSV data
df = CSV.read("BuswithRoute(APIgen).csv", DataFrame)

# Filter out invalid entries and route/destination headers
df = filter(row -> try
    !ismissing(row.latitude) && !ismissing(row.longitude) &&
    !startswith(row.stop_name, "Route") && !startswith(row.stop_name, "Destination")
catch
    false
end, df)

# Clean coordinate data by removing degree symbols and converting to float
df.latitude = parse.(Float64, replace.(df.latitude, "¬∞"=>""))
df.longitude = parse.(Float64, replace.(df.longitude, "¬∞"=>""))
df.num_passengers = coalesce.(df.num_passengers, 0)

# Extract cleaned data
raw_lat, raw_lon = df.latitude, df.longitude
raw_demand = Float64.(df.num_passengers)
raw_ids = df.stop_name

# =============================================================================
# Section 3: Depot Identification and Data Permutation
# =============================================================================

# Identify depot and permute so it's index 1
dest_idx = findfirst(df.is_destination .== true)
dest_idx === nothing && error("No destination found!")

n = length(raw_lat)
perm = [dest_idx; setdiff(1:n, dest_idx)]
lat, lon = raw_lat[perm], raw_lon[perm]
demand = raw_demand[perm]
ids = raw_ids[perm]

# =============================================================================
# Section 4: Distance and Heuristic Matrix Construction
# =============================================================================

# Haversine distance calculation for geographic coordinates
deg2rad(θ) = θ * π / 180

function haversine_distance(lat1, lon1, lat2, lon2)
    """
    Calculate geographic distance between two points using Haversine formula.
    
    Args:
        lat1, lon1: Coordinates of first point
        lat2, lon2: Coordinates of second point
    
    Returns:
        float: Distance in kilometers
    """
    φ1, φ2 = deg2rad(lat1), deg2rad(lat2)
    Δφ, Δλ = deg2rad(lat2 - lat1), deg2rad(lon2 - lon1)
    a = sin(Δφ/2)^2 + cos(φ1)*cos(φ2)*sin(Δλ/2)^2
    return 2*6371.0 * atan(sqrt(a), sqrt(1-a))
end

# Build distance matrix using haversine formula
D = [haversine_distance(lat[i], lon[i], lat[j], lon[j]) for i in 1:n, j in 1:n]

# Create heuristic matrix (inverse distance)
η = 1.0 ./ (D .+ eps())

# =============================================================================
# Section 5: ACO Parameters and Initialization
# =============================================================================

# ACO hyperparameters
const ALPHA = 1.0        # Pheromone importance factor
const BETA = 5.0         # Heuristic importance factor
const RHO = 0.1          # Evaporation rate
const Q = 500.0          # Pheromone deposit factor
const N_ANTS = 200       # Number of ants per iteration
const N_ITERS = 5000     # Number of iterations
const MAX_BUSES = 5      # Maximum number of vehicles
const CAPACITY = 20.0    # Vehicle capacity

# Initialize pheromone matrix
τ = ones(n, n)

# =============================================================================
# Section 6: Ant Colony Route Construction
# =============================================================================

function build_route(τ, η, demand, capacity, max_buses)
    """
    Build a single ant's route using ACO principles.
    
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
    route = [1]  # Start from depot
    load = 0.0
    buses = 1
    
    while !isempty(unvisited)
        i = route[end]
        
        # Find feasible next stops (within capacity)
        feas = [j for j in unvisited if load + demand[j] <= capacity]
        
        if isempty(feas)
            # Return to depot and start new bus
            push!(route, 1)
            buses += 1
            load = 0.0
            
            if buses > max_buses
                return route, Inf
            end
            continue
        end
        
        # Calculate selection weights using pheromone and heuristic
        w = (τ[i, feas].^ALPHA) .* (η[i, feas].^BETA)
        w ./= sum(w; init=0.0)
        
        # Select next stop probabilistically
        j = sample(feas, Weights(w))
        push!(route, j)
        delete!(unvisited, j)
        load += demand[j]
    end
    
    # Return to depot
    push!(route, 1)
    return route, buses
end

# =============================================================================
# Section 7: ACO Main Optimization Loop
# =============================================================================

function aco_cvrp(D, η, demand, capacity)
    """
    Main Ant Colony Optimization algorithm for CVRP.
    
    Args:
        D: Distance matrix
        η: Heuristic matrix
        demand: Demand at each stop
        capacity: Vehicle capacity
    
    Returns:
        tuple: (best_route, best_distance)
    """
    best_route, best_len = nothing, Inf
    
    for _ in 1:N_ITERS
        all_paths = Vector{Vector{Int}}(undef, N_ANTS)
        lengths = fill(Inf, N_ANTS)
        
        # Build routes for all ants
        for k in 1:N_ANTS
            path, bc = build_route(τ, η, demand, capacity, MAX_BUSES)
            all_paths[k] = path
            
            if bc <= MAX_BUSES
                lengths[k] = sum(D[path[i], path[i+1]] for i in 1:length(path)-1; init=0.0)
                
                if lengths[k] < best_len
                    best_route, best_len = path, lengths[k]
                end
            end
        end
        
        # Update pheromone trails
        τ .*= (1 - RHO)
        
        for (path, len) in zip(all_paths, lengths)
            if len < Inf
                Δ = Q / len
                for (a, b) in zip(path, path[2:end])
                    τ[a, b] += Δ
                    τ[b, a] += Δ
                end
            end
        end
    end
    
    return best_route, best_len
end

# =============================================================================
# Section 8: 2-Opt Local Search Implementation
# =============================================================================

function two_opt(route, dist_matrix)
    """
    Apply 2-Opt local search to improve route quality.
    
    Args:
        route: Current route
        dist_matrix: Distance matrix
    
    Returns:
        Vector{Int}: Improved route
    """
    best = copy(route)
    improved = true
    
    while improved
        improved = false
        
        for i in 2:length(route)-2
            for j in i+1:length(route)-1
                # Check if 2-opt swap improves the route
                if dist_matrix[best[i-1], best[j]] + dist_matrix[best[i], best[j+1]] <
                   dist_matrix[best[i-1], best[i]] + dist_matrix[best[j], best[j+1]]
                    
                    # Perform 2-opt swap
                    best[i:j] = reverse(best[i:j])
                    improved = true
                end
            end
        end
    end
    
    return best
end

# =============================================================================
# Section 9: Route Segmentation and Optimization
# =============================================================================

# Run ACO to get initial solution
best_route, _ = aco_cvrp(D, η, demand, CAPACITY)

if best_route === nothing
    println("❌ No feasible solution with ≤ $MAX_BUSES buses.")
else
    # Step 1: Initial segmentation at depots
    subroutes = Vector{Vector{Int}}()
    current_route = Int[]
    
    for stop in best_route[2:end]
        if stop == 1
            if !isempty(current_route)
                push!(subroutes, copy(current_route))
                empty!(current_route)
            end
        else
            push!(current_route, stop)
        end
    end
    
    # Add remaining route if not empty
    isempty(current_route) || push!(subroutes, copy(current_route))

    # Step 2: Merge tiny clusters (<3 stops) back to larger routes
    tiny_routes, filtered_routes = Vector{Vector{Int}}(), Vector{Vector{Int}}()
    
    for route in subroutes
        length(route) < 3 ? push!(tiny_routes, route) : push!(filtered_routes, route)
    end
    
    # Merge tiny routes to nearest larger route
    for tiny_route in tiny_routes
        tiny_points = [(lat[x], lon[x]) for x in tiny_route]
        tiny_centroid = (mean(first.(tiny_points)), mean(last.(tiny_points)))
        
        distances = [haversine_distance(tiny_centroid[1], tiny_centroid[2], 
                                      mean(lat[route]), mean(lon[route])) 
                    for route in filtered_routes]
        
        nearest_idx = argmin(distances)
        append!(filtered_routes[nearest_idx], tiny_route)
    end
    
    subroutes = filtered_routes

    # Step 3: Merge routes until exactly MAX_BUSES remain
    while length(subroutes) > MAX_BUSES
        # Compute centroids for all routes
        centroids = [(mean(lat[route]), mean(lon[route])) for route in subroutes]
        
        # Find the two closest routes
        best_distance, best_i, best_j = Inf, 0, 0
        
        for i in eachindex(centroids)
            for j in i+1:length(centroids)
                distance = haversine_distance(centroids[i]..., centroids[j]...)
                if distance < best_distance
                    best_distance, best_i, best_j = distance, i, j
                end
            end
        end
        
        # Merge route j into route i
        append!(subroutes[best_i], subroutes[best_j])
        deleteat!(subroutes, best_j)
    end

    # Step 4: Calculate route lengths and apply 2-Opt where beneficial
    route_lengths = [sum(D[route[k], route[k+1]] for k in 1:length(route)-1; init=0.0) 
                     for route in subroutes]
    average_length = mean(route_lengths)

    # Step 5: Print optimized routes with conditional 2-Opt application
    println("\n🚍 Final 5-Bus Network (Geographically Optimized):")
    total_distance = 0.0
    
    for (i, route) in enumerate(subroutes)
        full_route = vcat(1, route, 1)
        
        if route_lengths[i] > average_length
            # Apply 2-Opt to longer routes for improvement
            full_route = two_opt(full_route, D)
            optimization_note = " (2-Opt optimized)"
        else
            optimization_note = ""
        end
        
        # Extract stop names for display
        stop_names = ids[full_route[2:end-1]]
        route_distance = sum(D[full_route[k], full_route[k+1]] for k in 1:length(full_route)-1; init=0.0)
        total_distance += route_distance
        
        println("Bus $i: depot → ", join(stop_names, " → "), " → depot | Distance: $(round(route_distance, digits=1)) km$optimization_note")
    end

    println("\n📏 Total fleet distance: ", round(total_distance, digits=1), " km")
    println("🎯 Geographic optimization reduces zigzag patterns for more practical routes")
end