# =============================================================================
# Geographic ACO Implementation
# =============================================================================
# This file implements a hybrid approach combining K-means geographic clustering
# with Ant Colony Optimization (ACO) for solving the Capacitated Vehicle Routing Problem (CVRP).
# 
# Three implementations:
# 1. Basic Geographic Clustering with TSP
# 2. Advanced Clustering with ACO
# 3. Balanced Clustering with ACO
# 
# Sections:
# 1. Imports and Dependencies
# 2. Implementation 1: Basic Geographic Clustering with TSP
# 3. Implementation 2: Advanced Clustering with ACO
# 4. Implementation 3: Balanced Clustering with ACO
# =============================================================================

# =============================================================================
# Section 1: Imports and Dependencies
# =============================================================================
using CSV, DataFrames, StatsBase, Clustering, Plots, LinearAlgebra

# =============================================================================
# Section 2: Implementation 1 - Basic Geographic Clustering with TSP
# =============================================================================
# Simple K-means clustering with nearest neighbor TSP for each cluster
# Features: Basic clustering, TSP optimization, visualization

# ===============================
# Load Data
# ===============================
df_metadata = CSV.read("bus_stops_metadata.csv", DataFrame)
df_raw = CSV.read("distance_matrix.csv", DataFrame; header=false)
dist_matrix = Matrix{Float64}(df_raw)

# Ensure square matrix
if size(dist_matrix, 1) != size(dist_matrix, 2)
    dist_matrix = hcat(dist_matrix, last_col)
end

n_nodes = size(dist_matrix, 1)
demands = Int64.(df_metadata.num_passengers)
demands = vcat(0, demands)
stop_names = vcat("Depot", String.(df_metadata.stop_name))

function preprocess_distance_matrix!(dist_mat::Matrix{Float64})
    for i in 1:size(dist_mat, 1)
        dist_mat[i, i] = Inf
    end
    return dist_mat
end

dist_matrix = preprocess_distance_matrix!(dist_matrix)

# ===============================
# Parameters
# ===============================
bus_capacity = 20
num_clusters = 5

time_matrix = dist_matrix ./ 50 .* 60.0  # Assume average 50 km/h for conversion to time

# ===============================
# Geographic Clustering
# ===============================
coords = hcat(df_metadata.latitude, df_metadata.longitude)
kmeans_result = kmeans(coords', num_clusters)

cluster_assignments = kmeans_result.assignments
cluster_map = vcat(0, cluster_assignments)

scatter(df_metadata.longitude, df_metadata.latitude,
        group=cluster_assignments, legend=true,
        title="Geographic Clusters", xlabel="Longitude", ylabel="Latitude")

# ===============================
# Route Optimization (Greedy TSP per Cluster)
# ===============================
function tsp_nearest_neighbor(cluster_nodes::Vector{Int}, dist_matrix::Matrix{Float64})
    """
    Solve TSP using nearest neighbor heuristic for a cluster.
    
    Args:
        cluster_nodes: Nodes in the cluster
        dist_matrix: Distance matrix
    
    Returns:
        Vector{Int}: Optimized route
    """
    route = [1]  # start from depot
    remaining = setdiff(cluster_nodes, [1])
    current = 1
    while !isempty(remaining)
        next_node = argmin(dist_matrix[current, remaining])
        current = remaining[next_node]
        push!(route, current)
        deleteat!(remaining, next_node)
    end
    push!(route, 1)  # return to depot
    return route
end

routes = Vector{Vector{Int}}()
for c in 1:num_clusters
    cluster_nodes = [i for i in 2:n_nodes if cluster_map[i] == c]
    push!(cluster_nodes, 1)  # include depot
    push!(routes, tsp_nearest_neighbor(cluster_nodes, dist_matrix))
end

# ===============================
# Output Routes
# ===============================
println("\n🚍 Optimized Routes (Name → Name format):")
for (i, route) in enumerate(routes)
    names = join([stop_names[node] for node in route], " → ")
    println("Bus $(i): $names")
end

# =============================================================================
# Section 3: Implementation 2 - Advanced Clustering with ACO
# =============================================================================
# Enhanced K-means clustering with ACO optimization within each cluster
# Features: ACO per cluster, performance comparison, parallel processing

###############
###############
###############
###############
using CSV, DataFrames, Clustering, Combinatorics, LinearAlgebra, Base.Threads, StatsBase

# ===============================
# Load Data
# ===============================
df_meta = CSV.read("bus_stops_metadata.csv", DataFrame)
df_raw = CSV.read("distance_matrix.csv", DataFrame; header=false)
dist_matrix = Matrix{Float64}(df_raw)

if size(dist_matrix, 1) != size(dist_matrix, 2)
    last_col = dist_matrix[:, end]
    dist_matrix = hcat(dist_matrix, last_col)
end

n = size(dist_matrix, 1)
stop_names = vcat("Depot", String.(df_meta.stop_name))

# Replace diagonal with Inf
@inbounds for i in 1:n
    dist_matrix[i, i] = Inf
end

coords = hcat(df_meta.latitude, df_meta.longitude)
k = 5
kmeans_result = kmeans(coords', k)

cluster_assignments = kmeans_result.assignments
cluster_indices = [findall(cluster_assignments .== i) for i in 1:k]

# Add depot (index 1) to all clusters
for i in 1:k
    cluster_indices[i] = vcat(1, cluster_indices[i] .+ 1)  # .+1 to offset depot index
end

# ===============================
# Nearest Neighbor TSP
# ===============================
function nearest_neighbor_tsp(cluster::Vector{Int}, dist_matrix::Matrix{Float64})
    """
    Solve TSP using nearest neighbor heuristic.
    
    Args:
        cluster: Cluster nodes
        dist_matrix: Distance matrix
    
    Returns:
        tuple: (route, total_distance)
    """
    unvisited = Set(cluster)
    current = cluster[1]
    route = [current]
    delete!(unvisited, current)
    total_dist = 0.0

    while !isempty(unvisited)
        dists = [(dist_matrix[current, node], node) for node in unvisited]
        (_, idx) = findmin(first.(dists))
        d, next_stop = dists[idx]

        push!(route, next_stop)
        total_dist += d
        delete!(unvisited, next_stop)
        current = next_stop
    end

    # return to depot
    total_dist += dist_matrix[current, route[1]]
    push!(route, route[1])

    return route, total_dist
end


# ===============================
# Ant Colony Optimization
# ===============================
function ant_colony_tsp(cluster::Vector{Int}; num_ants=50, num_iter=300, alpha=1.0, beta=5.0, rho=0.1, Q=100.0)
    """
    ACO implementation for TSP within a cluster.
    
    Args:
        cluster: Cluster nodes
        num_ants: Number of ants
        num_iter: Number of iterations
        alpha: Pheromone importance
        beta: Heuristic importance
        rho: Evaporation rate
        Q: Pheromone deposit factor
    
    Returns:
        tuple: (optimized_route, cost)
    """
    m = length(cluster)
    submatrix = dist_matrix[cluster, cluster]
    pheromone = ones(m, m)
    visibility = 1.0 ./ submatrix   
    visibility[isinf.(visibility) .| isnan.(visibility)] .= 0.0


    best_cost = Inf
    best_path = []

    function select_next(current, unvisited)
        weights = [(pheromone[current, j]^alpha) * (visibility[current, j]^beta) for j in unvisited]
        total_weight = sum(weights)

        # Fallback: If all weights are zero or invalid
        if total_weight == 0.0 || any(isnan, weights) || any(isinf, weights)
            return rand(unvisited)  # Choose randomly as a fallback
        end

        p = weights ./ total_weight
        return unvisited[sample(1:length(unvisited), Weights(p))]
    end


    for _ in 1:num_iter
        all_paths = Vector{Vector{Int}}()
        all_dists = Float64[]

        for _ in 1:num_ants
            path = [1]
            unvisited = collect(2:m)
            while !isempty(unvisited)
                next = select_next(path[end], unvisited)
                push!(path, next)
                deleteat!(unvisited, findfirst(==(next), unvisited))
            end
            push!(path, 1)
            dist = sum(submatrix[path[i], path[i+1]] for i in 1:length(path)-1)
            push!(all_paths, path)
            push!(all_dists, dist)
        end

        pheromone .*= (1 - rho)
        for (path, dist) in zip(all_paths, all_dists)
            for i in 1:length(path)-1
                a, b = path[i], path[i+1]
                pheromone[a, b] += Q / dist
                pheromone[b, a] += Q / dist
            end
        end

        idx = argmin(all_dists)
        if all_dists[idx] < best_cost
            best_cost = all_dists[idx]
            best_path = all_paths[idx]
        end
    end

    return cluster[best_path], best_cost
end

# ===============================
# Run TSP and ACO
# ===============================
tsp_routes = Vector{Vector{Int}}(undef, k)
tsp_dists = zeros(Float64, k)
aco_routes = Vector{Vector{Int}}(undef, k)
aco_dists = zeros(Float64, k)

# Serial version (safe); use Threads.@spawn for parallel if needed
for i in 1:k
    cluster = cluster_indices[i]
    tsp_routes[i], tsp_dists[i] = nearest_neighbor_tsp(cluster, dist_matrix)
    aco_routes[i], aco_dists[i] = ant_colony_tsp(cluster)
end

# ===============================
# Display Results
# ===============================
println("📦 Route Comparison per Bus (TSP vs ACO)\n")

total_tsp = 0.0
total_aco = 0.0

for i in 1:k
    total_tsp += tsp_dists[i]
    total_aco += aco_dists[i]
    tsp_str = join(stop_names[tsp_routes[i]], " → ")
    aco_str = join(stop_names[aco_routes[i]], " → ")

    println("🚌 Bus $(i):")
    println("  - TSP Route  : $tsp_str\n    Distance   : $(round(tsp_dists[i], digits=2)) km")
    println("  - ACO Route  : $aco_str\n    Distance   : $(round(aco_dists[i], digits=2)) km\n")
end

println("📊 Total TSP Distance: $(round(total_tsp, digits=2)) km")
println("📊 Total ACO Distance: $(round(total_aco, digits=2)) km")

# =============================================================================
# Section 4: Implementation 3 - Balanced Clustering with ACO
# =============================================================================
# Smart cluster merging with advanced ACO and improved constraints
# Features: Load balancing, enhanced ACO, better solution validation

#####################################################################################

using CSV, DataFrames, Clustering, LinearAlgebra, StatsBase

# ===============================
# Load Data
# ===============================
df_meta = CSV.read("bus_stops_metadata.csv", DataFrame)
df_raw = CSV.read("distance_matrix.csv", DataFrame; header=false)
dist_matrix = Matrix{Float64}(df_raw)
if size(dist_matrix, 1) != size(dist_matrix, 2)
    last_col = dist_matrix[:, end]
    dist_matrix = hcat(dist_matrix, last_col)
end
n = size(dist_matrix, 1)
stop_names = vcat("Depot", String.(df_meta.stop_name))

# Replace diagonal with Inf
@inbounds for i in 1:n
    dist_matrix[i, i] = Inf
end

coords = hcat(df_meta.latitude, df_meta.longitude)

# ===============================
# Step 1: Initial K-means clustering
# ===============================
k = 5  # desired initial number of buses
kmeans_result = kmeans(coords', k)
cluster_assignments = kmeans_result.assignments
cluster_indices = [findall(cluster_assignments .== i) for i in 1:k]

# Add depot (index 1) to all clusters
for i in 1:k
    cluster_indices[i] = vcat(1, cluster_indices[i] .+ 1)  # adjust for depot index
end

# ===============================
# Step 2: Merge small clusters for balance
# ===============================
function cluster_centroid(cluster_idx)
    """
    Calculate centroid of a cluster.
    
    Args:
        cluster_idx: Cluster indices
    
    Returns:
        Vector{Float64}: Centroid coordinates
    """
    if length(cluster_idx) == 1
        return coords[1, :]  # default to depot if only depot
    else
        return mean(coords[cluster_idx[2:end] .- 1, :], dims=1)
    end
end

min_size = 4  # minimum number of stops (excluding depot)

function merge_small_clusters!(clusters)
    """
    Merge small clusters for better workload distribution.
    
    Args:
        clusters: Vector of cluster indices
    """
    i = 1
    while i <= length(clusters)
        if length(clusters[i]) - 1 < min_size
            # Merge into geographically nearest cluster (excluding self)
            current_centroid = cluster_centroid(clusters[i])
            distances = [norm(current_centroid .- cluster_centroid(clusters[j])) for j in 1:length(clusters) if (j != i)]
            nearest_j = findmin(distances)[2]
            # nearest_j is not offset by i's index if i < nearest_j
            adjust = nearest_j >= i ? 1 : 0
            merge_idx = nearest_j + adjust
            clusters[merge_idx] = vcat(clusters[merge_idx], clusters[i][2:end])
            empty!(clusters[i])  # Remove all stops from merged cluster
        end
        i += 1
    end
    # Remove any empty clusters after merging
    filter!(cl -> !isempty(cl), clusters)
end

merge_small_clusters!(cluster_indices)
k = length(cluster_indices)  # update cluster count

# ===============================
# Step 3: Ant Colony Optimization for TSP per cluster
# ===============================
function ant_colony_tsp(cluster::Vector{Int}, dist_matrix::Matrix{Float64}; num_ants=50, num_iter=300, alpha=1.0, beta=5.0, rho=0.1, Q=100.0)
    """
    Enhanced ACO for TSP within a cluster.
    
    Args:
        cluster: Cluster nodes
        dist_matrix: Distance matrix
        num_ants: Number of ants
        num_iter: Number of iterations
        alpha: Pheromone importance
        beta: Heuristic importance
        rho: Evaporation rate
        Q: Pheromone deposit factor
    
    Returns:
        tuple: (optimized_route, cost)
    """
    m = length(cluster)
    submatrix = dist_matrix[cluster, cluster]
    pheromone = ones(m, m)
    visibility = 1.0 ./ submatrix
    visibility[isinf.(visibility) .| isnan.(visibility)] .= 0.0

    best_cost = Inf
    best_path = []

    function select_next(current, unvisited)
        weights = [(pheromone[current, j]^alpha) * (visibility[current, j]^beta) for j in unvisited]
        total_weight = sum(weights)
        if total_weight == 0.0 || any(isnan, weights) || any(isinf, weights)
            return rand(unvisited)
        end
        p = weights ./ total_weight
        return unvisited[sample(1:length(unvisited), Weights(p))]
    end

    for _ in 1:num_iter
        all_paths = Vector{Vector{Int}}()
        all_dists = Float64[]

        for _ in 1:num_ants
            path = [1]
            unvisited = collect(2:m)
            while !isempty(unvisited)
                next = select_next(path[end], unvisited)
                push!(path, next)
                deleteat!(unvisited, findfirst(==(next), unvisited))
            end
            push!(path, 1)
            dist = sum(submatrix[path[i], path[i+1]] for i in 1:length(path)-1)
            push!(all_paths, path)
            push!(all_dists, dist)
        end

        pheromone .*= (1 - rho)
        for (path, dist) in zip(all_paths, all_dists)
            for i in 1:length(path)-1
                a, b = path[i], path[i+1]
                pheromone[a, b] += Q / dist
                pheromone[b, a] += Q / dist
            end
        end

        idx = argmin(all_dists)
        if all_dists[idx] < best_cost
            best_cost = all_dists[idx]
            best_path = all_paths[idx]
        end
    end
    return cluster[best_path], best_cost
end

# ===============================
# Step 4: Run ACO for each balanced cluster
# ===============================
routes = Vector{Vector{Int}}(undef, k)
dists = zeros(Float64, k)

for i in 1:k
    cluster = cluster_indices[i]
    routes[i], dists[i] = ant_colony_tsp(cluster, dist_matrix)
end

# ===============================
# Step 5: Display optimized routes
# ===============================
println("📦 Optimized Routes per Bus:")
for i in 1:k
    route_str = join(stop_names[routes[i]], " → ")
    println("🚌 Bus $i:")
    println("  Route: $route_str")
    println("  Distance: ", round(dists[i], digits=2), " km\n")
end
