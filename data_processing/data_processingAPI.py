# =============================================================================
# VRP Data Processing Script
# =============================================================================
# This script preprocesses raw bus stop data and generates distance matrix
# for Vehicle Routing Problem (VRP) optimization algorithms.
# 
# Sections:
# 1. Imports and Setup
# 2. Data Loading and Cleaning
# 3. Depot Identification
# 4. Distance Matrix Generation
# 5. Data Export
# =============================================================================

# =============================================================================
# Section 1: Imports and Setup
# =============================================================================
import pandas as pd
import openrouteservice
from openrouteservice import convert
import time

# =============================================================================
# Section 2: Data Loading and Cleaning
# =============================================================================

# Load file
file_path = "YOUR_FILE_LOCATION"  # Adjust path as needed
df = pd.read_csv(file_path)

# Clean latitude and longitude: remove ° if present, convert to float
def clean_coord(val):
    """
    Clean coordinate values by removing degree symbols and converting to float.
    
    Args:
        val: Coordinate value (string or float)
    
    Returns:
        float: Cleaned coordinate value
    """
    if isinstance(val, str):
        val = val.replace("°", "").strip()
    return float(val)

# Apply coordinate cleaning to latitude and longitude columns
df["latitude"] = df["latitude"].apply(clean_coord)
df["longitude"] = df["longitude"].apply(clean_coord)

# =============================================================================
# Section 3: Depot Identification
# =============================================================================

# Identify depot (destination point)
depot_name = "Mondelez International, Pune"
df["is_destination"] = df["stop_name"] == depot_name

# Sanity check: ensure depot exists in data
assert df["is_destination"].any(), "Depot not found in data!"

# =============================================================================
# Section 4: Distance Matrix Generation
# =============================================================================

# Create coordinate list in [lon, lat] format for OpenRouteService API
coordinates = df[["longitude", "latitude"]].values.tolist()

# Initialize OpenRouteService API client
client = openrouteservice.Client(key="YOUR_API_KEY")  # replace with your real key

# Call Distance Matrix API to get driving distances between all stops
matrix = client.distance_matrix(
    locations=coordinates,
    profile='driving-car',  # Use driving car profile for road distances
    metrics=['distance'],    # Get distance metrics
    units='km'              # Return distances in kilometers
)

# =============================================================================
# Section 5: Data Export
# =============================================================================

# Save cleaned CSVs for Julia algorithms
df.to_csv("bus_stops_metadata.csv", index=False)
pd.DataFrame(matrix["distances"]).to_csv("distance_matrix.csv", index=False)

print("✅ Files saved: bus_stops_metadata.csv and distance_matrix.csv")
print(f"📊 Processed {len(df)} bus stops")
print(f"📍 Depot location: {depot_name}")
print(f"🛣️  Distance matrix size: {len(matrix['distances'])}x{len(matrix['distances'][0])}")
