# Data Processing - Preprocessing and Data Preparation

## Overview
This folder contains the data preprocessing script that prepares raw bus stop data and generates the distance matrix needed for all VRP optimization algorithms.

## Purpose
**Foundation - Data Preparation**
- Clean and validate raw bus stop data
- Generate distance matrix using real road networks
- Prepare data for all optimization algorithms
- Ensure data quality and consistency

## Contents

### Files
- `data_processingAPI.py` - Main preprocessing script
- `data_processingAPI_README.md` - Detailed documentation

### Key Functions

#### 1. Data Cleaning
- Removes degree symbols (°) from coordinate data
- Converts coordinates to float format
- Validates data integrity

#### 2. Depot Identification
- Identifies depot location (Mondelez International, Pune)
- Validates depot existence in dataset
- Ensures proper depot labeling

#### 3. Distance Matrix Generation
- Uses OpenRouteService API for real road distances
- Computes driving distances between all stops
- Returns distances in kilometers

#### 4. Data Export
- Saves cleaned bus stop metadata
- Exports distance matrix for Julia algorithms
- Ensures proper data format

## Key Characteristics
- **Input**: Raw bus stop data with coordinates
- **Output**: Cleaned data and distance matrix
- **API Dependency**: OpenRouteService for road distances
- **Format**: CSV files for Julia compatibility

## Usage
```bash
cd data_processing
python data_processingAPI.py
```

## Dependencies
```bash
pip install pandas openrouteservice
```

## Configuration Required
1. **Set file path**: Update `file_path` variable with your raw data location
2. **Add API key**: Replace `"YOUR_API_KEY"` with your OpenRouteService API key
   - Get a free API key from: https://openrouteservice.org/dev/#/signup

## Output Files
- `bus_stops_metadata.csv` - Cleaned bus stop data with processed coordinates
- `distance_matrix.csv` - Distance matrix between all bus stops (in kilometers)

## Data Requirements
The script expects a CSV file with the following columns:
- `stop_name`: Name of the bus stop
- `latitude`: Latitude coordinate (may include ° symbol)
- `longitude`: Longitude coordinate (may include ° symbol)
- `num_passengers`: Number of passengers at each stop
- `is_destination`: Boolean indicating if it's the depot

## Error Handling
- Validates that depot exists in the dataset
- Ensures coordinate conversion is successful
- Checks API response validity
- Provides informative error messages

## Notes
- Uses 'driving-car' profile for realistic road distances
- Distances computed using real road networks, not straight-line
- API calls may have rate limits depending on your plan
- Coordinates are cleaned to remove degree symbols

## Next Steps
After data preparation, explore the optimization algorithms:
- `../vrp_solver/` - Professional optimization approaches
- `../aco_solver/` - Nature-inspired optimization approaches
- `../geoaco_solver/` - Hybrid geographic optimization approaches 