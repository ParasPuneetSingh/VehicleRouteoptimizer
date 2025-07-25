# Data Processing API Script

## Overview
`data_processingAPI.py` is a Python script that preprocesses raw bus stop data and generates the distance matrix needed for the VRP optimization algorithms.

## Purpose
This script performs the following key functions:
1. **Data Cleaning**: Removes degree symbols (°) from coordinate data and converts to float format
2. **Depot Identification**: Identifies the depot location (Mondelez International, Pune)
3. **Distance Matrix Generation**: Uses OpenRouteService API to compute distances between all bus stops
4. **Data Export**: Saves cleaned data in CSV format for Julia algorithms

## Dependencies
```bash
pip install pandas openrouteservice
```

## Configuration
Before running the script, you need to:

1. **Set your file path**: Update `file_path` variable with your raw data location
2. **Add API key**: Replace `"YOUR_API_KEY"` with your OpenRouteService API key
   - Get a free API key from: https://openrouteservice.org/dev/#/signup

## Input Data Format
The script expects a CSV file with the following columns:
- `stop_name`: Name of the bus stop
- `latitude`: Latitude coordinate (may include ° symbol)
- `longitude`: Longitude coordinate (may include ° symbol)
- `num_passengers`: Number of passengers at each stop
- `is_destination`: Boolean indicating if it's the depot

## Output Files
The script generates two CSV files:
1. **`bus_stops_metadata.csv`**: Cleaned bus stop data with processed coordinates
2. **`distance_matrix.csv`**: Distance matrix between all bus stops (in kilometers)

## Usage
```bash
python data_processingAPI.py
```

## Key Functions

### `clean_coord(val)`
- Removes degree symbols from coordinate strings
- Converts coordinates to float format
- Handles both string and numeric inputs

### Distance Matrix Generation
- Uses OpenRouteService Distance Matrix API
- Computes driving distances between all stops
- Returns distances in kilometers

## Error Handling
- Validates that depot exists in the dataset
- Ensures coordinate conversion is successful
- Checks API response validity

## Notes
- The script uses the 'driving-car' profile for distance calculations
- Distances are computed using real road networks, not straight-line distances
- API calls may have rate limits depending on your OpenRouteService plan

## Troubleshooting
- **API Key Issues**: Ensure your OpenRouteService API key is valid and has sufficient credits
- **File Path Errors**: Verify the input file path is correct
- **Coordinate Format**: The script handles common coordinate formats but may need adjustment for unusual formats 