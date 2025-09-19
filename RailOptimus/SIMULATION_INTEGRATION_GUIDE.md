# RailOptimus Simulation Integration Guide

## Overview
The simulation functionality has been successfully integrated with the frontend. Here's what has been implemented:

## Features Added

### 1. Popup Window Integration
- **Run Simulation Button**: Now opens the RailOptiSim application in a popup window
- **URL**: Opens `http://localhost:8050` in a new window (1400x900px)
- **Window Features**: Scrollable, resizable, focused automatically

### 2. Download Functionality
- **Export Format Selection**: Choose between CSV and PDF formats
- **Download Button**: Available in both header and simulation results section
- **Offline Viewing**: Download simulation results for offline analysis

## How to Use

### Running a Simulation
1. Go to the Simulation page in the frontend
2. Configure your scenario parameters:
   - Select scenario type (delay, block, accident)
   - Choose train/block if applicable
   - Set delay duration
   - Configure additional constraints if needed
3. Click "Run Simulation" button
4. The RailOptiSim application will open in a popup window
5. Use the advanced simulation controls in the popup

### Downloading Results
1. After running a simulation, results will appear in the Simulation Results card
2. Select your preferred format (CSV or PDF) from the dropdown
3. Click "Download Results" button (available in header or results section)
4. The file will be downloaded to your computer for offline viewing

## Technical Details

### Servers Required
- **Frontend**: Running on `http://localhost:8081`
- **RailOptiSim**: Running on `http://localhost:8050`

### File Formats
- **CSV**: Contains structured data with simulation parameters and results
- **PDF**: Professional report with formatted content and visualizations

### Integration Points
- Popup window opens RailOptiSim dashboard
- Download functionality uses existing export utilities
- Results are formatted for offline analysis

## Notes
- The RailOptiSim server must be running for the popup integration to work
- Download functionality works independently of the simulation server
- All existing functionality remains unchanged
- The integration is seamless and user-friendly

## Troubleshooting
- If popup doesn't open, ensure RailOptiSim server is running on port 8050
- If download fails, check browser permissions for file downloads
- Both servers must be running for full functionality

