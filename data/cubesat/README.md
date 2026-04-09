# CubeSat Experimental Data

Planar free-floating dynamics data from a CubeSat simulator on a 4m × 2m granite table.

## Format

Each CSV file is one trajectory with columns (no header):
1. **time** (seconds)
2. **theta** (yaw angle in degrees)
3. **x** (position in meters)
4. **y** (position in meters)

## Collection

- 30 trajectories from free-drift experiments
- ~5.5 Hz sampling rate, ~10 seconds per trajectory
- Position tracked via pseudo-galactic star tracking system
- Near-frictionless motion via three symmetrically placed air bearings

See Section 5.3 of the paper for details.
