import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration - ADJUST THESE FOR YOUR SETUP
vframe_folder = '/Users/zebra/projects/pjas2026n/piv_output/vframe_2sec'  # Folder with CSV files
rho = 1000  # kg/m³ (density of water)
h = 0.025   # meters (depth of water layer - CHANGE THIS!)
pixels_per_meter = 6925  # How many pixels = 1 meter? (CALIBRATE THIS!)

# Find all CSV files
vframe_path = Path(vframe_folder)
csv_files = sorted(vframe_path.glob('velocity_frame_*.csv'))

if len(csv_files) == 0:
    print(f"No CSV files found in {vframe_folder}")
    exit()

print(f"Found {len(csv_files)} CSV files")
print("="*60)

# Storage for results
timestamps = []
kinetic_energies = []
mean_velocities = []

# Process each CSV file
for i, csv_file in enumerate(csv_files):
    # Read the CSV
    df = pd.read_csv(csv_file)
    
    # Extract velocity components
    u = df['u'].values
    v = df['v'].values
    x = df['x'].values
    y = df['y'].values
    
    # Calculate grid spacing (from first file only)
    if i == 0:
        x_unique = np.unique(x)
        y_unique = np.unique(y)
        dx = x_unique[1] - x_unique[0]  # Grid spacing in x (pixels)
        dy = y_unique[1] - y_unique[0]  # Grid spacing in y (pixels)
        dA = dx * dy  # Area of each grid cell (pixels²)
        dA_physical = dA / (pixels_per_meter**2)  # m²
        
        print(f"Grid spacing: dx={dx:.2f} pixels, dy={dy:.2f} pixels")
        print(f"Grid cell area: {dA_physical:.6f} m²")
        print(f"Total area: {len(u) * dA_physical:.4f} m²")
        print("="*60 + "\n")
    
    # Calculate velocity magnitude squared
    velocity_squared = u**2 + v**2
    
    # Calculate kinetic energy
    # KE = (1/2) * ρ * h * ∫∫ |u|² dA
    KE = 0.5 * rho * h * np.sum(velocity_squared) * dA_physical
    
    # Extract timestamp from filename or use index
    # Assuming filename format: velocity_frame_XXXX.csv
    frame_num = int(csv_file.stem.split('_')[-1])
    
    # Try to get actual timestamp if NPZ file exists
    npz_file = csv_file.with_suffix('.npz')
    if npz_file.exists():
        data = np.load(npz_file)
        timestamp = float(data['timestamp'])
    else:
        # Estimate from frame number (assumes 30 fps)
        timestamp = frame_num / 30.0  # ADJUST FPS IF NEEDED
    
    timestamps.append(timestamp)
    kinetic_energies.append(KE)
    mean_velocities.append(np.sqrt(np.mean(velocity_squared)))
    
    # Progress
    if (i + 1) % 10 == 0 or (i + 1) == len(csv_files):
        print(f"Processed {i + 1}/{len(csv_files)} files", end='\r')

print(f"\n\nCompleted processing {len(csv_files)} files")

# Convert to arrays for plotting
timestamps = np.array(timestamps)
kinetic_energies = np.array(kinetic_energies)
mean_velocities = np.array(mean_velocities)

# Create plots
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(timestamps, kinetic_energies, linewidth=2, marker='o', markersize=4, color='blue')
ax.set_xlabel('Time (s)', fontsize=12)
ax.set_ylabel('Kinetic Energy (Joules)', fontsize=12)
ax.set_title('Kinetic Energy Over Time', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)



# Print summary statistics
print("\n" + "="*60)
print("KINETIC ENERGY STATISTICS")
print("="*60)
print(f"Number of frames: {len(timestamps)}")
print(f"Time range: {timestamps[0]:.2f}s to {timestamps[-1]:.2f}s")
print(f"\nKinetic Energy:")

print("="*60)

# Save data to CSV
summary_csv = Path(vframe_folder) / 'kinetic_energy_summary.csv'
summary_df = pd.DataFrame({
    'timestamp': timestamps,
    'kinetic_energy_J': kinetic_energies,
    'mean_velocity_px_per_frame': mean_velocities
})
summary_df.to_csv(summary_csv, index=False)
print(f"\nSummary data saved to {summary_csv}")
# Exponential curve fitting
from scipy.optimize import curve_fit

def exponential_decay(t, E0, tau):
    """
    Exponential decay model: E(t) = E0 * exp(-t/tau)
    
    E0 = initial energy
    tau = time constant (how long it takes to decay to E0/e)
    """
    return E0 * np.exp(-t / tau)

# Fit exponential curve
try:
    # Initial guess for parameters
    E0_guess = kinetic_energies[0]
    tau_guess = (timestamps[-1] - timestamps[0]) / 2
    
    popt, pcov = curve_fit(exponential_decay, timestamps, kinetic_energies, 
                          p0=[E0_guess, tau_guess])
    
    E0_fit, tau_fit = popt
    
    # Calculate R²
    residuals = kinetic_energies - exponential_decay(timestamps, E0_fit, tau_fit)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((kinetic_energies - np.mean(kinetic_energies))**2)
    r_squared = 1 - (ss_res / ss_tot)
    
    # Generate smooth curve
    t_smooth = np.linspace(timestamps[0], timestamps[-1], 200)
    E_smooth = exponential_decay(t_smooth, E0_fit, tau_fit)
    
    # Add fitted curve to the plot
    ax.plot(t_smooth, E_smooth, 'r--', linewidth=2.5, 
            label=f'Fit: E(t) = {E0_fit:.4f} × exp(-t/{tau_fit:.2f})\nR² = {r_squared:.4f}')
    
    fit_successful = True
    
    # Print fit results
    print("\n" + "="*60)
    print("EXPONENTIAL DECAY FIT")
    print("="*60)
    print(f"Model: E(t) = E₀ × exp(-t/τ)")
    print(f"\nFitted parameters:")
    print(f"  E₀ (initial energy) = {E0_fit:.6f} J")
    print(f"  τ (time constant)   = {tau_fit:.4f} s")
    print(f"\nQuality of fit:")
    print(f"  R² = {r_squared:.6f}")
    if r_squared > 0.9:
        print(f"  Excellent fit! Flow follows exponential decay.")
    elif r_squared > 0.7:
        print(f"  Good fit. Some deviations from pure exponential.")
    else:
        print(f"  Poor fit. Flow may not be purely exponential decay.")
    
    print(f"\nPhysical interpretation:")
    print(f"  Time to decay to 37% (1/e): {tau_fit:.2f} s")
    print(f"  Time to decay to 50%:       {tau_fit * np.log(2):.2f} s")
    print(f"  Time to decay to 10%:       {tau_fit * np.log(10):.2f} s")
    print("="*60)
    
    # Save fit parameters
    fit_summary = Path(vframe_folder) / 'exponential_fit_parameters.txt'
    with open(fit_summary, 'w') as f:
        f.write("EXPONENTIAL DECAY FIT RESULTS\n")
        f.write("="*60 + "\n")
        f.write(f"Model: E(t) = E₀ × exp(-t/τ)\n\n")
        f.write(f"E₀ (initial energy) = {E0_fit:.6f} J\n")
        f.write(f"τ (time constant)   = {tau_fit:.4f} s\n")
        f.write(f"R² = {r_squared:.6f}\n\n")
        f.write(f"Time to decay to 37%: {tau_fit:.2f} s\n")
        f.write(f"Time to decay to 50%: {tau_fit * np.log(2):.2f} s\n")
        f.write(f"Time to decay to 10%: {tau_fit * np.log(10):.2f} s\n")
    
    print(f"\nFit parameters saved to {fit_summary}")

except Exception as e:
    print(f"\nWarning: Could not fit exponential curve: {e}")
    fit_successful = False

# NOW save and show ONCE at the very end, outside the try block
plt.tight_layout()
output_plot = Path(vframe_folder) / 'kinetic_energy_vs_time.png'
plt.savefig(output_plot, dpi=150, bbox_inches='tight')
print(f"\nPlot saved to {output_plot}")
plt.show()