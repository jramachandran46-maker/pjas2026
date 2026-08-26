"""
Particle Image Velocimetry (PIV) Video Analysis
Processes video to extract 2D velocity fields as a function of time using OpenPIV
"""


import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import openpiv.tools
import openpiv.pyprocess
import openpiv.scaling
import openpiv.validation
import openpiv.filters
from pathlib import Path
import pickle
from scipy.ndimage import median_filter




class PIVVideoAnalyzer:
   """Process video files to extract PIV velocity fields"""
  
   def __init__(self, video_path, output_dir='piv_output'):
       """
       Initialize PIV analyzer
      
       Parameters:
       -----------
       video_path : str
           Path to input video file
       output_dir : str
           Directory to save output files
       """
       self.video_path = video_path
       self.output_dir = Path(output_dir)
       self.output_dir.mkdir(exist_ok=True)
      
       # PIV parameters (can be adjusted)
       self.window_size = 64  # ~7mm, should contain 9-12 particles
       self.overlap = 32   # 50% overlap
       self.search_area_size = 96  # conservative, covers up to ~50 pixel displacement
       self.dt = 1  # time between frames (adjust based on your video fps)
      
       # Storage for velocity fields
       self.velocity_fields = []
       self.x_coords = None
       self.y_coords = None
       self.timestamps = []
      
   def preprocess_frame(self, frame):
       """
       Preprocess video frame for PIV analysis
       """
       # Convert to grayscale if needed
       if len(frame.shape) == 3:
           gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
       else:
           gray = frame
      
      
       # Apply CLAHE for better contrast
       clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
       enhanced = clahe.apply(gray)
      
       return enhanced.astype(np.int32)
   def process_video(self, start_frame, frame_skip=1, max_frames=None):
      
       """
       Process entire video to extract velocity fields
      
       Parameters:
       -----------
       frame_skip : int
           Process every Nth frame (1 = process all frames)
       max_frames : int or None
           Maximum number of frame pairs to process (None = all)
       """
       cap = cv2.VideoCapture(self.video_path)
      
       if not cap.isOpened():
           raise ValueError(f"Could not open video file: {self.video_path}")
      
       fps = cap.get(cv2.CAP_PROP_FPS)
       total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
      
       print(f"Video info: {total_frames} frames at {fps} fps")
       print(f"Processing with frame_skip={frame_skip}")
       cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    
       frame_count = start_frame 
       processed_pairs = 0
       prev_frame = None
      
       while True:
           ret, frame = cap.read()
           if not ret:
               break
          
           # Skip frames if needed
           if frame_count % frame_skip != 0:
               frame_count += 1
               continue
          
           # Preprocess current frame
           current_frame = self.preprocess_frame(frame)
          
           # Need two consecutive frames for PIV
           if prev_frame is not None:
               print(f"Processing frame pair {processed_pairs + 1}...", end='\r')
              
               # Perform PIV analysis
               u, v, x, y, mask = self._analyze_frame_pair(prev_frame, current_frame)
              
               # Store results
               self.velocity_fields.append({
                   'u': u,
                   'v': v,
                   'mask': mask,
                   'frame_number': frame_count
               })
              
               # Store coordinates (same for all frames)
               if self.x_coords is None:
                   self.x_coords = x
                   self.y_coords = y
              
               # Store timestamp
               timestamp = frame_count / fps
               self.timestamps.append(timestamp)
              
               processed_pairs += 1
              
               if max_frames and processed_pairs >= max_frames:
                   break
          
           prev_frame = current_frame
           frame_count += 1
      
       cap.release()
       print(f"\nProcessed {processed_pairs} frame pairs")
   def visualize_particle_detection(self, frame_skip=1, max_frames=50, start_time=33):
       """
       Show detected particles overlaid on video to verify tracking
       """
       cap = cv2.VideoCapture(self.video_path)
      
       if not cap.isOpened():
           raise ValueError(f"Could not open video file: {self.video_path}")
      
       # JUMP TO START TIME
       fps = cap.get(cv2.CAP_PROP_FPS)
       start_frame = int(start_time * fps)
       cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
       frame_count = start_frame
      
       print(f"Starting particle detection at {start_time}s (frame {start_frame})")
       print("Showing particle detection - Press 'q' to quit, 'p' to pause")
       print("Adjust threshold with '+' and '-' keys")
      
       paused = False
       threshold_val = 140
       # ... rest of the function stays the same  # Start at middle value
      
       while True:
           if not paused:
               ret, frame = cap.read()
               if not ret:
                   break
              
               if frame_count % frame_skip != 0:
                   frame_count += 1
                   continue
              
               # Make a copy for display
               display_frame = frame.copy()
              
               # Preprocess to find particles
               gray = self.preprocess_frame(frame)
              
               # FOR BLUE ON WHITE: Use inverse threshold (detect dark particles)
               _, particle_mask = cv2.threshold(gray.astype(np.uint8), threshold_val, 255, cv2.THRESH_BINARY)
              
               # Find contours (particle locations)
               contours, _ = cv2.findContours(particle_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
              
               # Draw circles on detected particles
               for contour in contours:
                   # Filter by size (avoid noise)
                   area = cv2.contourArea(contour)
                   if area > 4:  # Minimum particle size
                       M = cv2.moments(contour)
                       if M['m00'] > 0:
                           cx = int(M['m10'] / M['m00'])
                           cy = int(M['m01'] / M['m00'])
                           cv2.circle(display_frame, (cx, cy), 5, (0, 255, 0), 2)
              
               # Show info
               cv2.putText(display_frame, f"Particles: {len([c for c in contours if cv2.contourArea(c) > 5])}",
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
               cv2.putText(display_frame, f"Threshold: {threshold_val} (use +/- keys)",
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
              
               # Show threshold mask too
               cv2.imshow('Threshold Mask', particle_mask)
               cv2.imshow('Particle Detection', display_frame)
              
               frame_count += 1
               if max_frames and (frame_count - start_frame) >= max_frames:
                   break
          
           # Handle keyboard
           key = cv2.waitKey(30 if not paused else 0) & 0xFF
           if key == ord('q'):
               break
           elif key == ord('p'):
               paused = not paused
               print(f"\n{'Paused' if paused else 'Resumed'} - Threshold: {threshold_val}")
           elif key == ord('+') or key == ord('='):
               threshold_val = min(255, threshold_val + 5)
               print(f"Threshold: {threshold_val}")
           elif key == ord('-') or key == ord('_'):
               threshold_val = max(0, threshold_val - 5)
               print(f"Threshold: {threshold_val}")
      
       cap.release()
       cv2.destroyAllWindows()
       print(f"\nOptimal threshold appears to be: {threshold_val}")
   def _analyze_frame_pair(self, frame_a, frame_b):
       """
       Perform PIV analysis on a pair of frames
      
       Parameters:
       -----------
       frame_a, frame_b : numpy arrays
           Consecutive frames
          
       Returns:
       --------
       u, v : numpy arrays
           Velocity components
       x, y : numpy arrays
           Grid coordinates
       mask : numpy array
           Valid velocity mask
       """
       # Perform PIV using extended search area
       u, v, sig2noise = openpiv.pyprocess.extended_search_area_piv(
           frame_a, frame_b,
           window_size=self.window_size,
           overlap=self.overlap,
           dt=self.dt,
           search_area_size=self.search_area_size,
           sig2noise_method='peak2peak'
       )
      
       # Get grid coordinates
       x, y = openpiv.pyprocess.get_coordinates(
           image_size=frame_a.shape,
           search_area_size=self.search_area_size,
           overlap=self.overlap
       )
      
       # Validation - remove outliers
      # Validation - create a mask (all False = all valid for now)
       mask = np.zeros_like(u, dtype=bool)


       # Replace outliers using interpolation - needs the mask!
       u, v = openpiv.filters.replace_outliers(
           u, v,
           mask,  # Add this line!
           method='localmean',
           max_iter=3,
           kernel_size=2
       )
       # Replace outliers using interpolation
      
       # Optional: Apply smoothing
       # u, v = openpiv.filters.gaussian(u, v, sigma=1)
       u, v = self._local_median_filter(u, v, kernel_size=5, threshold=1.5)
       fps = 30
       pixels_per_cm = 69.22  # 2110 pixels = 12 inches
       u = u * fps / pixels_per_cm
       v = v * fps / pixels_per_cm
       
       # DELETE vectors above max velocity (now in cm/s)
       max_velocity =7 # cm/s
       velocity_mag = np.sqrt(u**2 + v**2)  # Compute AFTER conversion
       too_fast = velocity_mag > max_velocity
       u[too_fast] = np.nan
       v[too_fast] = np.nan
       mask[too_fast] = True
   # ADD EDGE MASKING HERE - mask outer 2 rows/columns
       mask[:2, :] = True   # Top edge
       mask[-2:, :] = True  # Bottom edge
       mask[:, :2] = True   # Left edge
       mask[:, -2:] = True  # Right edge
      
       # Replace masked edge values with interpolation
       u, v = openpiv.filters.replace_outliers(
           u, v,
           mask,
           method='localmean',
           max_iter=5,
           kernel_size=3
       )
      
       return u, v, x, y, mask
   def save_velocity_data_range(self, start_time, end_time, num_frames=None, output_folder='vframe'):
    """
    Save velocity data for frames between start_time and end_time
    
    Parameters:
    -----------
    start_time : float
        Start time in seconds
    end_time : float
        End time in seconds
    num_frames : int or None
        Number of frames to save (evenly spaced). If None, save all frames in range.
    output_folder : str
        Folder name to save velocity data files
    """
    if not self.velocity_fields:
        raise ValueError("No velocity fields available. Run process_video first.")
    
    import pandas as pd
    
    # Create output folder
    output_dir = self.output_dir / output_folder
    output_dir.mkdir(exist_ok=True)
    
    # Find frame indices within time range
    timestamps = np.array(self.timestamps)
    indices_in_range = np.where((timestamps >= start_time) & (timestamps <= end_time))[0]
    
    if len(indices_in_range) == 0:
        print(f"No frames found between {start_time}s and {end_time}s")
        print(f"Available time range: {timestamps[0]:.2f}s to {timestamps[-1]:.2f}s")
        return
    
    # Select evenly spaced frames if num_frames specified
    if num_frames is not None:
        if num_frames > len(indices_in_range):
            print(f"Warning: Requested {num_frames} frames but only {len(indices_in_range)} available in range")
            indices = indices_in_range
        else:
            step = len(indices_in_range) / num_frames
            selected_positions = [int(i * step) for i in range(num_frames)]
            indices = indices_in_range[selected_positions]
    else:
        indices = indices_in_range
    
    print(f"\n{'='*60}")
    print(f"SAVING VELOCITY DATA FROM {start_time}s TO {end_time}s")
    print(f"{'='*60}")
    print(f"Available frames in range: {len(indices_in_range)}")
    print(f"Frames to save: {len(indices)}")
    print(f"Output folder: {output_dir}")
    
    for i, frame_idx in enumerate(indices):
        field = self.velocity_fields[frame_idx]
        u = field['u']
        v = field['v']
        
        # Calculate velocity magnitude
        velocity_mag = np.sqrt(u**2 + v**2)
        
        # Save as CSV
        csv_path = output_dir / f'velocity_frame_{frame_idx:04d}.csv'
        df = pd.DataFrame({
            'x': self.x_coords.flatten(),
            'y': self.y_coords.flatten(),
            'u': u.flatten(),
            'v': v.flatten(),
            'velocity_magnitude': velocity_mag.flatten()
        })
        df.to_csv(csv_path, index=False)
        
        # Progress indicator
        if (i + 1) % 10 == 0 or (i + 1) == len(indices):
            print(f"Saved {i + 1}/{len(indices)} frames", end='\r')
    
    print(f"\n\nSuccessfully saved {len(indices)} velocity fields to {output_dir}/")
    print(f"Time range: {timestamps[indices[0]]:.2f}s to {timestamps[indices[-1]]:.2f}s")
    print(f"{'='*60}\n")   
   def check_and_correct_divergence(self, tolerance=0.1, correct=False):
       """
       Check if velocity field is divergence-free (incompressible flow)
       Optionally correct the field to enforce zero divergence
      
       Parameters:
       -----------
       tolerance : float
           Maximum acceptable divergence magnitude
       correct : bool
           If True, apply divergence correction to velocity fields
      
       Returns:
       --------
       dict with divergence statistics
       """
       if not self.velocity_fields:
           raise ValueError("No velocity fields available. Run process_video first.")
      
       print("\n" + "="*70)
       print("DIVERGENCE CHECK FOR INCOMPRESSIBILITY")
       print("="*70)
      
       # Calculate spatial grid spacing
       dx = self.x_coords[0, 1] - self.x_coords[0, 0]
       dy = self.y_coords[1, 0] - self.y_coords[0, 0]
      
       divergence_stats = {
           'mean_divergence': [],
           'max_divergence': [],
           'rms_divergence': [],
           'timestamps': self.timestamps
       }
      
       for i, field in enumerate(self.velocity_fields):
           u = field['u']
           v = field['v']
          
           # Compute divergence: ∂u/∂x + ∂v/∂y
           du_dx = np.gradient(u, dx, axis=1)
           dv_dy = np.gradient(v, dy, axis=0)
           divergence = du_dx + dv_dy
          
           # Statistics
           mean_div = np.mean(np.abs(divergence))
           max_div = np.max(np.abs(divergence))
           rms_div = np.sqrt(np.mean(divergence**2))
          
           divergence_stats['mean_divergence'].append(mean_div)
           divergence_stats['max_divergence'].append(max_div)
           divergence_stats['rms_divergence'].append(rms_div)
          
           # Apply correction if requested
           if correct:
               u_corrected, v_corrected = self._apply_divergence_correction(u, v, dx, dy)
               self.velocity_fields[i]['u'] = u_corrected
               self.velocity_fields[i]['v'] = v_corrected
          
           if (i + 1) % 10 == 0:
               print(f"Processed {i + 1}/{len(self.velocity_fields)} frames", end='\r')
      
       print(f"\nCompleted divergence check for {len(self.velocity_fields)} frames\n")
      
       # Print summary
       mean_avg = np.mean(divergence_stats['mean_divergence'])
       max_avg = np.mean(divergence_stats['max_divergence'])
       rms_avg = np.mean(divergence_stats['rms_divergence'])
      
       print(f"Average mean |divergence|: {mean_avg:.6f}")
       print(f"Average max |divergence|:  {max_avg:.6f}")
       print(f"Average RMS divergence:     {rms_avg:.6f}")
       print(f"Tolerance threshold:        {tolerance:.6f}")
      
       if mean_avg < tolerance:
           print(f"✓ PASS: Field is approximately divergence-free")
       else:
           print(f"✗ FAIL: Divergence exceeds tolerance")
           if not correct:
               print(f"  Tip: Use correct=True to enforce incompressibility")
      
       print("="*70 + "\n")
      
       # Plot divergence over time
       self._plot_divergence_statistics(divergence_stats, tolerance)
      
       return divergence_stats




   def _apply_divergence_correction(self, u, v, dx, dy, iterations=50):
       """
       Apply divergence correction using FFT-based pressure projection
       This is much more effective than iterative relaxation
       """
       # Compute current divergence
       du_dx = np.gradient(u, dx, axis=1)
       dv_dy = np.gradient(v, dy, axis=0)
       divergence = du_dx + dv_dy
      
       # Use FFT to solve Poisson equation: ∇²p = div(V)
       # This is faster and more accurate than iterative methods
       ny, nx = divergence.shape
      
       # Create wavenumber grids
       kx = np.fft.fftfreq(nx, d=dx) * 2 * np.pi
       ky = np.fft.fftfreq(ny, d=dy) * 2 * np.pi
       kx_grid, ky_grid = np.meshgrid(kx, ky)
      
       # Laplacian in Fourier space: -(kx² + ky²)
       k_squared = kx_grid**2 + ky_grid**2
       k_squared[0, 0] = 1.0  # Avoid division by zero (set mean pressure to 0)
      
       # Solve for pressure in Fourier space
       divergence_fft = np.fft.fft2(divergence)
       pressure_fft = divergence_fft / (-k_squared)
       pressure_fft[0, 0] = 0  # Zero mean pressure
      
       # Transform back to real space
       pressure = np.fft.ifft2(pressure_fft).real
      
       # Compute pressure gradient
       dp_dx = np.gradient(pressure, dx, axis=1)
       dp_dy = np.gradient(pressure, dy, axis=0)
      
       # Correct velocity: V_corrected = V - ∇p
       u_corrected = u - dp_dx
       v_corrected = v - dp_dy
      
       # Verify correction
       du_dx_new = np.gradient(u_corrected, dx, axis=1)
       dv_dy_new = np.gradient(v_corrected, dy, axis=0)
       divergence_new = du_dx_new + dv_dy_new
      
       print(f"    Divergence reduced: {np.max(np.abs(divergence)):.6f} → {np.max(np.abs(divergence_new)):.6f}")
      
       return u_corrected, v_corrected
   def _plot_divergence_statistics(self, divergence_stats, tolerance):
       """Plot divergence statistics over time"""
       t = np.array(divergence_stats['timestamps'])
      
       fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
      
       # Plot 1: Mean and max divergence
       ax1.plot(t, divergence_stats['mean_divergence'],
               linewidth=2, label='Mean |Divergence|', color='blue')
       ax1.plot(t, divergence_stats['max_divergence'],
               linewidth=2, label='Max |Divergence|', color='red')
       ax1.axhline(y=tolerance, color='green', linestyle='--',
               linewidth=2, label=f'Tolerance ({tolerance})')
       ax1.set_ylabel('Divergence Magnitude', fontsize=12)
       ax1.set_title('Divergence Check (Incompressibility Test)', fontsize=14, fontweight='bold')
       ax1.legend(loc='best')
       ax1.grid(True, alpha=0.3)
       ax1.set_yscale('log')  # Log scale to see small values
      
       # Plot 2: RMS divergence
       ax2.plot(t, divergence_stats['rms_divergence'],
               linewidth=2, color='purple')
       ax2.axhline(y=tolerance, color='green', linestyle='--', linewidth=2)
       ax2.set_xlabel('Time (s)', fontsize=12)
       ax2.set_ylabel('RMS Divergence', fontsize=12)
       ax2.set_title('Root Mean Square Divergence', fontsize=14, fontweight='bold')
       ax2.grid(True, alpha=0.3)
       ax2.set_yscale('log')
      
       plt.tight_layout()
      
       output_path = self.output_dir / 'divergence_check.png'
       plt.savefig(output_path, dpi=150, bbox_inches='tight')
       print(f"Divergence plot saved to {output_path}")
       plt.show()   
   def load_results(self, filename='piv_results.pkl'):
       """Load previously saved results"""
       input_path = self.output_dir / filename
      
       with open(input_path, 'rb') as f:
           results = pickle.load(f)
      
       self.velocity_fields = results['velocity_fields']
       self.x_coords = results['x_coords']
       self.y_coords = results['y_coords']
       self.timestamps = results['timestamps']
      
       print(f"Results loaded from {input_path}")
   def save_results(self, filename='piv_results.pkl'):
       """Save velocity fields and metadata to file"""
       output_path = self.output_dir / filename
      
       results = {
           'velocity_fields': self.velocity_fields,
           'x_coords': self.x_coords,
           'y_coords': self.y_coords,
           'timestamps': self.timestamps,
           'parameters': {
               'window_size': self.window_size,
               'overlap': self.overlap,
               'search_area_size': self.search_area_size,
               'dt': self.dt
           }
       }
      
       with open(output_path, 'wb') as f:
           pickle.dump(results, f)
      
       print(f"Results saved to {output_path}")   
   def plot_velocity_field(self, frame_idx=0, scale=1.0, save=True):
       """
       Plot velocity field for a specific frame
      
       Parameters:
       -----------
       frame_idx : int
           Index of frame to plot
       scale : float
           Scaling factor for quiver arrows
       save : bool
           Whether to save the plot
       """
       if not self.velocity_fields:
           raise ValueError("No velocity fields available. Run process_video first.")
      
       field = self.velocity_fields[frame_idx]
       u = field['u']
       v = field['v']
      
       # Calculate velocity magnitude
       velocity_mag = np.sqrt(u**2 + v**2)
      
       fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
      
       # Quiver plot
  # Normalize to unit vectors (all same length), color by speed
       u_normalized = u / (velocity_mag + 1e-10)
       v_normalized = v / (velocity_mag + 1e-10)


       ax1.quiver(self.x_coords, self.y_coords, u_normalized, -v_normalized,
               velocity_mag, cmap='jet')
       ax1.set_aspect('equal')
       ax1.invert_yaxis()
       ax1.set_title(f'Velocity Vectors (t={self.timestamps[frame_idx]:.2f}s)')
       ax1.set_xlabel('X (pixels)')
       ax1.set_ylabel('Y (pixels)')
      
       # Contour plot
       im = ax2.contourf(self.x_coords, self.y_coords, velocity_mag,
                         levels=20, cmap='jet')
       ax2.set_aspect('equal')
       ax2.set_title('Velocity Magnitude')
       ax2.set_xlabel('X (pixels)')
       ax2.set_ylabel('Y (pixels)')
       plt.colorbar(im, ax=ax2, label='Velocity (pixels/frame)')
      
       plt.tight_layout()
      
       if save:
           output_path = self.output_dir / f'velocity_field_frame_{frame_idx:04d}.png'
           plt.savefig(output_path, dpi=150, bbox_inches='tight')
           print(f"Plot saved to {output_path}")
      
       plt.show()
      
   def create_animation(self, output_file='velocity_animation.mp4', fps=10):
       """
       Create animation of velocity fields over time
      
       Parameters:
       -----------
       output_file : str
           Output video filename
       fps : int
           Frames per second for output video
       """
       if not self.velocity_fields:
           raise ValueError("No velocity fields available. Run process_video first.")
      
       fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
      
       def update(frame_idx):
           ax1.clear()
           ax2.clear()
          
           field = self.velocity_fields[frame_idx]
           u = field['u']
           v = field['v']
           velocity_mag = np.sqrt(u**2 + v**2)
          
           # Quiver plot
           ax1.quiver(self.x_coords, self.y_coords, u, -v,
                      velocity_mag, cmap='jet', scale=50)
           ax1.set_aspect('equal')
           ax1.invert_yaxis()
           ax1.set_title(f'Velocity Vectors (t={self.timestamps[frame_idx]:.2f}s)')
           ax1.set_xlabel('X (pixels)')
           ax1.set_ylabel('Y (pixels)')
          
           # Contour plot
           im = ax2.contourf(self.x_coords, self.y_coords, velocity_mag,
                             levels=20, cmap='jet', vmin=0,
                             vmax=np.max([np.max(np.sqrt(f['u']**2 + f['v']**2))
                                         for f in self.velocity_fields]))
           ax2.set_aspect('equal')
           ax2.set_title('Velocity Magnitude')
           ax2.set_xlabel('X (pixels)')
           ax2.set_ylabel('Y (pixels)')
          
           return ax1, ax2
      
       anim = FuncAnimation(fig, update, frames=len(self.velocity_fields),
                           interval=1000/fps, blit=False)
      
       output_path = self.output_dir / output_file
       anim.save(output_path, writer='ffmpeg', fps=fps, dpi=100)
       print(f"Animation saved to {output_path}")
       plt.close()
      
   def get_velocity_statistics(self):
       """Calculate statistics of velocity fields over time"""
       if not self.velocity_fields:
           raise ValueError("No velocity fields available. Run process_video first.")
      
       stats = {
           'mean_velocity': [],
           'max_velocity': [],
           'std_velocity': [],
           'timestamps': self.timestamps
       }
      
       for field in self.velocity_fields:
           u = field['u']
           v = field['v']
           velocity_mag = np.sqrt(u**2 + v**2)
          
           # Remove masked/invalid values
           valid_vel = velocity_mag[~field['mask']]
          
           stats['mean_velocity'].append(np.mean(valid_vel))
           stats['max_velocity'].append(np.max(valid_vel))
           stats['std_velocity'].append(np.std(valid_vel))
      
       return stats
   def _local_median_filter(self, u, v, kernel_size=5, threshold=1):
       """
       Remove outliers using local median filtering
      
       Parameters:
       -----------
       u, v : arrays
           Velocity components
       kernel_size : int
           Size of neighborhood window (5 = 5x5)
       threshold : float
           How many standard deviations from median to allow (1.5 = stricter)
      
       Returns:
       --------
       u, v : arrays
           Filtered velocity components
       """
       from scipy.ndimage import median_filter
      
       # Compute local median in neighborhood
       u_median = median_filter(u, size=kernel_size)
       v_median = median_filter(v, size=kernel_size)
      
       # Compute local standard deviation
       u_diff = np.abs(u - u_median)
       v_diff = np.abs(v - v_median)
      
       u_std = median_filter(u_diff, size=kernel_size)
       v_std = median_filter(v_diff, size=kernel_size)
      
       # Flag outliers: vectors that differ too much from local median
       u_outliers = u_diff > (threshold * u_std + 1e-10)
       v_outliers = v_diff > (threshold * v_std + 1e-10)
       outliers = u_outliers | v_outliers
      
       # Replace outliers with local median
       u_filtered = np.where(outliers, u_median, u)
       v_filtered = np.where(outliers, v_median, v)
      
       print(f"  Filtered {np.sum(outliers)} outlier vectors")
      
       return u_filtered, v_filtered
   def plot_vorticity_field(self, frame_idx=0, save=True):
       """
       Plot the curl (vorticity) of the velocity field
      
       Vorticity (curl) = ∂v/∂x - ∂u/∂y
       For counterclockwise rotation, vorticity is positive
       """
       if not self.velocity_fields:
           raise ValueError("No velocity fields available. Run process_video first.")
      
       field = self.velocity_fields[frame_idx]
       u = field['u']
       v = field['v']
      
       # Calculate spatial grid spacing
       dx = self.x_coords[0, 1] - self.x_coords[0, 0]
       dy = self.y_coords[1, 0] - self.y_coords[0, 0]
      
       # Compute vorticity (curl in 2D)
       dv_dx = np.gradient(v, dx, axis=1)
       du_dy = np.gradient(u, dy, axis=0)
       vorticity = dv_dx - du_dy
      
       # Also compute velocity magnitude for reference
       velocity_mag = np.sqrt(u**2 + v**2)
      
       fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
      
       # Plot 1: Velocity vectors overlaid on vorticity
       im1 = ax1.contourf(self.x_coords, self.y_coords, vorticity,
                       levels=20, cmap='RdBu_r')
       ax1.quiver(self.x_coords[::2, ::2], self.y_coords[::2, ::2],
               u[::2, ::2], -v[::2, ::2], alpha=0.7)
       ax1.set_aspect('equal')
       ax1.invert_yaxis()
       ax1.set_title(f'Vorticity Field (t={self.timestamps[frame_idx]:.2f}s)')
       ax1.set_xlabel('X (pixels)')
       ax1.set_ylabel('Y (pixels)')
       cbar1 = plt.colorbar(im1, ax=ax1, label='Vorticity (1/frame)')
      
       # Plot 2: Pure vorticity contours
       im2 = ax2.contourf(self.x_coords, self.y_coords, vorticity,
                       levels=20, cmap='RdBu_r')
       ax2.contour(self.x_coords, self.y_coords, vorticity,
                   levels=10, colors='black', alpha=0.3, linewidths=0.5)
       ax2.set_aspect('equal')
       ax2.set_title('Vorticity Contours\n(Red=CCW, Blue=CW)')
       ax2.set_xlabel('X (pixels)')
       ax2.set_ylabel('Y (pixels)')
       cbar2 = plt.colorbar(im2, ax=ax2, label='Vorticity (1/frame)')
      
       # Plot 3: Vorticity statistics
       ax3.hist(vorticity.flatten(), bins=50, edgecolor='black', alpha=0.7)
       ax3.axvline(np.mean(vorticity), color='red', linewidth=2,
                   linestyle='--', label=f'Mean: {np.mean(vorticity):.3f}')
       ax3.axvline(0, color='black', linewidth=1, linestyle='-', alpha=0.5)
       ax3.set_xlabel('Vorticity (1/frame)', fontsize=12)
       ax3.set_ylabel('Count', fontsize=12)
       ax3.set_title('Vorticity Distribution')
       ax3.legend()
       ax3.grid(True, alpha=0.3)
      
       plt.tight_layout()
      
       if save:
           output_path = self.output_dir / f'vorticity_field_frame_{frame_idx:04d}.png'
           plt.savefig(output_path, dpi=150, bbox_inches='tight')
           print(f"Vorticity plot saved to {output_path}")
      
       plt.show()
      
       # Print statistics
       print(f"\nVorticity Statistics:")
       print(f"  Mean: {np.mean(vorticity):.4f} (1/frame)")
       print(f"  Std Dev: {np.std(vorticity):.4f}")
       print(f"  Min: {np.min(vorticity):.4f}")
       print(f"  Max: {np.max(vorticity):.4f}")
       print(f"  For CCW rotation, expect positive vorticity")
   def plot_vorticity_time_series(self, save=True):
       """
       Plot vorticity (curl) statistics as a function of time
       """
       if not self.velocity_fields:
           raise ValueError("No velocity fields available. Run process_video first.")
      
       mean_vorticity = []
       max_vorticity = []
       min_vorticity = []
       std_vorticity = []
      
       # Calculate spatial grid spacing (same for all frames)
       dx = self.x_coords[0, 1] - self.x_coords[0, 0]
       dy = self.y_coords[1, 0] - self.y_coords[0, 0]
      
       print("Computing vorticity over time...")
       for i, field in enumerate(self.velocity_fields):
           u = field['u']
           v = field['v']
          
           # Compute vorticity
           dv_dx = np.gradient(v, dx, axis=1)
           du_dy = np.gradient(u, dy, axis=0)
           vorticity = dv_dx - du_dy
          
           # Calculate statistics
           mean_vorticity.append(np.mean(vorticity))
           max_vorticity.append(np.max(vorticity))
           min_vorticity.append(np.min(vorticity))
           std_vorticity.append(np.std(vorticity))
          
           if (i + 1) % 10 == 0:
               print(f"  Processed {i + 1}/{len(self.velocity_fields)} frames", end='\r')
      
       print(f"\nCompleted vorticity analysis for {len(self.velocity_fields)} frames")
      
       t = np.array(self.timestamps)
      
       # Create plots
       fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
      
       # Plot 1: Max and mean vorticity
       ax1.plot(t, max_vorticity, linewidth=2, label='Max Vorticity', color='red')
       ax1.plot(t, mean_vorticity, linewidth=2, label='Mean Vorticity', color='blue')
       ax1.plot(t, min_vorticity, linewidth=2, label='Min Vorticity', color='green')
       ax1.fill_between(t,
                       np.array(mean_vorticity) - np.array(std_vorticity),
                       np.array(mean_vorticity) + np.array(std_vorticity),
                       alpha=0.3, label='±1 Std Dev', color='blue')
       ax1.axhline(y=0, color='black', linestyle='--', alpha=0.5)
       ax1.set_ylabel('Vorticity (1/frame)', fontsize=12)
       ax1.set_title('Vorticity Statistics Over Time', fontsize=14, fontweight='bold')
       ax1.legend(loc='best')
       ax1.grid(True, alpha=0.3)
      
       # Plot 2: Absolute max vorticity (to see magnitude regardless of sign)
       abs_max_vorticity = [max(abs(max_v), abs(min_v))
                           for max_v, min_v in zip(max_vorticity, min_vorticity)]
       ax2.plot(t, abs_max_vorticity, linewidth=2, color='purple')
       ax2.set_xlabel('Time (s)', fontsize=12)
       ax2.set_ylabel('|Max Vorticity| (1/frame)', fontsize=12)
       ax2.set_title('Maximum Vorticity Magnitude Over Time', fontsize=14, fontweight='bold')
       ax2.grid(True, alpha=0.3)
      
       plt.tight_layout()
      
       if save:
           output_path = self.output_dir / 'vorticity_time_series.png'
           plt.savefig(output_path, dpi=150, bbox_inches='tight')
           print(f"Vorticity time series saved to {output_path}")
      
       plt.show()
      
       # Print summary statistics
       print("\n" + "="*60)
       print("VORTICITY TIME SERIES SUMMARY")
       print("="*60)
       print(f"Mean vorticity across all times: {np.mean(mean_vorticity):.4f}")
       print(f"Mean max vorticity: {np.mean(max_vorticity):.4f}")
       print(f"Peak max vorticity: {np.max(max_vorticity):.4f} at t={t[np.argmax(max_vorticity)]:.2f}s")
       print(f"For solid body CCW rotation, expect consistent positive vorticity")
       print("="*60)
   def plot_velocity_time_series(self, save=True):
       """Plot velocity statistics as a function of time"""
       stats = self.get_velocity_statistics()
      
       fig, ax = plt.subplots(figsize=(12, 6))
      
       ax.plot(stats['timestamps'], stats['mean_velocity'],
               label='Mean Velocity', linewidth=2)
       ax.plot(stats['timestamps'], stats['max_velocity'],
               label='Max Velocity', linewidth=2, alpha=0.7)
       ax.fill_between(stats['timestamps'],
                       np.array(stats['mean_velocity']) - np.array(stats['std_velocity']),
                       np.array(stats['mean_velocity']) + np.array(stats['std_velocity']),
                       alpha=0.3, label='±1 Std Dev')
      
       ax.set_xlabel('Time (s)', fontsize=12)
       ax.set_ylabel('Velocity (pixels/frame)', fontsize=12)
       ax.set_title('Velocity Statistics Over Time', fontsize=14)
       ax.legend()
       ax.grid(True, alpha=0.3)
      
       plt.tight_layout()
      
       if save:
           output_path = self.output_dir / 'velocity_time_series.png'
           plt.savefig(output_path, dpi=150, bbox_inches='tight')
           print(f"Time series plot saved to {output_path}")
      
       plt.show()




def main():
   """Example usage"""
  
   # Example: Process a video file
   video_path = "/Users/zebra/projects/pjas2026n/testpiv.mp4"  # Change this to your video path
  
   # Initialize analyzer
   analyzer = PIVVideoAnalyzer(video_path, output_dir='piv_output')
   analyzer.visualize_particle_detection(frame_skip=1, max_frames=100)
   # Adjust PIV parameters if needed
   analyzer.window_size = 64  # smaller = more detail, but noisier
   analyzer.overlap = 32  # larger overlap = smoother fields
   analyzer.search_area_size = 128
   starttime= 30
   endtime= 90



   # Process video (adjust frame_skip to process  fewer frames for testing)
   start_frame = int(starttime * 30)  # Start at 2 seconds
   max_pairs = int((endtime-starttime) * 30)  
   analyzer.process_video(start_frame, frame_skip=2, max_frames=max_pairs)  
   div_stats = analyzer.check_and_correct_divergence(tolerance=0.01, correct=True) #divergence forcing
   # Save results
#    analyzer.save_results('piv_results.pkl')
  
   # Plot a single velocity field
   analyzer.plot_velocity_field(frame_idx=30)
#    analyzer.save_velocity_data_range(30, 90, num_frames=None, output_folder='vframe_10sec')
   # Plot velocity statistics over time
#    analyzer.plot_velocity_time_series()


#    analyzer.plot_vorticity_field(frame_idx=30)
#    analyzer.plot_vorticity_time_series()
   # Create animation (optional - requires ffmpeg)
   # analyzer.create_animation('velocity_animation.mp4', fps=10)
  
   # Later, you can load saved results
   # analyzer2 = PIVVideoAnalyzer(video_path)
   # analyzer2.load_results('piv_results.pkl')
   # analyzer2.plot_velocity_field(frame_idx=10)




if __name__ == '__main__':
   main()

