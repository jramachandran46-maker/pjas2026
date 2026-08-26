"""
Particle Image Velocimetry (PIV) Video Analysis
Processes video to extract 2D velocity fields as a function of time using OpenPIV
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import openpiv.tools
import openpiv.process
import openpiv.scaling
import openpiv.validation
import openpiv.filters
from pathlib import Path
import pickle


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
        self.window_size = 32  # interrogation window size in pixels
        self.overlap = 16  # overlap between windows
        self.search_area_size = 64  # search area for cross-correlation
        self.dt = 1  # time between frames (adjust based on your video fps)
        
        # Storage for velocity fields
        self.velocity_fields = []
        self.x_coords = None
        self.y_coords = None
        self.timestamps = []
        
    def preprocess_frame(self, frame):
        """
        Preprocess video frame for PIV analysis
        
        Parameters:
        -----------
        frame : numpy array
            Input frame (RGB or grayscale)
            
        Returns:
        --------
        numpy array
            Preprocessed grayscale frame
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
            
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Optional: Gaussian blur to reduce noise
        # enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        return enhanced.astype(np.int32)
    
    def process_video(self, frame_skip=1, max_frames=None):
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
        
        frame_count = 0
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
        u, v, sig2noise = openpiv.process.extended_search_area_piv(
            frame_a, frame_b,
            window_size=self.window_size,
            overlap=self.overlap,
            dt=self.dt,
            search_area_size=self.search_area_size,
            sig2noise_method='peak2peak'
        )
        
        # Get grid coordinates
        x, y = openpiv.process.get_coordinates(
            image_size=frame_a.shape,
            search_area_size=self.search_area_size,
            overlap=self.overlap
        )
        
        # Validation - remove outliers
        u, v, mask = openpiv.validation.sig2noise_val(
            u, v,
            sig2noise,
            threshold=1.3
        )
        
        # Replace outliers using interpolation
        u, v = openpiv.filters.replace_outliers(
            u, v,
            method='localmean',
            max_iter=3,
            kernel_size=2
        )
        
        # Optional: Apply smoothing
        # u, v = openpiv.filters.gaussian(u, v, sigma=1)
        
        return u, v, x, y, mask
    
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
        ax1.quiver(self.x_coords, self.y_coords, u, -v, 
                   velocity_mag, cmap='jet', scale=50*scale)
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
    video_path = 'path/to/your/video.mp4'  # Change this to your video path
    
    # Initialize analyzer
    analyzer = PIVVideoAnalyzer(video_path, output_dir='piv_output')
    
    # Adjust PIV parameters if needed
    analyzer.window_size = 32  # smaller = more detail, but noisier
    analyzer.overlap = 16  # larger overlap = smoother fields
    analyzer.search_area_size = 64
    
    # Process video (adjust frame_skip to process fewer frames for testing)
    analyzer.process_video(frame_skip=2, max_frames=50)  # Process first 50 frame pairs
    
    # Save results
    analyzer.save_results('piv_results.pkl')
    
    # Plot a single velocity field
    analyzer.plot_velocity_field(frame_idx=0)
    
    # Plot velocity statistics over time
    analyzer.plot_velocity_time_series()
    
    # Create animation (optional - requires ffmpeg)
    # analyzer.create_animation('velocity_animation.mp4', fps=10)
    
    # Later, you can load saved results
    # analyzer2 = PIVVideoAnalyzer(video_path)
    # analyzer2.load_results('piv_results.pkl')
    # analyzer2.plot_velocity_field(frame_idx=10)


if __name__ == '__main__':
    main()