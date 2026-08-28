# pjas2026 - See the file labled PJAS 2026 pdf for more info

Particle Image Velocimetry (PIV) analysis pipeline for extracting 2D fluid velocity fields from video, and studying how kinetic energy decays over time. Built for a PJAS (Pennsylvania Junior Academy of Science) 2026 project.

> See the file labeled **Presentation** in this repo for the full project writeup.

## What this does

The pipeline takes a video of a fluid seeded with tracer particles (e.g. water in a rotating container) and:

1. Tracks particle motion frame-by-frame to compute a 2D velocity field (`u`, `v`) at each point in the frame, using [OpenPIV](https://openpiv.readthedocs.io/).
2. Exports velocity data for a chosen time range to CSV files.
3. Computes the kinetic energy of the flow over time from those CSVs and fits it to an exponential decay model, `E(t) = E₀ · exp(-t/τ)`, to characterize how quickly the flow loses energy (e.g. due to viscous damping).
4. Optionally checks/enforces divergence-free (incompressible) flow and computes vorticity (rotation) of the field.

## Files

| File | Purpose |
|---|---|
| `piv.py` | Core `PIVVideoAnalyzer` class — baseline PIV pipeline (preprocess frames, run PIV, validate/filter outliers, plot velocity fields, animate, compute velocity statistics over time). |
| `sdf.py` | Extended version of the analyzer with: interactive particle-detection preview, pixel→physical unit calibration, max-velocity/edge outlier rejection, local median filtering, divergence checking + FFT-based divergence correction, vorticity field/time-series plots, and CSV export over a time range (`save_velocity_data_range`). |
| `energyte.py` | Reads the exported per-frame velocity CSVs, computes kinetic energy `KE = ½ρh∫∫|u|²dA` over time, plots KE vs. time, fits an exponential decay curve, and saves a summary CSV + fit parameters. |
| `test.py` | Test/scratch script. |
| `requirements.txt` | Python dependencies. |
| `PXL_20260109_225752957~2.mp4` | Sample/source video used for analysis. |

## Requirements

- Python 3.8+
- Dependencies in `requirements.txt`:
  - `numpy>=1.20.0`
  - `opencv-python>=4.5.0`
  - `matplotlib>=3.3.0`
  - `openpiv>=0.25.0`
  - `scipy>=1.7.0`
- `pandas` (used by `energyte.py` / CSV export)
- `ffmpeg` on your PATH if you want to render animations (`create_animation`)

Install:

```bash
pip install -r requirements.txt pandas
```

## Setup

Before running, open `piv.py` / `sdf.py` and `energyte.py` and adjust the configuration values for your own setup:

- **Video path** — path to your input video.
- **PIV parameters** — `window_size`, `overlap`, `search_area_size`, `dt` (frame interval).
- **Calibration** — `pixels_per_meter` / `pixels_per_cm` (how many pixels correspond to a real-world distance in your footage) and `fps`.
- **Physical constants** (in `energyte.py`) — `rho` (fluid density) and `h` (depth of the fluid layer).
- **Time range** — start/end times (seconds) for the section of video you want to analyze.

## Usage

### 1. Extract velocity fields from video

```bash
python sdf.py
```

This will (see `main()` for the exact sequence used):

- Preview particle detection to help you pick a good threshold (`visualize_particle_detection`).
- Run PIV over the chosen frame range (`process_video`).
- Check/correct divergence to enforce incompressible flow (`check_and_correct_divergence`).
- Plot a sample velocity field (`plot_velocity_field`).
- Export per-frame velocity CSVs for a time window (`save_velocity_data_range`).

`piv.py` contains the simpler baseline version of the same pipeline if you don't need the extra filtering/calibration steps.

### 2. Compute kinetic energy decay

Point `vframe_folder` in `energyte.py` at the folder of CSVs produced above, then run:

```bash
python energyte.py
```

This produces:

- `kinetic_energy_vs_time.png` — KE vs. time with the fitted exponential decay curve.
- `kinetic_energy_summary.csv` — timestamp, kinetic energy, and mean velocity per frame.
- `exponential_fit_parameters.txt` — fitted `E₀`, `τ`, R², and derived decay times (37%, 50%, 10%).

## Output

Running the full pipeline produces, under your chosen output directory:

- `velocity_field_frame_XXXX.png` — quiver + contour plots of the velocity field for a given frame.
- `velocity_time_series.png`, `vorticity_time_series.png`, `divergence_check.png` — statistics over time.
- `vframe_*/velocity_frame_XXXX.csv` — per-frame velocity data (x, y, u, v, magnitude).
- `kinetic_energy_vs_time.png`, `kinetic_energy_summary.csv`, `exponential_fit_parameters.txt` — energy decay analysis.
