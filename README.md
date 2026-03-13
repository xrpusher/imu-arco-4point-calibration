# IMU ArUco 4-Point Calibration

Local Windows desktop MVP for estimating the relative geometry of 4 IMU attachment points using ArUco markers, a laptop webcam, and robust frame aggregation.

The project runs fully offline and locally:

1. Generates 4 printable ArUco tags in `PNG` and `JPG`.
2. Supports both quick and chessboard-based camera calibration.
3. Shows live webcam preview with ArUco detection.
4. Collects only good-quality frames from multiple poses.
5. Estimates the 3D geometry of 4 points.
6. Exports:
   - `calibration.json`
   - `calibration_report.csv`
   - `calibration_report.txt`

Tech stack:

- Python 3.13
- PyQt6
- OpenCV (`opencv-contrib-python` is required for `cv2.aruco`)
- NumPy
- SciPy

## Project Structure

```text
project_root/
  main.py
  requirements.txt
  README.md
  app/
    __init__.py
    gui/
      __init__.py
      main_window.py
      generate_tags_widget.py
      camera_calibration_widget.py
      capture_calibrate_widget.py
      results_widget.py
    core/
      __init__.py
      aruco_utils.py
      camera_manager.py
      camera_calibration.py
      session_recorder.py
      geometry_solver.py
      export_utils.py
      models.py
    utils/
      __init__.py
      paths.py
      logging_utils.py
```

## Create a Virtual Environment on Windows

Open PowerShell in the project root:

```powershell
python -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Upgrade packaging tools:

```powershell
python -m pip install --upgrade pip setuptools wheel
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

## Run

```powershell
python main.py
```

The application contains 4 tabs:

- `Generate Tags`
- `Camera Calibration`
- `Capture & Calibrate`
- `Results`

All tabs are wrapped in scroll areas so controls and previews remain accessible even on smaller screens.

## Tag Generation

Open the `Generate Tags` tab and configure:

- ArUco dictionary, default `DICT_6X6_250`
- print layout preset
- 4 tag IDs, default `10, 20, 30, 40`
- physical tag size in millimeters
- output image size in pixels
- output folder

Click `Generate Tags`.

The app saves:

- `tag_10.png`, `tag_10.jpg`
- `tag_20.png`, `tag_20.jpg`
- `tag_30.png`, `tag_30.jpg`
- `tag_40.png`, `tag_40.jpg`
- `contact_sheet.png`
- `contact_sheet.jpg`

Important:

- `PNG` is recommended for printing
- `JPG` is also exported, but compression may slightly reduce marker quality

### B21S Preset

If you use a `B21S` thermal label printer with black-and-white `50x30 mm` labels at `203 dpi`, the app includes a ready-made preset:

- file canvas: `400x240 px`
- layout optimized for a `50x30 mm` label
- actual square marker size on the label: about `27.0 mm`
- the text label is moved to the right so the ArUco marker remains as large as possible

This preset is now the default configuration in the generator.

## Camera Calibration

Use the `Camera Calibration` tab.

### Quick Mode

Quick mode estimates camera intrinsics from frame size and an assumed horizontal FOV.

Use it only for rough testing when a chessboard is not available.

This mode is less accurate.

### Recommended Mode: Chessboard Calibration

1. Start the camera preview.
2. Show the chessboard fully inside the frame.
3. Capture at least 8 good frames from different viewpoints and distances.
4. Click `Calibrate and Save`.
5. Save the resulting `camera_calibration.json`.

That calibration file is then used for ArUco pose estimation.

## Dry Calibration Session

Use the `Capture & Calibrate` tab.

1. Select camera index and resolution.
2. Set the ArUco dictionary and the physical marker size.
3. Enter the 4 required tag IDs.
4. Start the camera.
5. Confirm that all 4 markers are visible and detected.
6. Start the session.
7. Perform 5-10 noticeably different poses or viewpoints.
8. Finish the session.

The live preview shows:

- marker boxes
- tag IDs
- pose axes, if camera calibration is available
- detected IDs in the current frame
- status from `Found 0/4` to `Found 4/4`
- approximate marker size in pixels
- blur score
- angle warning
- live pairwise distances
- large distance vectors directly overlaid on the camera image

### Good Frame Criteria

A frame is accepted only if:

- all 4 required IDs are visible
- pose estimation succeeds
- reprojection error is acceptable
- markers are not too small
- the frame is not too blurry
- the viewing angle is not too acute
- the frame is not too similar to the previous accepted frame

## Important Note About Phone Testing

If you simply open `contact_sheet.png` on a phone and show the full 2x2 sheet to the camera, that does not mean the markers are wrong.

The usual problem is different:

- each tag becomes too small on the phone screen
- the camera sees moire, glare, and motion blur
- only 1-2 tags are detected reliably instead of all 4

For phone testing it is better to:

- open individual files such as `tag_10.png`, `tag_20.png`, and so on
- set screen brightness close to 100%
- keep the screen flatter to the camera
- move the phone closer to the webcam
- avoid glare

For real calibration, printed `PNG` tags are strongly recommended.

## Geometry Estimation Modes

After `Finish Session`, the app estimates:

- local 3D coordinates of the 4 points
- pairwise distances
- robust summary statistics

### 1. Simple Mode

Simple mode:

- computes pairwise distances for every accepted frame
- applies robust filtering
- uses the median as the final distance
- reconstructs a relative local 3D shape from those distances

Important:

- local axes in this mode are only conventional
- the distances themselves are the main useful output

For the "arms extended sideways" scenario with 4 sensors, this is the recommended mode.

### 2. Torso Frame Mode

This mode is available when you want an explicit local coordinate frame.

The user selects:

- top / origin point
- lower point
- plane reference point

The application builds a torso-local coordinate frame and robustly aggregates coordinates across accepted frames.

## Exported Files

Every completed session creates a timestamped output folder containing at least:

- `camera_calibration_used.json`
- `calibration.json`
- `calibration_report.csv`
- `calibration_report.txt`

### `calibration.json`

Contains:

- version
- timestamp
- ArUco dictionary
- tag size
- tag IDs
- camera calibration file path
- used and rejected frame counts
- local 3D coordinates
- pairwise distances
- pairwise statistics
- quality notes

### `calibration_report.csv`

Contains one row per point pair:

- final distance
- mean
- median
- standard deviation
- MAD
- sample count

### `calibration_report.txt`

Human-readable session and geometry summary.

## Capture Recommendations

For better results:

- use good lighting
- keep markers large in the frame
- avoid motion blur
- do not move too far from the camera
- prefer printed `PNG` markers
- mount tags on a rigid flat surface if possible
- keep the camera static if possible
- collect multiple clearly different viewpoints instead of many nearly identical frames

## MVP Limitations

- works best with a static camera
- geometry estimation is more reliable when markers are rigidly attached
- if clothing stretches, folds, or shifts a lot, accuracy drops
- the current implementation is designed specifically for 4 markers
- quick camera calibration is approximate and reduces pose accuracy
- the contact sheet is a practical preview sheet, not a strict print-production layout
- the project is intended for local Windows execution and is not yet packaged as an `.exe`

## Common Issues

### `cv2.aruco` Is Unavailable

Reinstall the OpenCV contrib build:

```powershell
pip uninstall opencv-python opencv-contrib-python -y
pip install opencv-contrib-python
```

### Camera Does Not Open

- make sure the webcam is not already in use by another app
- try another camera index
- restart the application after reconnecting the camera

### A Lock Icon Appears in the Preview

This is usually a Windows privacy block or a hardware privacy feature, not a PyQt/OpenCV bug.

Check:

- `Settings -> Privacy & security -> Camera`
- `Camera access`
- `Let apps access your camera`
- `Let desktop apps access your camera`
- Zoom, Teams, Discord, OBS, browsers, or vendor camera utilities
- physical privacy shutter
- `Fn` key with a camera icon

### Detection Quality Is Poor

- move closer
- improve lighting
- reduce motion
- use fresh high-contrast tags
- avoid glare on glossy surfaces

## Future Improvements

The project is already split into clear modules:

- GUI: `app/gui`
- computer vision and geometry: `app/core`
- paths and logging: `app/utils`

This makes future work easier:

- camera presets
- `.exe` packaging
- richer quality metrics
- session replay
- alternative local coordinate systems
