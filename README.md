# MV-Human-Tracker

**Tracking** is a Python-based framework designed for multi-view, multi-person tracking and pose estimation. It integrates 2D detections, pose estimation, and multi-view association to reconstruct and track human poses across multiple camera views.

## Features

- **2D Detection and Pose Estimation**: Utilizes detectors like YOLO and pose estimators such as MMPose to extract 2D keypoints from video frames.
- **Multi-View Association**: Associates detections across different camera views to reconstruct 3D poses.
- **Tracking**: Implements tracking algorithms to maintain consistent identities over time.
- **Evaluation Metrics**: Provides tools to evaluate tracking performance using standard metrics.

## Directory Structure

```
tracking/
├── configs/           # Configuration files
├── videos/           # Input video files
├── data/             # Processed data and results
├── src/              # Source code
├── requirements.txt  # Python dependencies
├── README.md         # This file
└── LICENSE          # License file
```

## Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/marcoalves20/tracking.git
   cd tracking
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Prepare Input Videos**: Place your input videos in the `videos/` directory.

2. **Run Detection and Pose Estimation**:
   ```bash
   python detect_pose.py --config configs/detector_config.yaml
   ```
   Replace `detector_config.yaml` with your specific configuration file.

3. **Perform Multi-View Association**:
   ```bash
   python match_multiview.py --config configs/association_config.yaml
   ```

4. **Run Tracking**:
   ```bash
   python tracker.py --config configs/tracker_config.yaml
   ```

5. **Evaluate Tracking Performance**:
   ```bash
   python metrics.py --config configs/evaluation_config.yaml
   ```

## Configuration

All configurations are stored in the `configs/` directory. Modify these YAML files to set parameters for detectors, pose estimators, association algorithms, and tracking settings.

## Dependencies

- Python 3.7+
- OpenCV
- NumPy
- PyYAML
- MMPose
- YOLO or other detectors

Install all dependencies using the provided `requirements.txt` file.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your enhancements.

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.

## Acknowledgements

- [MMPose](https://github.com/open-mmlab/mmpose) for pose estimation tools.
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) for object detection.
