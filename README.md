# Object-based Sensor Data Fusion

This project implements a sensor data fusion pipeline that combines radar and camera data for object detection and tracking.

## Features

- Radar point cloud processing and clustering using custom DBSCAN algorithm
- YOLO-based object detection on camera images
- Sensor calibration and coordinate transformation
- Association of radar clusters with image bounding boxes
- Visualization of fused data on both image and ground planes

## Dependencies

- Python 3.x
- NumPy
- Pandas
- Matplotlib
- OpenCV (cv2)
- PyTorch
- Ultralytics YOLO

## Usage

1. Set up the required folder structure:
   - Images folder
   - Radar PCD folder
   - Calibration file

2. Update the paths in the `main()` function:
   - `path_to_images`
   - `path_to_pcd`
   - `calibration_file`
   - YOLO model path

3. Run the main script:
   ```
   python main.py
   ```

## Key Components

- `my_custom_dbscan`: Custom DBSCAN implementation for radar point clustering
- `radar_to_ground_transfomer`: Transforms radar points to ground plane
- `radar_to_camera_transformer`: Projects radar points onto the image plane
- `get_association_matrix`: Creates association matrix between radar clusters and image bounding boxes
- `get_filtered_cases`: Analyzes different association cases (one-to-one, one-to-many, many-to-one)
- `get_image_visualization`: Visualizes the associated objects on the image plane

## Output

The script generates visualizations showing:
1. Fused data on the image plane with bounding boxes and radar points
2. Top-down view of objects on the ground plane

## Note

This project is designed for research and development purposes in the field of multi-sensor fusion for autonomous systems.
