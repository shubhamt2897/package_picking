# Robotic Grasp Detection for Logistics Automation

## 1. Project Summary

This project presents a robust computer vision solution designed to identify and calculate optimal grasp points for packages in cluttered, real-world logistics environments. Given a single RGB camera feed, the system detects multiple packages, determines the most suitable surface for picking, and computes the necessary `(x, y, angle)` coordinates for a robotic arm.

The solution was developed through an iterative process, beginning with a foundational 2D geometric approach and evolving into a sophisticated 3D depth-based system, demonstrating adaptability and a deep understanding of both classic and modern computer vision techniques. The final implementation is optimized for suction grippers, which are ideal for handling the varied shapes and sizes of modern e-commerce packages.

### Core Competencies Demonstrated:

- **Machine Learning:** Training and fine-tuning deep learning models (YOLOv8) for instance segmentation
- **Computer Vision:** Advanced image processing, feature extraction, and 3D scene understanding from 2D inputs
- **Robotics & Automation:** Calculating precise grasp coordinates and surface normals for robotic manipulation
- **Algorithm Design:** Developing a multi-faceted scoring system to find optimal solutions based on flatness, centrality, and accessibility
- **Python & AI Libraries:** Proficient use of PyTorch, OpenCV, Ultralytics, and NumPy

## 2. Task Fulfillment & Key Features

This project successfully fulfills all core and bonus objectives outlined in the "Normal Angle Package Picking" challenge.

### Optimal Surface Detection
The 3D system actively searches for the best picking surface by analyzing hundreds of potential "patches" on an object. It scores each patch based on a weighted combination of:

- **Surface Flatness:** To quantify the flatness of a potential grasp area, the system calculates the standard deviation of depth values within a candidate "patch" from the MiDaS-generated depth map. A low standard deviation signifies a flat, stable surface ideal for a suction seal. A tolerance is included to ignore minor, acceptable creases.

- **Positional Centrality:** Prioritizes stable, central points and avoids unreliable edges.

- **Accessibility:** Identifies the highest, most reachable surfaces based on depth values.

### Advanced Features
- **Surface Normal Estimation:** The physically accurate 3D surface normal vector `[nx, ny, nz]` is computed from the gradients `(dz/dx, dz/dy)` of the depth map at the optimal grasp point.

- **Robot-Ready Output:** The system returns the grasp point in real-world millimeters and the true 3D angle of the surface normal, directly satisfying the bonus requirement.

- **Advanced Visualization:** The solution visualizes all potential flat surfaces, highlights the best choice with a distinct color, and overlays a clear arrow and text annotations.

- **Depth Integration:** The project successfully integrates monocular depth estimation (MiDaS) to perform true 3D analysis from a single RGB image, fulfilling another bonus requirement.

## 3. Models & Training Strategy

### Object Detection (YOLOv8)

**Model:** A YOLOv8 segmentation model is used to detect and generate pixel-perfect masks for each package.

**Dataset & Training Strategy:** The model was trained using a two-stage transfer learning approach to improve its robustness:

1. **Base Training:** The model was first trained for 200 epochs on the standard `package-seg.yaml` dataset provided by Ultralytics. This dataset primarily contains images of single, well-defined packages, which helped the model learn basic feature extraction.

2. **Fine-Tuning:** The model was then fine-tuned for an additional 100 epochs on a more complex dataset from Roboflow (`rf.workspace("jians").project("finalmerged-ikg5m")`, version 1). This second dataset includes images with multiple, overlapping packages, intended to improve performance in cluttered scenes.

### Depth Estimation (MiDaS)

**Model:** The `intel-isl/MiDaS` (small version) model is loaded directly from PyTorch Hub.

**Function:** It takes the 2D RGB image as input and outputs a relative depth map, which is the foundation for all 3D calculations.

## 4. Key Assumptions

- **Camera Orientation:** The system assumes a top-down or near-top-down camera angle for the grasp analysis to be effective.

- **Camera Calibration:** The conversion from pixel coordinates to millimeters relies on a `PIXELS_TO_MM_RATIO` constant. This value must be calibrated for the specific camera and mounting height.

- **Surface Properties:** The grasp logic assumes that the flattest, most central, and most accessible point is the optimal grasp point for a suction gripper.

## 5. Challenges & Solutions

### Challenge 1: Overlapping Package Detection
**Problem:** The initial object detection model struggled to separate overlapping packages, merging them into a single entity.

**Solution:** Implemented the two-stage training strategy detailed above. Fine-tuning on a dataset with cluttered scenes significantly improved the model's ability to perform instance separation.

### Challenge 2: Depth Estimation Artifacts
**Problem:** Monocular depth estimation (MiDaS) produced noisy or inaccurate depth values along the shiny, high-contrast edges of polybags.

**Solution:** Engineered a "centrality" score into the grasp selection algorithm. This successfully biases the system to choose stable surfaces near the object's center of mass, inherently avoiding the problematic edges.

### Challenge 3: Visualization Clarity
**Problem:** The mathematically correct 3D normal vector on a wrinkled surface often resulted in a visually counter-intuitive arrow direction.

**Solution:** Developed a hybrid visualization. The arrow's direction is based on the object's stable 2D shape for clarity, while the printed angle value remains the true, physically accurate 3D normal data required by the robot.

## 6. Project Structure

```
package_picking/
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies
├── package-seg.yaml                   # Dataset configuration
├── 
├── # Main Scripts
├── script_grasp_2d.py                 # 2D geometric grasp detection
├── script_grasp_3d                    # Primary 3D depth-based grasp detection
├── simple_grasp_detector.py           # Simple 2D geometric grasp detection
├── train_model.py                     # Model training script
├── 
├── # Pre-trained Models
├── yolo11n.pt                         # YOLO11 nano model
├── yolov8n-seg.pt                     # YOLOv8 nano segmentation model
├── yolov8s-seg.pt                     # YOLOv8 small segmentation model
├── 
├── # Trained Models
├── models/
│   ├── best.pt                        # Best performing model
│   ├── 200epoch/                      # 200 epoch training results
│   ├── 250epoch_best_pt/              # 250 epoch training results
│   └── 300epoch_best_pt/              # 300 epoch training results
├── 
├── # Results and Examples
├── results images/                    # Output examples and visualizations
│   ├── simple_grasp_result_object_instance_segmentation.jpg
│   ├── 2D Suction Grasp Detection_screenshot.png
│   ├── 3D Grasp Detection.png
│   └── grasp_visualization_3d_suction_patch.jpg
├── 
├── # Test Images
├── 3_normal_picking_angle/            # Test images with normal picking angles
│   ├── IMG_9102.jpeg
│   ├── IMG_9102_1.jpeg
│   ├── IMG_9103.jpeg
│   └── IMG_9104.jpeg
├── 
├── # Dataset
├── datasets/
│   └── package-seg/                   # Package segmentation dataset
│       ├── package-seg.yaml
│       ├── images/                    # Training, validation, and test images
│       └── labels/                    # Corresponding labels
├── 
└── # Training Results

```

## 7. Installation & Usage

### Prerequisites

```bash
pip install -r requirements.txt
```

or manually install the dependencies:

```bash
pip install ultralytics opencv-python torch torchvision numpy
```

### Running the Scripts

#### For 3D depth-based grasp detection (recommended):
```bash
python script_grasp_3d
```

#### For 2D geometric grasp detection:
```bash
python simple_grasp_detector.py
```

#### For alternative 2D approach:
```bash
python script_grasp_2d.py
```

### Configuration

**Important:** You must calibrate the `PIXELS_TO_MM_RATIO` constant for your specific camera setup:

1. Take a picture with your camera setup
2. Measure a known, flat object in the image in both pixels and real-world millimeters
3. Calculate the ratio: `ratio = real_world_mm / pixels_in_image`
4. Update the constant in the script

## 8. Key Features

- **Dual Approach:** Both 2D geometric and 3D depth-based methods
- **YOLOv8 Integration:** Accurate package segmentation with transfer learning
- **YOLO11 Support:** Compatibility with latest YOLO architecture
- **3D Surface Analysis:** MiDaS depth estimation for optimal grasp points
- **Real-world Coordinates:** Converts pixel coordinates to millimeter measurements
- **Multi-gripper Support:** Optimized for both suction and mechanical grippers
- **Visualization:** Clear visual feedback with grasp points and angles
- **Comprehensive Training:** 300+ epoch training with multiple datasets

## 9. Example Results

The project includes several example outputs in the `results images/` directory:

- **3D Grasp Detection:** Shows the advanced depth-based approach with optimal suction points
- **2D Suction Grasp Detection:** Demonstrates the geometric approach for comparison  
- **Object Instance Segmentation:** Displays the YOLOv8 segmentation accuracy
- **Grasp Visualization:** Shows the 3D suction patch selection process

## 10. Contributing

We welcome contributions to improve the package picking system:

### Priority Areas
- **Instance Segmentation:** Improve separation of overlapping packages
- **Depth Quality:** Enhance MiDaS depth estimation accuracy
- **Camera Calibration:** Automate pixel-to-millimeter ratio detection
