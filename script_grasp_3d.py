#!/usr/bin/env python3
"""
3D Package Grasp Detection for Suction Grippers with Millimeter Coordinates.

This script uses MiDaS depth estimation to find the most suitable flat surface
for a suction gripper. It scores regions based on a combination of flatness,
accessibility, and centrality. It visualizes the grasp with a stable 2D arrow
direction while displaying the true 3D angle value.
"""

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Any, Dict
from pathlib import Path
import warnings
import math
warnings.filterwarnings("ignore")

# --- Configuration ---
@dataclass(frozen=True)
class Config:
    """A centralized configuration for the detection system."""
    MODEL_PATH: Path = Path('models/300epoch_best_pt/best.pt')
    IMAGE_PATH: Path = Path('3_normal_picking_angle/IMG_9103.jpeg')
    OUTPUT_IMAGE_PATH: Path = Path('grasp_visualization_3d_suction_patch.jpg')
    
    # --- IMPORTANT: CALIBRATION REQUIRED ---
    PIXELS_TO_MM_RATIO: float = 0.4 

    # --- Detection & Grasping Parameters ---
    YOLO_CONFIDENCE: float = 0.4
    MIN_CONTOUR_AREA: int = 2000
    SUCTION_PATCH_SIZE: int = 70
    
    FLATNESS_TOLERANCE: float = 0.015 
    MIN_GRASP_SCORE_THRESH: float = 0.7
    
    # --- Scoring Weights (should sum to 1.0) ---
    FLATNESS_WEIGHT: float = 0.4
    DEPTH_WEIGHT: float = 0.2
    CENTRALITY_WEIGHT: float = 0.4

    # --- Visualization Parameters ---
    ARROW_LENGTH: int = 120
    ARROW_THICKNESS: int = 8
    FONT: int = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE: float = 2.0
    FONT_THICKNESS: int = 8
    BEST_PATCH_COLOR: Tuple[int, int, int] = (255, 0, 0)
    POSSIBLE_PATCH_COLOR: Tuple[int, int, int] = (0, 255, 0)
    PATCH_ALPHA: float = 0.4

# --- Data Structures ---
@dataclass
class GraspResult3D:
    """Stores the final 3D grasp information for the single best point."""
    position_px: Tuple[int, int]
    position_mm: Tuple[float, float]
    patch_coords_px: Optional[Tuple[int, int, int, int]]
    depth_value: float
    normal_vector_3d: np.ndarray
    visual_angle_degrees: float # The stable 2D angle for drawing the arrow
    true_angle_degrees: float   # The accurate 3D angle for text display
    score: float

# --- MiDaS Depth Estimation ---
class MiDaSDepthEstimator:
    """Encapsulates the MiDaS model for monocular depth estimation."""
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Initializing MiDaS on device: {self.device}")
        try:
            self.model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small").to(self.device)
            self.model.eval()
            midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
            self.transform = midas_transforms.small_transform
            print("MiDaS model loaded successfully.")
        except Exception as e:
            print(f"ERROR: Could not load MiDaS model. {e}")
            self.model = None

    def estimate_depth(self, image: np.ndarray) -> Optional[np.ndarray]:
        if self.model is None: return None
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        input_tensor = self.transform(rgb_image).to(self.device)
        with torch.no_grad():
            prediction = self.model(input_tensor)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1), size=rgb_image.shape[:2],
                mode="bicubic", align_corners=False
            ).squeeze()
        depth_map = prediction.cpu().numpy()
        return cv2.normalize(depth_map, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

# --- Core Logic ---
class GraspDetector3D:
    """Analyzes images to find 3D grasp points for suction grippers."""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.object_detector = YOLO(self.cfg.MODEL_PATH)
        self.depth_estimator = MiDaSDepthEstimator()

    def analyze_image(self, img: np.ndarray) -> Tuple[List[GraspResult3D], List[np.ndarray], Dict[int, List[Tuple]]]:
        print("1. Estimating depth from image...")
        depth_map = self.depth_estimator.estimate_depth(img)
        if depth_map is None: return [], [], {}

        print("2. Detecting objects...")
        results = self.object_detector(img, verbose=False, conf=self.cfg.YOLO_CONFIDENCE)
        if not results or not results[0].masks: return [], [], {}

        print("3. Analyzing surfaces for suction grasp points...")
        grad_x = cv2.Sobel(depth_map, cv2.CV_64F, 1, 0, ksize=5)
        grad_y = cv2.Sobel(depth_map, cv2.CV_64F, 0, 1, ksize=5)
        
        best_grasp_results, all_contours, all_possible_patches = [], [], {}
        h, w = img.shape[:2]

        for i, mask_data in enumerate(results[0].masks):
            contour = self.get_contour_from_mask(mask_data, w, h)
            if contour is None or cv2.contourArea(contour) < self.cfg.MIN_CONTOUR_AREA:
                continue
            all_contours.append(contour)
            mask = np.zeros(img.shape[:2], dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            
            best_grasp, possible_patches = self.process_object_for_suction(contour, mask, depth_map, grad_x, grad_y)
            
            if best_grasp:
                best_grasp_results.append(best_grasp)
                all_possible_patches[i] = possible_patches
        
        print(f"Finished analysis. Found {len(best_grasp_results)} suitable grasp points.")
        return best_grasp_results, all_contours, all_possible_patches

    def get_contour_from_mask(self, mask_data: Any, w: int, h: int) -> Optional[np.ndarray]:
        mask_raw = mask_data.data.cpu().numpy()[0]
        mask_resized = (cv2.resize(mask_raw, (w, h)) * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_resized, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return max(contours, key=cv2.contourArea) if contours else None

    def process_object_for_suction(self, contour: np.ndarray, mask: np.ndarray, depth_map: np.ndarray, grad_x: np.ndarray, grad_y: np.ndarray) -> Tuple[Optional[GraspResult3D], List[Tuple]]:
        """Finds the best grasp point and calculates both 3D normal and 2D visual angle."""
        patch_size = self.cfg.SUCTION_PATCH_SIZE
        candidate_patches = []
        
        M = cv2.moments(mask)
        if M["m00"] == 0: return None, []
        centroid_x, centroid_y = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])

        points = np.argwhere(mask == 255)
        max_dist = np.max(np.linalg.norm(points - (centroid_y, centroid_x), axis=1))
        if max_dist == 0: max_dist = 1.0

        num_points_to_check = min(len(points), 500)
        indices = np.random.choice(len(points), num_points_to_check, replace=False)

        for idx in indices:
            y, x = points[idx]
            half_patch = patch_size // 2
            y_start, y_end, x_start, x_end = y - half_patch, y + half_patch, x - half_patch, x + half_patch

            if y_start < 0 or y_end >= depth_map.shape[0] or x_start < 0 or x_end >= depth_map.shape[1]:
                continue
            
            depth_patch = depth_map[y_start:y_end, x_start:x_end]
            
            std_dev = np.std(depth_patch)
            if std_dev <= self.cfg.FLATNESS_TOLERANCE:
                flatness_score = 1.0
            else:
                penalty_range = 0.1
                scaled_std = (std_dev - self.cfg.FLATNESS_TOLERANCE) / (penalty_range - self.cfg.FLATNESS_TOLERANCE)
                flatness_score = 1.0 - np.clip(scaled_std, 0, 1)

            mean_depth = np.mean(depth_patch)
            dist_to_centroid = np.linalg.norm((y - centroid_y, x - centroid_x))
            centrality_score = 1.0 - (dist_to_centroid / max_dist)

            score = (self.cfg.FLATNESS_WEIGHT * flatness_score + 
                     self.cfg.DEPTH_WEIGHT * mean_depth +
                     self.cfg.CENTRALITY_WEIGHT * centrality_score)

            if score >= self.cfg.MIN_GRASP_SCORE_THRESH:
                candidate_patches.append({"center": (x, y), "coords": (x_start, y_start, x_end, y_end), "score": score})

        if not candidate_patches:
            best_patch = {"center": (centroid_x, centroid_y), "coords": None, "score": 0.0}
        else:
            best_patch = max(candidate_patches, key=lambda p: p["score"])
        
        grasp_x, grasp_y = best_patch["center"]
        final_depth = depth_map[grasp_y, grasp_y]
        
        # Calculate TRUE 3D normal and its corresponding angle for the robot/text
        dz_dx, dz_dy = grad_x[grasp_y, grasp_y], grad_y[grasp_y, grasp_y]
        normal_3d = np.array([-dz_dx, -dz_dy, 1.0])
        norm = np.linalg.norm(normal_3d)
        if norm > 0: normal_3d /= norm
        true_angle_rad = math.atan2(-normal_3d[1], normal_3d[0])
        true_angle_deg = math.degrees(true_angle_rad)
        
        # Calculate STABLE 2D angle from contour shape for visualization arrow
        if len(contour) >= 5:
            _, _, ellipse_angle_deg = cv2.fitEllipse(contour)
            visual_angle_deg = (ellipse_angle_deg + 90.0) % 360
        else:
            visual_angle_deg = 0.0

        pos_mm_x, pos_mm_y = grasp_x * self.cfg.PIXELS_TO_MM_RATIO, grasp_y * self.cfg.PIXELS_TO_MM_RATIO

        best_grasp_result = GraspResult3D(
            position_px=(grasp_x, grasp_y),
            position_mm=(pos_mm_x, pos_mm_y),
            patch_coords_px=best_patch["coords"],
            depth_value=final_depth,
            normal_vector_3d=normal_3d,
            visual_angle_degrees=visual_angle_deg,
            true_angle_degrees=true_angle_deg,
            score=best_patch["score"]
        )
        
        possible_patch_coords = [p["coords"] for p in candidate_patches]
        return best_grasp_result, possible_patch_coords

# --- Visualization ---
class Visualizer:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def draw_results(self, image: np.ndarray, best_grasps: List[GraspResult3D], contours: List[np.ndarray], all_patches: Dict[int, List[Tuple]]) -> np.ndarray:
        output_image = image.copy()
        overlay = np.zeros_like(output_image, dtype=np.uint8)
        
        for i, contour in enumerate(contours):
            if i in all_patches:
                for patch_coords in all_patches[i]:
                    x_start, y_start, x_end, y_end = patch_coords
                    cv2.rectangle(overlay, (x_start, y_start), (x_end, y_end), self.cfg.POSSIBLE_PATCH_COLOR, -1)

        for result in best_grasps:
            if result.patch_coords_px:
                x_start, y_start, x_end, y_end = result.patch_coords_px
                cv2.rectangle(overlay, (x_start, y_start), (x_end, y_end), self.cfg.BEST_PATCH_COLOR, -1)

        output_image = cv2.addWeighted(overlay, self.cfg.PATCH_ALPHA, output_image, 1 - self.cfg.PATCH_ALPHA, 0)
        
        cv2.drawContours(output_image, contours, -1, (0, 255, 100), 3)
        for result in best_grasps:
            self.draw_grasp_info(output_image, result)
            
        return output_image

    def draw_text(self, image: np.ndarray, text: str, pos: Tuple[int, int]):
        cv2.putText(image, text, pos, self.cfg.FONT, self.cfg.FONT_SCALE, (255, 255, 255), self.cfg.FONT_THICKNESS + 4, cv2.LINE_AA)
        cv2.putText(image, text, pos, self.cfg.FONT, self.cfg.FONT_SCALE, (0, 0, 0), self.cfg.FONT_THICKNESS, cv2.LINE_AA)

    def draw_grasp_info(self, image: np.ndarray, result: GraspResult3D):
        x, y = result.position_px
        x_mm, y_mm = result.position_mm
        
        # Use the stable 2D angle for drawing the arrow
        angle_rad = math.radians(result.visual_angle_degrees)
        nx_vis, ny_vis = math.cos(angle_rad), math.sin(angle_rad)
        
        end_x = int(x + self.cfg.ARROW_LENGTH * nx_vis)
        end_y = int(y + self.cfg.ARROW_LENGTH * ny_vis)
        
        cv2.arrowedLine(image, (x, y), (end_x, end_y), (0, 0, 255), self.cfg.ARROW_THICKNESS, tipLength=0.3)
        
        cv2.circle(image, (x, y), 10, (0, 0, 255), -1)
        cv2.circle(image, (x, y), 12, (255, 255, 255), 2)

        # Use the true 3D angle for the text display
        text_coords = f"({x_mm:.1f}, {y_mm:.1f}) mm"
        text_angle = f"Angle: {result.true_angle_degrees:.1f} deg"
        self.draw_text(image, text_coords, (x + 35, y - 35))
        self.draw_text(image, text_angle, (x + 35, y + 40))

# --- Main Execution ---
def main():
    print("--- 3D Suction Grasp Detection System Initializing ---")
    cfg = Config()

    try:
        img = cv2.imread(str(cfg.IMAGE_PATH))
        if img is None: raise FileNotFoundError(f"Image not found at {cfg.IMAGE_PATH}")
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    detector = GraspDetector3D(cfg)
    best_grasp_results, all_contours, all_possible_patches = detector.analyze_image(img)

    if not best_grasp_results:
        print("Analysis complete. No suitable grasp points found.")
        return

    print("\n--- Final Grasp Results ---")
    for i, result in enumerate(best_grasp_results):
        print(f"\n[Object {i}]")
        print(f"  - Grasp Point (px): {result.position_px}")
        print(f"  - Grasp Point (mm): ({result.position_mm[0]:.2f}, {result.position_mm[1]:.2f})")
        print(f"  - TRUE 3D Normal (for robot): [{result.normal_vector_3d[0]:.3f}, {result.normal_vector_3d[1]:.3f}, {result.normal_vector_3d[2]:.3f}]")
        print(f"  - TRUE Angle (from 3D normal): {result.true_angle_degrees:.2f}")
        print(f"  - VISUAL Angle (from 2D shape): {result.visual_angle_degrees:.2f}")
        print(f"  - Best Score: {result.score:.3f}")

    visualizer = Visualizer(cfg)
    result_image = visualizer.draw_results(img, best_grasp_results, all_contours, all_possible_patches)

    cv2.imwrite(str(cfg.OUTPUT_IMAGE_PATH), result_image)
    print(f"\nResult visualization saved to: {cfg.OUTPUT_IMAGE_PATH}")

    h, w = result_image.shape[:2]
    display_h, display_w = 900, int(w * (900 / h))
    cv2.imshow('3D Suction Grasp Detection', cv2.resize(result_image, (display_w, display_h)))

    print("Press any key to close the window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("--- Analysis Complete ---")

if __name__ == "__main__":
    main()
