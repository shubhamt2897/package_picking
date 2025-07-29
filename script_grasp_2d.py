#!/usr/bin/env python3
"""
Simplified 2D Package Grasp Detection for Suction Grippers.

This script detects packages, identifies the most stable central point (centroid),
and calculates a normal angle based on the object's 2D orientation.
"""

import cv2
import numpy as np
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
import math

# --- Configuration ---
@dataclass(frozen=True)
class Config:
    """A centralized configuration for the detection system."""
    MODEL_PATH: Path = Path('models/300epoch_best_pt/best.pt')
    IMAGE_PATH: Path = Path('3_normal_picking_angle/IMG_9103.jpeg')
    OUTPUT_IMAGE_PATH: Path = Path('grasp_visualization_2d_suction.jpg')
    
    PIXELS_TO_MM_RATIO: float = 0.4 
    YOLO_CONFIDENCE: float = 0.4
    MIN_CONTOUR_AREA: int = 2000

    # --- Visualization Parameters ---
    ARROW_LENGTH: int = 100
    ARROW_THICKNESS: int = 6
    FONT: int = cv2.FONT_HERSHEY_SIMPLEX
    FONT_SCALE: float = 1.6
    FONT_THICKNESS: int = 6

# --- Data Structure ---
@dataclass
class GraspResult2D:
    """Stores the simplified 2D grasp information."""
    position_px: Tuple[int, int]
    position_mm: Tuple[float, float]
    angle_degrees: float

# --- Core Logic ---
class GraspDetector2D:
    """Analyzes package contours to find 2D grasp points."""
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.model = YOLO(self.cfg.MODEL_PATH)
        print("YOLOv8 model loaded successfully.")

    def analyze_image(self, img: np.ndarray) -> Tuple[List[GraspResult2D], List[np.ndarray]]:
        """Runs the full 2D detection and analysis pipeline."""
        print("Running package detection...")
        results = self.model(img, verbose=False, conf=self.cfg.YOLO_CONFIDENCE)

        if not results or not results[0].masks:
            print("No packages detected.")
            return [], []

        grasp_results, all_contours = [], []
        h, w = img.shape[:2]

        for mask_data in results[0].masks:
            mask = mask_data.data.cpu().numpy()[0]
            mask_uint8 = (cv2.resize(mask, (w, h)) * 255).astype(np.uint8)
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if not contours: continue
            main_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(main_contour) < self.cfg.MIN_CONTOUR_AREA: continue

            all_contours.append(main_contour)
            grasp_result = self.process_contour(main_contour)
            if grasp_result:
                grasp_results.append(grasp_result)

        print(f"Finished analysis. Found {len(grasp_results)} suitable grasp points.")
        return grasp_results, all_contours

    def process_contour(self, contour: np.ndarray) -> Optional[GraspResult2D]:
        """Calculates the 2D grasp point (centroid) and angle for a single contour."""
        M = cv2.moments(contour)
        if M["m00"] == 0: return None
        
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        
        pos_mm_x = cx * self.cfg.PIXELS_TO_MM_RATIO
        pos_mm_y = cy * self.cfg.PIXELS_TO_MM_RATIO

        # Calculate angle from the overall shape
        if len(contour) >= 5:
            _, _, ellipse_angle_deg = cv2.fitEllipse(contour)
            angle_deg = (ellipse_angle_deg + 90.0) % 360
        else:
            angle_deg = 0.0

        return GraspResult2D(
            position_px=(cx, cy),
            position_mm=(pos_mm_x, pos_mm_y),
            angle_degrees=angle_deg
        )

# --- Visualization ---
class Visualizer:
    """Handles drawing all analysis results on the image."""
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def draw_results(self, image: np.ndarray, grasp_results: List[GraspResult2D], contours: List[np.ndarray]) -> np.ndarray:
        output_image = image.copy()
        cv2.drawContours(output_image, contours, -1, (0, 255, 100), 3)
        for result in grasp_results:
            self.draw_grasp_info(output_image, result)
        return output_image

    def draw_text(self, image: np.ndarray, text: str, pos: Tuple[int, int]):
        cv2.putText(image, text, pos, self.cfg.FONT, self.cfg.FONT_SCALE, (255, 255, 255), self.cfg.FONT_THICKNESS + 4, cv2.LINE_AA)
        cv2.putText(image, text, pos, self.cfg.FONT, self.cfg.FONT_SCALE, (0, 0, 0), self.cfg.FONT_THICKNESS, cv2.LINE_AA)

    def draw_grasp_info(self, image: np.ndarray, result: GraspResult2D):
        """Draws an arrow, marker, and text for the grasp point."""
        x, y = result.position_px
        x_mm, y_mm = result.position_mm
        angle = result.angle_degrees

        # Calculate arrow end point from angle
        angle_rad = math.radians(angle)
        end_x = int(x + self.cfg.ARROW_LENGTH * math.cos(angle_rad))
        end_y = int(y + self.cfg.ARROW_LENGTH * math.sin(angle_rad))

        # Draw the arrow and grasp point marker
        cv2.arrowedLine(image, (x, y), (end_x, end_y), (0, 0, 255), self.cfg.ARROW_THICKNESS, tipLength=0.3)
        cv2.circle(image, (x, y), 15, (0, 0, 255), 3)
        cv2.circle(image, (x, y), 5, (0, 0, 255), -1)
        
        # Draw the text labels
        text_coords = f"({x_mm:.1f}, {y_mm:.1f}) mm"
        text_angle = f"Angle: {angle:.1f} deg"
        self.draw_text(image, text_coords, (x + 30, y - 15))
        self.draw_text(image, text_angle, (x + 30, y + 45))

# --- Main Execution ---
def main():
    """Main function to run the detection, analysis, and visualization."""
    print("--- Simplified 2D Suction Grasp Detection ---")
    cfg = Config()

    try:
        img = cv2.imread(str(cfg.IMAGE_PATH))
        if img is None: raise FileNotFoundError(f"Image not found at {cfg.IMAGE_PATH}")
    except Exception as e:
        print(f"Error: {e}")
        return

    detector = GraspDetector2D(cfg)
    grasp_results, all_contours = detector.analyze_image(img)

    if not grasp_results:
        print("Analysis complete. No grasp points found.")
        return

    print("\n--- Analysis Results ---")
    for i, result in enumerate(grasp_results):
        print(f"\n[Object {i}]")
        print(f"  - Grasp Point (px): {result.position_px}")
        print(f"  - Grasp Point (mm): ({result.position_mm[0]:.2f}, {result.position_mm[1]:.2f})")
        print(f"  - Grasp Angle (deg): {result.angle_degrees:.2f}")

    visualizer = Visualizer(cfg)
    result_image = visualizer.draw_results(img, grasp_results, all_contours)

    cv2.imwrite(str(cfg.OUTPUT_IMAGE_PATH), result_image)
    print(f"\nResult visualization saved to: {cfg.OUTPUT_IMAGE_PATH}")

    h, w = result_image.shape[:2]
    display_h, display_w = 900, int(w * (900 / h))
    cv2.imshow('2D Suction Grasp Detection', cv2.resize(result_image, (display_w, display_h)))

    print("Press any key to close the window.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    print("--- Analysis Complete ---")

if __name__ == "__main__":
    main()
