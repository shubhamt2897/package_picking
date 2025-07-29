#!/usr/bin/env python3
"""
Package Grasp Detection System
Object-oriented implementation for package detection and grasp point calculation.
"""

import cv2
import numpy as np
import math
from ultralytics import YOLO
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

# Configuration
MODEL_PATH = '/home/shubham-0802/package_picking/models/nd3/best.pt'
IMAGE_PATH = '/home/shubham-0802/package_picking/3_normal_picking_angle/IMG_9103.jpeg'


@dataclass
class GraspPoint:
    """Data class for storing grasp point information"""
    package_id: int
    rank: int
    grasp_x: int
    grasp_y: int
    center_x: int
    center_y: int
    angle: float
    area: float
    confidence: float
    grasp_score: float
    surface_quality: float
    flat_surface_length: float
    flat_surface_center: Optional[Tuple[int, int]]
    is_best: bool


@dataclass
class PackageCandidate:
    """Data class for storing package analysis data"""
    index: int
    mask: np.ndarray
    main_contour: np.ndarray
    safe_region: np.ndarray
    package_area: float
    safe_area: float
    confidence: float
    surface_quality: float
    center_score: float
    grasp_score: float
    center_x: int
    center_y: int
    aspect_ratio: float

class PackageDetector:
    """Main class for package detection and grasp point calculation"""
    
    def __init__(self, model_path: str):
        """Initialize the detector with a YOLO model"""
        self.model_path = model_path
        self.model = None
        self.colors = [
            (255, 0, 0),    # Red
            (0, 255, 0),    # Green  
            (0, 0, 255),    # Blue
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        self.load_model()
    
    def load_model(self) -> bool:
        """Load the YOLO model"""
        print(f"Loading model: {self.model_path}")
        try:
            self.model = YOLO(self.model_path)
            print("Model loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model: {e}")
            return False
    
    def load_image(self, image_path: str) -> Optional[np.ndarray]:
        """Load and validate image"""
        print(f"Loading image: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error loading image: {image_path}")
            return None
        
        print(f"Image dimensions: {img.shape}")
        return img
    
    def run_detection(self, img: np.ndarray) -> Optional[Any]:
        """Run YOLO detection on image"""
        print("Running package detection...")
        results = self.model(img, verbose=False, conf=0.5)
        
        if not results or not results[0].masks:
            print("No packages detected")
            return None
        
        num_packages = len(results[0].masks)
        print(f"Detected {num_packages} package(s)")
        return results

    def analyze_package_candidates(self, results: Any, img: np.ndarray) -> List[PackageCandidate]:
        """Analyze all detected packages and calculate metrics"""
        package_candidates = []
        
        for i, mask_data in enumerate(results[0].masks):
            candidate = self._process_single_package(i, mask_data, results, img)
            if candidate:
                package_candidates.append(candidate)
        
        # Sort packages by grasp score (best first)
        package_candidates.sort(key=lambda x: x.grasp_score, reverse=True)
        
        print("Package Analysis Results:")
        for idx, pkg in enumerate(package_candidates):
            print(f"  Rank {idx+1}: Package {pkg.index+1} - Score: {pkg.grasp_score:.3f}")
            print(f"    Confidence: {pkg.confidence:.2f}, Surface: {pkg.surface_quality:.2f}, Position: {pkg.center_score:.2f}")
        
        return package_candidates
    
    def _process_single_package(self, index: int, mask_data: Any, results: Any, img: np.ndarray) -> Optional[PackageCandidate]:
        """Process a single package and calculate its metrics"""
        # Extract mask
        if hasattr(mask_data, 'data'):
            mask = mask_data.data.cpu().numpy()[0]
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
            mask = (mask * 255).astype(np.uint8)
        else:
            return None
        
        # Find package contour
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        
        # Get main package contour
        main_contour = max(contours, key=cv2.contourArea)
        package_area = cv2.contourArea(main_contour)
        
        # Calculate safe grasp zone (eroded area)
        kernel = np.ones((25, 25), np.uint8)
        safe_mask = cv2.erode(mask, kernel, iterations=1)
        safe_contours, _ = cv2.findContours(safe_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not safe_contours:
            return None
        
        # Get largest safe zone
        safe_region = max(safe_contours, key=cv2.contourArea)
        safe_area = cv2.contourArea(safe_region)
        
        # Calculate metrics
        aspect_ratio = self._calculate_aspect_ratio(main_contour)
        confidence = float(results[0].boxes[index].conf) if results[0].boxes else 1.0
        surface_quality = safe_area / package_area if package_area > 0 else 0
        
        # Calculate center position and score
        center_x, center_y, center_score = self._calculate_center_metrics(main_contour, img)
        
        # Calculate overall grasp score
        grasp_score = (
            confidence * 0.3 +           # Model confidence
            surface_quality * 0.3 +      # Safe grasp area
            center_score * 0.2 +         # Position preference
            min(package_area / 10000, 1.0) * 0.2  # Size preference (normalized)
        )
        
        return PackageCandidate(
            index=index,
            mask=mask,
            main_contour=main_contour,
            safe_region=safe_region,
            package_area=package_area,
            safe_area=safe_area,
            confidence=confidence,
            surface_quality=surface_quality,
            center_score=center_score,
            grasp_score=grasp_score,
            center_x=center_x,
            center_y=center_y,
            aspect_ratio=aspect_ratio
        )
    
    def _calculate_aspect_ratio(self, contour: np.ndarray) -> float:
        """Calculate aspect ratio of a contour"""
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            width, height = ellipse[1]
            return max(width, height) / min(width, height) if min(width, height) > 0 else 1.0
        return 1.0
    
    def _calculate_center_metrics(self, contour: np.ndarray, img: np.ndarray) -> Tuple[int, int, float]:
        """Calculate center position and position score"""
        M = cv2.moments(contour)
        if M["m00"] > 0:
            center_x = int(M["m10"] / M["m00"])
            center_y = int(M["m01"] / M["m00"])
            
            # Distance from image center (normalized)
            img_center_x, img_center_y = img.shape[1] // 2, img.shape[0] // 2
            distance_from_center = math.sqrt((center_x - img_center_x)**2 + (center_y - img_center_y)**2)
            max_distance = math.sqrt(img_center_x**2 + img_center_y**2)
            center_score = 1.0 - (distance_from_center / max_distance)
            
            return center_x, center_y, center_score
        return 0, 0, 0.0
    def detect_and_grasp(self, image_path: str) -> Tuple[List[GraspPoint], Optional[np.ndarray]]:
        """Main method to detect packages and calculate grasp points"""
        # Load image
        img = self.load_image(image_path)
        if img is None:
            return [], None
        
        # Run detection
        results = self.run_detection(img)
        if results is None:
            return [], img
        
        # Analyze package candidates
        package_candidates = self.analyze_package_candidates(results, img)
        if not package_candidates:
            return [], img
        
        # Create visualization
        visualizer = PackageVisualizer(self.colors)
        output_image = img.copy()
        grasp_points = []
        
        # Process packages in order of preference
        for rank, pkg_data in enumerate(package_candidates):
            grasp_point = self._create_grasp_point(pkg_data, rank)
            if grasp_point:
                grasp_points.append(grasp_point)
                
                # Visualize package
                visualizer.draw_package(
                    output_image, 
                    pkg_data, 
                    grasp_point, 
                    rank
                )
                
                # Print package info
                self._print_package_info(pkg_data, grasp_point, rank)
        
        return grasp_points, output_image
    
    def _create_grasp_point(self, pkg_data: PackageCandidate, rank: int) -> Optional[GraspPoint]:
        """Create grasp point from package candidate data"""
        # Calculate centroid of safe zone (grasp point)
        M = cv2.moments(pkg_data.safe_region)
        if M["m00"] == 0:
            print(f"Package {pkg_data.index+1}: Invalid moments")
            return None
        
        grasp_x = int(M["m10"] / M["m00"])
        grasp_y = int(M["m01"] / M["m00"])
        
        # Calculate orientation angle
        angle = self._calculate_orientation_angle(pkg_data.main_contour)
        
        # Find flat surface
        flat_surface_length, flat_surface_center = self._find_flat_surface(pkg_data.main_contour)
        
        return GraspPoint(
            package_id=pkg_data.index + 1,
            rank=rank + 1,
            grasp_x=grasp_x,
            grasp_y=grasp_y,
            center_x=pkg_data.center_x,
            center_y=pkg_data.center_y,
            angle=round(angle, 1),
            area=pkg_data.package_area,
            confidence=pkg_data.confidence,
            grasp_score=pkg_data.grasp_score,
            surface_quality=pkg_data.surface_quality,
            flat_surface_length=flat_surface_length,
            flat_surface_center=flat_surface_center,
            is_best=(rank == 0)
        )
    
    def _calculate_orientation_angle(self, contour: np.ndarray) -> float:
        """Calculate orientation angle of a contour"""
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            return (ellipse[2] + 90) % 360
        return 0.0
    
    def _find_flat_surface(self, contour: np.ndarray) -> Tuple[float, Optional[Tuple[int, int]]]:
        """Find the longest flat surface on a contour"""
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx_contour = cv2.approxPolyDP(contour, epsilon, True)
        
        max_edge_length = 0
        flat_surface_center = None
        
        if len(approx_contour) >= 3:
            for j in range(len(approx_contour)):
                pt1 = approx_contour[j][0]
                pt2 = approx_contour[(j + 1) % len(approx_contour)][0]
                edge_length = math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
                
                if edge_length > max_edge_length:
                    max_edge_length = edge_length
                    flat_surface_center = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        
        return max_edge_length, flat_surface_center
    
    def _print_package_info(self, pkg_data: PackageCandidate, grasp_point: GraspPoint, rank: int):
        """Print package information"""
        status = "BEST" if rank == 0 else f"Rank {rank+1}"
        print(f"{status} Package {pkg_data.index+1}:")
        print(f"  Grasp Point: ({grasp_point.grasp_x}, {grasp_point.grasp_y})")
        print(f"  Center Point: ({grasp_point.center_x}, {grasp_point.center_y})")
        print(f"  Angle: {grasp_point.angle} degrees")
        print(f"  Flat Surface: {grasp_point.flat_surface_length:.0f}px")
        print(f"  Score: {grasp_point.grasp_score:.3f}")


class PackageVisualizer:
    """Class for handling package visualization"""
    
    def __init__(self, colors: List[Tuple[int, int, int]]):
        self.colors = colors
    
    def draw_package(self, output_image: np.ndarray, pkg_data: PackageCandidate, 
                    grasp_point: GraspPoint, rank: int):
        """Draw package visualization on output image"""
        color = self.colors[pkg_data.index % len(self.colors)]
        line_thickness = 8 if rank == 0 else 5
        
        # Draw package outline
        cv2.drawContours(output_image, [pkg_data.main_contour], -1, color, line_thickness, cv2.LINE_AA)
        
        # Fill safe grasp zone with transparency
        self._draw_safe_zone(output_image, pkg_data.safe_region, color)
        
        # Draw flat surface
        self._draw_flat_surface(output_image, pkg_data.main_contour)
        
        # Draw center and grasp points
        self._draw_center_point(output_image, grasp_point.center_x, grasp_point.center_y)
        self._draw_grasp_point(output_image, grasp_point.grasp_x, grasp_point.grasp_y, color)
        
        # Draw orientation arrow
        self._draw_orientation_arrow(output_image, grasp_point, rank)
        
        # Draw text labels
        self._draw_text_labels(output_image, grasp_point)
    
    def _draw_safe_zone(self, output_image: np.ndarray, safe_region: np.ndarray, color: Tuple[int, int, int]):
        """Draw safe grasp zone with transparency"""
        overlay = output_image.copy()
        cv2.drawContours(overlay, [safe_region], -1, color, -1, cv2.LINE_AA)
        cv2.addWeighted(overlay, 0.25, output_image, 0.75, 0, output_image)
    
    def _draw_flat_surface(self, output_image: np.ndarray, main_contour: np.ndarray):
        """Draw flat surface detection"""
        epsilon = 0.02 * cv2.arcLength(main_contour, True)
        approx_contour = cv2.approxPolyDP(main_contour, epsilon, True)
        
        max_edge_length = 0
        flat_surface_start = None
        flat_surface_end = None
        flat_surface_center = None
        
        if len(approx_contour) >= 3:
            for j in range(len(approx_contour)):
                pt1 = approx_contour[j][0]
                pt2 = approx_contour[(j + 1) % len(approx_contour)][0]
                edge_length = math.sqrt((pt2[0] - pt1[0])**2 + (pt2[1] - pt1[1])**2)
                
                if edge_length > max_edge_length:
                    max_edge_length = edge_length
                    flat_surface_start = tuple(pt1)
                    flat_surface_end = tuple(pt2)
                    flat_surface_center = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
        
        if flat_surface_start and flat_surface_end:
            cv2.line(output_image, flat_surface_start, flat_surface_end, (0, 255, 0), 6, cv2.LINE_AA)
            cv2.circle(output_image, flat_surface_center, 8, (0, 255, 0), -1, cv2.LINE_AA)
    
    def _draw_center_point(self, output_image: np.ndarray, center_x: int, center_y: int):
        """Draw center point"""
        cv2.circle(output_image, (center_x, center_y), 12, (255, 0, 255), 3, cv2.LINE_AA)  # Magenta circle
        cv2.circle(output_image, (center_x, center_y), 4, (255, 255, 255), -1, cv2.LINE_AA)  # White center
    
    def _draw_grasp_point(self, output_image: np.ndarray, grasp_x: int, grasp_y: int, color: Tuple[int, int, int]):
        """Draw grasp point"""
        cv2.circle(output_image, (grasp_x, grasp_y), 15, (0, 0, 0), 3, cv2.LINE_AA)  # Black ring
        cv2.circle(output_image, (grasp_x, grasp_y), 12, (255, 255, 255), -1, cv2.LINE_AA)  # White center
        cv2.circle(output_image, (grasp_x, grasp_y), 8, color, -1, cv2.LINE_AA)  # Colored core
    
    def _draw_orientation_arrow(self, output_image: np.ndarray, grasp_point: GraspPoint, rank: int):
        """Draw orientation arrow"""
        arrow_length = 100 if rank == 0 else 80
        arrow_thickness = 8 if rank == 0 else 6
        
        end_x = int(grasp_point.grasp_x + arrow_length * math.cos(math.radians(grasp_point.angle)))
        end_y = int(grasp_point.grasp_y + arrow_length * math.sin(math.radians(grasp_point.angle)))
        
        # Draw arrow shaft
        cv2.line(output_image, (grasp_point.grasp_x, grasp_point.grasp_y), (end_x, end_y), 
                (0, 0, 0), arrow_thickness + 4, cv2.LINE_AA)
        cv2.line(output_image, (grasp_point.grasp_x, grasp_point.grasp_y), (end_x, end_y), 
                (255, 255, 255), arrow_thickness + 2, cv2.LINE_AA)
        cv2.line(output_image, (grasp_point.grasp_x, grasp_point.grasp_y), (end_x, end_y), 
                (255, 165, 0), arrow_thickness, cv2.LINE_AA)
        
        # Draw arrowhead
        self._draw_arrowhead(output_image, end_x, end_y, grasp_point.angle)
    
    def _draw_arrowhead(self, output_image: np.ndarray, end_x: int, end_y: int, angle: float):
        """Draw arrowhead"""
        arrow_head_length = 25
        arrow_head_angle = 30
        
        head_angle1 = angle + 180 - arrow_head_angle
        head_angle2 = angle + 180 + arrow_head_angle
        
        head_x1 = int(end_x + arrow_head_length * math.cos(math.radians(head_angle1)))
        head_y1 = int(end_y + arrow_head_length * math.sin(math.radians(head_angle1)))
        head_x2 = int(end_x + arrow_head_length * math.cos(math.radians(head_angle2)))
        head_y2 = int(end_y + arrow_head_length * math.sin(math.radians(head_angle2)))
        
        triangle_pts = np.array([[end_x, end_y], [head_x1, head_y1], [head_x2, head_y2]], np.int32)
        cv2.fillPoly(output_image, [triangle_pts], (255, 165, 0), cv2.LINE_AA)
        cv2.polylines(output_image, [triangle_pts], True, (0, 0, 0), 3, cv2.LINE_AA)
    
    def _draw_text_labels(self, output_image: np.ndarray, grasp_point: GraspPoint):
        """Draw text labels"""
        # Display X,Y coordinates
        coord_text = f"X:{grasp_point.grasp_x} Y:{grasp_point.grasp_y}"
        self._draw_enhanced_text(output_image, coord_text, 
                               (grasp_point.grasp_x - 90, grasp_point.grasp_y + 70), 
                               1.6, (0, 255, 255))
        
        # Display angle below coordinates
        angle_text = f"Angle: {grasp_point.angle} degrees"
        self._draw_enhanced_text(output_image, angle_text, 
                               (grasp_point.grasp_x - 90, grasp_point.grasp_y + 110), 
                               1.6, (255, 255, 0))
        
        # Display center position
        center_text = f"Center: ({grasp_point.center_x},{grasp_point.center_y})"
        self._draw_enhanced_text(output_image, center_text, 
                               (grasp_point.center_x - 100, grasp_point.center_y - 40), 
                               1.2, (255, 0, 255))
        
        # Display flat surface info
        if grasp_point.flat_surface_center:
            flat_text = f"Flat: {grasp_point.flat_surface_length:.0f}px"
            self._draw_enhanced_text(output_image, flat_text, 
                                   (grasp_point.flat_surface_center[0] - 60, 
                                    grasp_point.flat_surface_center[1] - 30), 
                                   1.2, (0, 255, 0))
    
    def _draw_enhanced_text(self, img: np.ndarray, text: str, pos: Tuple[int, int], 
                          scale: float = 1.4, color: Tuple[int, int, int] = (255, 255, 255), 
                          thickness: int = 4):
        """Draw enhanced text with outline"""
        font = cv2.FONT_HERSHEY_TRIPLEX
        # Black outline for visibility
        cv2.putText(img, text, pos, font, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
        cv2.putText(img, text, pos, font, scale, color, thickness, cv2.LINE_AA)


class PackageGraspApplication:
    """Main application class for package grasp detection"""
    
    def __init__(self, model_path: str):
        self.detector = PackageDetector(model_path)
    
    def run_detection(self, image_path: str) -> Tuple[List[GraspPoint], Optional[np.ndarray]]:
        """Run package detection and return results"""
        return self.detector.detect_and_grasp(image_path)
    
    def save_and_display_results(self, result_image: np.ndarray, output_path: str = 'simple_grasp_result.jpg'):
        """Save and display the result image"""
        if result_image is not None:
            # Save result
            cv2.imwrite(output_path, result_image)
            print(f"Result saved: {output_path}")
            
            # Display result
            display_height = 800
            if result_image.shape[0] > display_height:
                scale = display_height / result_image.shape[0]
                display_width = int(result_image.shape[1] * scale)
                result_display = cv2.resize(result_image, (display_width, display_height))
            else:
                result_display = result_image
            
            cv2.imshow('Package Grasp Detection', result_display)
            print("Press any key to close window...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    def print_robot_instructions(self, pick_data: List[GraspPoint]):
        """Print formatted robot pick instructions"""
        if not pick_data:
            print("\nNo packages detected for pickup")
            return
        
        print("\n" + "=" * 50)
        print("ROBOT PICK INSTRUCTIONS (Ranked by Quality)")
        print("=" * 50)
        
        # Sort by rank for display
        sorted_data = sorted(pick_data, key=lambda x: x.rank)
        
        for data in sorted_data:
            status = "PRIORITY TARGET" if data.is_best else "BACKUP OPTION"
            print(f"{status} - Package {data.package_id} (Rank #{data.rank}):")
            print(f"  Grasp Position: ({data.grasp_x}, {data.grasp_y})")
            print(f"  Rotation Angle: {data.angle} degrees")
            print(f"  Package Area: {data.area:.0f} pixels²")
            print(f"  Model Confidence: {data.confidence:.2f}")
            print(f"  Grasp Quality Score: {data.grasp_score:.3f}")
            print(f"  Surface Quality: {data.surface_quality:.2f}")
            print("-" * 40)
        
        best_package = next((p for p in pick_data if p.is_best), None)
        if best_package:
            print(f"\nRECOMMENDED ACTION:")
            print(f"   -> Target Package {best_package.package_id} at ({best_package.grasp_x}, {best_package.grasp_y})")
            print(f"   -> Rotate gripper to {best_package.angle} degrees before picking")
        
        print(f"\nTotal packages detected: {len(pick_data)}")
        print(f"Best package: #{best_package.package_id if best_package else 'None'}")


def main():
    """Main execution function"""
    print("Package Grasp Detection System")
    print("=" * 40)
    
    # Initialize application
    app = PackageGraspApplication(MODEL_PATH)
    
    # Run detection
    pick_data, result_image = app.run_detection(IMAGE_PATH)
    
    # Save and display results
    if result_image is not None:
        app.save_and_display_results(result_image)
    
    # Print robot instructions
    app.print_robot_instructions(pick_data)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
