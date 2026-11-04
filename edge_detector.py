#!/usr/bin/env python3

import os
import sys
import json
import argparse
from typing import List, Tuple
from tqdm import tqdm
import cv2
import numpy as np
from skimage import morphology


class CannyEdgeDetector3D:
    
    def __init__(self, config: dict = None):
        self.config = config or {
            'gaussian_sigma': None,
            'threshold_method': 'adaptive',
            't_low_factor': 0.6,
            't_high_factor': 1.2,
            'morphology_cleanup': True
        }
    
    def read_image(self, filepath: str) -> np.ndarray:
        image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image: {filepath}")
        return image.astype(np.float32) / 255.0
    
    def apply_gaussian_smoothing(self, image: np.ndarray) -> np.ndarray:
        sigma = self.config.get('gaussian_sigma') or np.maximum(np.min(image.shape) * 0.003, 0.5)
        kernel_size = int(2 * np.ceil(3 * sigma) + 1)
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), sigma)
    
    def compute_gradients(self, image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        angle = np.arctan2(grad_y, grad_x)
        return magnitude, angle
    
    def non_maximum_suppression(self, magnitude: np.ndarray, angle: np.ndarray) -> np.ndarray:
        height, width = magnitude.shape
        suppressed = np.zeros_like(magnitude)
        angle = np.rad2deg(angle) % 180
        
        for i in range(1, height - 1):
            for j in range(1, width - 1):
                direction = angle[i, j]
                current_mag = magnitude[i, j]
                
                if (0 <= direction < 22.5) or (157.5 <= direction < 180):
                    neighbors = [magnitude[i, j-1], magnitude[i, j+1]]
                elif 22.5 <= direction < 67.5:
                    neighbors = [magnitude[i-1, j+1], magnitude[i+1, j-1]]
                elif 67.5 <= direction < 112.5:
                    neighbors = [magnitude[i-1, j], magnitude[i+1, j]]
                else:
                    neighbors = [magnitude[i-1, j-1], magnitude[i+1, j+1]]
                
                if current_mag >= max(neighbors):
                    suppressed[i, j] = current_mag
        
        return suppressed
    
    def calculate_adaptive_thresholds(self, magnitude: np.ndarray) -> Tuple[float, float]:
        non_zero_mag = magnitude[magnitude > 0]
        if len(non_zero_mag) == 0:
            return 0.05, 0.15
        t_high = np.percentile(non_zero_mag, 70) * self.config['t_high_factor']
        t_low = np.percentile(non_zero_mag, 30) * self.config['t_low_factor']
        return t_low, t_high
    
    def hysteresis_thresholding(self, magnitude: np.ndarray, t_low: float, t_high: float) -> np.ndarray:
        if magnitude.max() > 0:
            magnitude = magnitude / magnitude.max()
        
        strong_edges = (magnitude >= t_high)
        weak_edges = ((magnitude >= t_low) & (magnitude < t_high))
        edges = np.zeros_like(magnitude, dtype=np.uint8)
        edges[strong_edges] = 1
        
        weak_uint8 = weak_edges.astype(np.uint8) * 255
        num_labels, labels = cv2.connectedComponents(weak_uint8, connectivity=8)
        strong_labels = np.unique(labels[strong_edges])
        
        for label in strong_labels:
            if label > 0:
                edges[labels == label] = 1
        return edges
    
    def apply_morphology(self, edges: np.ndarray) -> np.ndarray:
        if not self.config.get('morphology_cleanup', True):
            return edges
        cleaned = morphology.remove_small_objects(edges.astype(bool), min_size=3, connectivity=2).astype(np.uint8)
        kernel = morphology.disk(1)
        return morphology.closing(cleaned, kernel).astype(np.uint8)
    
    def canny_edge_detection(self, image: np.ndarray) -> np.ndarray:
        smoothed = self.apply_gaussian_smoothing(image)
        magnitude, angle = self.compute_gradients(smoothed)
        suppressed = self.non_maximum_suppression(magnitude, angle)
        t_low, t_high = self.calculate_adaptive_thresholds(suppressed)
        edges = self.hysteresis_thresholding(suppressed, t_low, t_high)
        return self.apply_morphology(edges)
    
    def link_24_connectivity(self, edge_images: List[np.ndarray]) -> List[np.ndarray]:
        if len(edge_images) < 2:
            return edge_images
        
        linked_images = [edge_images[0].copy()]
        for idx in tqdm(range(len(edge_images) - 1), desc="Linking"):
            current = edge_images[idx]
            next_img = edge_images[idx + 1].copy()
            linked_images.append(self._link_slices(current, next_img))
        return linked_images
    
    def _link_slices(self, current: np.ndarray, next_img: np.ndarray) -> np.ndarray:
        next_padded = np.pad(next_img, pad_width=2, mode='constant', constant_values=0)
        edge_pixels = np.argwhere(current == 1)
        
        for y, x in edge_pixels:
            py, px = y + 2, x + 2
            
            if next_padded[py, px] == 1:
                continue
            if np.any(next_padded[py-1:py+2, px-1:px+2] == 1):
                continue
            
            neighborhood_5x5 = next_padded[py-2:py+3, px-2:px+3]
            if np.sum(neighborhood_5x5) == 0:
                continue
            
            local_edges = np.argwhere(neighborhood_5x5 == 1)
            for local_y, local_x in local_edges:
                target_y, target_x = py - 2 + local_y, px - 2 + local_x
                self._draw_path(next_padded, px, py, target_x, target_y)
        
        return next_padded[2:-2, 2:-2]
    
    def _draw_path(self, image: np.ndarray, x1: int, y1: int, x2: int, y2: int):
        cx, cy = x1, y1
        while cx != x2 or cy != y2:
            image[cy, cx] = 1
            if cx < x2:
                cx += 1
            elif cx > x2:
                cx -= 1
            if cy < y2:
                cy += 1
            elif cy > y2:
                cy -= 1
        image[cy, cx] = 1
    
    def process_image_sequence(self, image_paths: List[str]) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
        original = [self.read_image(p) for p in tqdm(image_paths, desc="Loading")]
        edges = [self.canny_edge_detection(img) for img in tqdm(original, desc="Edges")]
        linked = self.link_24_connectivity(edges)
        return original, edges, linked


def save_results(linked_images: List[np.ndarray], image_names: List[str], output_dir: str = 'results'):
    os.makedirs(output_dir, exist_ok=True)
    for img, name in tqdm(zip(linked_images, image_names), total=len(linked_images), desc="Saving"):
        output_path = os.path.join(output_dir, name)
        cv2.imwrite(output_path, (img * 255).astype(np.uint8))


def main():
    parser = argparse.ArgumentParser(description='3D CT Edge Detection')
    parser.add_argument('directory', type=str)
    parser.add_argument('images', nargs='+')
    parser.add_argument('--config', type=str)
    parser.add_argument('--output', type=str, default='results')
    args = parser.parse_args()
    
    config = None
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            config = json.load(f)
    
    image_paths = [os.path.join(args.directory, img) for img in args.images]
    for path in image_paths:
        if not os.path.exists(path):
            print(f"Error: {path} not found")
            sys.exit(1)
    
    detector = CannyEdgeDetector3D(config)
    original, edges, linked = detector.process_image_sequence(image_paths)
    save_results(linked, args.images, args.output)


if __name__ == '__main__':
    main()
