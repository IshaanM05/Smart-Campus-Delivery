# ==================== ULTIMATE OPTIMIZED FINAL VERSION ====================
# Smart Campus Delivery System with Advanced GPS Tracking
# Enhanced with Entry System Variants and Complete Analytics

import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, scrolledtext
import networkx as nx
import time
from collections import deque
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors
import json
from dataclasses import dataclass
from typing import List, Dict, Tuple, Deque, Optional, Any
import threading
from queue import Queue
import logging
from datetime import datetime
import sys
import gc
import warnings
warnings.filterwarnings('ignore')

# Setup comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('smart_campus_simulation.log')
    ]
)
logger = logging.getLogger(__name__)

random.seed(42)
np.random.seed(42)

# ==================== ENHANCED SIMULATION PARAMETERS ====================
NUM_AGENTS = 25  # Balanced number of agents
MAX_STEPS = 800  # Increased for longer simulation
SPEED_VARIATION = {"petrol": 5.0, "e-bike": 7.0, "cycle": 4.0}
SMOOTHNESS_FACTOR = 3

# Enhanced entry system parameters with 0.25 ratio
ENTRY_SYSTEMS = {
    "written": {
        "base_time": 8.0,
        "congestion_multiplier": 2.5,
        "queue_penalty": 1.8,
        "color": "red",
        "name": "Written Entry",
        "marker": "o",  # Circle marker
        "efficiency_ratio": 1.0
    },
    "qr_code": {
        "base_time": 2.0,  # 8.0 * 0.25 = 2.0 (exactly 0.25 ratio)
        "congestion_multiplier": 1.1,  # Reduced congestion impact
        "queue_penalty": 1.1,  # Reduced queue penalty
        "color": "green",
        "name": "QR Code",
        "marker": "^",  # Triangle marker
        "efficiency_ratio": 0.25  # Exactly 0.25 ratio
    }
}

# Campus layout with enhanced node positioning
NODES = {
    "MainGate": (0, 50),
    "YPGate": (0, -50),
    "Hub1": (30, 20),
    "Hub2": (30, -20),
    "Hub3": (60, 0),
    "HostelA": (80, 30),
    "HostelB": (80, 0),
    "HostelC": (80, -30),
    "Acad1": (50, 20),
    "Acad2": (50, -20),
    "FoodCourt": (40, 0),
    "Library": (70, 25),
    "Sports": (70, -25)
}

# ==================== CAMPUS GRAPH SETUP ====================
def create_campus_graph():
    """Create enhanced campus graph with traffic flow optimization"""
    G = nx.Graph()
    for n, p in NODES.items():
        G.add_node(n, pos=p)

    edges_with_weights = [
        ("MainGate", "Hub1", 1.0), ("MainGate", "Hub2", 1.2),
        ("YPGate", "Hub1", 1.1), ("YPGate", "Hub2", 1.0),
        ("Hub1", "Hub3", 1.3), ("Hub2", "Hub3", 1.4),
        ("Hub1", "Acad1", 1.0), ("Hub1", "FoodCourt", 0.8),
        ("Hub2", "Acad2", 1.0), ("Hub2", "FoodCourt", 0.9),
        ("Hub3", "HostelA", 1.2), ("Hub3", "HostelB", 1.0), ("Hub3", "HostelC", 1.2),
        ("HostelA", "HostelB", 0.7), ("HostelB", "HostelC", 0.7),
        ("FoodCourt", "Acad1", 0.6), ("FoodCourt", "Acad2", 0.6),
        ("Hub1", "HostelB", 1.5), ("Hub2", "HostelB", 1.5),
        ("Hub3", "Library", 0.8), ("Library", "HostelA", 0.9),
        ("Hub3", "Sports", 0.8), ("Sports", "HostelC", 0.9)
    ]

    for a, b, weight_factor in edges_with_weights:
        xa, ya = NODES[a]
        xb, yb = NODES[b]
        base_distance = math.hypot(xa - xb, ya - yb)
        G.add_edge(a, b, weight=base_distance * weight_factor,
                  base_weight=base_distance * weight_factor)

    return G

# Create campus graph
CAMPUS_GRAPH = create_campus_graph()
CONGESTION_EDGES = [("Hub1", "Acad1"), ("Hub2", "Acad2"),
                   ("FoodCourt", "Acad1"), ("FoodCourt", "Acad2")]

# ==================== ENHANCED WIFI INFRASTRUCTURE ====================
WIFI_ACCESS_POINTS = {
    "MainGate": {"pos": (0, 50), "range": 35, "signal_variance": 1.0},
    "YPGate": {"pos": (0, -50), "range": 35, "signal_variance": 1.0},
    "Hub1": {"pos": (30, 20), "range": 25, "signal_variance": 1.2},
    "Hub2": {"pos": (30, -20), "range": 25, "signal_variance": 1.2},
    "Hub3": {"pos": (60, 0), "range": 28, "signal_variance": 1.3},
    "HostelA": {"pos": (80, 30), "range": 20, "signal_variance": 1.0},
    "HostelB": {"pos": (80, 0), "range": 22, "signal_variance": 1.1},
    "HostelC": {"pos": (80, -30), "range": 20, "signal_variance": 1.0},
    "Acad1": {"pos": (50, 20), "range": 22, "signal_variance": 1.0},
    "Acad2": {"pos": (50, -20), "range": 22, "signal_variance": 1.0},
    "FoodCourt": {"pos": (40, 0), "range": 26, "signal_variance": 1.4},
    "Library": {"pos": (70, 25), "range": 18, "signal_variance": 1.0},
    "Sports": {"pos": (70, -25), "range": 18, "signal_variance": 1.0}
}

# ==================== CLASS SCHEDULE WITH SWARM BEHAVIOR ====================
CLASS_SCHEDULE = {
    "morning_peak": {
        "start": 50, "end": 200,  # Extended duration
        "swarm_routes": [
            {"from": ["HostelA", "HostelB", "HostelC"], "to": ["Acad1", "Acad2", "Library"], "intensity": 4.0},
            {"from": ["MainGate", "YPGate"], "to": ["Acad1", "Acad2"], "intensity": 3.0}
        ],
        "areas": ["Acad1", "Acad2", "Hub1", "Hub2", "Library"],
        "intensity": 3.5
    },
    "lunch_peak": {
        "start": 250, "end": 350,  # Extended duration
        "swarm_routes": [
            {"from": ["Acad1", "Acad2", "Library"], "to": ["FoodCourt"], "intensity": 4.5},
            {"from": ["HostelA", "HostelB", "HostelC"], "to": ["FoodCourt"], "intensity": 4.0}
        ],
        "areas": ["FoodCourt", "Hub1", "Hub2"],
        "intensity": 4.0
    },
    "evening_peak": {
        "start": 400, "end": 600,  # Extended duration
        "swarm_routes": [
            {"from": ["Acad1", "Acad2", "Library", "Sports"], "to": ["HostelA", "HostelB", "HostelC"], "intensity": 4.0},
            {"from": ["FoodCourt"], "to": ["HostelA", "HostelB", "HostelC"], "intensity": 3.5}
        ],
        "areas": ["HostelA", "HostelB", "HostelC", "MainGate", "YPGate", "Sports"],
        "intensity": 3.0
    }
}

# ==================== ENHANCED GPS TRACKING SYSTEM ====================
class EnhancedGPSTracker:
    """Enhanced GPS tracking with complete algorithm integration - IDENTICAL TO WORKING CODE"""

    def __init__(self, agent_id=None):
        self.position_history = deque(maxlen=100)
        self.estimated_positions = deque(maxlen=100)
        self.kalman_state = None
        self.kalman_covariance = None
        self.filter_initialized = False
        self.fingerprint_database = self.create_enhanced_fingerprint_database()
        self.confidence_history = deque(maxlen=20)
        self.method_used = []
        self.algorithm_visualization_data = deque(maxlen=50)
        self.agent_id = agent_id
        self.last_valid_position = None
        self.consecutive_bad_estimates = 0
        self.path_constraint_active = False
        self.complete_journey_history = deque(maxlen=200)
        self.algorithm_performance = {
            'Trilateration': {'uses': 0, 'total_confidence': 0.0, 'avg_confidence': 0.0},
            'Fingerprinting': {'uses': 0, 'total_confidence': 0.0, 'avg_confidence': 0.0},
            'Centroid': {'uses': 0, 'total_confidence': 0.0, 'avg_confidence': 0.0},
            'Kalman Filter': {'uses': 0, 'total_confidence': 0.0, 'avg_confidence': 0.0},
            'Fusion': {'uses': 0, 'total_confidence': 0.0, 'avg_confidence': 0.0}
        }

    def create_enhanced_fingerprint_database(self):
        """Create enhanced WiFi fingerprint database"""
        fingerprints = []
        for x in range(0, 91, 5):
            for y in range(-50, 51, 5):
                if any(math.hypot(x - ap["pos"][0], y - ap["pos"][1]) <= ap["range"] * 1.2
                       for ap in WIFI_ACCESS_POINTS.values()):
                    fingerprint = {"position": (x, y)}
                    for ap_name, ap_info in WIFI_ACCESS_POINTS.items():
                        distance = math.hypot(x - ap_info["pos"][0], y - ap_info["pos"][1])
                        if distance <= ap_info["range"] * 1.2:
                            rssi = self.calculate_realistic_rssi(distance, ap_info)
                            fingerprint[ap_name] = max(-100, rssi)
                        else:
                            fingerprint[ap_name] = -100
                    fingerprints.append(fingerprint)
        return fingerprints

    def calculate_realistic_rssi(self, distance, ap_info):
        """Calculate realistic RSSI using log-distance path loss model"""
        tx_power = -20
        path_loss_exponent = 2.5
        shadow_std = ap_info["signal_variance"] * 0.5

        if distance <= 0:
            distance = 0.1

        rssi = tx_power - 10 * path_loss_exponent * math.log10(distance)
        rssi += random.gauss(0, shadow_std)

        return max(-95, rssi)

    def get_enhanced_wifi_signals(self, true_position):
        """Get enhanced WiFi signals with better noise modeling"""
        signals = {}
        for ap_name, ap_info in WIFI_ACCESS_POINTS.items():
            distance = math.hypot(true_position[0] - ap_info["pos"][0],
                                 true_position[1] - ap_info["pos"][1])

            if distance <= ap_info["range"] * 1.1:
                rssi = self.calculate_realistic_rssi(distance, ap_info)
                signals[ap_name] = rssi
            else:
                signals[ap_name] = -100

        return signals

    def apply_path_constraint(self, estimated_position, true_position, confidence):
        """Apply path constraints to prevent GPS divergence"""
        if confidence < 0.3 and self.last_valid_position:
            max_movement = 8.0
            distance_moved = math.hypot(estimated_position[0] - self.last_valid_position[0],
                                      estimated_position[1] - self.last_valid_position[1])

            if distance_moved > max_movement:
                scale_factor = max_movement / distance_moved
                constrained_x = self.last_valid_position[0] + (estimated_position[0] - self.last_valid_position[0]) * scale_factor
                constrained_y = self.last_valid_position[1] + (estimated_position[1] - self.last_valid_position[1]) * scale_factor

                self.consecutive_bad_estimates += 1
                self.path_constraint_active = True
                return (constrained_x, constrained_y)

        self.consecutive_bad_estimates = 0
        self.path_constraint_active = False
        return estimated_position

    def enhanced_trilateration(self, signals):
        """Improved trilateration with confidence weighting"""
        try:
            valid_aps = []
            for ap_name, rssi in signals.items():
                if rssi > -85:
                    distance_estimate = self.rssi_to_distance_enhanced(rssi)
                    confidence = min(1.0, (rssi + 85) / 40)
                    valid_aps.append((ap_name, rssi, distance_estimate, confidence))

            if len(valid_aps) < 3:
                return None, 0.0, "Trilateration (Insufficient APs)"

            def error_function(point):
                x, y = point
                total_error = 0
                total_weight = 0

                for ap_name, rssi, est_distance, confidence in valid_aps:
                    ap_pos = WIFI_ACCESS_POINTS[ap_name]["pos"]
                    actual_distance = math.hypot(x - ap_pos[0], y - ap_pos[1])
                    error = (est_distance - actual_distance) ** 2
                    weight = confidence
                    total_error += error * weight
                    total_weight += weight

                return total_error / total_weight if total_weight > 0 else total_error

            strong_aps = sorted(valid_aps, key=lambda x: x[2])[:4]
            ap_positions = [WIFI_ACCESS_POINTS[ap_name]["pos"] for ap_name, _, _, _ in strong_aps]

            if ap_positions:
                initial_guess = [np.mean([p[0] for p in ap_positions]),
                               np.mean([p[1] for p in ap_positions])]
            else:
                initial_guess = [45, 0]

            bounds = [(5, 85), (-45, 45)]
            result = minimize(error_function, initial_guess, bounds=bounds,
                            method='L-BFGS-B', options={'ftol': 1e-5})

            if result.success:
                final_error = result.fun
                confidence = max(0.1, 1.0 - min(1.0, final_error / 50))
                confidence *= min(1.0, len(valid_aps) / 6)
                
                # Track algorithm performance
                self.algorithm_performance['Trilateration']['uses'] += 1
                self.algorithm_performance['Trilateration']['total_confidence'] += confidence
                self.algorithm_performance['Trilateration']['avg_confidence'] = (
                    self.algorithm_performance['Trilateration']['total_confidence'] / 
                    self.algorithm_performance['Trilateration']['uses']
                )

                algo_data = {
                    'method': 'Trilateration',
                    'ap_positions': ap_positions,
                    'estimated_position': result.x,
                    'confidence': confidence,
                    'num_aps': len(valid_aps)
                }
                return result.x, confidence, algo_data
            else:
                return None, 0.0, "Trilateration (Optimization Failed)"

        except Exception as e:
            return None, 0.0, f"Trilateration (Error: {str(e)})"

    def rssi_to_distance_enhanced(self, rssi):
        """Enhanced RSSI to distance conversion with less noise"""
        if rssi >= -40:
            return 1.0 + random.gauss(0, 0.2)
        elif rssi >= -50:
            return 2.0 + random.gauss(0, 0.3)
        elif rssi >= -60:
            return 5.0 + random.gauss(0, 0.5)
        elif rssi >= -70:
            return 10.0 + random.gauss(0, 1.0)
        elif rssi >= -80:
            return 20.0 + random.gauss(0, 1.5)
        else:
            return 30.0 + random.gauss(0, 2.0)

    def enhanced_fingerprinting(self, signals):
        """Improved fingerprinting with signal strength weighting"""
        try:
            query_vector = []
            feature_names = list(WIFI_ACCESS_POINTS.keys())
            for ap_name in feature_names:
                query_vector.append(signals.get(ap_name, -100))

            if len(self.fingerprint_database) == 0:
                return None, 0.0, "Fingerprinting (No Database)"

            k = min(5, len(self.fingerprint_database) // 10)
            if k < 2:
                k = 2

            best_matches = []

            for fingerprint in self.fingerprint_database:
                similarity = 0
                total_weight = 0

                for i, ap_name in enumerate(feature_names):
                    db_rssi = fingerprint.get(ap_name, -100)
                    query_rssi = query_vector[i]

                    if query_rssi > -90 or db_rssi > -90:
                        weight = max(0, (max(query_rssi, db_rssi) + 90) / 10)
                        diff = abs(db_rssi - query_rssi)
                        similarity += weight * (100 - min(100, diff))
                        total_weight += weight

                if total_weight > 0:
                    normalized_similarity = similarity / total_weight
                    best_matches.append((fingerprint["position"], normalized_similarity))

            if not best_matches:
                return None, 0.0, "Fingerprinting (No Matches)"

            best_matches.sort(key=lambda x: x[1], reverse=True)
            top_matches = best_matches[:k]

            total_similarity = sum(sim for _, sim in top_matches)
            if total_similarity <= 0:
                return None, 0.0, "Fingerprinting (Zero Similarity)"

            weighted_x = 0
            weighted_y = 0

            for (x, y), similarity in top_matches:
                weight = similarity / total_similarity
                weighted_x += x * weight
                weighted_y += y * weight

            confidence = min(1.0, total_similarity / (k * 100))
            
            # Track algorithm performance
            self.algorithm_performance['Fingerprinting']['uses'] += 1
            self.algorithm_performance['Fingerprinting']['total_confidence'] += confidence
            self.algorithm_performance['Fingerprinting']['avg_confidence'] = (
                self.algorithm_performance['Fingerprinting']['total_confidence'] / 
                self.algorithm_performance['Fingerprinting']['uses']
            )

            algo_data = {
                'method': 'Fingerprinting',
                'reference_points': [match[0] for match in top_matches],
                'estimated_position': (weighted_x, weighted_y),
                'confidence': confidence,
                'k_neighbors': k
            }
            return (weighted_x, weighted_y), confidence, algo_data

        except Exception as e:
            return None, 0.0, f"Fingerprinting (Error: {str(e)})"

    def enhanced_kalman_filter(self, measured_position, measured_confidence=0.5):
        """Enhanced Kalman filter with adaptive noise - NOW PROPERLY INTEGRATED"""
        if not self.filter_initialized:
            self.kalman_state = np.array([measured_position[0], measured_position[1], 0, 0])
            self.kalman_covariance = np.eye(4) * 1.0
            self.filter_initialized = True
            return measured_position, "Kalman (Initialized)"

        dt = 1.0
        F = np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

        process_noise_scale = max(0.1, 1.0 - measured_confidence)
        Q = np.eye(4) * 0.03 * process_noise_scale

        self.kalman_state = F @ self.kalman_state
        self.kalman_covariance = F @ self.kalman_covariance @ F.T + Q

        H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])

        measurement_noise = max(0.3, 3.0 * (1.0 - measured_confidence))
        R = np.eye(2) * measurement_noise

        innovation = np.array(measured_position) - H @ self.kalman_state
        innovation_covariance = H @ self.kalman_covariance @ H.T + R
        kalman_gain = self.kalman_covariance @ H.T @ np.linalg.inv(innovation_covariance)

        self.kalman_state = self.kalman_state + kalman_gain @ innovation
        self.kalman_covariance = (np.eye(4) - kalman_gain @ H) @ self.kalman_covariance
        
        # Track algorithm performance
        kalman_confidence = min(1.0, measured_confidence * 1.1)  # Kalman typically improves confidence
        self.algorithm_performance['Kalman Filter']['uses'] += 1
        self.algorithm_performance['Kalman Filter']['total_confidence'] += kalman_confidence
        self.algorithm_performance['Kalman Filter']['avg_confidence'] = (
            self.algorithm_performance['Kalman Filter']['total_confidence'] / 
            self.algorithm_performance['Kalman Filter']['uses']
        )

        return self.kalman_state[0:2], "Kalman (Updated)"

    def estimate_enhanced_position(self, true_position, frame):
        """Enhanced position estimation with path constraints and ALL algorithms"""
        signals = self.get_enhanced_wifi_signals(true_position)

        estimates = []
        confidences = []
        algorithm_data = []
        methods_used = []

        # Method 1: Enhanced Trilateration
        tri_pos, tri_confidence, tri_algo = self.enhanced_trilateration(signals)
        if tri_pos is not None and tri_confidence > 0.1:  # Reduced threshold
            estimates.append(tri_pos)
            confidences.append(tri_confidence)
            algorithm_data.append(tri_algo)
            methods_used.append("Trilateration")

        # Method 2: Enhanced Fingerprinting
        fp_pos, fp_confidence, fp_algo = self.enhanced_fingerprinting(signals)
        if fp_pos is not None and fp_confidence > 0.1:  # Reduced threshold
            estimates.append(fp_pos)
            confidences.append(fp_confidence)
            algorithm_data.append(fp_algo)
            methods_used.append("Fingerprinting")

        # Method 3: AP Centroid with signal strength weighting
        strong_aps = [ap_name for ap_name, rssi in signals.items() if rssi > -75]
        if len(strong_aps) >= 2:
            ap_positions = [WIFI_ACCESS_POINTS[ap_name]["pos"] for ap_name in strong_aps]
            weights = [max(0.1, (signals[ap_name] + 75) / 25) for ap_name in strong_aps]
            total_weight = sum(weights)

            if total_weight > 0:
                centroid_x = sum(x * w for (x, y), w in zip(ap_positions, weights)) / total_weight
                centroid_y = sum(y * w for (x, y), w in zip(ap_positions, weights)) / total_weight

                centroid_confidence = min(0.8, len(strong_aps) / 8 + 0.2)
                estimates.append((centroid_x, centroid_y))
                confidences.append(centroid_confidence)
                
                # Track algorithm performance
                self.algorithm_performance['Centroid']['uses'] += 1
                self.algorithm_performance['Centroid']['total_confidence'] += centroid_confidence
                self.algorithm_performance['Centroid']['avg_confidence'] = (
                    self.algorithm_performance['Centroid']['total_confidence'] / 
                    self.algorithm_performance['Centroid']['uses']
                )
                
                algorithm_data.append({
                    'method': 'Centroid',
                    'ap_positions': ap_positions,
                    'estimated_position': (centroid_x, centroid_y),
                    'confidence': centroid_confidence,
                    'num_aps': len(strong_aps)
                })
                methods_used.append("Centroid")

        if len(estimates) == 0:
            if len(self.estimated_positions) > 0:
                last_pos = self.estimated_positions[-1]
                estimated = (last_pos[0] + random.gauss(0, 1.0),
                           last_pos[1] + random.gauss(0, 1.0))
                confidence = 0.1
                methods_used.append("Motion Model")
                algorithm_data.append({
                    'method': 'Motion Model',
                    'estimated_position': estimated,
                    'confidence': confidence
                })
            else:
                estimated = (true_position[0] + random.gauss(0, 2.0),
                           true_position[1] + random.gauss(0, 2.0))
                confidence = 0.05
                methods_used.append("Noisy True Position")
                algorithm_data.append({
                    'method': 'Noisy True Position',
                    'estimated_position': estimated,
                    'confidence': confidence
                })
        else:
            total_confidence = sum(confidences)
            weighted_x = sum(pos[0] * conf for pos, conf in zip(estimates, confidences)) / total_confidence
            weighted_y = sum(pos[1] * conf for pos, conf in zip(estimates, confidences)) / total_confidence
            estimated = (weighted_x, weighted_y)
            confidence = min(0.9, total_confidence / len(estimates))
            
            # Track fusion performance
            self.algorithm_performance['Fusion']['uses'] += 1
            self.algorithm_performance['Fusion']['total_confidence'] += confidence
            self.algorithm_performance['Fusion']['avg_confidence'] = (
                self.algorithm_performance['Fusion']['total_confidence'] / 
                self.algorithm_performance['Fusion']['uses']
            )
            
            methods_used.append("Fusion")
            algorithm_data.append({
                'method': 'Fusion',
                'component_estimates': estimates,
                'estimated_position': estimated,
                'confidence': confidence,
                'methods_used': methods_used[:-1]
            })

        # Apply path constraints to prevent divergence
        estimated = self.apply_path_constraint(estimated, true_position, confidence)

        # ALWAYS APPLY KALMAN FILTERING FOR SMOOTHING
        kalman_info = "Kalman (Not Applied)"
        if True:  # Always apply Kalman filter after frame 3
            estimated, kalman_info = self.enhanced_kalman_filter(estimated, confidence)
            methods_used.append("Kalman Filter")

        estimated_tuple = (float(estimated[0]), float(estimated[1]))

        # Update last valid position if confidence is good
        if confidence > 0.4:
            self.last_valid_position = estimated_tuple

        self.position_history.append(true_position)
        self.estimated_positions.append(estimated_tuple)
        self.confidence_history.append(confidence)
        self.method_used.append(methods_used)

        # Store complete journey data
        journey_point = {
            'frame': frame,
            'true_position': true_position,
            'estimated_position': estimated_tuple,
            'confidence': confidence,
            'methods_used': methods_used.copy(),
            'phase': 'unknown'
        }
        self.complete_journey_history.append(journey_point)

        self.algorithm_visualization_data.append({
            'true_position': true_position,
            'estimated_position': estimated_tuple,
            'methods_used': methods_used,
            'algorithm_data': algorithm_data,
            'kalman_info': kalman_info,
            'confidence': confidence,
            'frame': frame,
            'path_constrained': self.path_constraint_active
        })

        return estimated_tuple, signals, confidence, methods_used, algorithm_data

# ==================== ENTRY SYSTEM MANAGEMENT ====================
class EntrySystemManager:
    """Manage different entry systems and their time calculations"""
    
    def __init__(self):
        self.entry_log = []
        self.gate_congestion = {"MainGate": 0, "YPGate": 0}
        self.entry_efficiency_stats = {
            "written": {"total_time": 0, "count": 0, "avg_time": 0},
            "qr_code": {"total_time": 0, "count": 0, "avg_time": 0}
        }
        
    def calculate_entry_time(self, entry_type, gate, current_congestion, frame):
        """Calculate entry time based on system type and conditions"""
        system = ENTRY_SYSTEMS[entry_type]
        
        # Base time calculation with exact 0.25 ratio
        base_time = system["base_time"]
        
        # Enhanced congestion effect with compounding
        congestion_factor = 1.0 + (current_congestion / 8.0) * system["congestion_multiplier"]
        
        # Enhanced queue effect with random variation
        queue_factor = system["queue_penalty"] * random.uniform(0.7, 1.3)
        
        # Time-of-day effect
        time_factor = 1.0
        if 50 <= frame <= 200 or 400 <= frame <= 600:  # Extended peak hours
            time_factor = 1.4
        
        # Calculate total time with compounding effects
        total_time = base_time * congestion_factor * queue_factor * time_factor
        
        # Update efficiency statistics
        self.entry_efficiency_stats[entry_type]["total_time"] += total_time
        self.entry_efficiency_stats[entry_type]["count"] += 1
        self.entry_efficiency_stats[entry_type]["avg_time"] = (
            self.entry_efficiency_stats[entry_type]["total_time"] / 
            max(1, self.entry_efficiency_stats[entry_type]["count"])
        )

        # Log entry
        entry_record = {
            'frame': frame,
            'gate': gate,
            'entry_type': entry_type,
            'congestion': current_congestion,
            'calculated_time': total_time,
            'base_time': base_time,
            'congestion_factor': congestion_factor,
            'queue_factor': queue_factor,
            'time_factor': time_factor,
            'efficiency_ratio': system["efficiency_ratio"]
        }
        self.entry_log.append(entry_record)
        
        return total_time
    
    def update_gate_congestion(self, frame):
        """Update gate congestion based on swarm behavior"""
        for gate in self.gate_congestion:
            base_congestion = 2.0  # Base congestion
            
            # Class schedule effects
            for period, info in CLASS_SCHEDULE.items():
                if info["start"] <= frame <= info["end"]:
                    if gate in ["MainGate", "YPGate"]:
                        base_congestion += info["intensity"] * 0.7  # Increased impact
            
            # Time-of-day variations
            if 50 <= frame <= 200:  # Morning peak
                base_congestion += 4.0
            elif 250 <= frame <= 350:  # Lunch peak
                base_congestion += 3.0
            elif 400 <= frame <= 600:  # Evening peak
                base_congestion += 3.5
                
            self.gate_congestion[gate] = max(1.0, base_congestion + random.uniform(-1.5, 1.5))
            
    def get_efficiency_metrics(self):
        """Get entry system efficiency metrics"""
        metrics = {}
        for system in ['written', 'qr_code']:
            if self.entry_efficiency_stats[system]['count'] > 0:
                metrics[system] = {
                    'avg_time': self.entry_efficiency_stats[system]['avg_time'],
                    'total_entries': self.entry_efficiency_stats[system]['count'],
                    'efficiency_ratio': ENTRY_SYSTEMS[system]['efficiency_ratio']
                }
        
        # Calculate comparative efficiency
        if 'written' in metrics and 'qr_code' in metrics:
            time_ratio = metrics['qr_code']['avg_time'] / metrics['written']['avg_time']
            metrics['comparative_efficiency'] = {
                'time_ratio_actual': time_ratio,
                'time_ratio_target': 0.25,
                'efficiency_gap': abs(time_ratio - 0.25)
            }
            
        return metrics

# ==================== ENHANCED AGENT MANAGEMENT ====================
def create_enhanced_agents(num_agents):
    """Create agents with balanced entry system variants and enhanced tracking"""
    agents = []
    gates = ["MainGate", "YPGate"]
    destinations = ["HostelA", "HostelB", "HostelC", "Acad1", "Acad2", "Library", "Sports"]
    vendors = ["Zomato", "Swiggy", "Amazon", "Dunzo", "UberEats", "FoodPanda"]
    vehicle_types = ["petrol", "e-bike", "cycle"]
    
    # Balanced entry systems (50% QR, 50% Written)
    entry_types = ["qr_code"] * (num_agents // 2) + ["written"] * (num_agents - num_agents // 2)
    random.shuffle(entry_types)

    phase_colors = {'outward': 'blue', 'delivering': 'orange', 'returning': 'green', 'completed': 'purple'}

    for i in range(num_agents):
        try:
            gate = random.choice(gates)
            dest = random.choice(destinations)
            arrival = random.randint(0, MAX_STEPS // 2)  # Spread arrivals more
            vendor = random.choice(vendors)
            vehicle = random.choice(vehicle_types)
            entry_system = entry_types[i]
            speed = SPEED_VARIATION[vehicle]
            
            # Get marker based on entry system
            marker = ENTRY_SYSTEMS[entry_system]["marker"]

            # Calculate paths
            outward_path = find_optimal_path_with_congestion(gate, dest, {}, {})
            return_path = find_optimal_path_with_congestion(dest, gate, {}, {})

            if not outward_path or outward_path[0] != gate or outward_path[-1] != dest:
                outward_path = [gate, dest]
            if not return_path or return_path[0] != dest or return_path[-1] != gate:
                return_path = [dest, gate]

            outward_coords = create_smooth_path(outward_path, speed)
            return_coords = create_smooth_path(return_path, speed)

            if not outward_coords:
                outward_coords = [NODES[gate], NODES[dest]]
            if not return_coords:
                return_coords = [NODES[dest], NODES[gate]]

            # Enhanced time calculations with entry system
            conventional_outward_time = len(outward_coords) * 2.0
            conventional_return_time = len(return_coords) * 2.0
            delivery_duration = random.randint(3, 8)  # Increased variation
            
            # Add entry time to conventional time with enhanced calculation
            entry_manager = EntrySystemManager()
            entry_time = entry_manager.calculate_entry_time(entry_system, gate, 5.0, arrival)
            total_conventional_time = conventional_outward_time + conventional_return_time + delivery_duration + entry_time

            agent_data = {
                "agent_id": f"A{i+1:03d}",
                "gate": gate,
                "dest": dest,
                "arrival": arrival,
                "vendor": vendor,
                "vehicle": vehicle,
                "entry_system": entry_system,
                "entry_marker": marker,
                "phase": "outward",
                "outward_path_nodes": outward_path,
                "return_path_nodes": return_path,
                "outward_coords": outward_coords,
                "return_coords": return_coords,
                "current_path_coords": outward_coords,
                "current_idx": 0,
                "speed": speed,
                "color": plt.cm.Set3(i / num_agents),
                "phase_colors": phase_colors,
                "reroute_count": 0,
                "delivery_step": None,
                "return_step": None,
                "outward_time": None,
                "return_time": None,
                "total_optimized_time": None,
                "conventional_time": total_conventional_time,
                "time_saved": None,
                "efficiency_gain": None,
                "entry_time": entry_time,
                "actual_entry_time": None,
                "trail_positions": deque(maxlen=8),
                "gps_trail_positions": deque(maxlen=200),
                "delivery_duration": delivery_duration,
                "last_reroute_step": -10,
                "completed_step": None,
                "gps_tracker": EnhancedGPSTracker(agent_id=f"A{i+1:03d}"),
                "estimated_position": None,
                "current_signals": {},
                "gps_confidence": 0.0,
                "gps_methods_used": [],
                "gps_algorithm_data": [],
                "gps_complete_history": deque(maxlen=300),
                "journey_phase_history": [],
                "entry_processed": False,
                "continuous_activity": True,  # Enable continuous activity
                "journeys_completed": 0,
                "total_completed_journeys": 0  # Track all completions
            }
            agents.append(agent_data)

        except Exception as e:
            logger.error(f"Error creating agent {i}: {e}")
            # Fallback agent creation
            fallback_agent = {
                "agent_id": f"A{i+1:03d}",
                "gate": "MainGate",
                "dest": "HostelB",
                "arrival": 0,
                "vendor": "Fallback",
                "vehicle": "e-bike",
                "entry_system": "qr_code",
                "entry_marker": "^",
                "phase": "outward",
                "outward_path_nodes": ["MainGate", "HostelB"],
                "return_path_nodes": ["HostelB", "MainGate"],
                "outward_coords": [NODES["MainGate"], NODES["HostelB"]],
                "return_coords": [NODES["HostelB"], NODES["MainGate"]],
                "current_path_coords": [NODES["MainGate"], NODES["HostelB"]],
                "current_idx": 0,
                "speed": 7.0,
                "color": plt.cm.Set3(i / num_agents),
                "phase_colors": phase_colors,
                "reroute_count": 0,
                "delivery_step": None,
                "return_step": None,
                "outward_time": None,
                "return_time": None,
                "total_optimized_time": None,
                "conventional_time": 60.0,
                "time_saved": None,
                "efficiency_gain": None,
                "entry_time": 3.0,
                "actual_entry_time": None,
                "trail_positions": deque(maxlen=8),
                "gps_trail_positions": deque(maxlen=200),
                "delivery_duration": 4,
                "last_reroute_step": -10,
                "completed_step": None,
                "gps_tracker": EnhancedGPSTracker(agent_id=f"A{i+1:03d}"),
                "estimated_position": None,
                "current_signals": {},
                "gps_confidence": 0.0,
                "gps_methods_used": [],
                "gps_algorithm_data": [],
                "gps_complete_history": deque(maxlen=300),
                "journey_phase_history": [],
                "entry_processed": False,
                "continuous_activity": True,
                "journeys_completed": 0,
                "total_completed_journeys": 0
            }
            agents.append(fallback_agent)

    return agents

# ==================== SWARM TRAFFIC SYSTEM ====================
class SwarmParticle:
    """Swarm traffic particle with GPS tracking"""
    
    def __init__(self, particle_type="student"):
        self.type = particle_type
        self.speed = random.uniform(0.2, 0.4)
        self.color = self.get_particle_color()
        self.size = random.uniform(1.5, 3.5)
        self.alpha = random.uniform(0.5, 0.9)

        self.current_location = random.choice(list(NODES.keys()))
        self.target_location = None
        self.current_path = []
        self.path_index = 0
        self.progress = 0
        self.swarm_behavior = True
        self.last_route_change = 0
        self.route_cooldown = random.randint(30, 80)

        # GPS tracking
        self.gps_tracker = EnhancedGPSTracker()
        self.estimated_position = None
        self.current_signals = {}
        self.gps_confidence = 0.0
        self.gps_methods_used = []
        self.gps_algorithm_data = []

    def get_particle_color(self):
        """Different colors for different campus traffic types"""
        colors = {
            "student": ['lightblue', 'lightgreen', 'lightyellow', 'lightcyan'],
            "faculty": ['lightcoral', 'lightsalmon'],
            "service": ['silver', 'lightgray'],
            "visitor": ['plum', 'thistle']
        }
        return random.choice(colors.get(self.type, ['gray']))

    def update_swarm_route(self, frame):
        """Update route based on current swarm behavior patterns"""
        if frame - self.last_route_change < self.route_cooldown:
            return

        current_period = None
        for period, info in CLASS_SCHEDULE.items():
            if info["start"] <= frame <= info["end"]:
                current_period = period
                break

        if current_period and random.random() < 0.7:
            swarm_routes = CLASS_SCHEDULE[current_period]["swarm_routes"]
            matching_routes = []
            for route in swarm_routes:
                if self.current_location in route["from"]:
                    matching_routes.append(route)

            if matching_routes:
                chosen_route = random.choice(matching_routes)
                self.target_location = random.choice(chosen_route["to"])
                try:
                    self.current_path = nx.shortest_path(CAMPUS_GRAPH, self.current_location, self.target_location)
                    self.path_index = 0
                    self.progress = 0
                    self.last_route_change = frame
                    self.route_cooldown = random.randint(40, 100)
                except:
                    pass
        else:
            if random.random() < 0.1:
                available_nodes = list(NODES.keys())
                self.target_location = random.choice(available_nodes)
                if self.target_location != self.current_location:
                    try:
                        self.current_path = nx.shortest_path(CAMPUS_GRAPH, self.current_location, self.target_location)
                        self.path_index = 0
                        self.progress = 0
                        self.last_route_change = frame
                    except:
                        pass

    def update_gps_tracking(self, frame):
        """Update GPS position estimation"""
        true_position = self.get_position()
        self.estimated_position, self.current_signals, self.gps_confidence, self.gps_methods_used, self.gps_algorithm_data = self.gps_tracker.estimate_enhanced_position(true_position, frame)

    def update(self, frame):
        """Update particle position with swarm behavior"""
        self.update_swarm_route(frame)

        if not self.current_path or self.path_index >= len(self.current_path) - 1:
            if self.target_location and self.current_location != self.target_location:
                try:
                    self.current_path = nx.shortest_path(CAMPUS_GRAPH, self.current_location, self.target_location)
                    self.path_index = 0
                    self.progress = 0
                except:
                    self.current_path = []
            return

        # Move along current path segment
        current_node = self.current_path[self.path_index]
        next_node = self.current_path[self.path_index + 1]

        current_pos = NODES[current_node]
        next_pos = NODES[next_node]
        distance = math.hypot(next_pos[0] - current_pos[0], next_pos[1] - current_pos[1])

        if distance > 0:
            step_size = self.speed / distance
            self.progress += step_size

            if self.progress >= 1.0:
                self.progress = 0
                self.path_index += 1
                self.current_location = next_node

                if self.path_index >= len(self.current_path) - 1:
                    self.current_location = self.current_path[-1]
                    self.target_location = None
                    self.current_path = []

        # Update GPS tracking
        self.update_gps_tracking(frame)

    def get_position(self):
        """Get current position of particle"""
        if not self.current_path or self.path_index >= len(self.current_path) - 1:
            return NODES.get(self.current_location, (0, 0))

        current_node = self.current_path[self.path_index]
        next_node = self.current_path[self.path_index + 1]

        current_pos = NODES[current_node]
        next_pos = NODES[next_node]

        x = current_pos[0] + (next_pos[0] - current_pos[0]) * self.progress
        y = current_pos[1] + (next_pos[1] - current_pos[1]) * self.progress

        return (x, y)

# ==================== BACKGROUND SYSTEMS ====================
def get_background_congestion(frame):
    """Calculate background congestion based on swarm behavior"""
    background_congestion = {}

    # Base congestion
    for node in NODES:
        background_congestion[node] = 0.3 + 0.2 * random.random()

    # Class schedule peaks
    for period, info in CLASS_SCHEDULE.items():
        if info["start"] <= frame <= info["end"]:
            intensity = info["intensity"]

            for route in info["swarm_routes"]:
                route_intensity = route["intensity"]
                for area in route["from"] + route["to"]:
                    if area in background_congestion:
                        background_congestion[area] += route_intensity * (0.7 + 0.6 * random.random())

            for area in info["areas"]:
                if area in background_congestion:
                    background_congestion[area] += intensity * (0.5 + 0.5 * random.random())

    # Time-of-day variations
    if 50 <= frame <= 200:
        for node in ["MainGate", "YPGate", "Hub1", "Hub2", "Acad1", "Acad2", "Library"]:
            background_congestion[node] += 2.0
    elif 250 <= frame <= 350:
        for node in ["FoodCourt", "Hub1", "Hub2"]:
            background_congestion[node] += 2.5
    elif 400 <= frame <= 600:
        for node in ["HostelA", "HostelB", "HostelC", "MainGate", "YPGate", "Sports"]:
            background_congestion[node] += 2.0

    return background_congestion

def find_optimal_path_with_congestion(start, end, current_congestion, background_congestion):
    """Find optimal path using A* algorithm with congestion consideration"""
    try:
        def heuristic(u, v):
            pos_u = NODES[u]
            pos_v = NODES[v]
            return math.hypot(pos_u[0] - pos_v[0], pos_u[1] - pos_v[1])

        temp_G = CAMPUS_GRAPH.copy()
        for u, v in temp_G.edges():
            base_weight = temp_G[u][v]['base_weight']
            node_congestion = (current_congestion.get(u, 0) + current_congestion.get(v, 0)) / 2
            background_node_congestion = (background_congestion.get(u, 0) + background_congestion.get(v, 0)) / 2

            effective_congestion = 0.7 * background_node_congestion + 0.3 * node_congestion

            if effective_congestion > 2:
                congestion_penalty = 1 + min(4.0, 0.6 * effective_congestion)
            else:
                congestion_penalty = 1 + min(2.0, 0.4 * effective_congestion)

            temp_G[u][v]['weight'] = base_weight * congestion_penalty

        path = nx.astar_path(temp_G, start, end, heuristic=heuristic, weight='weight')
        return path
    except:
        try:
            return nx.shortest_path(CAMPUS_GRAPH, start, end, weight='weight')
        except:
            if start != end:
                return [start, end]
            return [start]

def create_smooth_path(path_nodes, speed):
    """Create smooth path coordinates"""
    if not path_nodes or len(path_nodes) < 2:
        return [NODES[path_nodes[0]]] if path_nodes else []

    coords = []
    total_distance = 0

    for j in range(len(path_nodes) - 1):
        start_node = path_nodes[j]
        end_node = path_nodes[j + 1]
        if start_node in NODES and end_node in NODES:
            start = NODES[start_node]
            end = NODES[end_node]
            total_distance += math.hypot(end[0] - start[0], end[1] - start[1])

    total_steps = max(5, int(total_distance * SMOOTHNESS_FACTOR / speed))

    for j in range(len(path_nodes) - 1):
        start_node = path_nodes[j]
        end_node = path_nodes[j + 1]

        if start_node not in NODES or end_node not in NODES:
            continue

        start = NODES[start_node]
        end = NODES[end_node]
        segment_distance = math.hypot(end[0] - start[0], end[1] - start[1])

        if total_distance > 0:
            segment_steps = max(2, int(total_steps * (segment_distance / total_distance)))
        else:
            segment_steps = 5

        for k in range(segment_steps):
            t = k / segment_steps
            x = start[0] + (end[0] - start[0]) * t
            y = start[1] + (end[1] - start[1]) * t
            coords.append((x, y))

    if path_nodes and path_nodes[-1] in NODES:
        coords.append(NODES[path_nodes[-1]])

    return coords

# ==================== ENHANCED SIMULATION ENGINE ====================
@dataclass
class AgentState:
    """Agent state for simulation"""
    agent_id: str
    true_position: Tuple[float, float]
    estimated_position: Optional[Tuple[float, float]]
    phase: str
    color: Any
    gps_confidence: float
    methods_used: List[str]
    trail_positions: Deque[Tuple[float, float]]
    current_signals: Dict[str, float]
    algorithm_data: List[Dict]
    entry_system: str
    entry_time: float
    entry_marker: str

@dataclass  
class FrameData:
    """Frame data for simulation"""
    frame: int
    agent_states: List[AgentState]
    statistics: Dict[str, Any]
    gps_accuracy: float
    algorithm_usage: Dict[str, int]
    heatmap_data: np.ndarray
    gps_heatmap_data: np.ndarray
    traffic_particle_positions: List[Tuple[float, float]]
    traffic_gps_positions: List[Tuple[float, float]]
    entry_stats: Dict[str, Any]
    algorithm_performance: Dict[str, Any]

class EnhancedSimulationEngine:
    """Enhanced simulation engine with entry systems and complete analytics"""

    def __init__(self):
        self.agents = create_enhanced_agents(NUM_AGENTS)
        self.traffic_particles = self._create_traffic_particles()
        self.entry_manager = EntrySystemManager()
        self.precomputed_frames = []
        self.is_precomputing = False
        self.precompute_queue = Queue()
        self.max_steps = MAX_STEPS
        self.G = CAMPUS_GRAPH
        self.nodes = NODES
        
        # Track cumulative completions
        self.total_completions = 0

        # Enhanced statistics tracking
        self.active_agents_history = np.zeros(MAX_STEPS, dtype=np.int16)
        self.delivered_agents_history = np.zeros(MAX_STEPS, dtype=np.int16)
        self.returned_agents_history = np.zeros(MAX_STEPS, dtype=np.int16)
        self.congestion_history = np.zeros(MAX_STEPS, dtype=np.float16)
        self.gps_accuracy_history = np.zeros(MAX_STEPS, dtype=np.float16)
        self.efficiency_history = np.zeros(MAX_STEPS, dtype=np.float16)
        self.entry_time_history = {"written": np.zeros(MAX_STEPS, dtype=np.float16),
                                 "qr_code": np.zeros(MAX_STEPS, dtype=np.float16)}

        self.algorithm_usage_history = {
            'Trilateration': np.zeros(MAX_STEPS, dtype=np.int16),
            'Fingerprinting': np.zeros(MAX_STEPS, dtype=np.int16),
            'Centroid': np.zeros(MAX_STEPS, dtype=np.int16),
            'Kalman Filter': np.zeros(MAX_STEPS, dtype=np.int16),
            'Fusion': np.zeros(MAX_STEPS, dtype=np.int16)
        }
        
        self.algorithm_performance_history = {
            'Trilateration': {'avg_confidence': np.zeros(MAX_STEPS, dtype=np.float16)},
            'Fingerprinting': {'avg_confidence': np.zeros(MAX_STEPS, dtype=np.float16)},
            'Centroid': {'avg_confidence': np.zeros(MAX_STEPS, dtype=np.float16)},
            'Kalman Filter': {'avg_confidence': np.zeros(MAX_STEPS, dtype=np.float16)},
            'Fusion': {'avg_confidence': np.zeros(MAX_STEPS, dtype=np.float16)}
        }

    def _create_traffic_particles(self):
        """Create swarm of traffic particles"""
        traffic_particles = []
        particle_types = ["student", "student", "student", "faculty", "service", "visitor"]

        for _ in range(250):  # Increased traffic
            particle_type = random.choice(particle_types)
            traffic_particles.append(SwarmParticle(particle_type))

        return traffic_particles

    def precompute_simulation(self):
        """Precompute entire simulation"""
        self.is_precomputing = True
        logger.info("Starting enhanced simulation precomputation...")
        threading.Thread(target=self._precompute_thread, daemon=True).start()

    def _precompute_thread(self):
        """Background thread for simulation precomputation"""
        try:
            start_time = time.time()

            for frame in range(MAX_STEPS):
                if frame % 50 == 0 or frame < 10:
                    progress = f"Computing frame {frame}/{MAX_STEPS}"
                    self.precompute_queue.put(("progress", progress))
                    logger.info(progress)
                    gc.collect()

                frame_data = self._compute_frame(frame)
                self.precomputed_frames.append(frame_data)

                if frame % 20 == 0:
                    gc.collect()

            computation_time = time.time() - start_time
            completion_msg = f"Precomputation completed in {computation_time:.2f}s - {len(self.precomputed_frames)} frames ready"
            self.precompute_queue.put(("complete", completion_msg))
            logger.info(completion_msg)

            self.is_precomputing = False

        except Exception as e:
            error_msg = f"Precomputation error: {str(e)}"
            self.precompute_queue.put(("error", error_msg))
            logger.error(error_msg)

    def _compute_frame(self, frame):
        """Compute a single simulation frame"""
        # Update traffic particles
        traffic_positions = []
        traffic_gps_positions = []

        for particle in self.traffic_particles:
            particle.update(frame)
            true_pos = particle.get_position()
            traffic_positions.append(true_pos)

            if particle.estimated_position:
                traffic_gps_positions.append(particle.estimated_position)

        # Update entry system
        self.entry_manager.update_gate_congestion(frame)
        background_congestion = get_background_congestion(frame)

        # Update agents and collect data
        agent_states = []
        node_counts = {n: 0 for n in self.nodes}
        active_count = 0
        delivered_count = 0
        returned_count = self.total_completions  # Use cumulative completions
        total_gps_error = 0
        gps_tracked_count = 0
        total_efficiency = 0
        efficiency_count = 0

        # Entry system statistics
        entry_processing = {"written": 0, "qr_code": 0}
        entry_times = {"written": [], "qr_code": []}

        algorithm_usage = {
            'Trilateration': 0,
            'Fingerprinting': 0,
            'Centroid': 0,
            'Kalman Filter': 0,
            'Fusion': 0
        }
        
        # Enhanced algorithm performance tracking
        algorithm_performance = {
            'Trilateration': {'total_confidence': 0, 'count': 0, 'avg_confidence': 0},
            'Fingerprinting': {'total_confidence': 0, 'count': 0, 'avg_confidence': 0},
            'Centroid': {'total_confidence': 0, 'count': 0, 'avg_confidence': 0},
            'Kalman Filter': {'total_confidence': 0, 'count': 0, 'avg_confidence': 0},
            'Fusion': {'total_confidence': 0, 'count': 0, 'avg_confidence': 0}
        }

        for agent in self.agents:
            agent_state = self._update_agent(agent, frame, node_counts, background_congestion)
            agent_states.append(agent_state)

            # Track algorithm usage and performance
            for method in agent_state.methods_used:
                if method in algorithm_usage:
                    algorithm_usage[method] += 1
                    
            # Track algorithm performance from GPS tracker
            tracker = agent['gps_tracker']
            for algo_name, perf in tracker.algorithm_performance.items():
                if perf['uses'] > 0:
                    algorithm_performance[algo_name]['total_confidence'] += perf['avg_confidence']
                    algorithm_performance[algo_name]['count'] += 1

            # Track phase counts
            if agent_state.phase in ['outward', 'returning']:
                active_count += 1
            elif agent_state.phase == 'delivering':
                delivered_count += 1

            # Track GPS accuracy
            if (agent_state.true_position and agent_state.estimated_position and
                agent_state.phase != 'completed'):
                error = math.hypot(
                    agent_state.true_position[0] - agent_state.estimated_position[0],
                    agent_state.true_position[1] - agent_state.estimated_position[1]
                )
                total_gps_error += error
                gps_tracked_count += 1

            # Track entry system processing
            if agent_state.phase == 'outward' and not agent.get('entry_processed', False):
                entry_processing[agent_state.entry_system] += 1
                entry_times[agent_state.entry_system].append(agent_state.entry_time)

            # Track efficiency for completed agents
            if agent_state.phase == 'completed' and agent.get('efficiency_gain'):
                total_efficiency += agent['efficiency_gain']
                efficiency_count += 1

        # Calculate algorithm performance averages
        for algo_name, perf in algorithm_performance.items():
            if perf['count'] > 0:
                perf['avg_confidence'] = perf['total_confidence'] / perf['count']

        # Calculate statistics
        congestion_level = sum(node_counts.values()) / len(self.nodes) if self.nodes else 0
        gps_accuracy = total_gps_error / gps_tracked_count if gps_tracked_count > 0 else 0
        avg_efficiency = total_efficiency / efficiency_count if efficiency_count > 0 else 0

        # Update history
        if frame < MAX_STEPS:
            self.active_agents_history[frame] = active_count
            self.delivered_agents_history[frame] = delivered_count
            self.returned_agents_history[frame] = returned_count
            self.congestion_history[frame] = congestion_level
            self.gps_accuracy_history[frame] = gps_accuracy
            self.efficiency_history[frame] = avg_efficiency

            # Update algorithm usage history
            for algo, count in algorithm_usage.items():
                self.algorithm_usage_history[algo][frame] = count
                
            # Update algorithm performance history
            for algo, perf in algorithm_performance.items():
                if algo in self.algorithm_performance_history:
                    self.algorithm_performance_history[algo]['avg_confidence'][frame] = perf['avg_confidence']

            # Update entry time history
            for system in ['written', 'qr_code']:
                if entry_times[system]:
                    self.entry_time_history[system][frame] = np.mean(entry_times[system])

        # Compute heatmaps
        heatmap_data = self._compute_heatmap(agent_states, traffic_positions, background_congestion)
        gps_heatmap_data = self._compute_gps_heatmap(agent_states, traffic_gps_positions)

        # Enhanced entry statistics with efficiency metrics
        entry_efficiency = self.entry_manager.get_efficiency_metrics()
        entry_stats = {
            'processing': entry_processing,
            'avg_times': {system: np.mean(times) if times else 0 for system, times in entry_times.items()},
            'gate_congestion': self.entry_manager.gate_congestion.copy(),
            'efficiency_metrics': entry_efficiency
        }

        statistics = {
            'active_count': active_count,
            'delivered_count': delivered_count,
            'returned_count': returned_count,  # This now shows cumulative completions
            'congestion_level': congestion_level,
            'gps_accuracy': gps_accuracy,
            'efficiency': avg_efficiency,
            'total_agents': NUM_AGENTS,
            'frame': frame,
            'continuous_activity': sum(1 for a in self.agents if a.get('continuous_activity', False))
        }

        return FrameData(
            frame=frame,
            agent_states=agent_states,
            statistics=statistics,
            gps_accuracy=gps_accuracy,
            algorithm_usage=algorithm_usage,
            heatmap_data=heatmap_data,
            gps_heatmap_data=gps_heatmap_data,
            traffic_particle_positions=traffic_positions,
            traffic_gps_positions=traffic_gps_positions,
            entry_stats=entry_stats,
            algorithm_performance=algorithm_performance
        )

    def _update_agent(self, agent, frame, node_counts, background_congestion):
        """Update a single agent's state with continuous activity"""
        if frame < agent['arrival']:
            return AgentState(
                agent_id=agent['agent_id'],
                true_position=None,
                estimated_position=None,
                phase=agent['phase'],
                color=agent['color'],
                gps_confidence=0.0,
                methods_used=[],
                trail_positions=deque(maxlen=8),
                current_signals={},
                algorithm_data=[],
                entry_system=agent['entry_system'],
                entry_time=0.0,
                entry_marker=agent['entry_marker']
            )

        true_position = None
        estimated_position = None
        gps_confidence = 0.0
        methods_used = []
        current_signals = {}
        algorithm_data = []

        # Enhanced continuous activity system
        if agent.get('continuous_activity', False) and agent['phase'] == 'completed':
            # Reset agent for new journey after a short break
            reset_delay = random.randint(10, 30)
            if frame >= agent['completed_step'] + reset_delay:
                agent['phase'] = 'outward'
                # FIXED: Corrected the list comprehension with proper bracket matching
                available_nodes = [n for n in NODES.keys() if n not in [agent['gate'], agent['dest']]]
                agent['dest'] = random.choice(available_nodes)
                agent['arrival'] = frame
                agent['entry_processed'] = False
                agent['journeys_completed'] += 1
                
                # Recalculate paths for new destination
                outward_path = find_optimal_path_with_congestion(agent['gate'], agent['dest'], {}, {})
                return_path = find_optimal_path_with_congestion(agent['dest'], agent['gate'], {}, {})
                
                if outward_path and outward_path[0] == agent['gate'] and outward_path[-1] == agent['dest']:
                    agent['outward_path_nodes'] = outward_path
                    agent['outward_coords'] = create_smooth_path(outward_path, agent['speed'])
                    agent['current_path_coords'] = agent['outward_coords']
                
                if return_path and return_path[0] == agent['dest'] and return_path[-1] == agent['gate']:
                    agent['return_path_nodes'] = return_path
                    agent['return_coords'] = create_smooth_path(return_path, agent['speed'])
                
                agent['current_idx'] = 0
                agent['trail_positions'].clear()
                agent['delivery_step'] = None
                agent['return_step'] = None
                agent['completed_step'] = None

        # Handle entry processing
        if not agent.get('entry_processed', False) and agent['phase'] == 'outward':
            gate_congestion = self.entry_manager.gate_congestion[agent['gate']]
            actual_entry_time = self.entry_manager.calculate_entry_time(
                agent['entry_system'], agent['gate'], gate_congestion, frame
            )
            agent['actual_entry_time'] = actual_entry_time
            
            # Simulate entry processing delay
            if frame < agent['arrival'] + int(actual_entry_time):
                true_position = NODES[agent['gate']]
                node_counts[agent['gate']] += 1
                agent['entry_processed'] = False
            else:
                agent['entry_processed'] = True
                true_position = NODES[agent['gate']]  # Start moving after entry

        # Enhanced phase management
        if agent['phase'] == 'outward' and agent.get('entry_processed', True):
            if agent['current_idx'] < len(agent['current_path_coords']):
                true_position = agent['current_path_coords'][agent['current_idx']]
                agent['trail_positions'].append(true_position)

                # Update nearest node count
                nearest_node = min(self.nodes, key=lambda n: math.hypot(true_position[0] - self.nodes[n][0],
                                                                      true_position[1] - self.nodes[n][1]))
                node_counts[nearest_node] += 1

            else:
                agent['phase'] = 'delivering'
                agent['delivery_step'] = frame
                agent['journey_phase_history'].append(('outward_to_delivering', frame))
                true_position = self.nodes[agent['dest']]

        elif agent['phase'] == 'delivering':
            true_position = self.nodes[agent['dest']]
            node_counts[agent['dest']] += 1

            if frame >= agent['delivery_step'] + agent['delivery_duration']:
                agent['phase'] = 'returning'
                agent['current_path_coords'] = agent['return_coords']
                agent['current_idx'] = 0
                agent['journey_phase_history'].append(('delivering_to_returning', frame))

        elif agent['phase'] == 'returning':
            if agent['current_idx'] < len(agent['current_path_coords']):
                true_position = agent['current_path_coords'][agent['current_idx']]
                agent['trail_positions'].append(true_position)

                # Update nearest node count
                nearest_node = min(self.nodes, key=lambda n: math.hypot(true_position[0] - self.nodes[n][0],
                                                                      true_position[1] - self.nodes[n][1]))
                node_counts[nearest_node] += 1
            else:
                agent['phase'] = 'completed'
                agent['return_step'] = frame
                agent['completed_step'] = frame
                agent['total_optimized_time'] = frame - agent['arrival']
                agent['time_saved'] = agent['conventional_time'] - agent['total_optimized_time']
                if agent['conventional_time'] > 0:
                    agent['efficiency_gain'] = (agent['time_saved'] / agent['conventional_time']) * 100
                agent['journey_phase_history'].append(('returning_to_completed', frame))
                true_position = self.nodes[agent['gate']]
                
                # Increment total completions counter
                self.total_completions += 1
                agent['total_completed_journeys'] += 1

        # Enhanced GPS tracking with all algorithms
        if true_position:
            estimated_position, current_signals, gps_confidence, methods_used, algorithm_data = agent['gps_tracker'].estimate_enhanced_position(true_position, frame)
            if estimated_position:
                agent['gps_trail_positions'].append(estimated_position)
                agent['gps_complete_history'].append(estimated_position)

        # Increment movement index for moving agents
        if agent['phase'] in ['outward', 'returning'] and true_position and agent.get('entry_processed', True):
            agent['current_idx'] += 1

        return AgentState(
            agent_id=agent['agent_id'],
            true_position=true_position,
            estimated_position=estimated_position,
            phase=agent['phase'],
            color=agent['color'],
            gps_confidence=gps_confidence,
            methods_used=methods_used,
            trail_positions=agent['trail_positions'].copy(),
            current_signals=current_signals,
            algorithm_data=algorithm_data,
            entry_system=agent['entry_system'],
            entry_time=agent.get('actual_entry_time', 0.0),
            entry_marker=agent['entry_marker']
        )

    def _compute_heatmap(self, agent_states, traffic_positions, background_congestion):
        """Compute heatmap for visualization"""
        heat = np.zeros((20, 18), dtype=np.float16)

        # Add agent heat
        for state in agent_states:
            if state.true_position:
                x, y = state.true_position
                ix = int(np.clip((x + 10) / 100 * 17, 0, 17))
                iy = int(np.clip((y + 60) / 120 * 19, 0, 19))

                intensity = 0.3
                if state.phase == 'delivering':
                    intensity = 0.6
                elif state.phase in ['outward', 'returning']:
                    intensity = 0.4

                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        idx_x = min(max(0, ix + dx), 17)
                        idx_y = min(max(0, iy + dy), 19)
                        heat[idx_y, idx_x] += intensity

        # Add traffic heat
        for x, y in traffic_positions:
            ix = int(np.clip((x + 10) / 100 * 17, 0, 17))
            iy = int(np.clip((y + 60) / 120 * 19, 0, 19))
            for dx in [-1, 0, 1]:
                for dy in [-1, 0, 1]:
                    idx_x = min(max(0, ix + dx), 17)
                    idx_y = min(max(0, iy + dy), 19)
                    heat[idx_y, idx_x] += 0.1

        return heat

    def _compute_gps_heatmap(self, agent_states, traffic_gps_positions):
        """Compute GPS heatmap for visualization"""
        gps_heat = np.zeros((20, 18), dtype=np.float16)

        # Add GPS position heat
        for state in agent_states:
            if state.estimated_position:
                x, y = state.estimated_position
                ix = int(np.clip((x + 10) / 100 * 17, 0, 17))
                iy = int(np.clip((y + 60) / 120 * 19, 0, 19))
                if 0 <= ix < 18 and 0 <= iy < 20:
                    intensity = 0.4 * state.gps_confidence
                    gps_heat[iy, ix] += intensity

        # Add traffic GPS heat
        for x, y in traffic_gps_positions:
            ix = int(np.clip((x + 10) / 100 * 17, 0, 17))
            iy = int(np.clip((y + 60) / 120 * 19, 0, 19))
            if 0 <= ix < 18 and 0 <= iy < 20:
                gps_heat[iy, ix] += 0.1

        return gps_heat

    def generate_post_simulation_analytics(self):
        """Generate comprehensive post-simulation analytics"""
        analytics = {}
        
        # Basic completion statistics
        completed_agents = [a for a in self.agents if a.get('phase') == 'completed']
        total_journeys = sum(a.get('total_completed_journeys', 0) for a in self.agents)
        
        analytics['completion_stats'] = {
            'total_agents': NUM_AGENTS,
            'completed_agents': len(completed_agents),
            'completion_rate': (len(completed_agents) / NUM_AGENTS) * 100,
            'total_journeys': total_journeys,
            'avg_journeys_per_agent': total_journeys / NUM_AGENTS
        }
        
        # Time efficiency analysis
        if completed_agents:
            total_conventional = sum(a['conventional_time'] for a in completed_agents)
            total_optimized = sum(a.get('total_optimized_time', 0) for a in completed_agents)
            total_saved = total_conventional - total_optimized
            
            analytics['time_efficiency'] = {
                'total_conventional_time': total_conventional,
                'total_optimized_time': total_optimized,
                'total_time_saved': total_saved,
                'avg_efficiency_gain': np.mean([a.get('efficiency_gain', 0) for a in completed_agents if a.get('efficiency_gain')])
            }
        
        # GPS accuracy analysis
        gps_errors = []
        for agent in self.agents:
            if hasattr(agent['gps_tracker'], 'position_history') and hasattr(agent['gps_tracker'], 'estimated_positions'):
                min_len = min(len(agent['gps_tracker'].position_history), 
                            len(agent['gps_tracker'].estimated_positions))
                for i in range(min_len):
                    true_pos = agent['gps_tracker'].position_history[i]
                    est_pos = agent['gps_tracker'].estimated_positions[i]
                    if true_pos and est_pos:
                        error = math.hypot(true_pos[0] - est_pos[0], true_pos[1] - est_pos[1])
                        gps_errors.append(error)
        
        analytics['gps_performance'] = {
            'avg_error': np.mean(gps_errors) if gps_errors else 0,
            'max_error': max(gps_errors) if gps_errors else 0,
            'min_error': min(gps_errors) if gps_errors else 0
        }
        
        # Entry system analysis
        written_times = []
        qr_times = []
        for agent in self.agents:
            if agent.get('actual_entry_time'):
                if agent['entry_system'] == 'written':
                    written_times.append(agent['actual_entry_time'])
                else:
                    qr_times.append(agent['actual_entry_time'])
        
        analytics['entry_systems'] = {
            'written': {
                'avg_time': np.mean(written_times) if written_times else 0,
                'count': len(written_times),
                'target_efficiency': 1.0
            },
            'qr_code': {
                'avg_time': np.mean(qr_times) if qr_times else 0,
                'count': len(qr_times),
                'target_efficiency': 0.25
            }
        }
        
        # Calculate actual efficiency ratio
        if written_times and qr_times:
            actual_ratio = np.mean(qr_times) / np.mean(written_times)
            analytics['entry_efficiency'] = {
                'actual_ratio': actual_ratio,
                'target_ratio': 0.25,
                'efficiency_gap': abs(actual_ratio - 0.25),
                'achievement_percentage': (0.25 / actual_ratio) * 100 if actual_ratio > 0 else 0
            }
        
        # Algorithm usage analysis
        algorithm_totals = {}
        for algo in self.algorithm_usage_history:
            algorithm_totals[algo] = np.sum(self.algorithm_usage_history[algo])
        
        analytics['algorithm_usage'] = algorithm_totals
        
        # Algorithm performance analysis
        algorithm_performance = {}
        for algo in self.algorithm_performance_history:
            non_zero_values = self.algorithm_performance_history[algo]['avg_confidence'][
                self.algorithm_performance_history[algo]['avg_confidence'] > 0
            ]
            if len(non_zero_values) > 0:
                algorithm_performance[algo] = {
                    'avg_confidence': np.mean(non_zero_values),
                    'max_confidence': np.max(non_zero_values),
                    'min_confidence': np.min(non_zero_values)
                }
        
        analytics['algorithm_performance'] = algorithm_performance
        
        return analytics

# ==================== ENHANCED GUI WITH SCROLLABLE CONTENT ====================
class EnhancedCampusGUI:
    """Professional GUI with scrollable algorithm performance displays"""

    def __init__(self, root):
        self.root = root
        self.root.title("Smart Campus Delivery Dashboard - Enhanced Version")
        self.root.geometry("1800x1200")
        self.root.configure(bg='#0f1116')

        # Initialize enhanced simulation engine
        self.simulation = EnhancedSimulationEngine()
        self.post_simulation_analytics = None

        # Enhanced GUI state
        self.state = {
            'current_frame': 0,
            'is_playing': False,
            'animation_speed': 50,
            'active_tab': 'main',
            'view_mode': 'delivery',
            'theme': 'dark',
            'metrics': {
                'active_agents': 0,
                'delivering': 0,
                'completed': 0,
                'efficiency': '0%',
                'gps_accuracy': '0%',
                'avg_speed': '0.0'
            },
            'show_traffic': True,
            'show_heatmap': True,
            'enhanced_graphics': True,
            'show_entry_markers': True
        }

        # Visualization elements
        self.agent_artists = []
        self.traffic_artists = []
        self.gps_artists = []
        self.heatmap_artists = []

        # Create enhanced UI
        self.setup_enhanced_styles()
        self.create_enhanced_layout()
        self.setup_state_management()

        # Start precomputation
        self.simulation.precompute_simulation()
        self.monitor_precomputation()

    def setup_enhanced_styles(self):
        """Configure modern styling"""
        self.style = ttk.Style()
        self.style.theme_use('clam')

        # Enhanced color palette
        self.colors = {
            'bg_primary': '#0f1116',
            'bg_secondary': '#1e1e1e',
            'bg_card': '#252526',
            'bg_hover': '#2a2d2e',
            'bg_active': '#37373d',
            'text_primary': '#ffffff',
            'text_secondary': '#cccccc',
            'text_muted': '#888888',
            'accent_primary': '#007acc',
            'accent_secondary': '#4CAF50',
            'accent_warning': '#FF9800',
            'accent_error': '#F44336',
            'accent_info': '#2196F3',
            'accent_success': '#4CAF50',
            'success': '#4CAF50',
            'warning': '#FF9800',
            'error': '#F44336',
            'info': '#2196F3',
            'border_light': '#3c3c3c',
            'border_dark': '#2d2d30',
            'animation_primary': '#00ff88',
            'animation_secondary': '#ff6b6b',
            'animation_tertiary': '#4ecdc4'
        }

        # Configure styles
        self.style.configure('Modern.TFrame', background=self.colors['bg_primary'])
        self.style.configure('Card.TFrame', background=self.colors['bg_card'], relief='flat', borderwidth=1)

        # Enhanced label styles
        self.style.configure('Title.TLabel',
                           background=self.colors['bg_card'],
                           foreground=self.colors['text_primary'],
                           font=('Segoe UI', 16, 'bold'))

        self.style.configure('Subtitle.TLabel',
                           background=self.colors['bg_card'],
                           foreground=self.colors['text_secondary'],
                           font=('Segoe UI', 11))

        self.style.configure('Metric.TLabel',
                           background=self.colors['bg_card'],
                           foreground=self.colors['text_primary'],
                           font=('Segoe UI', 20, 'bold'))

    def create_enhanced_layout(self):
        """Create enhanced modern layout"""
        main_container = ttk.Frame(self.root, style='Modern.TFrame')
        main_container.pack(fill='both', expand=True, padx=24, pady=24)

        # Enhanced header
        self.create_enhanced_header(main_container)

        # Enhanced content area
        content_container = ttk.Frame(main_container, style='Modern.TFrame')
        content_container.pack(fill='both', expand=True, pady=(24, 0))

        # Enhanced sidebar
        self.create_enhanced_sidebar(content_container)

        # Enhanced main content
        main_content_frame = ttk.Frame(content_container, style='Modern.TFrame')
        main_content_frame.pack(side='right', fill='both', expand=True)

        # Create enhanced main content
        self.create_enhanced_main_content(main_content_frame)

    def create_enhanced_sidebar(self, parent):
        """Create enhanced sidebar with scrollable functionality"""
        sidebar_frame = ttk.Frame(parent, style='Modern.TFrame', width=380)
        sidebar_frame.pack(side='left', fill='y', padx=(0, 16))
        sidebar_frame.pack_propagate(False)

        # Create canvas and scrollbar
        sidebar_canvas = tk.Canvas(sidebar_frame, bg=self.colors['bg_primary'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(sidebar_frame, orient="vertical", command=sidebar_canvas.yview)
        scrollable_frame = ttk.Frame(sidebar_canvas, style='Modern.TFrame')

        scrollable_frame.bind(
            "<Configure>",
            lambda e: sidebar_canvas.configure(scrollregion=sidebar_canvas.bbox("all"))
        )

        sidebar_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=360)
        sidebar_canvas.configure(yscrollcommand=scrollbar.set)

        sidebar_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Create sidebar content
        self.create_enhanced_sidebar_content(scrollable_frame)

    def create_enhanced_sidebar_content(self, parent):
        """Create enhanced sidebar content"""
        # Enhanced metrics section
        metrics_card = self.create_modern_card(parent, "Enhanced Live Metrics")
        metrics_card.pack(fill='x', pady=(0, 16))
        self.create_enhanced_metrics(metrics_card)

        # Enhanced controls section
        controls_card = self.create_modern_card(parent, "Enhanced Controls")
        controls_card.pack(fill='x', pady=(0, 16))
        self.create_enhanced_controls(controls_card)

        # Entry system analytics
        entry_card = self.create_modern_card(parent, "Entry System Analytics")
        entry_card.pack(fill='x', pady=(0, 16))
        self.create_entry_system_analytics(entry_card)

        # Enhanced algorithm stats with scrollable content
        algo_card = self.create_modern_card(parent, "Algorithm Performance")
        algo_card.pack(fill='x', pady=(0, 16))
        self.create_enhanced_algorithm_stats(algo_card)

    def create_enhanced_header(self, parent):
        """Create enhanced header"""
        header_frame = ttk.Frame(parent, style='Card.TFrame')
        header_frame.pack(fill='x', pady=(0, 24))

        header_left = ttk.Frame(header_frame, style='Card.TFrame')
        header_left.pack(side='left', padx=24, pady=16)

        # Enhanced logo and title
        title_label = ttk.Label(header_left, text="Smart Campus Delivery - ENHANCED",
                               style='Title.TLabel', font=('Segoe UI', 20, 'bold'))
        title_label.pack(side='left')

        subtitle_label = ttk.Label(header_left,
                                 text="Advanced GPS Tracking & Entry System Optimization",
                                 style='Subtitle.TLabel')
        subtitle_label.pack(anchor='w', pady=(4, 0))

        # Enhanced status section
        header_right = ttk.Frame(header_frame, style='Card.TFrame')
        header_right.pack(side='right', padx=24, pady=16)

        self.status_indicator = ttk.Frame(header_right, style='Card.TFrame')
        self.status_indicator.pack(side='right', padx=(16, 0))

        self.status_dot = tk.Canvas(self.status_indicator, width=16, height=16,
                              bg=self.colors['accent_warning'], highlightthickness=0)
        self.status_dot.pack(side='left')
        self.status_dot.create_oval(4, 4, 12, 12, fill=self.colors['accent_warning'], outline='')

        self.status_label = ttk.Label(self.status_indicator, text="Precomputing Simulation...",
                                     style='Subtitle.TLabel')
        self.status_label.pack(side='left', padx=(8, 0))

        # Enhanced progress bar
        self.progress_frame = ttk.Frame(header_right, style='Card.TFrame')
        self.progress_frame.pack(side='right', padx=(24, 0))

        self.progress_bar = ttk.Progressbar(self.progress_frame, orient='horizontal',
                                      length=200, mode='determinate')
        self.progress_bar.pack(side='left', padx=(0, 8))

        self.progress_label = ttk.Label(self.progress_frame, text="0%",
                                  style='Subtitle.TLabel')
        self.progress_label.pack(side='left')

    def create_enhanced_metrics(self, parent):
        """Create enhanced metrics display"""
        metrics_content = ttk.Frame(parent, style='Card.TFrame')
        metrics_content.pack(fill='x', padx=20, pady=20)

        self.enhanced_metrics_data = {
            'active_agents': {'icon': '🚚', 'label': 'Active Agents', 'color': self.colors['accent_info']},
            'delivering': {'icon': '📦', 'label': 'Delivering', 'color': self.colors['accent_warning']},
            'completed': {'icon': '✅', 'label': 'Total Completed', 'color': self.colors['success']},
            'efficiency': {'icon': '⚡', 'label': 'Efficiency', 'color': self.colors['animation_primary']},
            'gps_accuracy': {'icon': '🎯', 'label': 'GPS Accuracy', 'color': self.colors['accent_primary']},
            'entry_time': {'icon': '⏱️', 'label': 'Avg Entry Time', 'color': self.colors['info']},
            'continuous_activity': {'icon': '🔄', 'label': 'Continuous Agents', 'color': self.colors['animation_tertiary']}
        }

        self.enhanced_metric_widgets = {}

        metrics_grid = ttk.Frame(metrics_content, style='Card.TFrame')
        metrics_grid.pack(fill='x')

        row, col = 0, 0
        for metric_id, metric_info in self.enhanced_metrics_data.items():
            metric_card = ttk.Frame(metrics_grid, style='Card.TFrame')
            metric_card.grid(row=row, column=col, sticky='nsew', padx=8, pady=8)

            content_frame = ttk.Frame(metric_card, style='Card.TFrame')
            content_frame.pack(fill='both', expand=True, padx=16, pady=16)

            # Icon and label
            icon_frame = ttk.Frame(content_frame, style='Card.TFrame')
            icon_frame.pack(anchor='w')

            icon_label = ttk.Label(icon_frame, text=metric_info['icon'],
                             font=('Segoe UI', 24),
                             background=self.colors['bg_card'])
            icon_label.pack(side='left', padx=(0, 8))

            label = ttk.Label(content_frame, text=metric_info['label'],
                        style='Subtitle.TLabel', font=('Segoe UI', 12, 'bold'))
            label.pack(anchor='w', pady=(8, 0))

            # Value display
            value_frame = ttk.Frame(content_frame, style='Card.TFrame')
            value_frame.pack(anchor='w', pady=(8, 0))

            value_label = ttk.Label(value_frame, text="0",
                              style='Metric.TLabel', font=('Segoe UI', 24, 'bold'),
                              foreground=metric_info['color'])
            value_label.pack(side='left')

            self.enhanced_metric_widgets[metric_id] = value_label

            # Update grid position
            col += 1
            if col >= 2:
                col = 0
                row += 1

        metrics_grid.columnconfigure(0, weight=1)
        metrics_grid.columnconfigure(1, weight=1)

    def create_enhanced_controls(self, parent):
        """Create enhanced control panel"""
        controls_content = ttk.Frame(parent, style='Card.TFrame')
        controls_content.pack(fill='x', padx=20, pady=20)

        # Playback controls
        playback_frame = ttk.Frame(controls_content, style='Card.TFrame')
        playback_frame.pack(fill='x', pady=(0, 16))

        control_buttons = ttk.Frame(playback_frame, style='Card.TFrame')
        control_buttons.pack(fill='x', pady=(0, 12))

        self.play_btn = ttk.Button(control_buttons, text="Play",
                             command=self.toggle_play)
        self.play_btn.pack(side='left', padx=(0, 8))

        ttk.Button(control_buttons, text="Stop",
             command=self.stop_animation).pack(side='left', padx=(0, 8))

        ttk.Button(control_buttons, text="Reset",
             command=self.reset_animation).pack(side='left')

        # Speed control
        speed_frame = ttk.Frame(playback_frame, style='Card.TFrame')
        speed_frame.pack(fill='x', pady=(0, 12))

        ttk.Label(speed_frame, text="Speed:",
             style='Subtitle.TLabel', font=('Segoe UI', 12, 'bold')).pack(anchor='w')

        speed_controls = ttk.Frame(speed_frame, style='Card.TFrame')
        speed_controls.pack(fill='x', pady=(8, 0))

        self.speed_var = tk.StringVar(value="1x")
        speeds = [("0.5x", 100), ("1x", 50), ("2x", 25), ("5x", 10)]

        for speed_text, speed_value in speeds:
            btn = ttk.Radiobutton(speed_controls, text=speed_text,
                            variable=self.speed_var, value=speed_text,
                            command=lambda v=speed_value: self.on_speed_change(v))
            btn.pack(side='left', padx=(0, 8))

        # Frame slider
        slider_frame = ttk.Frame(playback_frame, style='Card.TFrame')
        slider_frame.pack(fill='x', pady=(0, 8))

        slider_header = ttk.Frame(slider_frame, style='Card.TFrame')
        slider_header.pack(fill='x')

        ttk.Label(slider_header, text="Frame Control:",
             style='Subtitle.TLabel', font=('Segoe UI', 12, 'bold')).pack(side='left')

        self.frame_label = ttk.Label(slider_header, text="Frame: 0/0",
                               style='Subtitle.TLabel')
        self.frame_label.pack(side='right')

        self.frame_slider = ttk.Scale(slider_frame, from_=0, to=100,
                                 orient='horizontal',
                                 command=self.on_slider_change)
        self.frame_slider.pack(fill='x', pady=(8, 0))

        # Visualization options
        viz_frame = ttk.Frame(playback_frame, style='Card.TFrame')
        viz_frame.pack(fill='x', pady=(8, 0))

        ttk.Label(viz_frame, text="Visualization:",
             style='Subtitle.TLabel', font=('Segoe UI', 12, 'bold')).pack(anchor='w')

        viz_controls = ttk.Frame(viz_frame, style='Card.TFrame')
        viz_controls.pack(fill='x', pady=(8, 0))

        self.show_traffic_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(viz_controls, text="Show Traffic",
                       variable=self.show_traffic_var,
                       command=self.on_viz_change).pack(side='left', padx=(0, 8))

        self.show_heatmap_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(viz_controls, text="Show Heatmap",
                       variable=self.show_heatmap_var,
                       command=self.on_viz_change).pack(side='left', padx=(0, 8))

        self.show_markers_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(viz_controls, text="Show Entry Markers",
                       variable=self.show_markers_var,
                       command=self.on_viz_change).pack(side='left')

    def create_entry_system_analytics(self, parent):
        """Create entry system analytics display"""
        entry_content = ttk.Frame(parent, style='Card.TFrame')
        entry_content.pack(fill='x', padx=20, pady=20)

        # Entry system comparison
        comparison_frame = ttk.Frame(entry_content, style='Card.TFrame')
        comparison_frame.pack(fill='x', pady=(0, 12))

        ttk.Label(comparison_frame, text="Entry System Performance",
                 style='Subtitle.TLabel', font=('Segoe UI', 14, 'bold')).pack(anchor='w')

        # Entry system stats
        self.entry_stats_frame = ttk.Frame(entry_content, style='Card.TFrame')
        self.entry_stats_frame.pack(fill='x')

        # Initialize entry system widgets
        self.entry_widgets = {}
        for system_id, system_info in ENTRY_SYSTEMS.items():
            system_frame = ttk.Frame(self.entry_stats_frame, style='Card.TFrame')
            system_frame.pack(fill='x', pady=(8, 0))

            # System header
            header_frame = ttk.Frame(system_frame, style='Card.TFrame')
            header_frame.pack(fill='x')

            color_label = ttk.Label(header_frame, text="■",
                                  foreground=system_info['color'],
                                  font=('Segoe UI', 16))
            color_label.pack(side='left', padx=(0, 8))

            ttk.Label(header_frame, text=system_info['name'],
                     style='Subtitle.TLabel', font=('Segoe UI', 12, 'bold')).pack(side='left')

            # Stats
            stats_frame = ttk.Frame(system_frame, style='Card.TFrame')
            stats_frame.pack(fill='x', pady=(4, 0))

            time_label = ttk.Label(stats_frame, text="Avg Time: 0.0s",
                                 style='Subtitle.TLabel', font=('Segoe UI', 10))
            time_label.pack(anchor='w')

            count_label = ttk.Label(stats_frame, text="Processing: 0",
                                  style='Subtitle.TLabel', font=('Segoe UI', 10))
            count_label.pack(anchor='w')
            
            efficiency_label = ttk.Label(stats_frame, text="Efficiency Ratio: 0.0",
                                       style='Subtitle.TLabel', font=('Segoe UI', 9),
                                       foreground=self.colors['accent_secondary'])
            efficiency_label.pack(anchor='w')

            self.entry_widgets[system_id] = {
                'time_label': time_label,
                'count_label': count_label,
                'efficiency_label': efficiency_label
            }

        # Comparative efficiency
        efficiency_frame = ttk.Frame(entry_content, style='Card.TFrame')
        efficiency_frame.pack(fill='x', pady=(12, 0))

        ttk.Label(efficiency_frame, text="Comparative Efficiency",
                 style='Subtitle.TLabel', font=('Segoe UI', 12, 'bold')).pack(anchor='w')

        self.efficiency_comparison_label = ttk.Label(efficiency_frame,
                                                   text="QR/Written Ratio: - (Target: 0.25)",
                                                   style='Subtitle.TLabel', font=('Segoe UI', 10))
        self.efficiency_comparison_label.pack(anchor='w', pady=(4, 0))

    def create_enhanced_algorithm_stats(self, parent):
        """Create enhanced algorithm performance statistics with scrollable content"""
        algo_content = ttk.Frame(parent, style='Card.TFrame')
        algo_content.pack(fill='both', expand=True, padx=20, pady=20)

        # Create scrollable frame for algorithm details
        algo_canvas = tk.Canvas(algo_content, bg=self.colors['bg_card'], highlightthickness=0, height=300)
        scrollbar = ttk.Scrollbar(algo_content, orient="vertical", command=algo_canvas.yview)
        scrollable_frame = ttk.Frame(algo_canvas, style='Card.TFrame')

        scrollable_frame.bind(
            "<Configure>",
            lambda e: algo_canvas.configure(scrollregion=algo_canvas.bbox("all"))
        )

        algo_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=340)
        algo_canvas.configure(yscrollcommand=scrollbar.set)

        algo_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Algorithm performance details
        self.algorithm_widgets = {}

        algorithms = [
            ("Trilateration", "📡", "High precision positioning using distance measurements from multiple APs", "Requires 3+ APs for accuracy"),
            ("Fingerprinting", "👆", "Pattern matching against pre-recorded signal database", "Database-dependent, high accuracy in known areas"),
            ("Centroid", "🎯", "Signal strength weighted average of AP positions", "Good for dense AP coverage"),
            ("Kalman Filter", "🔮", "Noise reduction and predictive smoothing", "Excellent for tracking moving agents"),
            ("Fusion", "⚡", "Combines multiple methods for optimal accuracy", "Adaptive, robust performance")
        ]

        for algo_name, icon, description, details in algorithms:
            algo_frame = ttk.Frame(scrollable_frame, style='Card.TFrame')
            algo_frame.pack(fill='x', pady=(0, 12))

            # Algorithm header
            header_frame = ttk.Frame(algo_frame, style='Card.TFrame')
            header_frame.pack(fill='x')

            icon_label = ttk.Label(header_frame, text=icon,
                             font=('Segoe UI', 16),
                             background=self.colors['bg_card'])
            icon_label.pack(side='left', padx=(0, 8))

            name_label = ttk.Label(header_frame, text=algo_name,
                             style='Subtitle.TLabel', font=('Segoe UI', 12, 'bold'))
            name_label.pack(side='left')

            # Algorithm usage indicator
            usage_frame = ttk.Frame(algo_frame, style='Card.TFrame')
            usage_frame.pack(fill='x', pady=(4, 0))

            usage_label = ttk.Label(usage_frame, text="Usage: 0 (0.0%)",
                              style='Subtitle.TLabel', font=('Segoe UI', 11),
                              foreground=self.colors['accent_secondary'])
            usage_label.pack(side='left')

            # Description
            desc_label = ttk.Label(algo_frame, text=description,
                             style='Subtitle.TLabel', font=('Segoe UI', 9),
                             foreground=self.colors['text_muted'])
            desc_label.pack(anchor='w', pady=(2, 0))

            # Technical details
            details_label = ttk.Label(algo_frame, text=details,
                               style='Subtitle.TLabel', font=('Segoe UI', 8),
                               foreground=self.colors['text_muted'])
            details_label.pack(anchor='w', pady=(1, 0))

            # Performance metrics
            metrics_frame = ttk.Frame(algo_frame, style='Card.TFrame')
            metrics_frame.pack(fill='x', pady=(4, 0))

            accuracy_label = ttk.Label(metrics_frame, text="Avg Confidence: -",
                                 style='Subtitle.TLabel', font=('Segoe UI', 9))
            accuracy_label.pack(side='left', padx=(0, 12))

            performance_label = ttk.Label(metrics_frame, text="Performance: -",
                                   style='Subtitle.TLabel', font=('Segoe UI', 9))
            performance_label.pack(side='left')

            self.algorithm_widgets[algo_name] = {
                'usage_label': usage_label,
                'accuracy_label': accuracy_label,
                'performance_label': performance_label
            }

    def create_enhanced_main_content(self, parent):
        """Create enhanced main content area"""
        tab_container = ttk.Frame(parent, style='Modern.TFrame')
        tab_container.pack(fill='both', expand=True)

        # Create enhanced notebook
        self.enhanced_notebook = ttk.Notebook(tab_container)
        self.enhanced_notebook.pack(fill='both', expand=True)

        # Enhanced tabs
        enhanced_tabs = [
            ("Simulation", self.create_simulation_tab),
            ("Advanced GPS Tracking", self.create_gps_tab),
            ("Analytics", self.create_analytics_tab),
            ("Entry System Analysis", self.create_entry_analysis_tab),
            ("Complete GPS Loops", self.create_loops_tab),
            ("Algorithm Details", self.create_algorithm_details_tab),
            ("Post-Simulation Analysis", self.create_post_simulation_tab)
        ]

        self.tab_frames = {}

        for tab_name, tab_creator in enhanced_tabs:
            tab_frame = ttk.Frame(self.enhanced_notebook, style='Modern.TFrame')
            self.enhanced_notebook.add(tab_frame, text=tab_name)
            self.tab_frames[tab_name] = tab_frame
            tab_creator(tab_frame)

        self.enhanced_notebook.bind('<<NotebookTabChanged>>', self.on_tab_change)

    def create_simulation_tab(self, parent):
        """Create simulation tab"""
        tab_content = ttk.Frame(parent, style='Modern.TFrame')
        tab_content.pack(fill='both', expand=True)

        # Create a frame for the plot that expands fully
        plot_frame = ttk.Frame(tab_content, style='Modern.TFrame')
        plot_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.fig_main = Figure(figsize=(12, 8), facecolor=self.colors['bg_card'], dpi=100)
        self.ax_main = self.fig_main.add_subplot(111)
        self.ax_main.set_facecolor(self.colors['bg_card'])

        self.ax_main.tick_params(colors=self.colors['text_secondary'], labelsize=10)
        for spine in self.ax_main.spines.values():
            spine.set_color(self.colors['border_light'])
            spine.set_linewidth(2)

        self.canvas_main = FigureCanvasTkAgg(self.fig_main, plot_frame)
        self.canvas_main.get_tk_widget().pack(fill='both', expand=True)

        self.draw_campus_map()

    def create_gps_tab(self, parent):
        """Create GPS tracking tab"""
        tab_content = ttk.Frame(parent, style='Modern.TFrame')
        tab_content.pack(fill='both', expand=True)

        # Create a frame for the plot that expands fully
        plot_frame = ttk.Frame(tab_content, style='Modern.TFrame')
        plot_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.fig_gps = Figure(figsize=(12, 8), facecolor=self.colors['bg_card'], dpi=100)
        self.ax_gps = self.fig_gps.add_subplot(111)
        self.ax_gps.set_facecolor(self.colors['bg_card'])

        self.ax_gps.tick_params(colors=self.colors['text_secondary'], labelsize=10)
        for spine in self.ax_gps.spines.values():
            spine.set_color(self.colors['border_light'])
            spine.set_linewidth(2)

        self.canvas_gps = FigureCanvasTkAgg(self.fig_gps, plot_frame)
        self.canvas_gps.get_tk_widget().pack(fill='both', expand=True)

        self.draw_wifi_access_points()

    def create_analytics_tab(self, parent):
        """Create analytics tab"""
        tab_content = ttk.Frame(parent, style='Modern.TFrame')
        tab_content.pack(fill='both', expand=True)

        # Create a frame for the plot that expands fully
        plot_frame = ttk.Frame(tab_content, style='Modern.TFrame')
        plot_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.fig_analytics = Figure(figsize=(12, 8), facecolor=self.colors['bg_card'], dpi=100)

        gs = self.fig_analytics.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        self.ax_analytics1 = self.fig_analytics.add_subplot(gs[0, 0])
        self.ax_analytics2 = self.fig_analytics.add_subplot(gs[0, 1])
        self.ax_analytics3 = self.fig_analytics.add_subplot(gs[1, 0])
        self.ax_analytics4 = self.fig_analytics.add_subplot(gs[1, 1])

        for ax in [self.ax_analytics1, self.ax_analytics2, self.ax_analytics3, self.ax_analytics4]:
            ax.set_facecolor(self.colors['bg_card'])
            ax.tick_params(colors=self.colors['text_secondary'], labelsize=9)
            for spine in ax.spines.values():
                spine.set_color(self.colors['border_light'])
                spine.set_linewidth(2)

        self.canvas_analytics = FigureCanvasTkAgg(self.fig_analytics, plot_frame)
        self.canvas_analytics.get_tk_widget().pack(fill='both', expand=True)

    def create_entry_analysis_tab(self, parent):
        """Create entry system analysis tab"""
        tab_content = ttk.Frame(parent, style='Modern.TFrame')
        tab_content.pack(fill='both', expand=True)

        # Create a frame for the plot that expands fully
        plot_frame = ttk.Frame(tab_content, style='Modern.TFrame')
        plot_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.fig_entry = Figure(figsize=(12, 8), facecolor=self.colors['bg_card'], dpi=100)

        gs = self.fig_entry.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

        self.ax_entry1 = self.fig_entry.add_subplot(gs[0, 0])  # Entry time comparison
        self.ax_entry2 = self.fig_entry.add_subplot(gs[0, 1])  # Congestion impact
        self.ax_entry3 = self.fig_entry.add_subplot(gs[1, 0])  # Time efficiency
        self.ax_entry4 = self.fig_entry.add_subplot(gs[1, 1])  # System usage

        for ax in [self.ax_entry1, self.ax_entry2, self.ax_entry3, self.ax_entry4]:
            ax.set_facecolor(self.colors['bg_card'])
            ax.tick_params(colors=self.colors['text_secondary'], labelsize=9)
            for spine in ax.spines.values():
                spine.set_color(self.colors['border_light'])
                spine.set_linewidth(2)

        self.canvas_entry = FigureCanvasTkAgg(self.fig_entry, plot_frame)
        self.canvas_entry.get_tk_widget().pack(fill='both', expand=True)

    def create_loops_tab(self, parent):
        """Create GPS loops visualization tab"""
        tab_content = ttk.Frame(parent, style='Modern.TFrame')
        tab_content.pack(fill='both', expand=True)

        # Create a frame for the plot that expands fully
        plot_frame = ttk.Frame(tab_content, style='Modern.TFrame')
        plot_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.fig_loops = Figure(figsize=(12, 8), facecolor=self.colors['bg_card'], dpi=100)
        self.ax_loops = self.fig_loops.add_subplot(111)
        self.ax_loops.set_facecolor(self.colors['bg_card'])

        self.ax_loops.tick_params(colors=self.colors['text_secondary'], labelsize=10)
        for spine in self.ax_loops.spines.values():
            spine.set_color(self.colors['border_light'])
            spine.set_linewidth(2)

        self.canvas_loops = FigureCanvasTkAgg(self.fig_loops, plot_frame)
        self.canvas_loops.get_tk_widget().pack(fill='both', expand=True)

        self.draw_campus_map_for_loops()

    def create_algorithm_details_tab(self, parent):
        """Create algorithm details tab with comprehensive information"""
        tab_content = ttk.Frame(parent, style='Modern.TFrame')
        tab_content.pack(fill='both', expand=True)

        # Create scrollable text area for algorithm details
        text_frame = ttk.Frame(tab_content, style='Modern.TFrame')
        text_frame.pack(fill='both', expand=True, padx=20, pady=20)

        self.algorithm_text = scrolledtext.ScrolledText(
            text_frame,
            wrap=tk.WORD,
            width=100,
            height=30,
            bg=self.colors['bg_card'],
            fg=self.colors['text_primary'],
            font=('Consolas', 10)
        )
        self.algorithm_text.pack(fill='both', expand=True)
        self.algorithm_text.config(state=tk.DISABLED)

    def create_post_simulation_tab(self, parent):
        """Create post-simulation analysis tab"""
        tab_content = ttk.Frame(parent, style='Modern.TFrame')
        tab_content.pack(fill='both', expand=True)

        # Create frame for post-simulation controls
        control_frame = ttk.Frame(tab_content, style='Modern.TFrame')
        control_frame.pack(fill='x', padx=20, pady=10)

        ttk.Button(control_frame, text="Generate Post-Simulation Analysis",
                  command=self.generate_post_simulation_analysis).pack(side='left', padx=(0, 10))

        ttk.Button(control_frame, text="Generate GPS Loop Visualization",
                  command=self.generate_gps_loop_visualization).pack(side='left')

        # Create notebook for different analyses
        analysis_notebook = ttk.Notebook(tab_content)
        analysis_notebook.pack(fill='both', expand=True, padx=20, pady=10)

        # Summary tab
        summary_frame = ttk.Frame(analysis_notebook, style='Modern.TFrame')
        analysis_notebook.add(summary_frame, text="Summary")

        self.summary_text = scrolledtext.ScrolledText(
            summary_frame,
            wrap=tk.WORD,
            width=100,
            height=20,
            bg=self.colors['bg_card'],
            fg=self.colors['text_primary'],
            font=('Consolas', 10)
        )
        self.summary_text.pack(fill='both', expand=True, padx=10, pady=10)
        self.summary_text.config(state=tk.DISABLED)

        # Detailed analytics tab
        details_frame = ttk.Frame(analysis_notebook, style='Modern.TFrame')
        analysis_notebook.add(details_frame, text="Detailed Analytics")

        self.details_text = scrolledtext.ScrolledText(
            details_frame,
            wrap=tk.WORD,
            width=100,
            height=20,
            bg=self.colors['bg_card'],
            fg=self.colors['text_primary'],
            font=('Consolas', 9)
        )
        self.details_text.pack(fill='both', expand=True, padx=10, pady=10)
        self.details_text.config(state=tk.DISABLED)

    def create_modern_card(self, parent, title):
        """Create a modern card component"""
        card = ttk.Frame(parent, style='Card.TFrame')

        header = ttk.Frame(card, style='Card.TFrame')
        header.pack(fill='x', padx=20, pady=(16, 12))

        ttk.Label(header, text=title, style='Title.TLabel',
             font=('Segoe UI', 14, 'bold')).pack(side='left')

        return card

    # ==================== VISUALIZATION METHODS ====================
    def draw_campus_map(self):
        """Draw campus map visualization"""
        self.ax_main.clear()
        self.ax_main.set_xlim(-10, 90)
        self.ax_main.set_ylim(-60, 60)
        self.ax_main.set_facecolor(self.colors['bg_card'])

        # Draw edges
        for u, v in self.simulation.G.edges():
            x1, y1 = self.simulation.nodes[u]
            x2, y2 = self.simulation.nodes[v]

            edge_color = self.colors['accent_primary'] if (u, v) in CONGESTION_EDGES else self.colors['border_light']
            edge_alpha = 0.8 if (u, v) in CONGESTION_EDGES else 0.5
            edge_width = 3 if (u, v) in CONGESTION_EDGES else 2

            self.ax_main.plot([x1, x2], [y1, y2],
                        color=edge_color,
                        alpha=edge_alpha,
                        linewidth=edge_width,
                        zorder=1)

        # Draw nodes
        for n, (x, y) in self.simulation.nodes.items():
            if 'Gate' in n:
                color = self.colors['success']
                size = 150
                marker = 'D'
            elif 'Hub' in n:
                color = self.colors['accent_primary']
                size = 120
                marker = 's'
            elif 'Hostel' in n:
                color = self.colors['accent_warning']
                size = 100
                marker = '^'
            else:
                color = self.colors['accent_info']
                size = 100
                marker = 'o'

            self.ax_main.scatter(x, y, c=color, s=size, marker=marker, alpha=0.9,
                       edgecolors=self.colors['text_primary'],
                       linewidth=2, zorder=2)

            self.ax_main.text(x + 2, y + 2, n, fontsize=10, fontweight='bold',
                    color=self.colors['text_primary'],
                    bbox=dict(boxstyle="round,pad=0.4",
                            facecolor=self.colors['bg_card'],
                            alpha=0.9,
                            edgecolor=self.colors['border_light']),
                    zorder=3)

        self.ax_main.set_title("Campus Delivery Network",
                     color=self.colors['text_primary'],
                     fontsize=16, pad=20, fontweight='bold')

        self.canvas_main.draw_idle()

    def draw_wifi_access_points(self):
        """Draw WiFi access points visualization"""
        self.ax_gps.clear()
        self.ax_gps.set_xlim(-10, 90)
        self.ax_gps.set_ylim(-60, 60)
        self.ax_gps.set_facecolor(self.colors['bg_card'])

        # Draw campus map background
        for u, v in self.simulation.G.edges():
            x1, y1 = self.simulation.nodes[u]
            x2, y2 = self.simulation.nodes[v]
            self.ax_gps.plot([x1, x2], [y1, y2],
                       color=self.colors['border_light'],
                       alpha=0.3, linewidth=1, zorder=1)

        # Draw WiFi access points
        for ap_name, ap_info in WIFI_ACCESS_POINTS.items():
            x, y = ap_info["pos"]

            # Signal range visualization
            inner_range = plt.Circle((x, y), ap_info["range"] * 0.3,
                           color='green', alpha=0.2, fill=True, zorder=2)
            mid_range = plt.Circle((x, y), ap_info["range"] * 0.6,
                         color='yellow', alpha=0.15, fill=True, zorder=2)
            outer_range = plt.Circle((x, y), ap_info["range"],
                           color='red', alpha=0.1, fill=True, zorder=2)

            self.ax_gps.add_patch(outer_range)
            self.ax_gps.add_patch(mid_range)
            self.ax_gps.add_patch(inner_range)

            self.ax_gps.scatter(x, y, c='red', s=120, marker='^',
                      zorder=4, edgecolors='white', linewidth=3)

            self.ax_gps.text(x, y - 10, f"{ap_name}\nRange: {ap_info['range']}m",
                   fontsize=9, ha='center', color=self.colors['text_primary'],
                   bbox=dict(boxstyle="round,pad=0.3",
                           facecolor=self.colors['bg_card'],
                           alpha=0.9,
                           edgecolor=self.colors['border_light']),
                   zorder=5)

        self.ax_gps.set_title("WiFi Access Points & Signal Coverage",
                    color=self.colors['text_primary'], fontsize=16, fontweight='bold')

        self.canvas_gps.draw_idle()

    def draw_campus_map_for_loops(self):
        """Draw campus map for GPS loops visualization"""
        self.ax_loops.clear()
        self.ax_loops.set_xlim(-10, 90)
        self.ax_loops.set_ylim(-60, 60)
        self.ax_loops.set_facecolor(self.colors['bg_card'])

        # Draw campus map
        for u, v in self.simulation.G.edges():
            x1, y1 = self.simulation.nodes[u]
            x2, y2 = self.simulation.nodes[v]
            self.ax_loops.plot([x1, x2], [y1, y2],
                         color=self.colors['border_light'],
                         alpha=0.4, linewidth=1.5, zorder=1)

        for n, (x, y) in self.simulation.nodes.items():
            self.ax_loops.scatter(x, y, c=self.colors['accent_primary'],
                        s=80, alpha=0.8, zorder=2, edgecolors='white', linewidth=2)
            self.ax_loops.text(x + 1, y + 1, n, fontsize=9,
                         color=self.colors['text_primary'])

        self.ax_loops.set_title("GPS Journey Loops Visualization",
                      color=self.colors['text_primary'], fontsize=16, fontweight='bold')

        self.canvas_loops.draw_idle()

    # ==================== INTERACTION METHODS ====================
    def toggle_play(self):
        """Toggle animation"""
        self.state['is_playing'] = not self.state['is_playing']

        if self.state['is_playing']:
            self.play_btn.configure(text="Pause")
            self.animate()
        else:
            self.play_btn.configure(text="Play")

    def stop_animation(self):
        """Stop animation"""
        self.state['is_playing'] = False
        self.play_btn.configure(text="Play")
        self.state['current_frame'] = 0
        self.frame_slider.set(0)
        self.update_display()

    def reset_animation(self):
        """Reset animation to beginning"""
        self.stop_animation()
        self.state['current_frame'] = 0
        self.update_display()

    def on_speed_change(self, speed_value):
        """Handle speed change"""
        self.state['animation_speed'] = speed_value

    def on_slider_change(self, value):
        """Handle slider change"""
        if self.simulation.precomputed_frames:
            new_frame = min(int(float(value)), len(self.simulation.precomputed_frames)-1)
            self.state['current_frame'] = new_frame
            if not self.state['is_playing']:
                self.update_display()

    def on_viz_change(self):
        """Handle visualization changes"""
        self.state['show_traffic'] = self.show_traffic_var.get()
        self.state['show_heatmap'] = self.show_heatmap_var.get()
        self.state['show_entry_markers'] = self.show_markers_var.get()
        self.update_display()

    def on_tab_change(self, event):
        """Handle tab changes"""
        selected_tab = self.enhanced_notebook.tab(self.enhanced_notebook.select(), "text")
        self.state['active_tab'] = selected_tab
        self.update_display()

    # ==================== ANIMATION AND UPDATE METHODS ====================
    def animate(self):
        """Animation loop"""
        if self.state['is_playing'] and self.simulation.precomputed_frames:
            frames = self.simulation.precomputed_frames
            current = self.state['current_frame']
            next_frame = (current + 1) % len(frames)

            self.state['current_frame'] = next_frame
            self.frame_slider.set(next_frame)
            self.update_display()

            self.root.after(self.state['animation_speed'], self.animate)

    def update_display(self):
        """Update all displays"""
        if (self.simulation.precomputed_frames and
            self.state['current_frame'] < len(self.simulation.precomputed_frames)):

            frame_data = self.simulation.precomputed_frames[self.state['current_frame']]

            # Update metrics
            self.update_metrics(frame_data)

            # Update visualizations based on active tab
            active_tab = self.state['active_tab']
            if active_tab == "Simulation":
                self.update_simulation_visualization(frame_data)
            elif active_tab == "Advanced GPS Tracking":
                self.update_gps_visualization(frame_data)
            elif active_tab == "Analytics":
                self.update_analytics_visualization(frame_data)
            elif active_tab == "Entry System Analysis":
                self.update_entry_analysis_visualization(frame_data)
            elif active_tab == "Complete GPS Loops":
                self.update_loops_visualization()
            elif active_tab == "Algorithm Details":
                self.update_algorithm_details(frame_data)

            # Update frame label
            total_frames = len(self.simulation.precomputed_frames) - 1
            self.frame_label.configure(
                text=f"Frame: {self.state['current_frame']}/{total_frames}"
            )

    def update_metrics(self, frame_data):
        """Update metrics display"""
        stats = frame_data.statistics

        self.enhanced_metric_widgets['active_agents'].configure(text=str(stats['active_count']))
        self.enhanced_metric_widgets['delivering'].configure(text=str(stats['delivered_count']))
        self.enhanced_metric_widgets['completed'].configure(text=str(stats['returned_count']))  # Now shows cumulative

        efficiency = f"{(stats['active_count'] / stats['total_agents'] * 100):.1f}%"
        self.enhanced_metric_widgets['efficiency'].configure(text=efficiency)

        gps_accuracy = f"{(1 - frame_data.gps_accuracy / 10) * 100:.1f}%" if frame_data.gps_accuracy > 0 else "100%"
        self.enhanced_metric_widgets['gps_accuracy'].configure(text=gps_accuracy)

        # Update entry system metrics
        if frame_data.entry_stats['avg_times']:
            avg_entry_time = np.mean(list(frame_data.entry_stats['avg_times'].values()))
            self.enhanced_metric_widgets['entry_time'].configure(text=f"{avg_entry_time:.1f}s")

        # Update continuous activity metric
        continuous_count = stats.get('continuous_activity', 0)
        self.enhanced_metric_widgets['continuous_activity'].configure(text=str(continuous_count))

        # Update algorithm usage
        total_algorithm_usage = sum(frame_data.algorithm_usage.values())
        for algo_name, usage in frame_data.algorithm_usage.items():
            if algo_name in self.algorithm_widgets:
                usage_pct = (usage / total_algorithm_usage * 100) if total_algorithm_usage > 0 else 0
                self.algorithm_widgets[algo_name]['usage_label'].configure(
                    text=f"Usage: {usage} ({usage_pct:.1f}%)"
                )
                
                # Update algorithm performance
                if algo_name in frame_data.algorithm_performance:
                    perf = frame_data.algorithm_performance[algo_name]
                    if perf['avg_confidence'] > 0:
                        self.algorithm_widgets[algo_name]['accuracy_label'].configure(
                            text=f"Avg Confidence: {perf['avg_confidence']:.3f}"
                        )
                        
                        # Performance rating
                        if perf['avg_confidence'] > 0.7:
                            rating = "Excellent"
                            color = self.colors['success']
                        elif perf['avg_confidence'] > 0.5:
                            rating = "Good"
                            color = self.colors['accent_secondary']
                        elif perf['avg_confidence'] > 0.3:
                            rating = "Fair"
                            color = self.colors['warning']
                        else:
                            rating = "Poor"
                            color = self.colors['error']
                            
                        self.algorithm_widgets[algo_name]['performance_label'].configure(
                            text=f"Performance: {rating}",
                            foreground=color
                        )

        # Update entry system widgets
        for system_id, widgets in self.entry_widgets.items():
            avg_time = frame_data.entry_stats['avg_times'].get(system_id, 0)
            count = frame_data.entry_stats['processing'].get(system_id, 0)
            efficiency_ratio = ENTRY_SYSTEMS[system_id]['efficiency_ratio']
            
            widgets['time_label'].configure(text=f"Avg Time: {avg_time:.1f}s")
            widgets['count_label'].configure(text=f"Processing: {count}")
            widgets['efficiency_label'].configure(text=f"Efficiency Ratio: {efficiency_ratio}")

        # Update comparative efficiency
        if 'efficiency_metrics' in frame_data.entry_stats:
            efficiency_metrics = frame_data.entry_stats['efficiency_metrics']
            if 'comparative_efficiency' in efficiency_metrics:
                comp = efficiency_metrics['comparative_efficiency']
                ratio_text = f"QR/Written Ratio: {comp['time_ratio_actual']:.3f} (Target: {comp['time_ratio_target']})"
                self.efficiency_comparison_label.configure(text=ratio_text)

    def update_simulation_visualization(self, frame_data):
        """Update simulation visualization"""
        self.ax_main.clear()
        self.draw_campus_map()

        time_of_day = "Morning" if frame_data.frame < 200 else "Afternoon" if frame_data.frame < 350 else "Evening"
        self.ax_main.set_title(f"Campus Delivery - {time_of_day} (Frame {frame_data.frame})",
                     color=self.colors['text_primary'], fontsize=16, fontweight='bold')

        # Heatmap
        if self.state['show_heatmap']:
            self.ax_main.imshow(frame_data.heatmap_data, extent=(-10, 90, -60, 60),
                      origin='lower', cmap='hot', alpha=0.4, vmin=0, vmax=10,
                      interpolation='bilinear')

        # Traffic particles
        if self.state['show_traffic']:
            traffic_x = [pos[0] for pos in frame_data.traffic_particle_positions]
            traffic_y = [pos[1] for pos in frame_data.traffic_particle_positions]
            self.ax_main.scatter(traffic_x, traffic_y, c='lightblue', s=20, alpha=0.6,
                       zorder=3, edgecolors='none')

        # Agent visualization
        for state in frame_data.agent_states:
            if state.true_position:
                self.draw_agent(state)

        # Add legend for entry systems
        if self.state['show_entry_markers']:
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Written Entry'),
                plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=10, label='QR Code')
            ]
            self.ax_main.legend(handles=legend_elements, loc='upper right', 
                               facecolor=self.colors['bg_card'], 
                               edgecolor=self.colors['border_light'])

        self.canvas_main.draw_idle()

    def draw_agent(self, state):
        """Draw agent visualization with entry system markers"""
        x, y = state.true_position

        phase_config = {
            'outward': {'color': self.colors['accent_info'], 'size': 100, 'symbol': '→'},
            'delivering': {'color': self.colors['accent_warning'], 'size': 120, 'symbol': '📦'},
            'returning': {'color': self.colors['success'], 'size': 100, 'symbol': '←'},
            'completed': {'color': self.colors['text_muted'], 'size': 80, 'symbol': '✓'}
        }

        config = phase_config.get(state.phase, phase_config['outward'])
        
        # Use entry system marker
        marker = state.entry_marker if self.state['show_entry_markers'] else 'o'

        self.ax_main.scatter(x, y, c=config['color'], marker=marker,
                   s=config['size'], alpha=0.9,
                   edgecolors=self.colors['text_primary'],
                   linewidth=2, zorder=5)

        # Trail
        if len(state.trail_positions) > 1:
            trail_x, trail_y = zip(*state.trail_positions)
            self.ax_main.plot(trail_x, trail_y, '-',
                    color=config['color'], alpha=0.7, linewidth=2.5, zorder=4)

        # Label with GPS confidence
        confidence_color = self.colors['success'] if state.gps_confidence > 0.7 else self.colors['warning'] if state.gps_confidence > 0.4 else self.colors['error']
        label = f"{state.agent_id} {config['symbol']}\nConf: {state.gps_confidence:.2f}"

        # Add entry system indicator
        entry_color = ENTRY_SYSTEMS[state.entry_system]['color']
        label += f"\nEntry: {state.entry_system[:2]}"

        self.ax_main.text(x, y - 8, label, fontsize=8, ha='center',
                color=self.colors['text_primary'],
                bbox=dict(boxstyle="round,pad=0.3",
                        facecolor=confidence_color,
                        alpha=0.8,
                        edgecolor=self.colors['border_light']),
                zorder=6)

    def update_gps_visualization(self, frame_data):
        """Update GPS visualization"""
        self.ax_gps.clear()
        self.draw_wifi_access_points()

        self.ax_gps.set_title(f"GPS Tracking - Frame {frame_data.frame}",
                    color=self.colors['text_primary'], fontsize=16, fontweight='bold')

        # GPS heatmap
        if self.state['show_heatmap']:
            self.ax_gps.imshow(frame_data.gps_heatmap_data, extent=(-10, 90, -60, 60),
                     origin='lower', cmap='viridis', alpha=0.4, vmin=0, vmax=8,
                     interpolation='bilinear')

        # Traffic GPS positions
        if self.state['show_traffic']:
            traffic_gps_x = [pos[0] for pos in frame_data.traffic_gps_positions]
            traffic_gps_y = [pos[1] for pos in frame_data.traffic_gps_positions]
            self.ax_gps.scatter(traffic_gps_x, traffic_gps_y, c='lightgreen', s=15, alpha=0.4,
                      zorder=3, edgecolors='none')

        # Agent GPS positions
        for state in frame_data.agent_states:
            if state.estimated_position:
                x, y = state.estimated_position

                # GPS confidence visualization
                if state.gps_confidence > 0.7:
                    color = self.colors['success']
                    size = 80
                elif state.gps_confidence > 0.4:
                    color = self.colors['warning']
                    size = 70
                else:
                    color = self.colors['error']
                    size = 60

                # Use entry system marker for GPS positions too
                marker = state.entry_marker if self.state['show_entry_markers'] else 'o'
                
                self.ax_gps.scatter(x, y, c=color, s=size, marker=marker, alpha=0.9,
                      edgecolors='white', linewidth=2, zorder=6)

                # Confidence circle
                confidence_circle = plt.Circle((x, y), max(5, 15 * (1 - state.gps_confidence)),
                                     color=color, alpha=0.3, fill=True, zorder=5)
                self.ax_gps.add_patch(confidence_circle)

                # Connection to true position
                if state.true_position:
                    true_x, true_y = state.true_position
                    self.ax_gps.plot([true_x, x], [true_y, y], '--',
                           color=color, alpha=0.6, linewidth=1.5, zorder=4)

                    # Error visualization
                    error = math.hypot(true_x - x, true_y - y)
                    self.ax_gps.text(x, y + 10, f"Err: {error:.1f}u", fontsize=8,
                           color=self.colors['text_primary'],
                           bbox=dict(boxstyle="round,pad=0.2",
                                   facecolor=self.colors['bg_card'],
                                   alpha=0.8))

        # Add legend for entry systems in GPS view
        if self.state['show_entry_markers']:
            legend_elements = [
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=8, label='Written Entry'),
                plt.Line2D([0], [0], marker='^', color='w', markerfacecolor='green', markersize=8, label='QR Code')
            ]
            self.ax_gps.legend(handles=legend_elements, loc='upper right', 
                              facecolor=self.colors['bg_card'], 
                              edgecolor=self.colors['border_light'])

        self.canvas_gps.draw_idle()

    def update_analytics_visualization(self, frame_data):
        """Update analytics visualization"""
        for ax in [self.ax_analytics1, self.ax_analytics2, self.ax_analytics3, self.ax_analytics4]:
            ax.clear()
            ax.set_facecolor(self.colors['bg_card'])
            ax.tick_params(colors=self.colors['text_secondary'], labelsize=9)
            for spine in ax.spines.values():
                spine.set_color(self.colors['border_light'])
                spine.set_linewidth(2)

        # Plot 1: Agent activity
        frames_to_plot = min(frame_data.frame + 1, len(self.simulation.active_agents_history))
        x_data = range(frames_to_plot)

        self.ax_analytics1.plot(x_data, self.simulation.active_agents_history[:frames_to_plot],
                       color=self.colors['accent_info'], linestyle='-', label='Active', linewidth=3, alpha=0.8)
        self.ax_analytics1.plot(x_data, self.simulation.delivered_agents_history[:frames_to_plot],
                       color=self.colors['accent_warning'], linestyle='-', label='Delivering', linewidth=3, alpha=0.8)
        self.ax_analytics1.plot(x_data, self.simulation.returned_agents_history[:frames_to_plot],
                       color=self.colors['success'], linestyle='-', label='Total Completed', linewidth=3, alpha=0.8)
        self.ax_analytics1.set_title("Agent Activity", color=self.colors['text_primary'], fontsize=12, fontweight='bold')
        self.ax_analytics1.legend(fontsize=8, facecolor=self.colors['bg_card'])
        self.ax_analytics1.grid(True, alpha=0.3)

        # Plot 2: Congestion levels
        self.ax_analytics2.plot(x_data, self.simulation.congestion_history[:frames_to_plot],
                       color=self.colors['animation_secondary'], linewidth=3, alpha=0.8)
        self.ax_analytics2.fill_between(x_data, self.simulation.congestion_history[:frames_to_plot],
                                   alpha=0.3, color=self.colors['animation_secondary'])
        self.ax_analytics2.set_title("Campus Congestion", color=self.colors['text_primary'], fontsize=12, fontweight='bold')
        self.ax_analytics2.grid(True, alpha=0.3)

        # Plot 3: GPS accuracy
        self.ax_analytics3.plot(x_data, self.simulation.gps_accuracy_history[:frames_to_plot],
                       color=self.colors['accent_primary'], linewidth=3, alpha=0.8)
        self.ax_analytics3.fill_between(x_data, self.simulation.gps_accuracy_history[:frames_to_plot],
                                   alpha=0.3, color=self.colors['accent_primary'])
        self.ax_analytics3.set_title("GPS Accuracy", color=self.colors['text_primary'], fontsize=12, fontweight='bold')
        self.ax_analytics3.set_ylabel("Average Error (units)", color=self.colors['text_secondary'])
        self.ax_analytics3.grid(True, alpha=0.3)

        # Plot 4: Algorithm usage - FIXED VERSION
        algorithms_to_plot = ['Trilateration', 'Fingerprinting', 'Centroid', 'Kalman Filter', 'Fusion']
        colors = [self.colors['accent_info'], self.colors['accent_warning'], self.colors['accent_primary'], 
                 self.colors['success'], self.colors['animation_primary']]
        
        for i, algo in enumerate(algorithms_to_plot):
            if algo in self.simulation.algorithm_usage_history:
                # Ensure we don't plot beyond available data
                data_to_plot = self.simulation.algorithm_usage_history[algo][:frames_to_plot]
                if len(data_to_plot) > 0:
                    self.ax_analytics4.plot(x_data[:len(data_to_plot)], data_to_plot,
                              label=algo, linewidth=2, alpha=0.7, color=colors[i])

        self.ax_analytics4.set_title("Algorithm Usage", color=self.colors['text_primary'], fontsize=12, fontweight='bold')
        self.ax_analytics4.legend(fontsize=7, facecolor=self.colors['bg_card'])
        self.ax_analytics4.grid(True, alpha=0.3)

        self.fig_analytics.tight_layout()
        self.canvas_analytics.draw_idle()

    def update_entry_analysis_visualization(self, frame_data):
        """Update entry system analysis visualization"""
        for ax in [self.ax_entry1, self.ax_entry2, self.ax_entry3, self.ax_entry4]:
            ax.clear()
            ax.set_facecolor(self.colors['bg_card'])
            ax.tick_params(colors=self.colors['text_secondary'], labelsize=9)
            for spine in ax.spines.values():
                spine.set_color(self.colors['border_light'])
                spine.set_linewidth(2)

        frames_to_plot = min(frame_data.frame + 1, MAX_STEPS)
        x_data = range(frames_to_plot)

        # Plot 1: Entry time comparison
        written_times = self.simulation.entry_time_history['written'][:frames_to_plot]
        qr_times = self.simulation.entry_time_history['qr_code'][:frames_to_plot]
        
        # Filter out zeros for proper plotting
        written_nonzero = written_times[written_times > 0]
        qr_nonzero = qr_times[qr_times > 0]
        
        if len(written_nonzero) > 0:
            self.ax_entry1.plot(range(len(written_nonzero)), written_nonzero, 
                               color=ENTRY_SYSTEMS['written']['color'], 
                               label='Written Entry', linewidth=3, alpha=0.8)
        
        if len(qr_nonzero) > 0:
            self.ax_entry1.plot(range(len(qr_nonzero)), qr_nonzero, 
                               color=ENTRY_SYSTEMS['qr_code']['color'], 
                               label='QR Code', linewidth=3, alpha=0.8)
        
        self.ax_entry1.set_title("Entry Time Comparison", color=self.colors['text_primary'], fontsize=12, fontweight='bold')
        self.ax_entry1.legend(fontsize=8, facecolor=self.colors['bg_card'])
        self.ax_entry1.grid(True, alpha=0.3)

        # Plot 2: Congestion impact
        congestion = self.simulation.congestion_history[:frames_to_plot]
        self.ax_entry2.plot(x_data, congestion, color=self.colors['accent_warning'], linewidth=3, alpha=0.8)
        self.ax_entry2.set_title("Gate Congestion Impact", color=self.colors['text_primary'], fontsize=12, fontweight='bold')
        self.ax_entry2.grid(True, alpha=0.3)

        # Plot 3: Time efficiency
        efficiency = self.simulation.efficiency_history[:frames_to_plot]
        efficiency_nonzero = efficiency[efficiency > 0]
        if len(efficiency_nonzero) > 0:
            self.ax_entry3.plot(range(len(efficiency_nonzero)), efficiency_nonzero, 
                               color=self.colors['animation_primary'], linewidth=3, alpha=0.8)
            self.ax_entry3.fill_between(range(len(efficiency_nonzero)), efficiency_nonzero, 
                                      alpha=0.3, color=self.colors['animation_primary'])
        self.ax_entry3.set_title("Delivery Efficiency", color=self.colors['text_primary'], fontsize=12, fontweight='bold')
        self.ax_entry3.set_ylabel("Efficiency %", color=self.colors['text_secondary'])
        self.ax_entry3.grid(True, alpha=0.3)

        # Plot 4: System usage
        written_count = np.array([self.simulation.entry_time_history['written'][i] > 0 for i in range(frames_to_plot)]).cumsum()
        qr_count = np.array([self.simulation.entry_time_history['qr_code'][i] > 0 for i in range(frames_to_plot)]).cumsum()
        
        self.ax_entry4.plot(x_data, written_count, color=ENTRY_SYSTEMS['written']['color'], 
                           label='Written', linewidth=3, alpha=0.8)
        self.ax_entry4.plot(x_data, qr_count, color=ENTRY_SYSTEMS['qr_code']['color'], 
                           label='QR Code', linewidth=3, alpha=0.8)
        self.ax_entry4.set_title("Entry System Usage", color=self.colors['text_primary'], fontsize=12, fontweight='bold')
        self.ax_entry4.legend(fontsize=8, facecolor=self.colors['bg_card'])
        self.ax_entry4.grid(True, alpha=0.3)

        self.fig_entry.tight_layout()
        self.canvas_entry.draw_idle()

    def update_loops_visualization(self):
        """Update GPS loops visualization - FIXED VERSION"""
        self.ax_loops.clear()
        self.draw_campus_map_for_loops()

        # Plot complete GPS journeys
        completed_agents = [a for a in self.simulation.agents if a.get('phase') == 'completed']

        for agent in self.simulation.agents:  # Plot for all agents, not just completed ones
            tracker = agent['gps_tracker']
            if hasattr(tracker, 'complete_journey_history') and tracker.complete_journey_history:
                journey_data = list(tracker.complete_journey_history)

                if len(journey_data) > 1:
                    positions = [point['estimated_position'] for point in journey_data]
                    phases = [point.get('phase', 'unknown') for point in journey_data]

                    if len(positions) > 5:
                        gps_x, gps_y = zip(*positions)

                        # Phase-based coloring
                        for i in range(len(gps_x) - 1):
                            phase = phases[i]
                            if phase == 'outward':
                                color = 'blue'
                                alpha = 0.8
                                linewidth = 3
                            elif phase == 'delivering':
                                color = 'orange'
                                alpha = 0.6
                                linewidth = 4
                            elif phase == 'returning':
                                color = 'green'
                                alpha = 0.8
                                linewidth = 3
                            else:
                                color = agent['color']
                                alpha = 0.6
                                linewidth = 2
                            
                            # Plot the journey segment
                            self.ax_loops.plot([gps_x[i], gps_x[i+1]], [gps_y[i], gps_y[i+1]],
                                         '-', color=color, alpha=alpha, linewidth=linewidth, zorder=10)

        # Enhanced statistics
        completed_count = len(completed_agents)
        total_journeys = sum(a.get('total_completed_journeys', 0) for a in self.simulation.agents)
        stats_text = f"Analysis:\nActive Agents: {len([a for a in self.simulation.agents if a['phase'] != 'completed'])}\nTotal Journeys Completed: {total_journeys}"
        self.ax_loops.text(0.02, 0.98, stats_text, transform=self.ax_loops.transAxes,
                  fontsize=11, verticalalignment='top', color=self.colors['text_primary'],
                  bbox=dict(boxstyle="round,pad=0.6", facecolor=self.colors['bg_card'], alpha=0.9))

        self.canvas_loops.draw_idle()

    def update_algorithm_details(self, frame_data):
        """Update algorithm details text"""
        if not self.simulation.precomputed_frames:
            return

        # Get current frame data for algorithm usage
        current_frame = min(self.state['current_frame'], len(self.simulation.precomputed_frames) - 1)

        algorithm_text = "GPS ALGORITHM PERFORMANCE ANALYSIS\n"
        algorithm_text += "=" * 50 + "\n\n"

        # Current frame usage
        algorithm_text += f"FRAME {current_frame} - ALGORITHM USAGE:\n"
        algorithm_text += "-" * 30 + "\n"
        
        total_usage = sum(frame_data.algorithm_usage.values())
        for algo, count in frame_data.algorithm_usage.items():
            percentage = (count / total_usage * 100) if total_usage > 0 else 0
            algorithm_text += f"{algo:<15}: {count:>3} uses ({percentage:>5.1f}%)\n"

        algorithm_text += "\nCUMULATIVE PERFORMANCE:\n"
        algorithm_text += "-" * 25 + "\n"

        # Cumulative statistics
        for algo in self.simulation.algorithm_usage_history:
            total_uses = np.sum(self.simulation.algorithm_usage_history[algo][:current_frame + 1])
            algorithm_text += f"{algo:<15}: {total_uses:>5} total uses\n"

        algorithm_text += "\nALGORITHM TECHNICAL DETAILS:\n"
        algorithm_text += "-" * 30 + "\n"

        algorithm_details = {
            "Trilateration": "• Uses distance measurements from 3+ APs\n• Mathematical optimization for position\n• High precision in good conditions\n• Requires strong signal from multiple APs",
            "Fingerprinting": "• Pattern matching against database\n• High accuracy in known areas\n• Database-dependent performance\n• Robust to signal variations",
            "Centroid": "• Signal strength weighted average\n• Good for dense AP coverage\n• Simple and computationally efficient\n• Less precise than other methods",
            "Kalman Filter": "• Noise reduction and prediction\n• Excellent for tracking moving agents\n• Adaptive to signal quality changes\n• Provides smooth position estimates",
            "Fusion": "• Combines multiple methods\n• Adaptive algorithm selection\n• Most robust overall performance\n• Computationally intensive"
        }

        for algo, details in algorithm_details.items():
            algorithm_text += f"\n{algo}:\n{details}\n"
            
        # Enhanced performance metrics
        algorithm_text += "\nPERFORMANCE METRICS:\n"
        algorithm_text += "-" * 20 + "\n"
        
        for algo_name, perf in frame_data.algorithm_performance.items():
            if perf['count'] > 0:
                algorithm_text += f"{algo_name}: Avg Confidence = {perf['avg_confidence']:.3f}\n"

        # Update the text widget
        self.algorithm_text.config(state=tk.NORMAL)
        self.algorithm_text.delete(1.0, tk.END)
        self.algorithm_text.insert(1.0, algorithm_text)
        self.algorithm_text.config(state=tk.DISABLED)

    def generate_post_simulation_analysis(self):
        """Generate comprehensive post-simulation analysis"""
        if not self.simulation.precomputed_frames:
            return

        # Generate analytics
        self.post_simulation_analytics = self.simulation.generate_post_simulation_analytics()

        # Update summary tab
        summary_text = "POST-SIMULATION ANALYSIS SUMMARY\n"
        summary_text += "=" * 40 + "\n\n"

        # Completion statistics
        completion = self.post_simulation_analytics['completion_stats']
        summary_text += f"COMPLETION RATE: {completion['completion_rate']:.1f}%\n"
        summary_text += f"Completed Agents: {completion['completed_agents']}/{completion['total_agents']}\n"
        summary_text += f"Total Journeys Completed: {completion['total_journeys']}\n"
        summary_text += f"Average Journeys per Agent: {completion['avg_journeys_per_agent']:.2f}\n\n"

        # Time efficiency
        if 'time_efficiency' in self.post_simulation_analytics:
            efficiency = self.post_simulation_analytics['time_efficiency']
            summary_text += f"TIME EFFICIENCY: {efficiency['avg_efficiency_gain']:.1f}%\n"
            summary_text += f"Total Time Saved: {efficiency['total_time_saved']:.1f} steps\n"
            summary_text += f"Optimized vs Conventional: {efficiency['total_optimized_time']:.1f} vs {efficiency['total_conventional_time']:.1f} steps\n\n"

        # GPS performance
        gps_perf = self.post_simulation_analytics['gps_performance']
        summary_text += f"GPS PERFORMANCE:\n"
        summary_text += f"Average Error: {gps_perf['avg_error']:.2f} units\n"
        summary_text += f"Maximum Error: {gps_perf['max_error']:.2f} units\n"
        summary_text += f"Minimum Error: {gps_perf['min_error']:.2f} units\n\n"

        # Entry system performance
        entry_systems = self.post_simulation_analytics['entry_systems']
        summary_text += "ENTRY SYSTEM PERFORMANCE:\n"
        for system_id, stats in entry_systems.items():
            system_name = ENTRY_SYSTEMS[system_id]['name']
            summary_text += f"{system_name}: {stats['avg_time']:.1f}s avg ({stats['count']} agents)\n"

        # Entry efficiency analysis
        if 'entry_efficiency' in self.post_simulation_analytics:
            entry_eff = self.post_simulation_analytics['entry_efficiency']
            summary_text += f"\nENTRY EFFICIENCY ANALYSIS:\n"
            summary_text += f"Actual QR/Written Ratio: {entry_eff['actual_ratio']:.3f}\n"
            summary_text += f"Target Ratio: {entry_eff['target_ratio']}\n"
            summary_text += f"Efficiency Gap: {entry_eff['efficiency_gap']:.3f}\n"
            summary_text += f"Achievement Percentage: {entry_eff['achievement_percentage']:.1f}%\n"

        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete(1.0, tk.END)
        self.summary_text.insert(1.0, summary_text)
        self.summary_text.config(state=tk.DISABLED)

        # Update details tab
        details_text = "DETAILED POST-SIMULATION ANALYTICS\n"
        details_text += "=" * 45 + "\n\n"

        # Algorithm usage breakdown
        details_text += "ALGORITHM USAGE BREAKDOWN:\n"
        details_text += "-" * 25 + "\n"
        algo_totals = self.post_simulation_analytics['algorithm_usage']
        total_algo_uses = sum(algo_totals.values())
        
        for algo, total in algo_totals.items():
            percentage = (total / total_algo_uses * 100) if total_algo_uses > 0 else 0
            details_text += f"{algo:<15}: {total:>6} uses ({percentage:>6.2f}%)\n"

        details_text += "\nALGORITHM PERFORMANCE:\n"
        details_text += "-" * 20 + "\n"
        
        if 'algorithm_performance' in self.post_simulation_analytics:
            algo_perf = self.post_simulation_analytics['algorithm_performance']
            for algo, perf in algo_perf.items():
                details_text += f"{algo}: Avg Confidence = {perf['avg_confidence']:.3f}\n"

        details_text += "\nPERFORMANCE METRICS:\n"
        details_text += "-" * 20 + "\n"

        # Additional metrics
        avg_congestion = np.mean(self.simulation.congestion_history)
        max_congestion = np.max(self.simulation.congestion_history)
        avg_gps_accuracy = np.mean(self.simulation.gps_accuracy_history)
        
        details_text += f"Average Campus Congestion: {avg_congestion:.2f}\n"
        details_text += f"Peak Campus Congestion: {max_congestion:.2f}\n"
        details_text += f"Average GPS Error: {avg_gps_accuracy:.2f} units\n"

        # Entry system efficiency gain
        if (entry_systems['written']['count'] > 0 and entry_systems['qr_code']['count'] > 0 and
            'entry_efficiency' in self.post_simulation_analytics):
            entry_eff = self.post_simulation_analytics['entry_efficiency']
            time_saved_per_agent = entry_systems['written']['avg_time'] - entry_systems['qr_code']['avg_time']
            total_time_saved = time_saved_per_agent * entry_systems['qr_code']['count']
            details_text += f"\nQR Code Time Savings: {total_time_saved:.1f} steps total\n"
            details_text += f"Per-agent savings: {time_saved_per_agent:.1f} steps\n"
            details_text += f"Efficiency Achievement: {entry_eff['achievement_percentage']:.1f}% of target\n"

        self.details_text.config(state=tk.NORMAL)
        self.details_text.delete(1.0, tk.END)
        self.details_text.insert(1.0, details_text)
        self.details_text.config(state=tk.DISABLED)

    def generate_gps_loop_visualization(self):
        """Generate enhanced GPS loop visualization using the provided code structure"""
        if not self.simulation.precomputed_frames:
            return

        # Create a new window for GPS loop visualization
        gps_window = tk.Toplevel(self.root)
        gps_window.title("Complete GPS Loop Visualization")
        gps_window.geometry("1200x800")

        # Create figure for GPS loops
        fig_loops = Figure(figsize=(12, 8), facecolor=self.colors['bg_card'], dpi=100)
        ax_loops = fig_loops.add_subplot(111)
        ax_loops.set_facecolor(self.colors['bg_card'])

        ax_loops.set_xlim(-10, 90)
        ax_loops.set_ylim(-60, 60)
        ax_loops.set_title("Complete GPS Tracking Loops - All Agents", 
                          color=self.colors['text_primary'], fontsize=16, fontweight='bold')

        # Draw campus map
        for n, p in NODES.items():
            color = '#28a745' if 'Gate' in n else '#007bff' if 'Hub' in n else '#dc3545'
            ax_loops.scatter(p[0], p[1], c=color, s=150, alpha=0.9, zorder=5, 
                           edgecolors='white', linewidth=2)
            ax_loops.text(p[0] + 2, p[1] + 2, n, color='black', fontsize=10, zorder=5,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9))

        # Draw edges
        for u, v in CAMPUS_GRAPH.edges():
            x1, y1 = NODES[u]
            x2, y2 = NODES[v]
            ax_loops.plot([x1, x2], [y1, y2], 'gray', alpha=0.4, linewidth=1.5, zorder=1)

        # Plot GPS journeys for all agents
        completed_count = 0
        total_journeys = 0
        for agent in self.simulation.agents:
            tracker = agent['gps_tracker']
            if hasattr(tracker, 'complete_journey_history') and tracker.complete_journey_history:
                journey_data = list(tracker.complete_journey_history)

                if len(journey_data) > 1:
                    positions = [point['estimated_position'] for point in journey_data]
                    phases = [point.get('phase', 'unknown') for point in journey_data]

                    if len(positions) > 5:
                        gps_x, gps_y = zip(*positions)

                        # Plot with phase-based coloring
                        for i in range(len(gps_x) - 1):
                            phase = phases[i]
                            if phase == 'outward':
                                color = 'blue'
                                alpha = 0.7
                            elif phase == 'delivering':
                                color = 'orange'
                                alpha = 0.5
                            elif phase == 'returning':
                                color = 'green'
                                alpha = 0.7
                            else:
                                color = agent['color']
                                alpha = 0.6

                            ax_loops.plot([gps_x[i], gps_x[i+1]], [gps_y[i], gps_y[i+1]],
                                        '-', color=color, alpha=alpha, linewidth=2, zorder=10)

                        # Removed start and end markers as requested

                        if agent.get('phase') == 'completed':
                            completed_count += 1
                            total_journeys += agent.get('total_completed_journeys', 0)

        # Add statistics
        stats_text = f"Active Agents: {len([a for a in self.simulation.agents if a['phase'] != 'completed'])}\nTotal Journeys: {total_journeys}"
        ax_loops.text(0.02, 0.98, stats_text, transform=ax_loops.transAxes, fontsize=12,
                     verticalalignment='top', color=self.colors['text_primary'],
                     bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9))

        # Add legend
        legend_elements = [
            plt.Line2D([0], [0], color='blue', linewidth=2, label='Outward Journey'),
            plt.Line2D([0], [0], color='orange', linewidth=2, label='Delivery Phase'),
            plt.Line2D([0], [0], color='green', linewidth=2, label='Return Journey')
        ]
        ax_loops.legend(handles=legend_elements, loc='upper right', fontsize=10)

        # Embed in tkinter window
        canvas = FigureCanvasTkAgg(fig_loops, gps_window)
        canvas.draw()
        canvas.get_tk_widget().pack(fill='both', expand=True)

        # Add toolbar
        toolbar_frame = ttk.Frame(gps_window)
        toolbar_frame.pack(fill='x')
        ttk.Button(toolbar_frame, text="Save Image", 
                  command=lambda: fig_loops.savefig('gps_loops_visualization.png', dpi=300, bbox_inches='tight')).pack(side='left', padx=5)

    def monitor_precomputation(self):
        """Monitor precomputation progress"""
        try:
            while True:
                msg_type, data = self.simulation.precompute_queue.get_nowait()

                if msg_type == "progress":
                    self.status_label.configure(text=data)

                    if "frame" in data:
                        try:
                            frame_num = int(data.split(' ')[2].split('/')[0])
                            progress_pct = (frame_num / self.simulation.max_steps) * 100
                            self.progress_bar['value'] = progress_pct
                            self.progress_label.configure(text=f"{progress_pct:.1f}%")

                            if progress_pct < 33:
                                color = self.colors['accent_warning']
                            elif progress_pct < 66:
                                color = self.colors['accent_info']
                            else:
                                color = self.colors['success']

                            self.status_dot.configure(bg=color)
                            self.status_dot.delete("all")
                            self.status_dot.create_oval(4, 4, 12, 12, fill=color, outline='')

                        except:
                            pass

                elif msg_type == "complete":
                    self.status_label.configure(text="Ready for Visualization!")
                    self.progress_bar['value'] = 100
                    self.progress_label.configure(text="100%")
                    self.status_dot.configure(bg=self.colors['success'])
                    self.status_dot.delete("all")
                    self.status_dot.create_oval(4, 4, 12, 12, fill=self.colors['success'], outline='')

                    self.frame_slider.configure(to=len(self.simulation.precomputed_frames)-1)
                    self.frame_label.configure(
                        text=f"Frame: 0/{len(self.simulation.precomputed_frames)-1}"
                    )
                    self.play_btn.state(['!disabled'])

                    logger.info("Precomputation monitoring completed")
                    break

        except:
            pass

        if self.simulation.is_precomputing:
            self.root.after(100, self.monitor_precomputation)

    def setup_state_management(self):
        """Setup state management"""
        pass

# ==================== MAIN APPLICATION ====================
def main():
    """Enhanced main application entry point"""
    print("Starting Enhanced Smart Campus Delivery Dashboard...")
    print("Features:")
    print("  - Enhanced GPS tracking with ALL algorithms integrated")
    print("  - Balanced entry system variants (50% QR, 50% Written)")
    print("  - Continuous agent activity throughout simulation")
    print("  - Different markers for QR (triangle) and Written (circle) agents")
    print("  - Cumulative completion counter (doesn't reset)")
    print("  - Comprehensive analytics and performance metrics")
    print("  - Scrollable algorithm performance displays")
    print("  - Post-simulation GPS loop visualization")
    print("  - Real-time efficiency tracking")
    print("  - 25 Agents (Balanced), 250 Particles, 800 Steps for extended simulation")
    print("Precomputing simulation for smooth playback...")

    try:
        gc.enable()
        gc.set_threshold(700, 10, 10)

        root = tk.Tk()
        app = EnhancedCampusGUI(root)
        root.mainloop()
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"Application error: {e}")


if __name__ == "__main__":
    main()