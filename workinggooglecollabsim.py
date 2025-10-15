# Step 0: Enhanced Imports
import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from IPython.display import HTML, display
import networkx as nx
import time
from collections import deque
from scipy.optimize import minimize
from sklearn.neighbors import NearestNeighbors

random.seed(42)
np.random.seed(42)

# Increase animation size limit for enhanced simulation
plt.rcParams['animation.embed_limit'] = 400  # Increased to 400MB

# Step 1: Enhanced parameters for swarm simulation
num_agents = 15
max_steps = 500  # Full day simulation
speed_variation = {"petrol": 5.0, "e-bike": 7.0, "cycle": 4.0}
smoothness_factor = 3

# Campus layout
nodes = {
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
    "FoodCourt": (40, 0)
}

# Step 2: Enhanced campus graph with better traffic flow
G = nx.Graph()
for n, p in nodes.items():
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
    ("Hub1", "HostelB", 1.5), ("Hub2", "HostelB", 1.5)
]

for a, b, weight_factor in edges_with_weights:
    xa, ya = nodes[a]
    xb, yb = nodes[b]
    base_distance = math.hypot(xa - xb, ya - yb)
    G.add_edge(a, b, weight=base_distance * weight_factor, base_weight=base_distance * weight_factor)

congestion_edges = [("Hub1", "Acad1"), ("Hub2", "Acad2"), ("FoodCourt", "Acad1"), ("FoodCourt", "Acad2")]

# ENHANCED: WiFi Access Points with better coverage at gates
wifi_access_points = {
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
    "FoodCourt": {"pos": (40, 0), "range": 26, "signal_variance": 1.4}
}

# FIXED: Enhanced GPS Tracking System with Path-Constrained Estimation
class EnhancedGPSTracker:
    def __init__(self, agent_id=None):
        self.position_history = deque(maxlen=100)  # Increased for complete journey tracking
        self.estimated_positions = deque(maxlen=100)
        self.kalman_state = None
        self.kalman_covariance = None
        self.filter_initialized = False
        self.fingerprint_database = self.create_enhanced_fingerprint_database()
        self.confidence_history = deque(maxlen=20)
        self.method_used = []
        self.algorithm_visualization_data = []
        self.agent_id = agent_id
        self.last_valid_position = None
        self.consecutive_bad_estimates = 0
        self.path_constraint_active = False
        self.complete_journey_history = []  # NEW: Store complete journey

    def create_enhanced_fingerprint_database(self):
        """Create high-resolution WiFi fingerprint database"""
        fingerprints = []
        for x in range(0, 91, 3):
            for y in range(-50, 51, 3):
                if any(math.hypot(x - ap["pos"][0], y - ap["pos"][1]) <= ap["range"] * 1.2
                       for ap in wifi_access_points.values()):
                    fingerprint = {"position": (x, y)}
                    for ap_name, ap_info in wifi_access_points.items():
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
        for ap_name, ap_info in wifi_access_points.items():
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
                    ap_pos = wifi_access_points[ap_name]["pos"]
                    actual_distance = math.hypot(x - ap_pos[0], y - ap_pos[1])

                    error = (est_distance - actual_distance) ** 2
                    weight = confidence

                    total_error += error * weight
                    total_weight += weight

                return total_error / total_weight if total_weight > 0 else total_error

            strong_aps = sorted(valid_aps, key=lambda x: x[2])[:4]
            ap_positions = [wifi_access_points[ap_name]["pos"] for ap_name, _, _, _ in strong_aps]

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
            feature_names = list(wifi_access_points.keys())
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
        """Enhanced Kalman filter with adaptive noise"""
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

        return self.kalman_state[0:2], "Kalman (Updated)"

    def estimate_enhanced_position(self, true_position, frame):
        """Enhanced position estimation with path constraints"""
        signals = self.get_enhanced_wifi_signals(true_position)

        estimates = []
        confidences = []
        algorithm_data = []
        methods_used = []

        # Method 1: Enhanced Trilateration
        tri_pos, tri_confidence, tri_algo = self.enhanced_trilateration(signals)
        if tri_pos is not None and tri_confidence > 0.2:
            estimates.append(tri_pos)
            confidences.append(tri_confidence)
            algorithm_data.append(tri_algo)
            methods_used.append("Trilateration")

        # Method 2: Enhanced Fingerprinting
        fp_pos, fp_confidence, fp_algo = self.enhanced_fingerprinting(signals)
        if fp_pos is not None and fp_confidence > 0.2:
            estimates.append(fp_pos)
            confidences.append(fp_confidence)
            algorithm_data.append(fp_algo)
            methods_used.append("Fingerprinting")

        # Method 3: AP Centroid with signal strength weighting
        strong_aps = [ap_name for ap_name, rssi in signals.items() if rssi > -75]
        if len(strong_aps) >= 2:
            ap_positions = [wifi_access_points[ap_name]["pos"] for ap_name in strong_aps]
            weights = [max(0.1, (signals[ap_name] + 75) / 25) for ap_name in strong_aps]
            total_weight = sum(weights)

            if total_weight > 0:
                centroid_x = sum(x * w for (x, y), w in zip(ap_positions, weights)) / total_weight
                centroid_y = sum(y * w for (x, y), w in zip(ap_positions, weights)) / total_weight

                centroid_confidence = min(0.8, len(strong_aps) / 8 + 0.2)
                estimates.append((centroid_x, centroid_y))
                confidences.append(centroid_confidence)
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

        # Apply enhanced Kalman filtering
        kalman_info = "Kalman (Not Applied)"
        if self.filter_initialized or frame > 3:
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

        # NEW: Store complete journey data
        journey_point = {
            'frame': frame,
            'true_position': true_position,
            'estimated_position': estimated_tuple,
            'confidence': confidence,
            'methods_used': methods_used.copy(),
            'phase': 'unknown'  # Will be set by agent
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

# ENHANCED: IIT Bombay Class Schedule with Swarm Behavior
class_schedule = {
    "morning_peak": {
        "start": 50, "end": 150,
        "swarm_routes": [
            {"from": ["HostelA", "HostelB", "HostelC"], "to": ["Acad1", "Acad2"], "intensity": 4.0},
            {"from": ["MainGate", "YPGate"], "to": ["Acad1", "Acad2"], "intensity": 3.0}
        ],
        "areas": ["Acad1", "Acad2", "Hub1", "Hub2"],
        "intensity": 3.5
    },
    "lunch_peak": {
        "start": 200, "end": 280,
        "swarm_routes": [
            {"from": ["Acad1", "Acad2"], "to": ["FoodCourt"], "intensity": 4.5},
            {"from": ["HostelA", "HostelB", "HostelC"], "to": ["FoodCourt"], "intensity": 4.0}
        ],
        "areas": ["FoodCourt", "Hub1", "Hub2"],
        "intensity": 4.0
    },
    "evening_peak": {
        "start": 350, "end": 450,
        "swarm_routes": [
            {"from": ["Acad1", "Acad2"], "to": ["HostelA", "HostelB", "HostelC"], "intensity": 4.0},
            {"from": ["FoodCourt"], "to": ["HostelA", "HostelB", "HostelC"], "intensity": 3.5}
        ],
        "areas": ["HostelA", "HostelB", "HostelC", "MainGate", "YPGate"],
        "intensity": 3.0
    }
}

# ENHANCED: Swarm Traffic Particle with intelligent routing
class SwarmParticle:
    def __init__(self, particle_type="student"):
        self.type = particle_type
        self.speed = random.uniform(0.2, 0.4)
        self.color = self.get_particle_color()
        self.size = random.uniform(1.5, 3.5)
        self.alpha = random.uniform(0.5, 0.9)

        self.current_location = random.choice(list(nodes.keys()))
        self.target_location = None
        self.current_path = []
        self.path_index = 0
        self.progress = 0
        self.swarm_behavior = True
        self.last_route_change = 0
        self.route_cooldown = random.randint(30, 80)

        # Add GPS tracking
        self.gps_tracker = EnhancedGPSTracker()
        self.estimated_position = None
        self.current_signals = {}
        self.gps_confidence = 0.0
        self.gps_methods_used = []
        self.gps_algorithm_data = []

    def get_particle_color(self):
        """Different colors for different types of campus traffic"""
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
        for period, info in class_schedule.items():
            if info["start"] <= frame <= info["end"]:
                current_period = period
                break

        if current_period and random.random() < 0.7:
            swarm_routes = class_schedule[current_period]["swarm_routes"]
            matching_routes = []
            for route in swarm_routes:
                if self.current_location in route["from"]:
                    matching_routes.append(route)

            if matching_routes:
                chosen_route = random.choice(matching_routes)
                self.target_location = random.choice(chosen_route["to"])
                try:
                    self.current_path = nx.shortest_path(G, self.current_location, self.target_location)
                    self.path_index = 0
                    self.progress = 0
                    self.last_route_change = frame
                    self.route_cooldown = random.randint(40, 100)
                except:
                    pass
        else:
            if random.random() < 0.1:
                available_nodes = list(nodes.keys())
                self.target_location = random.choice(available_nodes)
                if self.target_location != self.current_location:
                    try:
                        self.current_path = nx.shortest_path(G, self.current_location, self.target_location)
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
                    self.current_path = nx.shortest_path(G, self.current_location, self.target_location)
                    self.path_index = 0
                    self.progress = 0
                except:
                    self.current_path = []
            return

        # Move along current path segment
        current_node = self.current_path[self.path_index]
        next_node = self.current_path[self.path_index + 1]

        current_pos = nodes[current_node]
        next_pos = nodes[next_node]
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
            return nodes.get(self.current_location, (0, 0))

        current_node = self.current_path[self.path_index]
        next_node = self.current_path[self.path_index + 1]

        current_pos = nodes[current_node]
        next_pos = nodes[next_node]

        x = current_pos[0] + (next_pos[0] - current_pos[0]) * self.progress
        y = current_pos[1] + (next_pos[1] - current_pos[1]) * self.progress

        return (x, y)

# ENHANCED: Create dense swarms of traffic particles with GPS tracking
traffic_particles = []
particle_types = ["student", "student", "student", "faculty", "service", "visitor"]

for _ in range(200):
    particle_type = random.choice(particle_types)
    traffic_particles.append(SwarmParticle(particle_type))

# ENHANCED: Background congestion with swarm intensity
def get_background_congestion(frame):
    """Calculate background congestion based on swarm behavior and class schedule"""
    background_congestion = {}

    # Base congestion (always present)
    for node in nodes:
        background_congestion[node] = 0.3 + 0.2 * random.random()

    # Enhanced class schedule peaks with swarm behavior
    for period, info in class_schedule.items():
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

    # Time-of-day variations with swarm patterns
    if 50 <= frame <= 150:
        for node in ["MainGate", "YPGate", "Hub1", "Hub2", "Acad1", "Acad2"]:
            background_congestion[node] += 2.0
    elif 200 <= frame <= 280:
        for node in ["FoodCourt", "Hub1", "Hub2"]:
            background_congestion[node] += 2.5
    elif 350 <= frame <= 450:
        for node in ["HostelA", "HostelB", "HostelC", "MainGate", "YPGate"]:
            background_congestion[node] += 2.0

    return background_congestion

# Step 3: Robust path optimization with enhanced congestion consideration
def find_optimal_path_with_congestion(start, end, current_congestion, background_congestion):
    """Find optimal path using A* algorithm with enhanced congestion consideration"""
    try:
        def heuristic(u, v):
            pos_u = nodes[u]
            pos_v = nodes[v]
            return math.hypot(pos_u[0] - pos_v[0], pos_u[1] - pos_v[1])

        temp_G = G.copy()
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
            return nx.shortest_path(G, start, end, weight='weight')
        except:
            if start != end:
                return [start, end]
            return [start]

# Step 4: Fixed agent creation with proper timing calculations
def create_smooth_path(path_nodes, speed):
    """Create smooth path coordinates with guaranteed completion"""
    if not path_nodes or len(path_nodes) < 2:
        return [nodes[path_nodes[0]]] if path_nodes else []

    coords = []
    total_distance = 0

    for j in range(len(path_nodes) - 1):
        start_node = path_nodes[j]
        end_node = path_nodes[j + 1]
        if start_node in nodes and end_node in nodes:
            start = nodes[start_node]
            end = nodes[end_node]
            total_distance += math.hypot(end[0] - start[0], end[1] - start[1])

    total_steps = max(5, int(total_distance * smoothness_factor / speed))

    for j in range(len(path_nodes) - 1):
        start_node = path_nodes[j]
        end_node = path_nodes[j + 1]

        if start_node not in nodes or end_node not in nodes:
            continue

        start = nodes[start_node]
        end = nodes[end_node]
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

    if path_nodes and path_nodes[-1] in nodes:
        coords.append(nodes[path_nodes[-1]])

    return coords

# FIXED: Enhanced agent creation with proper loop closure
def create_optimized_agents(num_agents):
    agents = []
    gates = ["MainGate", "YPGate"]
    destinations = ["HostelA", "HostelB", "HostelC", "Acad1", "Acad2"]
    vendors = ["Zomato", "Swiggy", "Amazon", "Dunzo"]
    vehicle_types = ["petrol", "e-bike", "cycle"]

    phase_colors = {'outward': 'blue', 'delivering': 'orange', 'returning': 'green', 'completed': 'purple'}

    for i in range(num_agents):
        try:
            gate = random.choice(gates)
            dest = random.choice(destinations)
            arrival = random.randint(0, max_steps // 3)
            vendor = random.choice(vendors)
            vehicle = random.choice(vehicle_types)
            speed = speed_variation[vehicle]

            # FIXED: Ensure proper return paths are calculated
            outward_path = find_optimal_path_with_congestion(gate, dest, {}, {})
            return_path = find_optimal_path_with_congestion(dest, gate, {}, {})

            if not outward_path or outward_path[0] != gate or outward_path[-1] != dest:
                outward_path = [gate, dest]
            if not return_path or return_path[0] != dest or return_path[-1] != gate:
                return_path = [dest, gate]

            outward_coords = create_smooth_path(outward_path, speed)
            return_coords = create_smooth_path(return_path, speed)

            if not outward_coords:
                outward_coords = [nodes[gate], nodes[dest]]
            if not return_coords:
                return_coords = [nodes[dest], nodes[gate]]

            conventional_outward_time = len(outward_coords) * 2.0
            conventional_return_time = len(return_coords) * 2.0
            delivery_duration = random.randint(3, 6)
            total_conventional_time = conventional_outward_time + conventional_return_time + delivery_duration

            agent_data = {
                "agent_id": f"A{i+1:03d}",
                "gate": gate,
                "dest": dest,
                "arrival": arrival,
                "vendor": vendor,
                "vehicle": vehicle,
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
                "trail_positions": deque(maxlen=8),
                "gps_trail_positions": deque(maxlen=200),  # Increased for complete loop
                "delivery_duration": delivery_duration,
                "last_reroute_step": -10,
                "completed_step": None,
                # FIXED: Enhanced GPS tracking with agent-specific initialization
                "gps_tracker": EnhancedGPSTracker(agent_id=f"A{i+1:03d}"),
                "estimated_position": None,
                "current_signals": {},
                "gps_confidence": 0.0,
                "gps_methods_used": [],
                "gps_algorithm_data": [],
                "gps_complete_history": deque(maxlen=300),  # Increased for complete journey
                "journey_phase_history": []  # NEW: Track phase changes
            }
            agents.append(agent_data)

        except Exception as e:
            print(f"Error creating agent {i}: {e}")
            fallback_agent = {
                "agent_id": f"A{i+1:03d}",
                "gate": "MainGate",
                "dest": "HostelB",
                "arrival": 0,
                "vendor": "Fallback",
                "vehicle": "e-bike",
                "phase": "outward",
                "outward_path_nodes": ["MainGate", "HostelB"],
                "return_path_nodes": ["HostelB", "MainGate"],
                "outward_coords": [nodes["MainGate"], nodes["HostelB"]],
                "return_coords": [nodes["HostelB"], nodes["MainGate"]],
                "current_path_coords": [nodes["MainGate"], nodes["HostelB"]],
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
                "journey_phase_history": []
            }
            agents.append(fallback_agent)

    return agents

agents = create_optimized_agents(num_agents)

# Step 5: Enhanced visualization with complete GPS loop tracking
print("Creating enhanced visualization with complete GPS loop tracking...")

# Create larger figure with better spacing
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(36, 12))

# Main simulation plot
ax1.set_xlim(-10, 90)
ax1.set_ylim(-60, 60)
ax1.set_title("Smart Campus Delivery - Complete Loop GPS Tracking", fontsize=20, fontweight='bold', pad=25)
ax1.set_xlabel("Campus X Coordinate", fontsize=14)
ax1.set_ylabel("Campus Y Coordinate", fontsize=14)
ax1.set_facecolor('#f0f8ff')
ax1.tick_params(axis='both', which='major', labelsize=12)

# Statistics plot
ax2.set_xlim(0, max_steps)
ax2.set_ylim(0, num_agents + 8)
ax2.set_title("Delivery Performance & GPS Algorithm Usage", fontsize=16, fontweight='bold')
ax2.set_xlabel("Time Step", fontsize=14)
ax2.set_ylabel("Count", fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.set_facecolor('#f8f9fa')
ax2.tick_params(axis='both', which='major', labelsize=12)

# Enhanced GPS Tracking plot
ax3.set_xlim(-10, 90)
ax3.set_ylim(-60, 60)
ax3.set_title("Advanced GPS Tracking - Complete Loop Visualization", fontsize=20, fontweight='bold', pad=25)
ax3.set_xlabel("Campus X Coordinate", fontsize=14)
ax3.set_ylabel("Campus Y Coordinate", fontsize=14)
ax3.set_facecolor('#f8f3f0')
ax3.tick_params(axis='both', which='major', labelsize=12)

# Draw campus elements with enhanced styling and larger labels
for ax in [ax1, ax3]:
    for n, p in nodes.items():
        color = '#28a745' if 'Gate' in n else '#007bff' if 'Hub' in n else '#dc3545'
        ax.scatter(p[0], p[1], c=color, s=200, alpha=0.9, zorder=5, edgecolors='white', linewidth=3)
        ax.text(p[0] + 1, p[1] + 1, n, color='black', fontsize=12, zorder=5,
                bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.9, edgecolor='gray'))

    # Draw edges with enhanced styling
    for u, v in G.edges():
        x1, y1 = nodes[u]
        x2, y2 = nodes[v]
        if (u, v) in congestion_edges or (v, u) in congestion_edges:
            ax.plot([x1, x2], [y1, y2], 'red', alpha=0.5, linewidth=4, zorder=1, linestyle='--')
        else:
            ax.plot([x1, x2], [y1, y2], 'gray', alpha=0.4, linewidth=3, zorder=1)

    # Add campus zone annotations with swarm information
    ax.fill_between([70, 90], [-60, -60], [60, 60], color='lightcoral', alpha=0.15, label='Hostel Zone')
    ax.fill_between([40, 60], [10, 10], [60, 60], color='lightblue', alpha=0.15, label='Academic Zone')
    ax.fill_between([30, 50], [-60, -60], [10, 10], color='lightgreen', alpha=0.15, label='Common Area')

# Draw WiFi access points on GPS plot with larger labels
for ap_name, ap_info in wifi_access_points.items():
    x, y = ap_info["pos"]
    range_circle = plt.Circle((x, y), ap_info["range"], color='orange', alpha=0.2, zorder=2)
    ax3.add_patch(range_circle)
    ax3.scatter(x, y, c='red', s=120, marker='^', zorder=6, edgecolors='white', linewidth=3)
    ax3.text(x, y - 10, f"AP:{ap_name}", fontsize=10, ha='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

# Initialize agent visualization with larger elements
agent_dots = []
agent_trails = []
agent_path_lines = []
agent_labels = []
agent_status_labels = []
agent_time_labels = []

# Initialize enhanced GPS visualization with larger elements
gps_estimated_dots = []
gps_confidence_circles = []
gps_signal_lines = []
gps_accuracy_texts = []
gps_trail_lines = []
gps_algorithm_visualizations = []
gps_method_texts = []

for i, agent in enumerate(agents):
    # Main simulation visualization with larger elements
    dot, = ax1.plot([], [], 'o', markersize=14, color=agent['color'],
                   markeredgecolor='white', markeredgewidth=3, zorder=15, alpha=0.9)
    agent_dots.append(dot)

    trail, = ax1.plot([], [], '-', color=agent['color'], alpha=0.5, linewidth=3.5, zorder=12)
    agent_trails.append(trail)

    path_line, = ax1.plot([], [], '--', color=agent['color'], alpha=0.7, linewidth=2.5, zorder=10)
    agent_path_lines.append(path_line)

    label = ax1.text(0, 0, f"{agent['agent_id']}", fontsize=11, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.9), zorder=16)
    agent_labels.append(label)

    status_label = ax1.text(0, 0, "", fontsize=10,
                          bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.9), zorder=17)
    agent_status_labels.append(status_label)

    time_label = ax1.text(0, 0, "", fontsize=9,
                         bbox=dict(boxstyle="round,pad=0.2", facecolor='lightblue', alpha=0.8), zorder=17)
    agent_time_labels.append(time_label)

    # GPS tracking visualization with larger elements
    gps_dot, = ax3.plot([], [], 'X', markersize=15, color=agent['color'],
                       markeredgecolor='black', markeredgewidth=2.5, zorder=20, alpha=0.9)
    gps_estimated_dots.append(gps_dot)

    confidence_circle = plt.Circle((0, 0), 5, color=agent['color'], alpha=0.15, fill=True, zorder=18)
    ax3.add_patch(confidence_circle)
    gps_confidence_circles.append(confidence_circle)

    signal_line, = ax3.plot([], [], '--', color=agent['color'], alpha=0.3, linewidth=2, zorder=16)
    gps_signal_lines.append(signal_line)

    # GPS trail for complete loop
    gps_trail, = ax3.plot([], [], '-', color=agent['color'], alpha=0.6, linewidth=3, zorder=17)
    gps_trail_lines.append(gps_trail)

    accuracy_text = ax3.text(0, 0, "", fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.9), zorder=21)
    gps_accuracy_texts.append(accuracy_text)

    # Method used text
    method_text = ax3.text(0, 0, "", fontsize=9,
                         bbox=dict(boxstyle="round,pad=0.3", facecolor='lightyellow', alpha=0.8), zorder=22)
    gps_method_texts.append(method_text)

    # Algorithm visualization elements
    algorithm_elements = {
        'trilateration_circles': [],
        'fingerprint_points': [],
        'centroid_points': [],
        'fusion_lines': []
    }
    gps_algorithm_visualizations.append(algorithm_elements)

# Multiple swarm particle visualizations for density
swarm_dots = []
swarm_gps_dots = []

for i in range(5):
    dots, = ax1.plot([], [], 'o', markersize=3, alpha=0.6, zorder=4+i)
    swarm_dots.append(dots)

    gps_dots, = ax3.plot([], [], 'o', markersize=2, alpha=0.4, zorder=4+i)
    swarm_gps_dots.append(gps_dots)

# Status text with larger font
status_text = ax1.text(0.02, 0.98, '', transform=ax1.transAxes, fontsize=13,
                     verticalalignment='top',
                     bbox=dict(boxstyle="round,pad=0.7", facecolor='white', alpha=0.95))

gps_status_text = ax3.text(0.02, 0.98, '', transform=ax3.transAxes, fontsize=12,
                         verticalalignment='top',
                         bbox=dict(boxstyle="round,pad=0.6", facecolor='white', alpha=0.95))

# Algorithm legend and status with larger font
algorithm_status_text = ax3.text(0.02, 0.02, '', transform=ax3.transAxes, fontsize=11,
                               verticalalignment='bottom',
                               bbox=dict(boxstyle="round,pad=0.6", facecolor='white', alpha=0.9))

# Add legend for swarm types with larger font
ax1.legend(loc='upper right', fontsize=12, framealpha=0.9)
ax3.legend(loc='upper right', fontsize=11, framealpha=0.9)

# Statistics tracking
active_agents = [0] * max_steps
delivered_agents = [0] * max_steps
returned_agents = [0] * max_steps
congestion_levels = [0] * max_steps
background_congestion_levels = [0] * max_steps
swarm_intensity_levels = [0] * max_steps
gps_accuracy_levels = [0] * max_steps
algorithm_usage = {
    'Trilateration': [0] * max_steps,
    'Fingerprinting': [0] * max_steps,
    'Centroid': [0] * max_steps,
    'Kalman Filter': [0] * max_steps,
    'Fusion': [0] * max_steps
}

active_line, = ax2.plot([], [], color='blue', linestyle='-', label='Active Agents', linewidth=4)
delivered_line, = ax2.plot([], [], color='orange', linestyle='-', label='At Destination', linewidth=4)
returned_line, = ax2.plot([], [], color='green', linestyle='-', label='Returned', linewidth=4)
congestion_line, = ax2.plot([], [], color='red', linestyle='-', label='Rider Congestion', linewidth=3, alpha=0.7)
background_line, = ax2.plot([], [], color='purple', linestyle='-', label='Swarm Traffic', linewidth=3, alpha=0.7)
swarm_line, = ax2.plot([], [], color='brown', linestyle='-', label='Swarm Intensity', linewidth=3, alpha=0.7)
gps_accuracy_line, = ax2.plot([], [], color='cyan', linestyle='-', label='GPS Accuracy', linewidth=3, alpha=0.8)

# Algorithm usage lines with thicker lines
trilateration_line, = ax2.plot([], [], color='red', linestyle='--', label='Trilateration', linewidth=3, alpha=0.7)
fingerprinting_line, = ax2.plot([], [], color='green', linestyle='--', label='Fingerprinting', linewidth=3, alpha=0.7)
centroid_line, = ax2.plot([], [], color='orange', linestyle='--', label='Centroid', linewidth=3, alpha=0.7)
kalman_line, = ax2.plot([], [], color='purple', linestyle='--', label='Kalman', linewidth=3, alpha=0.7)
fusion_line, = ax2.plot([], [], color='brown', linestyle='--', label='Fusion', linewidth=3, alpha=0.7)

ax2.legend(loc='upper left', fontsize=11)

# Enhanced heatmap for swarm visualization
heatmap_data = np.zeros((60, 50))
heatmap = ax1.imshow(heatmap_data, extent=(-10, 90, -60, 60), origin='lower',
                   cmap='hot', alpha=0.4, vmin=0, vmax=12)

# GPS tracking heatmap
gps_heatmap_data = np.zeros((60, 50))
gps_heatmap = ax3.imshow(gps_heatmap_data, extent=(-10, 90, -60, 60), origin='lower',
                       cmap='viridis', alpha=0.3, vmin=0, vmax=8)

# NEW: Additional data visualization elements
# Real-time efficiency gain plot
efficiency_gains = [0] * max_steps
efficiency_line, = ax2.plot([], [], color='magenta', linestyle='-', label='Avg Efficiency %', linewidth=3, alpha=0.8)

# NEW: Phase transition markers
phase_markers = []

# Step 6: Enhanced swarm management with complete GPS loop tracking
def update_plot(frame):
    current_status = f"Step: {frame}/500 (8am-8pm)\n"
    gps_status = f"Advanced GPS Tracking - Step: {frame}/500\n"
    active_count = 0
    delivered_count = 0
    returned_count = 0
    total_congestion = 0
    total_background_congestion = 0
    swarm_intensity = 0
    total_gps_accuracy = 0
    gps_tracked_count = 0
    total_efficiency = 0
    efficiency_count = 0

    # ENHANCED: Get background congestion with swarm behavior
    background_congestion = get_background_congestion(frame)

    # Update swarm particles
    for particle in traffic_particles:
        particle.update(frame)

    # Calculate swarm intensity
    current_period = None
    for period, info in class_schedule.items():
        if info["start"] <= frame <= info["end"]:
            current_period = period
            swarm_intensity = info["intensity"]
            break

    # Update congestion weights with swarm behavior
    node_counts = {n: 0 for n in nodes}
    positions = []
    gps_positions = []

    # Reset algorithm counts for this frame
    current_algorithm_counts = {algo: 0 for algo in algorithm_usage.keys()}

    for agent in agents:
        if frame >= agent['arrival'] and agent['phase'] != 'completed':
            if agent['phase'] == 'outward' and agent['current_idx'] < len(agent['current_path_coords']):
                x, y = agent['current_path_coords'][agent['current_idx']]
                positions.append((x, y))
                nearest_node = min(nodes, key=lambda n: math.hypot(x - nodes[n][0], y - nodes[n][1]))
                node_counts[nearest_node] += 1

                # Update GPS tracking for agent
                true_position = (x, y)
                agent['estimated_position'], agent['current_signals'], agent['gps_confidence'], methods_used, algo_data = agent['gps_tracker'].estimate_enhanced_position(true_position, frame)
                agent['gps_methods_used'] = methods_used
                agent['gps_algorithm_data'] = algo_data

                # NEW: Update journey phase in GPS tracker
                if agent['gps_tracker'].complete_journey_history:
                    agent['gps_tracker'].complete_journey_history[-1]['phase'] = 'outward'

                if agent['estimated_position'] is not None:
                    gps_positions.append(agent['estimated_position'])
                    error = math.hypot(agent['estimated_position'][0] - x, agent['estimated_position'][1] - y)
                    total_gps_accuracy += error
                    gps_tracked_count += 1

                    for method in methods_used:
                        if method in current_algorithm_counts:
                            current_algorithm_counts[method] += 1

            elif agent['phase'] == 'delivering':
                x, y = nodes[agent['dest']]
                positions.append((x, y))
                node_counts[agent['dest']] += 1

                # NEW: Update GPS during delivery phase
                true_position = (x, y)
                agent['estimated_position'], agent['current_signals'], agent['gps_confidence'], methods_used, algo_data = agent['gps_tracker'].estimate_enhanced_position(true_position, frame)

                # NEW: Update journey phase in GPS tracker
                if agent['gps_tracker'].complete_journey_history:
                    agent['gps_tracker'].complete_journey_history[-1]['phase'] = 'delivering'

            elif agent['phase'] == 'returning' and agent['current_idx'] < len(agent['current_path_coords']):
                x, y = agent['current_path_coords'][agent['current_idx']]
                positions.append((x, y))
                nearest_node = min(nodes, key=lambda n: math.hypot(x - nodes[n][0], y - nodes[n][1]))
                node_counts[nearest_node] += 1

                # Update GPS tracking for agent
                true_position = (x, y)
                agent['estimated_position'], agent['current_signals'], agent['gps_confidence'], methods_used, algo_data = agent['gps_tracker'].estimate_enhanced_position(true_position, frame)

                # NEW: Update journey phase in GPS tracker
                if agent['gps_tracker'].complete_journey_history:
                    agent['gps_tracker'].complete_journey_history[-1]['phase'] = 'returning'

                if agent['estimated_position'] is not None:
                    gps_positions.append(agent['estimated_position'])
                    error = math.hypot(agent['estimated_position'][0] - x, agent['estimated_position'][1] - y)
                    total_gps_accuracy += error
                    gps_tracked_count += 1

                    for method in methods_used:
                        if method in current_algorithm_counts:
                            current_algorithm_counts[method] += 1

    # Update algorithm usage tracking
    for algo, count in current_algorithm_counts.items():
        if frame < max_steps:
            algorithm_usage[algo][frame] = count

    # Update edge weights with enhanced swarm congestion
    for u, v in G.edges():
        base_weight = G[u][v]['base_weight']
        node_congestion = (node_counts[u] + node_counts[v]) / 2
        background_node_congestion = (background_congestion.get(u, 0) + background_congestion.get(v, 0)) / 2

        effective_congestion = 0.7 * background_node_congestion + 0.3 * node_congestion

        if effective_congestion > 2:
            congestion_penalty = 1 + min(5.0, 0.7 * effective_congestion)
        else:
            congestion_penalty = 1 + min(2.0, 0.4 * effective_congestion)

        G[u][v]['weight'] = base_weight * congestion_penalty
        total_congestion += node_congestion
        total_background_congestion += background_node_congestion

    # ENHANCED: Update heatmap with swarm visualization
    heat = np.zeros((60, 50))
    gps_heat = np.zeros((60, 50))

    # Add rider heat
    for x, y in positions:
        ix = int(np.clip((x + 10) / 100 * 49, 0, 49))
        iy = int(np.clip((y + 60) / 120 * 59, 0, 59))
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                idx_x = min(max(0, ix + dx), 49)
                idx_y = min(max(0, iy + dy), 59)
                heat[idx_y, idx_x] += 0.2

    # Add GPS position heat
    for x, y in gps_positions:
        ix = int(np.clip((x + 10) / 100 * 49, 0, 49))
        iy = int(np.clip((y + 60) / 120 * 59, 0, 59))
        if 0 <= ix < 50 and 0 <= iy < 60:
            gps_heat[iy, ix] += 0.3

    # Add swarm heat with enhanced visualization
    for particle in traffic_particles:
        x, y = particle.get_position()
        ix = int(np.clip((x + 10) / 100 * 49, 0, 49))
        iy = int(np.clip((y + 60) / 120 * 59, 0, 59))

        for dx in range(-2, 3):
            for dy in range(-2, 3):
                idx_x = min(max(0, ix + dx), 49)
                idx_y = min(max(0, iy + dy), 59)
                distance = math.sqrt(dx*dx + dy*dy)
                if distance <= 2:
                    intensity = 0.8 * (1 - distance/3)
                    heat[idx_y, idx_x] += intensity

        if hasattr(particle, 'estimated_position') and particle.estimated_position is not None:
            gps_x, gps_y = particle.estimated_position
            gps_ix = int(np.clip((gps_x + 10) / 100 * 49, 0, 49))
            gps_iy = int(np.clip((gps_y + 60) / 120 * 59, 0, 59))
            if 0 <= gps_ix < 50 and 0 <= gps_iy < 60:
                gps_heat[gps_iy, gps_ix] += 0.1

    # Add background congestion heat
    for node, congestion in background_congestion.items():
        x, y = nodes[node]
        ix = int(np.clip((x + 10) / 100 * 49, 0, 49))
        iy = int(np.clip((y + 60) / 120 * 59, 0, 59))
        for dx in range(-3, 4):
            for dy in range(-3, 4):
                idx_x = min(max(0, ix + dx), 49)
                idx_y = min(max(0, iy + dy), 59)
                distance = math.sqrt(dx*dx + dy*dy)
                if distance <= 3:
                    intensity = congestion * 0.3 * (1 - distance/4)
                    heat[idx_y, idx_x] += intensity

    heatmap.set_data(heat)
    gps_heatmap.set_data(gps_heat)

    # Update swarm particle visualization with layered effect
    swarm_positions = [[] for _ in range(5)]
    swarm_gps_positions = [[] for _ in range(5)]

    for i, particle in enumerate(traffic_particles):
        layer = i % 5
        x, y = particle.get_position()
        swarm_positions[layer].append((x, y))

        if hasattr(particle, 'estimated_position') and particle.estimated_position is not None:
            gps_x, gps_y = particle.estimated_position
            swarm_gps_positions[layer].append((gps_x, gps_y))

    for i, dots in enumerate(swarm_dots):
        if swarm_positions[i]:
            x_vals, y_vals = zip(*swarm_positions[i])
            dots.set_data(x_vals, y_vals)
            if i == 0:
                dots.set_color('lightblue')
            elif i == 1:
                dots.set_color('lightgreen')
            elif i == 2:
                dots.set_color('lightyellow')
            elif i == 3:
                dots.set_color('lightcoral')
            else:
                dots.set_color('plum')
            dots.set_markersize(3)
            dots.set_alpha(0.6)
        else:
            dots.set_data([], [])

    for i, dots in enumerate(swarm_gps_dots):
        if swarm_gps_positions[i]:
            x_vals, y_vals = zip(*swarm_gps_positions[i])
            dots.set_data(x_vals, y_vals)
            if i == 0:
                dots.set_color('lightblue')
            elif i == 1:
                dots.set_color('lightgreen')
            elif i == 2:
                dots.set_color('lightyellow')
            elif i == 3:
                dots.set_color('lightcoral')
            else:
                dots.set_color('plum')
            dots.set_markersize(2)
            dots.set_alpha(0.4)
        else:
            dots.set_data([], [])

    # Update each agent with swarm-aware routing
    for i, agent in enumerate(agents):
        if frame < agent['arrival']:
            agent_dots[i].set_data([], [])
            agent_trails[i].set_data([], [])
            agent_path_lines[i].set_data([], [])
            agent_labels[i].set_position((-100, -100))
            agent_status_labels[i].set_position((-100, -100))
            agent_time_labels[i].set_position((-100, -100))

            gps_estimated_dots[i].set_data([], [])
            gps_confidence_circles[i].set_radius(0)
            gps_signal_lines[i].set_data([], [])
            gps_trail_lines[i].set_data([], [])
            gps_accuracy_texts[i].set_position((-100, -100))
            gps_method_texts[i].set_position((-100, -100))
            continue

        if agent['phase'] == 'completed':
            x, y = nodes[agent['gate']]
            agent_dots[i].set_data([x], [y])
            agent_labels[i].set_position((x, y - 8))
            agent_status_labels[i].set_position((x, y + 10))
            agent_status_labels[i].set_text("COMPLETED")
            agent_status_labels[i].set_bbox(dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.9))

            if agent['time_saved'] is not None:
                if agent['time_saved'] >= 0:
                    time_color = 'lightgreen'
                    time_text = f"Saved: {agent['time_saved']:.1f}s"
                else:
                    time_color = 'lightcoral'
                    time_text = f"Lost: {abs(agent['time_saved']):.1f}s"
                agent_time_labels[i].set_position((x, y - 15))
                agent_time_labels[i].set_text(time_text)
                agent_time_labels[i].set_bbox(dict(boxstyle="round,pad=0.2", facecolor=time_color, alpha=0.9))

            returned_count += 1

            # Calculate efficiency for completed agents
            if agent['efficiency_gain'] is not None:
                total_efficiency += agent['efficiency_gain']
                efficiency_count += 1

            # GPS tracking for completed agents
            if hasattr(agent['gps_tracker'], 'estimated_positions') and agent['gps_tracker'].estimated_positions:
                final_gps_pos = agent['gps_tracker'].estimated_positions[-1]
                gps_x, gps_y = final_gps_pos
                gps_estimated_dots[i].set_data([gps_x], [gps_y])

                error = math.hypot(gps_x - x, gps_y - y)

                gps_confidence_circles[i].set_center((gps_x, gps_y))
                gps_confidence_circles[i].set_radius(max(3, min(10, error * 1.2)))

                gps_accuracy_texts[i].set_position((gps_x, gps_y + 12))
                gps_accuracy_texts[i].set_text(f"Final Err: {error:.1f}u")

                if hasattr(agent['gps_tracker'], 'method_used') and agent['gps_tracker'].method_used:
                    final_methods = agent['gps_tracker'].method_used[-1] if agent['gps_tracker'].method_used else []
                    method_str = " + ".join(final_methods[-2:]) if len(final_methods) > 2 else " + ".join(final_methods)
                    gps_method_texts[i].set_position((gps_x, gps_y - 15))
                    gps_method_texts[i].set_text(f"Final: {method_str}")
            continue

        elif agent['phase'] == 'returning':
            if agent['current_idx'] >= len(agent['current_path_coords']):
                agent['phase'] = 'completed'
                agent['return_step'] = frame
                agent['completed_step'] = frame
                agent['total_optimized_time'] = frame - agent['arrival']
                agent['time_saved'] = agent['conventional_time'] - agent['total_optimized_time']
                if agent['conventional_time'] > 0:
                    agent['efficiency_gain'] = (agent['time_saved'] / agent['conventional_time']) * 100
                continue

        elif agent['phase'] == 'delivering':
            if frame >= agent['delivery_step'] + agent['delivery_duration']:
                agent['phase'] = 'returning'
                agent['current_path_coords'] = agent['return_coords']
                agent['current_idx'] = 0
                # NEW: Record phase transition
                agent['journey_phase_history'].append(('delivering_to_returning', frame))
                continue
            else:
                x, y = nodes[agent['dest']]
                agent_dots[i].set_data([x], [y])
                agent_labels[i].set_position((x, y - 8))
                remaining_delivery = (agent['delivery_step'] + agent['delivery_duration']) - frame
                agent_status_labels[i].set_position((x, y + 10))
                agent_status_labels[i].set_text(f"Delivering ({remaining_delivery}s)")
                agent_status_labels[i].set_bbox(dict(boxstyle="round,pad=0.3", facecolor='orange', alpha=0.9))

                elapsed_time = frame - agent['arrival']
                agent_time_labels[i].set_position((x, y - 15))
                agent_time_labels[i].set_text(f"ET: {elapsed_time}s")
                agent_time_labels[i].set_bbox(dict(boxstyle="round,pad=0.2", facecolor='lightblue', alpha=0.8))

                delivered_count += 1
                continue

        elif agent['phase'] == 'outward':
            if agent['current_idx'] >= len(agent['current_path_coords']):
                agent['phase'] = 'delivering'
                agent['delivery_step'] = frame
                agent['outward_time'] = frame - agent['arrival']
                # NEW: Record phase transition
                agent['journey_phase_history'].append(('outward_to_delivering', frame))
                continue

        # Movement for both outward and returning phases
        if agent['phase'] in ['outward', 'returning']:
            if agent['current_idx'] < len(agent['current_path_coords']):
                x, y = agent['current_path_coords'][agent['current_idx']]
                agent_dots[i].set_data([x], [y])
                agent_labels[i].set_position((x, y - 8))

                dot_color = agent['phase_colors'][agent['phase']]
                agent_dots[i].set_color(dot_color)

                status_text_map = {
                    'outward': f">> {agent['dest']}",
                    'returning': f"<< {agent['gate']}",
                    'delivering': f"DLV @ {agent['dest']}"
                }
                agent_status_labels[i].set_position((x, y + 10))
                agent_status_labels[i].set_text(status_text_map.get(agent['phase'], agent['phase']))
                status_color = dot_color if agent['phase'] != 'delivering' else 'orange'
                agent_status_labels[i].set_bbox(dict(boxstyle="round,pad=0.3", facecolor=status_color, alpha=0.7))

                elapsed_time = frame - agent['arrival']
                agent_time_labels[i].set_position((x, y - 15))
                agent_time_labels[i].set_text(f"ET: {elapsed_time}s")
                agent_time_labels[i].set_bbox(dict(boxstyle="round,pad=0.2", facecolor='lightblue', alpha=0.8))

                agent['trail_positions'].append((x, y))
                if len(agent['trail_positions']) > 1:
                    trail_x, trail_y = zip(*agent['trail_positions'])
                    agent_trails[i].set_data(trail_x, trail_y)

                if agent['current_idx'] < len(agent['current_path_coords']) - 1:
                    future_x = [agent['current_path_coords'][j][0] for j in range(agent['current_idx'], len(agent['current_path_coords']))]
                    future_y = [agent['current_path_coords'][j][1] for j in range(agent['current_idx'], len(agent['current_path_coords']))]
                    agent_path_lines[i].set_data(future_x, future_y)
                else:
                    agent_path_lines[i].set_data([], [])

                active_count += 1

                # Update GPS tracking visualization for moving agents
                true_position = (x, y)
                agent['estimated_position'], agent['current_signals'], agent['gps_confidence'], methods_used, algo_data = agent['gps_tracker'].estimate_enhanced_position(true_position, frame)
                agent['gps_methods_used'] = methods_used
                agent['gps_algorithm_data'] = algo_data

                if agent['estimated_position'] is not None:
                    gps_x, gps_y = agent['estimated_position']
                    gps_estimated_dots[i].set_data([gps_x], [gps_y])

                    # Store GPS position for complete trail
                    agent['gps_trail_positions'].append((gps_x, gps_y))
                    agent['gps_complete_history'].append((gps_x, gps_y))

                    # Update GPS trail for complete loop
                    if len(agent['gps_trail_positions']) > 1:
                        trail_x, trail_y = zip(*agent['gps_trail_positions'])
                        gps_trail_lines[i].set_data(trail_x, trail_y)

                    error = math.hypot(gps_x - x, gps_y - y)

                    gps_confidence_circles[i].set_center((gps_x, gps_y))
                    gps_confidence_circles[i].set_radius(max(3, min(15, error * 1.5)))

                    connected_aps = []
                    for ap_name, rssi in agent['current_signals'].items():
                        if rssi > -80:
                            ap_pos = wifi_access_points[ap_name]["pos"]
                            connected_aps.extend([gps_x, gps_y, ap_pos[0], ap_pos[1]])

                    if connected_aps:
                        gps_signal_lines[i].set_data(connected_aps[0::2], connected_aps[1::2])
                    else:
                        gps_signal_lines[i].set_data([], [])

                    gps_accuracy_texts[i].set_position((gps_x, gps_y + 12))
                    gps_accuracy_texts[i].set_text(f"Err: {error:.1f}u\nConf: {agent['gps_confidence']:.2f}")

                    if error < 3:
                        color = 'lightgreen'
                    elif error < 6:
                        color = 'lightyellow'
                    else:
                        color = 'lightcoral'
                    gps_accuracy_texts[i].set_bbox(dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.9))

                    method_str = " + ".join(agent['gps_methods_used'][-2:]) if len(agent['gps_methods_used']) > 2 else " + ".join(agent['gps_methods_used'])
                    gps_method_texts[i].set_position((gps_x, gps_y - 15))
                    gps_method_texts[i].set_text(f"Methods: {method_str}")
                    gps_method_texts[i].set_bbox(dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.8))

                    visualize_algorithms(agent, i, frame)

                # Enhanced rerouting with swarm awareness
                if (frame % 35 == 0 and agent['current_idx'] > 3 and
                    agent['phase'] in ['outward', 'returning'] and
                    frame > agent['last_reroute_step'] + 30 and
                    agent['reroute_count'] < 4):

                    try:
                        current_node = min(nodes, key=lambda n: math.hypot(x - nodes[n][0], y - nodes[n][1]))
                        target = agent['dest'] if agent['phase'] == 'outward' else agent['gate']

                        if current_node != target:
                            current_congestion_val = node_counts.get(current_node, 0)
                            background_congestion_val = background_congestion.get(current_node, 0)
                            effective_current_congestion = 0.7 * background_congestion_val + 0.3 * current_congestion_val

                            if effective_current_congestion > 3.0:
                                new_path = find_optimal_path_with_congestion(current_node, target, node_counts, background_congestion)

                                if new_path and len(new_path) > 1:
                                    new_coords = create_smooth_path(new_path, agent['speed'])

                                    if new_coords and len(new_coords) < len(agent['current_path_coords']) * 1.5:
                                        if agent['phase'] == 'outward':
                                            agent['outward_coords'] = new_coords
                                        else:
                                            agent['return_coords'] = new_coords

                                        agent['current_path_coords'] = new_coords
                                        agent['current_idx'] = 0
                                        agent['reroute_count'] += 1
                                        agent['last_reroute_step'] = frame
                    except Exception as e:
                        pass

                agent['current_idx'] += 1

    # Update statistics
    if frame < max_steps:
        active_agents[frame] = active_count
        delivered_agents[frame] = delivered_count
        returned_agents[frame] = returned_count
        congestion_levels[frame] = total_congestion / len(G.edges()) if G.edges() else 0
        background_congestion_levels[frame] = total_background_congestion / len(nodes) if nodes else 0
        swarm_intensity_levels[frame] = swarm_intensity
        if gps_tracked_count > 0:
            gps_accuracy_levels[frame] = total_gps_accuracy / gps_tracked_count
        else:
            gps_accuracy_levels[frame] = 0

        # NEW: Update efficiency gains
        if efficiency_count > 0:
            efficiency_gains[frame] = total_efficiency / efficiency_count
        else:
            efficiency_gains[frame] = 0

    # Update statistics plot
    frames_to_plot = min(frame + 1, max_steps)
    active_line.set_data(range(frames_to_plot), active_agents[:frames_to_plot])
    delivered_line.set_data(range(frames_to_plot), delivered_agents[:frames_to_plot])
    returned_line.set_data(range(frames_to_plot), returned_agents[:frames_to_plot])
    congestion_line.set_data(range(frames_to_plot), congestion_levels[:frames_to_plot])
    background_line.set_data(range(frames_to_plot), background_congestion_levels[:frames_to_plot])
    swarm_line.set_data(range(frames_to_plot), swarm_intensity_levels[:frames_to_plot])
    gps_accuracy_line.set_data(range(frames_to_plot), gps_accuracy_levels[:frames_to_plot])
    efficiency_line.set_data(range(frames_to_plot), efficiency_gains[:frames_to_plot])  # NEW

    # Update algorithm usage lines
    trilateration_line.set_data(range(frames_to_plot), algorithm_usage['Trilateration'][:frames_to_plot])
    fingerprinting_line.set_data(range(frames_to_plot), algorithm_usage['Fingerprinting'][:frames_to_plot])
    centroid_line.set_data(range(frames_to_plot), algorithm_usage['Centroid'][:frames_to_plot])
    kalman_line.set_data(range(frames_to_plot), algorithm_usage['Kalman Filter'][:frames_to_plot])
    fusion_line.set_data(range(frames_to_plot), algorithm_usage['Fusion'][:frames_to_plot])

    # Status text with swarm information
    time_of_day = "Morning"
    if 150 < frame <= 280:
        time_of_day = "Afternoon"
    elif frame > 280:
        time_of_day = "Evening"

    current_status += f"Time: {time_of_day} | Active: {active_count} | Delivering: {delivered_count} | Returned: {returned_count}\n"
    current_status += f"Reroutes: {sum(a['reroute_count'] for a in agents)}\n"
    current_status += f"Rider Congestion: {congestion_levels[min(frame, max_steps-1)]:.1f}\n"
    current_status += f"Swarm Traffic: {background_congestion_levels[min(frame, max_steps-1)]:.1f}\n"
    current_status += f"Swarm Intensity: {swarm_intensity:.1f}\n"

    # NEW: Add efficiency information
    if efficiency_count > 0:
        current_status += f"Avg Efficiency: {efficiency_gains[min(frame, max_steps-1)]:.1f}%"

    gps_status += f"Tracked Agents: {gps_tracked_count}/{active_count + delivered_count + returned_count}\n"
    if gps_tracked_count > 0:
        avg_accuracy = total_gps_accuracy / gps_tracked_count
        gps_status += f"Avg GPS Error: {avg_accuracy:.1f} units\n"
        if avg_accuracy < 2:
            gps_status += "Tracking Quality: Excellent"
        elif avg_accuracy < 4:
            gps_status += "Tracking Quality: Good"
        elif avg_accuracy < 6:
            gps_status += "Tracking Quality: Fair"
        else:
            gps_status += "Tracking Quality: Poor"
    else:
        gps_status += "No active GPS tracking"

    # Show current swarm patterns
    current_peaks = []
    swarm_patterns = []
    for period, info in class_schedule.items():
        if info["start"] <= frame <= info["end"]:
            current_peaks.append(period.replace('_', ' '))
            for route in info["swarm_routes"]:
                pattern = f"{'→'.join(route['from'][:1])}→{'→'.join(route['to'][:1])}"
                swarm_patterns.append(pattern)

    if current_peaks:
        current_status += f"Peak: {', '.join(current_peaks)}\n"
    if swarm_patterns:
        unique_patterns = list(set(swarm_patterns))
        display_patterns = unique_patterns[:2]
        current_status += f"Swarm: {', '.join(display_patterns)}\n"

    congested_nodes = sorted([(node, count) for node, count in node_counts.items()],
                           key=lambda x: x[1], reverse=True)[:2]
    if congested_nodes and any(count > 0 for _, count in congested_nodes):
        congestion_info = ", ".join([f"{node}({count})" for node, count in congested_nodes])
        current_status += f"Rider Hotspots: {congestion_info}"

    status_text.set_text(current_status)
    gps_status_text.set_text(gps_status)

    # ENHANCED: Update algorithm status
    update_algorithm_status(frame)

    return (agent_dots + agent_trails + agent_path_lines +
            agent_labels + agent_status_labels + agent_time_labels + swarm_dots +
            gps_estimated_dots + gps_signal_lines + gps_trail_lines +
            gps_accuracy_texts + gps_method_texts + swarm_gps_dots +
            [status_text, gps_status_text, algorithm_status_text, active_line,
             delivered_line, returned_line, congestion_line, background_line,
             swarm_line, gps_accuracy_line, efficiency_line, trilateration_line, fingerprinting_line,
             centroid_line, kalman_line, fusion_line, heatmap, gps_heatmap])

def visualize_algorithms(agent, agent_idx, frame):
    """Visualize the GPS algorithms being used"""
    tracker = agent['gps_tracker']
    if not tracker.algorithm_visualization_data:
        return

    latest_algo = tracker.algorithm_visualization_data[-1]

    for element_type in gps_algorithm_visualizations[agent_idx]:
        for artist in gps_algorithm_visualizations[agent_idx][element_type]:
            artist.remove()
        gps_algorithm_visualizations[agent_idx][element_type] = []

    for algo_data in latest_algo['algorithm_data']:
        method = algo_data.get('method', '')
        est_pos = algo_data.get('estimated_position')

        if method == 'Trilateration' and 'ap_positions' in algo_data:
            for ap_pos in algo_data['ap_positions']:
                circle = plt.Circle(ap_pos, 5, color='red', alpha=0.2, fill=True)
                ax3.add_patch(circle)
                gps_algorithm_visualizations[agent_idx]['trilateration_circles'].append(circle)

        elif method == 'Fingerprinting' and 'reference_points' in algo_data:
            for ref_point in algo_data['reference_points']:
                point, = ax3.plot([ref_point[0]], [ref_point[1]], 's',
                                markersize=6, color='green', alpha=0.6)
                gps_algorithm_visualizations[agent_idx]['fingerprint_points'].append(point)

        elif method == 'Centroid' and 'ap_positions' in algo_data:
            for ap_pos in algo_data['ap_positions']:
                point, = ax3.plot([ap_pos[0]], [ap_pos[1]], '^',
                                markersize=7, color='orange', alpha=0.5)
                gps_algorithm_visualizations[agent_idx]['centroid_points'].append(point)

def update_algorithm_status(frame):
    """Update the algorithm usage status"""
    algorithm_counts = {
        'Trilateration': 0,
        'Fingerprinting': 0,
        'Centroid': 0,
        'Kalman Filter': 0,
        'Fusion': 0
    }

    total_active = 0
    for agent in agents:
        if (frame >= agent['arrival'] and agent['phase'] != 'completed' and
            hasattr(agent['gps_tracker'], 'method_used') and agent['gps_tracker'].method_used):
            total_active += 1
            latest_methods = agent['gps_tracker'].method_used[-1] if agent['gps_tracker'].method_used else []
            for method in latest_methods:
                if method in algorithm_counts:
                    algorithm_counts[method] += 1

    status_text = "GPS Algorithms Active:\n"
    for method, count in algorithm_counts.items():
        if count > 0:
            percentage = (count / total_active * 100) if total_active > 0 else 0
            status_text += f"{method}: {count} ({percentage:.0f}%)\n"

    algorithm_status_text.set_text(status_text)

# Step 7: Create and display enhanced swarm animation with complete GPS tracking
print("Creating dense swarm animation with complete GPS loop tracking...")

try:
    ani = animation.FuncAnimation(
        fig, update_plot, frames=min(max_steps, 300),  # Reduced frames to prevent size issues
        interval=100, blit=False, repeat=True
    )

    plt.close(fig)

    display(HTML(ani.to_jshtml()))

except Exception as e:
    print(f"HTML display failed: {e}")
    try:
        ani = animation.FuncAnimation(
            fig, update_plot, frames=min(max_steps, 150),
            interval=100, blit=False, repeat=True
        )

        ani.save('complete_gps_loop_simulation.gif', writer='pillow', fps=10, dpi=120)

        from IPython.display import Image
        display(Image(filename='complete_gps_loop_simulation.gif'))

    except Exception as e2:
        print(f"GIF method failed: {e2}")

# Step 8: FIXED GPS loop visualization to show complete paths
def calculate_gps_accuracy(agent):
    """Calculate GPS accuracy for an agent with proper array handling"""
    gps_errors = []
    tracker = agent.get('gps_tracker')
    if (tracker and hasattr(tracker, 'position_history') and
        hasattr(tracker, 'estimated_positions')):

        min_length = min(len(tracker.position_history), len(tracker.estimated_positions))

        for i in range(min_length):
            true_pos = tracker.position_history[i]
            est_pos = tracker.estimated_positions[i]

            if (true_pos is not None and est_pos is not None and
                not (isinstance(true_pos, np.ndarray) and np.any(np.isnan(true_pos))) and
                not (isinstance(est_pos, np.ndarray) and np.any(np.isnan(est_pos)))):

                if isinstance(true_pos, np.ndarray):
                    true_pos = tuple(true_pos)
                if isinstance(est_pos, np.ndarray):
                    est_pos = tuple(est_pos)

                if (isinstance(true_pos, (tuple, list)) and isinstance(est_pos, (tuple, list)) and
                    len(true_pos) == 2 and len(est_pos) == 2):

                    error = math.hypot(true_pos[0] - est_pos[0], true_pos[1] - est_pos[1])
                    gps_errors.append(error)

    return np.mean(gps_errors) if gps_errors else 0

# FIXED: Enhanced GPS loop visualization to show complete paths for ALL agents
def visualize_complete_gps_loops():
    """Create a separate visualization showing complete GPS loops for all agents"""
    # Create a much larger figure for better visibility
    fig_loops, ax_loops = plt.subplots(1, 1, figsize=(20, 14))
    ax_loops.set_xlim(-10, 90)
    ax_loops.set_ylim(-60, 60)
    ax_loops.set_title("Complete GPS Tracking Loops - All Agents (Entry + Delivery + Return)",
                      fontsize=20, fontweight='bold', pad=25)
    ax_loops.set_xlabel("Campus X Coordinate", fontsize=16)
    ax_loops.set_ylabel("Campus Y Coordinate", fontsize=16)
    ax_loops.set_facecolor('#f8f8f8')
    ax_loops.tick_params(axis='both', which='major', labelsize=14)

    # Draw campus map with larger elements
    for n, p in nodes.items():
        color = '#28a745' if 'Gate' in n else '#007bff' if 'Hub' in n else '#dc3545'
        ax_loops.scatter(p[0], p[1], c=color, s=200, alpha=0.9, zorder=5, edgecolors='white', linewidth=3)
        ax_loops.text(p[0] + 2, p[1] + 2, n, color='black', fontsize=14, zorder=5,
                     bbox=dict(boxstyle="round,pad=0.4", facecolor='white', alpha=0.9))

    # Draw edges with thicker lines
    for u, v in G.edges():
        x1, y1 = nodes[u]
        x2, y2 = nodes[v]
        if (u, v) in congestion_edges or (v, u) in congestion_edges:
            ax_loops.plot([x1, x2], [y1, y2], 'red', alpha=0.5, linewidth=3, zorder=1, linestyle='--')
        else:
            ax_loops.plot([x1, x2], [y1, y2], 'gray', alpha=0.4, linewidth=2, zorder=1)

    # FIXED: Use the complete journey history from GPS tracker
    completed_loops = 0
    partial_loops = 0

    for agent in agents:
        # Use the complete journey history that includes phase information
        tracker = agent['gps_tracker']
        if hasattr(tracker, 'complete_journey_history') and tracker.complete_journey_history:
            journey_data = tracker.complete_journey_history

            if len(journey_data) > 1:
                # Extract positions and phases
                positions = [point['estimated_position'] for point in journey_data]
                phases = [point.get('phase', 'unknown') for point in journey_data]

                if len(positions) > 5:
                    gps_x, gps_y = zip(*positions)

                    # Plot the complete GPS trail with phase-based coloring
                    for i in range(len(gps_x)-1):
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
                                    '-', color=color, alpha=alpha, linewidth=2.5, zorder=10)

                    # Mark key points
                    if gps_x and gps_y:
                        # Start point (gate)
                        ax_loops.scatter(gps_x[0], gps_y[0], color='green', s=150, marker='o',
                                       zorder=15, edgecolors='white', linewidth=3)

                        # End point
                        ax_loops.scatter(gps_x[-1], gps_y[-1], color='red', s=150, marker='s',
                                       zorder=15, edgecolors='white', linewidth=3)

                        # Delivery location
                        if 'delivering' in phases:
                            delivery_idx = phases.index('delivering')
                            if delivery_idx < len(gps_x):
                                ax_loops.scatter(gps_x[delivery_idx], gps_y[delivery_idx],
                                               color='orange', s=120, marker='^', zorder=16,
                                               edgecolors='black', linewidth=2)

                        # Add agent ID label
                        ax_loops.text(gps_x[-1] + 3, gps_y[-1] + 3, agent['agent_id'], fontsize=12,
                                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

                    # Check completion status
                    if agent.get('phase') == 'completed':
                        completed_loops += 1
                        ax_loops.scatter(gps_x[-1], gps_y[-1], color='green', s=200, marker='*',
                                       zorder=20, edgecolors='black', linewidth=3)
                    else:
                        partial_loops += 1
                        ax_loops.scatter(gps_x[-1], gps_y[-1], color='orange', s=150, marker='D',
                                       zorder=19, edgecolors='black', linewidth=2)

    # Add comprehensive legend and statistics
    stats_text = f"Completed Loops: {completed_loops}/{num_agents}\nPartial Trails: {partial_loops}/{num_agents}"
    ax_loops.text(0.02, 0.98, stats_text, transform=ax_loops.transAxes, fontsize=16,
                 verticalalignment='top',
                 bbox=dict(boxstyle="round,pad=0.6", facecolor='white', alpha=0.9))

    # Enhanced legend
    legend_elements = [
        plt.Line2D([0], [0], color='blue', linewidth=3, label='Outward Journey'),
        plt.Line2D([0], [0], color='orange', linewidth=3, label='Delivery Phase'),
        plt.Line2D([0], [0], color='green', linewidth=3, label='Return Journey'),
        plt.Line2D([0], [0], marker='o', color='green', markersize=10, label='Start (Gate)'),
        plt.Line2D([0], [0], marker='s', color='red', markersize=10, label='Current Position'),
        plt.Line2D([0], [0], marker='^', color='orange', markersize=10, label='Delivery Point'),
        plt.Line2D([0], [0], marker='*', color='green', markersize=15, label='Completed Loop'),
        plt.Line2D([0], [0], marker='D', color='orange', markersize=10, label='Partial Trail')
    ]

    ax_loops.legend(handles=legend_elements, loc='upper right', fontsize=12)
    ax_loops.grid(True, alpha=0.3)
    plt.tight_layout()

    print(f"GPS Loop Visualization: {completed_loops} agents completed full loops, {partial_loops} partial trails")
    return fig_loops

# Generate additional analysis plots
def create_individual_agent_analysis():
    """Create individual analysis plots for each agent showing their complete journey"""
    completed_agents = [a for a in agents if a.get('phase') == 'completed']

    if not completed_agents:
        print("No completed agents for individual analysis")
        return

    n_agents = len(completed_agents)
    cols = min(3, n_agents)
    rows = (n_agents + cols - 1) // cols

    fig_individual, axes = plt.subplots(rows, cols, figsize=(6*cols, 5*rows))
    if n_agents == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, (agent, ax) in enumerate(zip(completed_agents, axes)):
        if idx >= len(axes):
            break

        # Plot individual agent journey using complete journey history
        tracker = agent['gps_tracker']
        if hasattr(tracker, 'complete_journey_history') and tracker.complete_journey_history:
            journey_data = tracker.complete_journey_history

            if len(journey_data) > 1:
                positions = [point['estimated_position'] for point in journey_data]
                phases = [point.get('phase', 'unknown') for point in journey_data]

                if len(positions) > 5:
                    gps_x, gps_y = zip(*positions)

                    # Color code by phase
                    for i in range(len(gps_x)-1):
                        phase = phases[i]
                        if phase == 'outward':
                            color = 'blue'
                        elif phase == 'delivering':
                            color = 'orange'
                        elif phase == 'returning':
                            color = 'green'
                        else:
                            color = 'gray'

                        ax.plot([gps_x[i], gps_x[i+1]], [gps_y[i], gps_y[i+1]],
                               '-', color=color, alpha=0.7, linewidth=2)

        # Draw simplified campus map
        for n, p in nodes.items():
            color = '#28a745' if 'Gate' in n else '#007bff' if 'Hub' in n else '#dc3545'
            ax.scatter(p[0], p[1], c=color, s=50, alpha=0.7, zorder=2)
            ax.text(p[0] + 1, p[1] + 1, n, fontsize=8, zorder=3)

        ax.set_xlim(-10, 90)
        ax.set_ylim(-60, 60)
        ax.set_title(f"{agent['agent_id']} - Complete Journey", fontsize=12)
        ax.grid(True, alpha=0.3)

        # Add performance metrics
        time_saved = agent.get('time_saved', 0)
        efficiency = agent.get('efficiency_gain', 0)
        gps_error = calculate_gps_accuracy(agent)

        metrics_text = f"Time Saved: {time_saved:.1f}s\nEfficiency: {efficiency:.1f}%\nGPS Error: {gps_error:.1f}u"
        ax.text(0.02, 0.98, metrics_text, transform=ax.transAxes, fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))

    for idx in range(len(completed_agents), len(axes)):
        axes[idx].set_visible(False)

    plt.tight_layout()
    return fig_individual

# After the animation completes, show the complete loop visualization
print("\nGenerating complete GPS loop visualization...")
complete_loops_fig = visualize_complete_gps_loops()
plt.show()

print("\nGenerating individual agent analysis...")
individual_analysis_fig = create_individual_agent_analysis()
if individual_analysis_fig:
    plt.show()

print("\n" + "="*80)
print("ENHANCED ANALYSIS: COMPLETE GPS LOOP TRACKING WITH ALGORITHM VISUALIZATION")
print("="*80)

completed_agents = [a for a in agents if a.get('phase') == 'completed']
if completed_agents:
    total_conventional_time = sum(a['conventional_time'] for a in completed_agents)
    total_optimized_time = sum(a['total_optimized_time'] for a in completed_agents)
    total_time_saved = sum(a['time_saved'] for a in completed_agents if a['time_saved'] is not None)
    avg_conventional_time = total_conventional_time / len(completed_agents)
    avg_optimized_time = total_optimized_time / len(completed_agents)
    avg_time_saved = total_time_saved / len(completed_agents)

    if total_conventional_time > 0:
        overall_efficiency_gain = ((total_conventional_time - total_optimized_time) / total_conventional_time) * 100
    else:
        overall_efficiency_gain = 0

    print(f"\nOVERALL STATISTICS ({len(completed_agents)}/{num_agents} loops completed):")
    print(f"  Total Conventional Time: {total_conventional_time:.1f} steps")
    print(f"  Total Optimized Time: {total_optimized_time:.1f} steps")
    print(f"  Total Time Saved: {total_time_saved:.1f} steps")
    print(f"  Average Conventional Time: {avg_conventional_time:.1f} steps")
    print(f"  Average Optimized Time: {avg_optimized_time:.1f} steps")
    print(f"  Average Time Saved: {avg_time_saved:.1f} steps")
    print(f"  Overall Efficiency Gain: {overall_efficiency_gain:.1f}%")

    # Enhanced GPS tracking analysis
    print(f"\nENHANCED GPS TRACKING ANALYSIS:")
    all_gps_errors = []
    for agent in agents:
        accuracy = calculate_gps_accuracy(agent)
        if accuracy > 0:
            all_gps_errors.append(accuracy)

    if all_gps_errors:
        avg_gps_error = np.mean(all_gps_errors)
        max_gps_error = max(all_gps_errors)
    else:
        avg_gps_error = 0
        max_gps_error = 0

    print(f"  Average GPS Error: {avg_gps_error:.2f} units")
    print(f"  Maximum GPS Error: {max_gps_error:.2f} units")

    if avg_gps_error < 2.0:
        accuracy_rating = "Excellent"
    elif avg_gps_error < 4.0:
        accuracy_rating = "Good"
    elif avg_gps_error < 6.0:
        accuracy_rating = "Fair"
    else:
        accuracy_rating = "Poor"

    print(f"  Tracking Quality: {accuracy_rating}")

    print(f"\nKEY IMPROVEMENTS:")
    print("  ✓ Complete journey tracking with phase information")
    print("  ✓ Real-time efficiency gain visualization")
    print("  ✓ Enhanced GPS loop visualization showing full paths")
    print("  ✓ Individual agent journey analysis")
    print("  ✓ Phase-based coloring in GPS trails")
    print("  ✓ Improved animation performance with optimized frame count")

    print(f"\nVISUALIZATION FEATURES:")
    print("  • Complete GPS trails showing entry → delivery → return paths")
    print("  • Phase-based coloring (Blue: Outward, Orange: Delivery, Green: Return)")
    print("  • Individual agent journey analysis plots")
    print("  • Real-time efficiency tracking in statistics")
    print("  • Enhanced algorithm visualization")
    print("  • Larger plots with better visibility")

else:
    print("No delivery loops completed within simulation time")

print("\n" + "="*80)
print("COMPLETE GPS LOOP TRACKING SIMULATION WITH ENHANCED VISUALIZATION COMPLETED")
print("="*80)