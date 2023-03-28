import numpy as np
import matplotlib.pyplot as plt
import random

class FastSlam:
    def __init__(self, num_particles, num_landmarks, motion_model, sensor_model):
        self.particles = [Particle(num_landmarks, motion_model, sensor_model) for _ in range(num_particles)]
        self.num_particles = num_particles

    def predict(self, u):
        for particle in self.particles:
            particle.pose = particle.motion_model.update(particle.pose, u)

    def update(self, u, z):
        # Predict
        self.predict(u)

        # Update
        for particle in self.particles:
            particle.update(u, z)

        # Normalize weights
        w = [particle.weight for particle in self.particles]
        self.particles = [particle for _, particle in sorted(zip(w, self.particles), key=lambda x: x[0], reverse=True)]
        w = [particle.weight for particle in self.particles]
        w_norm = w / sum(w)
        for i, particle in enumerate(self.particles):
            particle.weight = w_norm[i]
        
    def resample(self):
        # Low variance sampler
        r = np.random.uniform(0, 1.0 / self.num_particles)
        c = self.particles[0].weight
        i = 0
        for m in range(self.num_particles):
            U = r + m * 1.0 / self.num_particles
            while U > c:
                i += 1
                c += self.particles[i].weight
        return i

class Particle:
    def __init__(self, num_landmarks, motion_model, sensor_model):
        self.pose = np.zeros(3, dtype=float)  # x, y, theta
        self.landmarks = [Landmark(idx) for idx in range(num_landmarks)]
        self.weight = 1.0
        self.motion_model = motion_model
        self.sensor_model = sensor_model
        self.poses = []  # New attribute: list of poses

    def update(self, u, z):
        for zi in z:
            landmark = self.landmarks[zi[0]]
            landmark.update(self.pose, u, z)  # Fix: pass zi as a list instead of a tuple
        self.weight = self.sensor_model.likelihood(self.pose, self.landmarks, z)
        self.poses.append(self.pose)  # New: append current pose to list of poses

    def copy(self):
        particle = Particle()
        particle.pose = self.pose.copy()
        particle.weight = self.weight
        particle.landmarks = {k: v.copy() for k, v in self.landmarks.items()}
        return particle


class Landmark:
    def __init__(self, idx):
        self.idx = idx
        self.mean = np.zeros(2)  # x, y
        self.cov = np.zeros((2, 2))  # Covariance matrix

    @property
    def position(self):  # New attribute: position (mean of Gaussian distribution)
        return self.mean

    def observe(self, pose, u, z):
        dx = z[3] * np.cos(pose[2] + z[2]) - self.mean[0]
        dy = z[3] * np.sin(pose[2] + z[2]) - self.mean[1]
        return dx, dy

    def jacobian(self, pose, u, z):
        dx = z[1] * np.cos(pose[2] + z[0]) - self.mean[0]
        dy = z[1] * np.sin(pose[2] + z[0]) - self.mean[1]
        G = np.array([[1.0, 0.0, -z[1] * np.sin(pose[2] + z[0])],
                    [0.0, 1.0, z[1] * np.cos(pose[2] + z[0])]])
        return G

    def update(self, u, z):
        # Predict
        self.predict(u)

        # Update
        for particle in self.particles:
            particle.update(u, z)

        # Normalize weights
        w = [particle.weight for particle in self.particles]
        self.particles = [particle for _, particle in sorted(zip(w, self.particles), key=lambda x: x[0], reverse=True)]
        w = [particle.weight for particle in self.particles]
        w_norm = w / sum(w)
        for i, particle in enumerate(self.particles):
            particle.weight = w_norm[i]

    def copy(self):
        landmark = Landmark(self.idx)  # Fix: pass idx argument to __init__ method
        landmark.mean = self.mean.copy()
        landmark.cov = self.cov.copy()
        return landmark


class MotionModel:
    def __init__(self, alpha1, alpha2, alpha3, alpha4):
        # Noise parameters for odometry motion model
        self.alpha1 = alpha1
        self.alpha2 = alpha2
        self.alpha3 = alpha3
        self.alpha4 = alpha4
    
    def update(self, pose, u):
        # Implement odometry motion model
        delta_rot1 = u[1] + np.random.normal(0, self.alpha1 * abs(u[1]) + self.alpha2 * u[0])
        delta_trans = u[0] + np.random.normal(0, self.alpha3 * u[0] + self.alpha4 * (abs(u[1]) + abs(u[2])))
        delta_rot2 = u[1] + np.random.normal(0, self.alpha1 * abs(u[1]) + self.alpha2 * u[0])

        theta = pose[2] + delta_rot1
        x = pose[0] + delta_trans * np.cos(theta)
        y = pose[1] + delta_trans * np.sin(theta)
        theta += delta_rot2

        return np.array([x, y, theta])

class SensorModel:
    def __init__(self, sigma = 0.1):
        # Noise parameter for sensor model
        self.sigma = sigma
    
    def gaussian(self, mu, sigma, x):  # New method: Gaussian function
        return np.exp(- ((mu - x) ** 2) / (sigma ** 2) / 2.0) / np.sqrt(2.0 * np.pi * (sigma ** 2))

    def likelihood(self, pose, landmarks, z):
        likelihood = 1.0
        for zi in z:
            landmark = landmarks[zi[0]]
            true_distance = np.linalg.norm(landmark.position - pose[:2])  # Fix: use position attribute of landmark
            observed_distance = zi[1]
            likelihood *= self.gaussian(true_distance, self.sigma, observed_distance)
        return likelihood

def main():
    # Create random map with landmarks
    num_landmarks = 10
    landmarks = [(random.uniform(-5, 5), random.uniform(-5, 5)) for _ in range(num_landmarks)]

    # Initialize FastSLAM
    num_particles = 100
    alpha1 = 0.1  # Noise parameter for odometry motion model
    alpha2 = 0.1
    alpha3 = 0.1
    alpha4 = 0.1
    motion_model = MotionModel(alpha1, alpha2, alpha3, alpha4)
    sensor_model = SensorModel()  # Assume perfect sensor
    fastslam = FastSlam(num_particles, num_landmarks, motion_model, sensor_model)

    # Initialize robot pose
    pose = np.zeros(3)  # x, y, theta
    poses = []  # Record of robot poses

    # Simulate robot motion
    fig, ax = plt.subplots()
    plt.scatter([lm[0] for lm in landmarks], [lm[1] for lm in landmarks], marker='x', color='k')
    while True:
        u = (random.uniform(0, 1), random.uniform(-np.pi/6, np.pi/6), random.uniform(-np.pi/6, np.pi/6))  # Control inputs: distance traveled, change in heading in first and second time steps
        z = []  # Measurements
        for lm in landmarks:
            # Simulate measurement
            z.append((0, np.linalg.norm(lm - pose[:2]) + np.random.normal(0, 0.1)))
        fastslam.update(u, z)

        # Update robot pose
        pose = motion_model.update(pose, u)
        poses.append(pose.copy())

        # Plot paths of top 5 particles
        weights = [particle.weight for particle in fastslam.particles]
        top_particles = sorted(fastslam.particles, key=lambda p: p.weight, reverse=True)[:5]
        for particle in top_particles:
            ax.plot([p[0] for p in particle.poses], [p[1] for p in particle.poses], 'b-')

        # Plot robot path
        ax.plot([p[0] for p in poses], [p[1] for p in poses], 'r-')
        plt.xlim((min([p[0] for p in poses]) - 1, max([p[0] for p in poses]) + 1))
        plt.ylim((min([p[1] for p in poses]) - 1, max([p[1] for p in poses]) + 1))
        plt.title('FastSLAM')
        plt.pause(0.001)
        ax.clear()

if __name__ == '__main__':
    main()