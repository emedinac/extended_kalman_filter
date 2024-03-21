import datetime

import numpy as np
import pandas as pd
import pymap3d as pm

# references:
# Probabilistic Robotics was pretty hard to understand, so I read other sources :DI
# [1] https://arxiv.org/pdf/1204.0375.pdf
# [2] https://medium.com/@ab.jannatpour/kalman-filter-with-python-code-98641017a2bd
# [3] https://machinelearningspace.com/2d-object-tracking-using-kalman-filter/
# [4]

'''
# get from the format data.
lat:   latitude of the oxts-unit (deg)
lon:   longitude of the oxts-unit (deg)
alt:   altitude of the oxts-unit (m)
roll:  roll angle (rad),    0 = level, positive = left side up,      range: -pi   .. +pi
pitch: pitch angle (rad),   0 = level, positive = front down,        range: -pi/2 .. +pi/2
yaw:   heading (rad),       0 = east,  positive = counter clockwise, range: -pi   .. +pi
vn:    velocity towards north (m/s)
ve:    velocity towards east (m/s)
vf:    forward velocity, i.e. parallel to earth-surface (m/s)
vl:    leftward velocity, i.e. parallel to earth-surface (m/s)
vu:    upward velocity, i.e. perpendicular to earth-surface (m/s)
ax:    acceleration in x, i.e. in direction of vehicle front (m/s^2)
ay:    acceleration in y, i.e. in direction of vehicle left (m/s^2)
ay:    acceleration in z, i.e. in direction of vehicle top (m/s^2)
af:    forward acceleration (m/s^2)
al:    leftward acceleration (m/s^2)
au:    upward acceleration (m/s^2)
wx:    angular rate around x (rad/s)
wy:    angular rate around y (rad/s)
wz:    angular rate around z (rad/s)
wf:    angular rate around forward axis (rad/s)
wl:    angular rate around leftward axis (rad/s)
wu:    angular rate around upward axis (rad/s)
pos_accuracy:  velocity accuracy (north/east in m)
vel_accuracy:  velocity accuracy (north/east in m/s)
navstat:       navigation status (see navstat_to_string)
numsats:       number of satellites tracked by primary GPS receiver
posmode:       position mode of primary GPS receiver (see gps_mode_to_string)
velmode:       velocity mode of primary GPS receiver (see gps_mode_to_string)
orimode:       orientation mode of primary GPS receiver (see gps_mode_to_string)
'''


class IMUDataLoader():
    def __init__(self, path, idx, seed=23456):
        np.random.seed(seed)
        self.path = path
        self.idx = idx
        self.data = pd.read_csv(f'{self.path}/oxts_{self.idx:02d}.txt', sep=" ")
        self.features = open(f'{self.path}/dataformat.txt', 'r').readlines()
        self.features = [f.split(":")[0] for f in self.features]
        self.data.columns = self.features

        self.timestamp = open(f'{self.path}/timestamps_oxts_{self.idx:02d}.txt', 'r').readlines()
        self.timestamp = [datetime.datetime.strptime(t[:-4], '%Y-%m-%d %H:%M:%S.%f') for t in self.timestamp]

        self.timestamp = np.array(self.timestamp) - self.timestamp[0]
        self.timestamp = [t.total_seconds() for t in self.timestamp]

    def set_robot_data(self, features):
        x_ = self.data[features].values
        return x_

    def get_feature(self, f):
        return self.data[f].values

    @staticmethod
    def lla2xyz(data, idx_ref=0):
        lon, lat, h = data.T
        lon0, lat0, h0 = data[idx_ref]
        xyz = np.array(pm.geodetic2enu(lat, lon, h, lat0, lon0, h0)).T
        return xyz

    @staticmethod
    def _normalize_angles(angles):
        return np.arctan2(np.sin(angles), np.cos(angles))

    @staticmethod
    def _add_normal_noise(data: np.array, std: (float, int)) -> np.array:
        return data + np.random.normal(0.0, std, data.shape)

    @staticmethod
    def _add_uniform_noise(data: np.array, std: (float, int)) -> np.array:
        return data + np.random.uniform(0.0, std, data.shape)


class ExtendedKalmanFilter:
    def __init__(self, x, xyz_std, yaw_std, forward_velocity_std, yaw_rate_std):
        """
        Args:
            x: state to estimate: (x, y, yaw)
        """
        self.x = x  # X,Y,YAW we could add another angle too.

        self.P = np.array([xyz_std, xyz_std, yaw_std]) ** 2.
        self.P = np.diag(self.P)

        self.MQ = np.array([xyz_std, xyz_std]) ** 2
        self.MQ = np.diag(self.MQ)

        self.MR = np.array([forward_velocity_std, forward_velocity_std, yaw_rate_std]) ** 2.
        self.MR = np.diag(self.MR)

        self.MH = np.eye(len(x) - 1, len(x))  # Jacobian of observations.

    def prediction(self, u, dt):
        """propagate x and P based on state transition model defined as eq. (5.9) in [1]
        Args:
            u (numpy.array): control input: [v, omega]^T
            dt (float): time interval in second
            R (numpy.array): state transition noise covariance
        """
        # predict state x and variance P
        x, y, theta = self.x
        v, omega = u
        radius = v / omega  # turning radius

        dtheta = omega * dt
        dx = - radius * np.sin(theta) + radius * np.sin(theta + dtheta)
        dy = + radius * np.cos(theta) - radius * np.cos(theta + dtheta)
        self.x += np.array([dx, dy, dtheta])

        # predict covariance P
        dx = - radius * np.cos(theta) + radius * np.cos(theta + dtheta)
        dy = - radius * np.sin(theta) + radius * np.sin(theta + dtheta)
        G = np.array([[1., 0., dx],
                      [0., 1., dy],
                      [0., 0., 1.]])  # Jacobian of state transition
        self.P = G @ self.P @ G.T + self.MR

    def correction(self, z):
        """
        Args:
            z: XYZ observation [x, y]
        """
        # Kalman gain - tricky to find it....
        K = self.P @ self.MH.T @ np.linalg.inv(self.MH @ self.P @ self.MH.T + self.MQ)

        # update state x
        x, y, theta = self.x
        z_ = np.array([x, y])  # observation from estimated state
        self.x = self.x + K @ (z - z_)
        # update covariance P
        self.P = self.P - K @ self.MH @ self.P
