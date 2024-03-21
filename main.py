import argparse

import numpy as np

import algorithm as alg
import visualizer as plot


# Ref: Probabilistic Robotics book Pag 51 Section 3.3.2 Algorithm 3.3

def main(path, file_idx, plot_map):
    loader = alg.IMUDataLoader(path, file_idx)  # data [m] and timestamp [s]
    lla = loader.set_robot_data(["lon", "lat", "alt"])
    yaw = loader.get_feature("yaw")  # [rad]
    yaw_rate = loader.get_feature("wz")  # [rad/s]
    forward_velocity = loader.get_feature("vf")  # [m/s]

    # noise for measurements and observations and others.
    xyz_std = 5.0
    yaw_std = np.pi
    yaw_rate_std = 0.02
    forward_velocity_std = 0.3
    xyz = loader.lla2xyz(lla)  # lon, lat, alt
    measured_xyz = loader._add_normal_noise(xyz.copy(), xyz_std)
    measured_yaw = loader._add_normal_noise(yaw.copy(), yaw_std)
    measured_yaw_rate = loader._add_normal_noise(yaw_rate.copy(), yaw_rate_std)
    measured_forward_velocity = loader._add_normal_noise(forward_velocity.copy(), forward_velocity_std)

    # Algorithm: Extended Kalman Filter.
    mu_ = np.array([measured_xyz[0, 0], measured_xyz[0, 1], measured_yaw[0]])
    EKF = alg.ExtendedKalmanFilter(mu_, xyz_std, yaw_std, forward_velocity_std, yaw_rate_std)
    mus_ = [mu_]
    vars_ = [EKF.P]
    for idx in range(1, len(loader.data)):
        dt = loader.timestamp[idx] - loader.timestamp[idx - 1]
        u = np.array([measured_forward_velocity[idx], measured_yaw_rate[idx]])  # input control u = [v, theta]

        EKF.prediction(u, dt)
        EKF.correction(measured_xyz[idx][:2])  # z = [x, y]

        mus_.append([EKF.x[0], EKF.x[1], loader._normalize_angles(EKF.x[2])])  # X, Y, Theta
        vars_.append(EKF.P)

    mus_ = np.array(mus_)
    vars_ = np.array(vars_)

    print(f'\nMean Square Error (MSE): {((mus_ - xyz) ** 2).mean():.4f}\n\n')

    plot.plot_route(mus_, vars_, xyz, measured_xyz, std=3, filename=f'output/map{file_idx}.png')

    graphdata = plot.PlotOnMap(lla, measured_xyz)
    if plot_map:
        graphdata.visualize_on_map("map.html")
    else:
        graphdata.update_graph()
        graphdata._generate_image(filename=f'map_html{file_idx}.png', figsize=(12, 12))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='What the program does')

    parser.add_argument('--path', help="path to data")  # positional argument
    parser.add_argument('--idx', default=0)  # on/off flag
    parser.add_argument('--plot-map', action='store_true')  # on/off flag
    args = parser.parse_args()
    main(args.path, int(args.idx), args.plot_map)
    main(args.path, int(0), args.plot_map)
    main(args.path, int(1), args.plot_map)
    main(args.path, int(5), args.plot_map)
    main(args.path, int(6), args.plot_map)
    main(args.path, int(7), args.plot_map)
    main(args.path, int(8), args.plot_map)

