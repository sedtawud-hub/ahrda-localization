import random
import pandas as pd
import math
import numpy as np
import matplotlib.pyplot as plt


def get_dist(p1, p2):
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def get_sensor_data(cur_pose, map):
    LIDAR_RANGE = 20
    data = []  # dist, theta
    for point in map:
        dist = get_dist((cur_pose[0, 0], cur_pose[1, 0]), point)
        if dist <= LIDAR_RANGE:
            theta = math.atan2(point[1] - cur_pose[1, 0], point[0] - cur_pose[0, 0]) + cur_pose[2, 0]
            dist_noise = dist + random.uniform(-1, 1) * 0.2
            data.append([dist_noise, theta])
    data = data[1::5]
    xy_sensor = []
    for dist, theta in data:
        Tranformation_Matrix = [
            [math.cos(cur_pose[2, 0]), math.sin(cur_pose[2, 0]), cur_pose[0, 0]],
            [-math.sin(cur_pose[2, 0]), math.cos(cur_pose[2, 0]), cur_pose[1, 0]],
            [0, 0, 1],
        ]
        x, y = dist * math.cos(theta), dist * math.sin(theta)
        xy_sensor.append(np.dot(Tranformation_Matrix, np.array([x, y, 1]).T))

    return np.array(xy_sensor)[:, 0:2]


def add_input_noise(u, R):
    ud1 = u[0] + random.uniform(-1.0, 1) * R[0, 0] ** 0.5
    ud2 = u[1] + random.uniform(-1.0, 1) * R[1, 1] ** 0.5
    return np.array([ud1, ud2])


def motion_model(x, u):
    """
    param:
        state: [[x], [y], [yaw], [v]]
        input: [[v],[yaw_rate]]
    return:
        state: [[x], [y], [yaw], [v]]
    """
    F = np.array([[1.0, 0, 0, 0], [0, 1.0, 0, 0], [0, 0, 1.0, 0], [0, 0, 0, 0]])

    B = np.array([[DT * math.cos(x[2, 0]), 0], [DT * math.sin(x[2, 0]), 0], [0.0, DT], [1.0, 0.0]])

    x = F.dot(x) + B.dot(u)
    #
    # x = 1, 0, 0, 0   x    dt*cos(r), 0      v
    # y   0, 1, 0, 0 X y +  dt*sin(r), 0   X  Yaw_rate
    # r   0, 0, 1, 0   r    0,         dt
    #     0, 0, 0, 0   1    1,         0
    return x


def pf_localization(prev_pose, cur_pose, px, pw, z, u, R, Q, NP):
    """
    Localization with Particle filter
    """

    for ip in range(NP):
        x = np.array([px[:, ip]]).T
        w = pw[0, ip]
        ud = add_input_noise(u, R)
        x = motion_model(prev_pose, ud)  # predict particle state some model use random state
        plt.plot(x[0, 0], x[1, 0], ".r", markersize=1)
        #  Calc Importance Weight
        for z_point in z:
            particle_dx = x[0, 0] - z_point[0]
            particle_dy = x[1, 0] - z_point[1]
            ego_pose_dx = cur_pose[0, 0] - z_point[0]
            ego_pose_dy = cur_pose[0, 0] - z_point[1]
            dist_particle = math.hypot(particle_dx, particle_dy)
            dist_ego_pose = math.hypot(ego_pose_dx, ego_pose_dy)
            dr = dist_particle - dist_ego_pose
            w += 1 / math.exp(dr)

        px[:, ip] = x[:, 0]
        pw[0, ip] = w
    pw = pw / pw.sum()  # normalize
    x_est = px.dot(pw.T)  # estimate = sum of weighted state

    N_eff = 1.0 / (pw.dot(pw.T))[0, 0]  # Effective particle number

    if N_eff < NTh:
        px, pw = re_sampling(px, pw, NP)
    return x_est, px, pw


def re_sampling(px, pw, NP):
    """
    low variance re-sampling
    """

    w_cum = np.cumsum(pw)
    base = np.arange(0.0, 1.0, 1 / NP)
    re_sample_id = base + np.random.uniform(0, 1 / NP)
    indexes = []
    ind = 0
    for ip in range(NP):
        while re_sample_id[ip] > w_cum[ind]:
            ind += 1
        indexes.append(ind)
    px = px[:, indexes]
    pw = np.zeros((1, NP)) + 1.0 / NP  # init weight

    return px, pw


if __name__ == "__main__":
    # Estimation parameter of PF
    Q = np.diag([0.2]) ** 2  # range error
    R = np.diag([5, np.deg2rad(20.0)]) ** 2  # input error

    NP = 100  # number of particles
    NTh = NP / 2.0
    DT = 0.1
    map = pd.read_csv("./map.csv").to_numpy()
    px = np.zeros((4, NP))  # Particle store
    pw = np.zeros((1, NP)) + 1.0 / NP  # Particle weight
    cur_pose_T = np.zeros((4, 1))
    prev_pose = cur_pose_T
    u = np.array([[5, math.radians(0)]]).T
    h_x_t = cur_pose_T
    h_x_est = cur_pose_T
    for i in range(180):
        print(i)
        if i >= 110:
            u = np.array([[5, math.radians(-40)]]).T
        if i >= 132:
            u = np.array([[5, math.radians(0)]]).T
        plt.cla()
        xy = get_sensor_data(cur_pose=cur_pose_T, map=map)
        x_est, px, pw = pf_localization(prev_pose, cur_pose_T, px, pw, xy, u, R, Q, NP)
        plt.title(
            f"est_pose: {x_est[0, 0]:.2f},{x_est[1, 0]:.2f} act_pose: {cur_pose_T[0, 0]:.2f},{ cur_pose_T[1, 0]:.2f}"
        )
        h_x_t = np.hstack((h_x_t, cur_pose_T))
        h_x_est = np.hstack((h_x_est, x_est))
        plt.plot(map[:, 0], map[:, 1], ".k", markersize=0.5)
        plt.plot(xy[:, 0], xy[:, 1], "+r", markersize=3)
        plt.plot(x_est[0, 0], x_est[1, 0], "og", markersize=7)
        plt.plot(cur_pose_T[0, 0], cur_pose_T[1, 0], "xb", markersize=10)
        plt.pause(1 / 30)
        prev_pose = cur_pose_T
        cur_pose_T = motion_model(cur_pose_T, u)
    plt.cla()
    plt.plot(map[:, 0], map[:, 1], ".k", markersize=0.5)
    plt.plot(np.array(h_x_t[0, :]).flatten(), np.array(h_x_t[1, :]).flatten(), "-b")
    plt.plot(np.array(h_x_est[0, :]).flatten(), np.array(h_x_est[1, :]).flatten(), "-g")
    plt.show()
