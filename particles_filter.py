import math
import random

class ParticlesFilter:
    def __init__(self, particles:int = 100, sensor:any,)


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