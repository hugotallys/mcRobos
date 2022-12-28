import numpy as np

from five_dof import FiveDOF
from matplotlib import pyplot as plt

MAX_ITER = 500
ERROR = 0.001

if __name__ == "__main__":

    robot = FiveDOF()

    # Initial joint configuration
    q0 = np.array([0., -2.199, 1.948, 0.754, 0.])
    robot.setJointPositions(q0)
    robot.delay(1500)

    joints_values = []
    position_values = []
    joint_distance_values = []
    absolute_error_values = []

    dt = robot.getTimeStep()
    x0 = robot.fkine(q0, pose=True)
    x = x0

    iter_ = 0

    while robot.step() != -1:

        dx = robot.getDummyPos() - x

        error = np.linalg.norm(dx)

        if (iter_ > MAX_ITER) and (error < ERROR):
            break

        q = robot.getJointPositions()
        jac = robot.jacobian(q)
        jac_pinv = np.linalg.pinv(jac)
        q0dot = robot.q0dot(q, k0=150)

        dq = jac_pinv @ dx + (np.eye(5) - jac_pinv @ jac) @ q0dot
        q = q + dq.flatten() * dt
        robot.setJointPositions(q)
        x = robot.fkine(q, pose=True)
        joint_dist = robot.joint_distance(q)

        joints_values.append(q)
        joint_distance_values.append(joint_dist)
        position_values.append(x)
        absolute_error_values.append(error)

        iter_ += 1

    t = np.arange(0, np.array(joints_values).shape[0]*dt, dt)

    fig, axs = plt.subplots(2, 2)
    fig.set_size_inches(18.5, 10.5)

    joints_values = np.array(joints_values)
    position_values = np.array(position_values)

    for i in range(5):
        axs[0, 0].plot(t, joints_values[:, i], label=str(i+1))
    axs[0, 0].legend(loc=(1.04, 0))
    axs[0, 0].set_title('Joint space')

    axs[0, 1].plot(t, joint_distance_values, 'tab:blue')
    axs[0, 1].set_title('Distance from joint limits')

    labels = ['x', 'y', 'z']
    for i in range(3):
        axs[1, 0].plot(t, position_values[:, i], label=labels[i])
    axs[1, 0].legend(loc=(1.04, 0))
    axs[1, 0].set_title('End effector position')

    axs[1, 1].plot(t, absolute_error_values, 'tab:red')
    axs[1, 1].set_title('Position absolute error')

    for ax in axs.flat:
        ax.set(xlabel='time (s)', ylabel='')
        ax.grid()

    axs[0, 0].set(xlabel='', ylabel='')
    axs[0, 1].set(xlabel='', ylabel='')

    axs[0, 1].yaxis.tick_right()
    axs[1, 1].yaxis.tick_right()

    plt.show()
