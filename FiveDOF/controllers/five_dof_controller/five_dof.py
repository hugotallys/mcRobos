import numpy as np

from controller import Supervisor  # type: ignore


class FiveDOF:

    def __init__(self) -> None:
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())

        self.joints = [
            self.robot.getDevice(f"joint{i}") for i in range(1, 6)
        ]

        self.joints_ranges = [
            (
                joint.getMinPosition(), joint.getMaxPosition()
            ) for joint in self.joints
        ]

        self.sensors = [
            self.robot.getDevice(f"sensor{i}") for i in range(1, 6)
        ]

        self.dummy = self.robot.getFromDef("DUMMY")

        self.TB0 = self.dh_transform(
            theta=0., d=0.11, a=0., alpha=0.
        )
        self.TV = self.dh_transform(
            theta=np.pi/2, d=0., a=-0.0275, alpha=np.pi/2
        )

        for s in self.sensors:
            s.enable(self.timestep)

    def dh_transform(self, theta, d, a, alpha, offset=0.):
        theta = theta + offset
        ct, st = np.cos(theta), np.sin(theta)
        ca, sa = np.cos(alpha), np.sin(alpha)
        return np.array([
            ct, -st*ca, st*sa, a*ct,
            st, ct*ca, -ct*sa, a*st,
            0., sa, ca, d,
            0., 0., 0., 1.
        ]).reshape(4, 4)

    def T01(self, q1):
        return self.dh_transform(
            theta=q1, d=0., a=0., alpha=-np.pi/2
        )

    def T12(self, q2):
        return self.dh_transform(
            theta=q2, d=0., a=0.1175, alpha=0.
        )

    def T23(self, q3):
        return self.dh_transform(
            theta=q3, d=0., a=0.096, alpha=0.
        )

    def T34(self, q4):
        return self.dh_transform(
            theta=q4, d=0., a=0.065, alpha=0.
        )

    def T45(self, q5):
        return self.dh_transform(
            theta=q5, d=0., a=0., alpha=0.
        )

    @staticmethod
    def pose(r):
        return r[0:3, -1].reshape(3, 1)

    def fkine(self, q, pose=False):
        r = (
            self.TB0 @ self.T01(q[0]) @
            self.T12(q[1]) @ self.T23(q[2]) @
            self.T34(q[3]) @ self.TV @ self.T45(q[4])
        )
        return self.pose(r) if pose else r

    def jacobian(self, q):

        P1 = self.T01(q[0])
        P2 = P1 @ self.T12(q[1])
        P3 = P2 @ self.T23(q[2])
        P4 = P3 @ self.T34(q[3])
        P5 = P4 @ self.T45(q[4])

        PE = P5[0:3, -1]

        JO1 = np.array([0., 0., 1.]).T
        JP1 = np.cross(JO1, PE)

        JO2 = P1[0:3, 2]
        JP2 = np.cross(JO2, PE - P1[0:3, -1])

        JO3 = P2[0:3, 2]
        JP3 = np.cross(JO3, PE - P2[0:3, -1])

        JO4 = P3[0:3, 2]
        JP4 = np.cross(JO4, PE - P3[0:3, -1])

        JO5 = P4[0:3, 2]
        JP5 = np.cross(JO5, PE - P4[0:3, -1])

        return np.array([JP1, JP2, JP3, JP4, JP5]).T

    def setJointPositions(self, q):
        for i, joint in enumerate(self.joints):
            joint.setPosition(q[i])

    def getJointPositions(self):
        return np.array(
            list(map(lambda s: s.getValue(), self.sensors))
        )

    def step(self):
        return self.robot.step(self.timestep)

    def getDummyPos(self):
        return np.array(
            self.dummy.getPosition()
        ).reshape(3, 1)

    def delay(self, ms):
        counter = ms / self.timestep
        while (counter > 0) and (self.step() != -1):
            counter -= 1

    def joint_distance(self, q):
        n = len(q)
        values = []
        for i in range(5):
            q_min, q_max = self.joints_ranges[i][0], self.joints_ranges[i][1]
            q_mean = 0.5*(q_max + q_min)
            values.append(
                (q[i] - q_mean) / (q_max - q_min)
            )
        return (-1. / (2*n)) * (np.array(values)**2).sum()

    def q0dot(self, q, k0=1.0):
        n = len(q)
        values = []
        for i in range(5):
            q_min, q_max = self.joints_ranges[i][0], self.joints_ranges[i][1]
            q_mean = 0.5*(q_max + q_min)
            values.append(
                (q[i] - q_mean) / (q_max - q_min)**2
            )
        return (-k0 / n) * (np.array(values)).reshape(n, 1)
