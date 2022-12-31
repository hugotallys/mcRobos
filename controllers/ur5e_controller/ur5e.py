import numpy as np

from controller import Supervisor  # type: ignore


class UR5e:

    def __init__(self) -> None:
        self.robot = Supervisor()
        self.timestep = int(self.robot.getBasicTimeStep())

        self.joints_names = [
            "shoulder_pan_joint",   # Joint 1
            "shoulder_lift_joint",  # Joint 2
            "elbow_joint",          # Joint 3
            "wrist_1_joint",        # Joint 4
            "wrist_2_joint",        # Joint 5
            "wrist_3_joint"         # Joint 6
        ]

        self.joints = [
            self.robot.getDevice(jname) for jname in self.joints_names
        ]

        self.joints_ranges = [
            (
                joint.getMinPosition(), joint.getMaxPosition()
            ) for joint in self.joints
        ]

        self.sensors = [
            self.robot.getDevice(jname + "_sensor") for jname in self.joints_names
        ]

        self.dummy = self.robot.getFromDef("DUMMY")

        for s in self.sensors:
            s.enable(self.timestep)

    @staticmethod
    def cross_product(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.cross(a,b)

    @staticmethod
    def pose(r):
        return r[0:3, -1].reshape(3, 1)

    @staticmethod
    def dh_transform(theta, d, a, alpha):
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
            theta=q1, d=0.1625, a=0., alpha=-np.pi/2
        )

    def T12(self, q2):
        return self.dh_transform(
            theta=q2, d=0., a=0.4250, alpha=0.
        )

    def T23(self, q3):
        return self.dh_transform(
            theta=q3, d=0., a=0.3922, alpha=0.
        )

    def T34(self, q4):
        return self.dh_transform(
            theta=q4, d=0.1333, a=0., alpha=-np.pi/2
        )

    def T45(self, q5):
        return self.dh_transform(
            theta=q5, d=0.0997, a=0., alpha=np.pi/2
        )
    
    def T56(self, q6):
        return self.dh_transform(
            theta=q6, d=0.0996, a=0., alpha=0.
        )

    def fkine(self, q, pose=False):
        r = (
            self.T01(q[0]) @ self.T12(q[1]) @ self.T23(q[2]) @
            self.T34(q[3]) @ self.T45(q[4]) @ self.T56(q[5])
        )
        return self.pose(r) if pose else r

    def jacobian(self, q):

        P1 = self.T01(q[0])
        P2 = P1 @ self.T12(q[1])
        P3 = P2 @ self.T23(q[2])
        P4 = P3 @ self.T34(q[3])
        P5 = P4 @ self.T45(q[4])
        P6 = P5 @ self.T56(q[5])

        PE = P5[0:3, -1]

        JO1 = np.array([0., 0., 1.]).T
        JP1 = self.cross_product(JO1, PE)

        JO2 = P1[0:3, 2]
        JP2 = self.cross_product(JO2, PE - P1[0:3, -1])

        JO3 = P2[0:3, 2]
        JP3 = self.cross_product(JO3, PE - P2[0:3, -1])

        JO4 = P3[0:3, 2]
        JP4 = self.cross_product(JO4, PE - P3[0:3, -1])

        JO5 = P4[0:3, 2]
        JP5 = self.cross_product(JO5, PE - P4[0:3, -1])

        JO6 = P5[0:3, 2]
        JP6 = self.cross_product(JO6, PE - P5[0:3, -1])

        return np.array([JP1, JP2, JP3, JP4, JP5, JP6]).T

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

    def getInitialPose(self):
        return [0.5*(min_q + max_q) for (min_q, max_q) in self.joints_ranges]

    def getTimeStep(self):
        return int(self.timestep) * 1e-3

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
        for i in range(6):
            q_min, q_max = self.joints_ranges[i][0], self.joints_ranges[i][1]
            q_mean = 0.5*(q_max + q_min)
            values.append(
                (q[i] - q_mean) / (q_max - q_min)**2
            )
        return (-k0 / n) * (np.array(values)).reshape(n, 1)
