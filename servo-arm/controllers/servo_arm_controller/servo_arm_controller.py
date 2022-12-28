"""servo_arm_controller controller."""

# You may need to import some classes of the controller module. Ex:
#  from controller import Robot, Motor, DistanceSensor

import numpy as np
from controller import Supervisor

EPS = 0.01
VOID_CONTROLLER = False


class ServoArm:

    def __init__(self) -> None:

        self.robot = Supervisor()

        self.timestep = int(self.robot.getBasicTimeStep())

        self.joints = [
            self.robot.getDevice("joint%d" % i) for i in range(1, 6)
        ]
        self.sensors = [
            self.robot.getDevice("sensor%d" % i) for i in range(1, 6)
        ]

        for s in self.sensors:
            s.enable(self.timestep)

        self.TB = self.dh_transform(np.pi/2, 0.0680, 0., 0.)
        self.virtual = self.dh_transform(np.pi/2, 0., 0., np.pi/2)
        self.vec_to_zero = np.vectorize(self.to_zero)

    def step(self):
        return self.robot.step(self.timestep)

    def setPositions(self, q):
        assert len(q) == len(self.joints)
        for i, pos in enumerate(q):
            self.joints[i].setPosition(pos)

    def getPositions(self):
        return [sensor.getValue() for sensor in self.sensors]

    def delay(self, ms):
        counter = ms / self.timestep
        while (counter > 0) and (self.step() != -1):
            counter -= 1

    @staticmethod
    def dh_transform(theta, d, a, alpha):
        return np.array([
            np.cos(theta), -np.sin(theta)*np.cos(alpha),
            np.sin(theta)*np.sin(alpha), a*np.cos(theta),
            np.sin(theta), np.cos(theta)*np.cos(alpha),
            -np.cos(theta)*np.sin(alpha), a*np.sin(theta),
            0., np.sin(alpha), np.cos(alpha), d,
            0., 0., 0., 1.
        ]).reshape(4, 4)
    
    @staticmethod
    def to_zero(x):
        if np.abs(x) < 0.01:
            return 0.
        return x
    
    def pose(self, r):
        r = self.vec_to_zero(r)
        x, y, z = r[0:3, -1]
        wz = np.arctan2(r[1, 0], r[0, 0])
        wy = np.arctan2(-r[2, 0], np.sqrt(r[2, 1]**2 + r[2, 2]**2))
        wx = np.arctan2(r[2, 1], r[2, 2])
        return np.array([x, y, z, wx, wy, wz]).reshape(6, 1)

    def T01(self, q1):
        return self.dh_transform(
            theta=q1, d=0., a=0.0192, alpha=np.pi/2
        )

    def T12(self, q2):
        return self.dh_transform(
            theta=q2, d=0., a=0.1028, alpha=0.
        )

    def T23(self, q3):
        return self.dh_transform(
            theta=q3, d=0., a=0.0747, alpha=0.
        )

    def T34(self, q4):
        return self.dh_transform(
            theta=q4, d=-0.0116, a=0.0688, alpha=0.
        )

    def T45(self, q5):
        return self.dh_transform(
            theta=q5, d=0.0541, a=0., alpha=0.
        )

    def fkine(self, q, pose=False):
        r = (
            self.TB @
            self.T01(q[0]) @ self.T12(q[1]) @
            self.T23(q[2]) @ self.T34(q[3]) @
            self.virtual @ self.T45(q[4])
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
        JPO1 = np.concatenate([JP1, JO1])

        JO2 = P1[0:3, 2]
        JP2 = np.cross(JO2, PE - P1[0:3, -1])
        JPO2 = np.concatenate([JP2, JO2])

        JO3 = P2[0:3, 2]
        JP3 = np.cross(JO3, PE - P2[0:3, -1])
        JPO3 = np.concatenate([JP3, JO3])

        JO4 = P3[0:3, 2]
        JP4 = np.cross(JO4, PE - P3[0:3, -1])
        JPO4 = np.concatenate([JP4, JO4])

        JO5 = P4[0:3, 2]
        JP5 = np.cross(JO5, PE - P4[0:3, -1])
        JPO5 = np.concatenate([JP5, JO5])

        return np.array([JPO1, JPO2, JPO3, JPO4, JPO5]).T


if __name__ == "__main__":

    np.set_printoptions(precision=4, suppress=True)

    servo_arm = ServoArm()

    begin_ball = servo_arm.robot.getFromDef("BEGIN")
    end_ball = servo_arm.robot.getFromDef("END")

    q0 = np.array([-1.257, 1.193, -1.822, 1.194, 0.])
    servo_arm.setPositions(q0)
    servo_arm.delay(1000)

    x0 = servo_arm.fkine(q0, pose=True)
    begin_ball.getField("translation").setSFVec3f(x0[0:3, 0].tolist())

    DX = 0.
    DY = 0.
    DZ = -0.05

    x1 = x0 + np.array([DX, DY, DZ, 0., 0., 0.]).reshape(6, 1)
    end_ball.getField("translation").setSFVec3f(x1[0:3, 0].tolist())

    x = x0

    dt = servo_arm.timestep * 1e-3

    # Main loop:
    # - perform simulation steps until Webots is stopping the controller
    while servo_arm.step() != -1:

        if VOID_CONTROLLER:
            continue

        if np.linalg.norm(x - x1) < EPS:
            break

        q = np.array(servo_arm.getPositions())
        dx = x1 - x
        jac = servo_arm.jacobian(q)
        dq = np.linalg.pinv(jac) @ dx
        q = q + dq.flatten() * dt
        servo_arm.setPositions(q)
        x = servo_arm.fkine(q, pose=True)

    # Enter here exit cleanup code.
