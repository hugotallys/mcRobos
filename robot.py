import roboticstoolbox as rtb
import numpy as np
import time
from zmqRemoteApi import RemoteAPIClient
import plots

from time import sleep

class ServoArm:

    def __init__(self) -> None:
        pass

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

    # | j   | theta   | d      | a      | alpha |
    # | --------------------------------------- |
    # | T01 | q1      | 0.1103 | 0.     | -90   |
    # | T12 | q2 - 90 | 0.     | 0.125  | 0.    |
    # | T23 | q3 + 90 | 0.     | 0.096  | 0.    |
    # | T34 | q4 + 90 | 0.     | -0.0275| 90    |
    # | T45 | q5      | 0.065  | 0.     | 0.    |

    def T01(self, q1):
        return self.dh_transform(
            theta=q1, d=0.1103, a=0., alpha=-np.pi/2
        )

    def T12(self, q2):
        return self.dh_transform(
            theta=q2, d=0., a=0.125, alpha=0., offset=-np.pi/2
        )

    def T23(self, q3):
        return self.dh_transform(
            theta=q3, d=0., a=0.096, alpha=0., offset=np.pi/2
        )

    def T34(self, q4):
        return self.dh_transform(
            theta=q4, d=0., a=-0.0275, alpha=np.pi/2, offset=np.pi/2
        )

    def T45(self, q5):
        return self.dh_transform(
            theta=q5, d=0.065, a=0., alpha=0.
        )

    @staticmethod
    def pose(r):
        x, y, z = r[0:3, -1]
        for i in range(3):
            for j in range(3):
                if (np.abs(r[i,j]) < 0.01):
                    r[i,j] = 0.0
        wz = np.arctan2(r[1, 0], r[0, 0])
        wy = np.arctan2(-r[2, 0], np.sqrt(r[2, 1]**2 + r[2, 2]**2))
        wx = np.arctan2(r[2, 1], r[2, 2])
        return np.array([x, y, z, wx, wy, wz]).reshape(6, 1)

    def fkine(self, q, pose=False):
        r = (
            self.T01(q[0]) @ self.T12(q[1]) @
            self.T23(q[2]) @ self.T34(q[3]) @
            self.T45(q[4])
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


def setJointPositions(q):

    for i, joint in enumerate(joints):
        sim.setJointTargetPosition(joint, q[i])


def getJointPositions():
    r = []
    for i, joint in enumerate(joints):
        r.append(
            sim.getJointPosition(joint)
        )
    return r

EPS = 0.01



if __name__ == "__main__":
  
    servo_arm = ServoArm()
    
    np.set_printoptions(suppress=True, precision=4)

    client = RemoteAPIClient()
    sim = client.getObject('sim')

    client.setStepping(True)
    sim.startSimulation()

    # Resgatando as referências do modelo
    joints = [sim.getObject(f"/theta{i}") for i in range(1, 6)]
    dummyHandle = sim.getObject('/Dummy')

    # Definindo os valores iniciais de cada junta
    q0 = np.zeros(5)
    setJointPositions(q0)

    x0 = servo_arm.fkine(q0, pose=True) # pose inicial

    # DX = -0.05
    # DY = 0
    # DZ = 0.05

    # x1 = x0 + np.array([DX, DY, DZ, 0., 0., 0.]).reshape(6, 1)
    
    x = x0

    # Definindo o passo da simulação
    dt = 50 * 1e-3

    # Criando as listas para guardar os valores das juntas, pose do atuador e manipularidade para plotar
    jointsList = []
    poseList = []
    M = []

    # print(
    #     servo_arm.T01(0.) @ servo_arm.T12(0.) # servo_arm.fkine([0., 0., 0., 0., 0.])
    # )

    while True:

        # Resganto a pose do dummy
        dummyPos = np.array(sim.getObjectPosition(dummyHandle, sim.handle_world) + sim.getObjectOrientation(dummyHandle, sim.handle_world)).reshape(6, 1)
        #print(dummyPos)

        # Caso atinja a posição do dommy, a simulação termina
        if np.linalg.norm(x[0:3] - dummyPos[0:3]) < EPS: # x1
            break

        # Pegando os valores de cada junta
        q = np.array(getJointPositions())
        jointsList.append(q)
        #print(x)
        dx = dummyPos - x # dx = x1 - x
        #print(dx)

        jac = servo_arm.jacobian(q)
        M.append(np.sqrt(round(np.linalg.det(jac @ jac.T), 2))) # medida de manipularidade
        dq = np.linalg.pinv(jac) @ dx
        q = q + dq.flatten() * dt
        setJointPositions(q)
        x = servo_arm.fkine(q, pose=True)

        poseList.append(x)
        #print(x)
        client.step()
        time.sleep(dt)
    
    # Criação de uma lista para o tempo utilizada na plotagem
    t = np.arange(0, np.array(jointsList).shape[0]*dt, dt)

    # 3 Plots(valores das juntas, manipulabilidade, pose do efetuador)
    plots.jointsPlot(t, jointsList)
    plots.mPlot(t, M)
    plots.posePlot(t, poseList)
