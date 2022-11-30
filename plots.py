from matplotlib import pyplot as plt
import numpy as np

def jointsPlot(t, jointsList):

    plt.figure()

    jointsList = np.array(jointsList)

    plt.title('Joints Angles')
    for i in range(5):
        plt.plot(t, jointsList[:, i], label=f"{i+1}º Joint")
    plt.legend(loc="best")
    plt.show()

def mPlot(t, M):
    plt.figure()

    plt.title('Manipulability')
    plt.plot(t, M)
    plt.show()

def posePlot(t, poseList):

    plt.figure()

    poseList = np.array(poseList)
    labels = ['x', 'y', 'z', 'alpha(wx)', 'beta(wy)', 'gama(wz)'] 

    plt.title('Effectuator Pose')
    for i in range(6):
        plt.plot(t, poseList[:, i], label=labels[i])
    plt.legend(loc="best")
    plt.show()