from pathlib import Path

from matplotlib import pyplot
import numpy

from zmqRemoteApi import RemoteAPIClient

buffer_len = 10


def main():
    client = RemoteAPIClient()
    client.setStepping(True)

    time_span = numpy.arange(0, buffer_len)

    sim = client.getObject('sim')
    sim.loadScene(str(Path('solid_simple.ttt').absolute()).replace('\\', '/'))

    joints = [sim.getObjectHandle(f'theta{i + 1}') for i in range(5)]
    buffer = numpy.zeros((len(joints), buffer_len))
    fig, ax = pyplot.subplots(1, len(joints), figsize=(5 * len(joints), 2))

    sim.startSimulation()

    try:
        t = 0
        while t < buffer_len:
            client.step()

            buffer[:, t] = [sim.getJointForce(joint) for joint in joints]

            for i in range(len(joints)):
                ax[i].clear()
                ax[i].set_title(f'Joint {i + 1} force')
                ax[i].plot(time_span[:t], buffer[i, :t])
                ax[i].set_xlim(0, buffer_len)
                ax[i].set_ylim(0, 0.2)
            pyplot.tight_layout()
            pyplot.show(block=False)
            pyplot.pause(0.0001)

            t += 1
    finally:
        sim.stopSimulation()

    pyplot.close(fig)
    while sim.getSimulationState() != sim.simulation_stopped:
        pass
    sim.loadScene(str(Path('solid_force.ttt').absolute()).replace('\\', '/'))
    sim.startSimulation()

    try:
        while True:

            for i, joint in enumerate(joints):
                print(f'Setting joint{i + 1} force to {buffer[i, -1]}...')
                sim.setJointForce(joint, 2 * buffer[i, -1])
                print(f'Read joint{i + 1} force: {sim.getJointForce(joint)} target: {sim.getJointTargetForce(joint)}')

            client.step()
    finally:
        sim.stopSimulation()


if __name__ == '__main__':
    main()
