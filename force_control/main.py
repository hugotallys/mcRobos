from matplotlib import pyplot
import numpy

from zmqRemoteApi import RemoteAPIClient

buffer_len = 100


def main():
    client = RemoteAPIClient()
    sim = client.getObject('sim')

    joints = [sim.getObjectHandle(f'theta{i + 1}') for i in range(5)]
    buffer = numpy.zeros((len(joints), buffer_len))
    time_span = numpy.arange(0, buffer_len)
    t = 0

    client.setStepping(True)
    print('Starting simulation...')
    sim.startSimulation()
    fig, ax = pyplot.subplots(1, len(joints), figsize=(5 * len(joints), 2))

    try:
        while pyplot.fignum_exists(fig.number):
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
            t = t % buffer_len
    finally:
        sim.stopSimulation()


if __name__ == '__main__':
    main()
