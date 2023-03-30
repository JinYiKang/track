import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    data = np.loadtxt('./traj.txt', dtype=float, delimiter=',')
    data_1 = np.array([d for d in data if d[0] == 1])
    data_2 = np.array([d for d in data if d[0] == 2])
    data_3 = np.array([d for d in data if d[0] == 3])
    data_4 = np.array([d for d in data if d[0] == 4])
    data_5 = np.array([d for d in data if d[0] == 5])
    data_6 = np.array([d for d in data if d[0] == 6])

    plt.plot(data_1[:,1], data_1[:, 2], label="1")
    plt.plot(data_2[:,1], data_2[:, 2], label="2")
    plt.plot(data_3[:,1], data_3[:, 2], label="3")
    plt.plot(data_4[:,1], data_4[:, 2], label="4")
    plt.plot(data_5[:,1], data_5[:, 2], label="5")
    plt.plot(data_6[:,1], data_6[:, 2], label="6")

    plt.legend()
    # plt.savefig('./data/30.jpg')
    plt.show()
    # plt.close()
