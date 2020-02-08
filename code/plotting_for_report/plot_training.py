import matplotlib.pyplot as plt
import numpy as np


if __name__ == "__main__":
    def main():
        training_logs_path = "cloud_training_logs.txt"
        with open(training_logs_path) as f:
            lines = f.readlines()
        last_epoch = None
        losses = []
        for line in lines:
            if line.startswith("Epoch"):
                words = line.split()
                yaw = float(words[6][:-1])
                pitch = float(words[8][:-1])
                roll = float(words[10][:-1])
                epoch_word = words[1]
                epoch_num = int(epoch_word.split('/')[0][1:])

                if last_epoch != epoch_num:
                    last_epoch = epoch_num
                    losses.append((epoch_num, yaw, pitch, roll))

        losses = np.array(losses)
        plt.figure()
        plt.title("training losses per epoch")
        plt.plot(losses[:, 0], losses[:, 1])
        plt.plot(losses[:, 0], losses[:, 2])
        plt.plot(losses[:, 0], losses[:, 3])
        plt.plot(losses[:, 0], np.mean(losses[:, 1:], axis=1))
        plt.legend(['yaw loss', 'pitch loss', 'roll loss', 'average loss'], loc='upper left')
        plt.show()


    main()

