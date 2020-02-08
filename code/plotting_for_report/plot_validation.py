import pandas as pd
import matplotlib.pyplot as plt


if __name__ == "__main__":
    def main():
        validation_output_csv = "validation_error_per_epoch_cloud.csv"
        validation_error_per_epoch_df = pd.read_csv(validation_output_csv)
        validation_error_per_epoch_np = validation_error_per_epoch_df.to_numpy()
        plt.figure()
        plt.title("validation error by Rodrigues' formula ")
        plt.plot(validation_error_per_epoch_np[:, 0], validation_error_per_epoch_np[:, 1])
        # plt.plot(validation_error_per_epoch_np[:, 0], validation_error_per_epoch_np[:, 2])
        plt.legend(['mean error'], loc='upper left')
        plt.show()
    main()

