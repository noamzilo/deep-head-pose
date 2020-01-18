import torch
import sys


def validate():
    print("validating cuda is on")
    print(torch.cuda.current_device())
    print(torch.cuda.device(0))
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name(0))
    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        print("cuda is on")
    else:
        sys.stderr.write("******** cuda is not on *********\r\n")

if __name__ == "__main__":
    def main():
        validate()

    main()
