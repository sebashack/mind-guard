import os
import sys
import torch


def main():
    if torch.cuda.is_available():
        print("CUDA Version:", torch.version.cuda)
    else:
        raise Exception("CUDA is not available")


if __name__ == "__main__":
    sys.exit(main())
