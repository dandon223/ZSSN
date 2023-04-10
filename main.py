
from dataset import BatchLoader


def main():
    batch_size = 1
    seq_length = 50
    batch_loader = BatchLoader(batch_size, seq_length)

if __name__ == "__main__":
    main()