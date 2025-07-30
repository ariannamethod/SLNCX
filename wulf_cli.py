import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description='Wulf1 CLI interface')
    parser.add_argument('prompt', help='prompt text')
    parser.add_argument('--ckpt', help='path to checkpoint file')
    args = parser.parse_args()

    if args.ckpt:
        os.environ['CKPT_PATH'] = args.ckpt

    import wulf_inference
    if args.ckpt:
        wulf_inference.CKPT_PATH = Path(args.ckpt)

    print(wulf_inference.generate(args.prompt))


if __name__ == '__main__':
    main()
