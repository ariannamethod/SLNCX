import argparse
from wulf_inference import generate


def main() -> None:
    parser = argparse.ArgumentParser(description='Wulf1 CLI interface')
    parser.add_argument('prompt', help='prompt text')
    args = parser.parse_args()
    print(generate(args.prompt))


if __name__ == '__main__':
    main()
