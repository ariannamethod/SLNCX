import argparse

from wulf_inference import infer_and_log


def main() -> None:
    parser = argparse.ArgumentParser(description="Interact with Wulf via CLI")
    parser.add_argument("prompt", nargs="?", help="Prompt to send. If omitted, enter interactive mode")
    parser.add_argument("--tokens", type=int, default=100, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.8)
    args = parser.parse_args()

    if args.prompt:
        print(infer_and_log(args.prompt, max_new_tokens=args.tokens, temperature=args.temperature))
    else:
        try:
            while True:
                prompt = input("wulf> ")
                if not prompt:
                    continue
                reply = infer_and_log(prompt, max_new_tokens=args.tokens, temperature=args.temperature)
                print(reply)
        except (EOFError, KeyboardInterrupt):
            print()


if __name__ == "__main__":
    main()
