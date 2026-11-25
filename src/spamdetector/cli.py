"""Command-line interface:  spam-detector train | predict | interactive."""

from __future__ import annotations

import argparse
import logging
import sys

from spamdetector import __version__
from spamdetector.predict import SpamClassifier


def _print_prediction(message: str, prediction) -> None:
    verdict = "SPAM" if prediction.is_spam else "NOT SPAM"
    print(f"\nMessage : {message}")
    print(f"Result  : {verdict}  (spam probability {prediction.spam_probability:.1%})")


def _cmd_train(args) -> int:
    from spamdetector.train import train

    summary = train(quick=args.quick)
    print("\nModel comparison (test set)")
    print(f"{'Model':<22}{'CV F1':>8}{'Acc':>8}{'Prec':>8}{'Recall':>8}{'F1':>8}")
    for name, res in summary["models"].items():
        t = res["test"]
        print(f"{name:<22}{res['cv_f1_mean']:>8.4f}{t['accuracy']:>8.4f}{t['precision']:>8.4f}"
              f"{t['recall']:>8.4f}{t['f1']:>8.4f}")
    if "unseen" in summary:
        print(f"\nUnseen-message check ({summary['unseen']['messages']} hand-written messages)")
        for name, res in summary["unseen"]["models"].items():
            print(f"{name:<22}{'':>8}{res['accuracy']:>8.4f}{res['precision']:>8.4f}"
                  f"{res['recall']:>8.4f}{res['f1']:>8.4f}")
    print(f"\nBest model (by cross-validation): {summary['best_model']}")
    print("Saved: models/spam_model.joblib, reports/metrics.json, reports/figures/*.png")
    return 0


def _cmd_predict(args) -> int:
    classifier = SpamClassifier.load()
    for message, prediction in zip(args.messages, classifier.predict_many(args.messages)):
        _print_prediction(message, prediction)
    return 0


def _cmd_interactive(_args) -> int:
    classifier = SpamClassifier.load()
    print("Spam detector. Type a message and press Enter ('quit' to stop).")
    while True:
        try:
            message = input("\nMessage> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if message.lower() in {"quit", "exit"}:
            return 0
        if not message:
            print("Please type something.")
            continue
        _print_prediction(message, classifier.predict(message))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="spam-detector", description="SMS / email spam detection.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p_train = sub.add_parser("train", help="tune, compare and save the models")
    p_train.add_argument("--quick", action="store_true", help="tiny grids for a fast smoke test")
    p_train.set_defaults(func=_cmd_train)

    p_predict = sub.add_parser("predict", help="classify one or more messages")
    p_predict.add_argument("messages", nargs="+", help="message text (use quotes)")
    p_predict.set_defaults(func=_cmd_predict)

    p_inter = sub.add_parser("interactive", help="type messages one by one")
    p_inter.set_defaults(func=_cmd_interactive)
    return parser


def main(argv: list | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except (FileNotFoundError, ValueError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
