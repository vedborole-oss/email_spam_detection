"""Replace the YOUR_NAME / YOUR_GITHUB_USERNAME placeholders in one go.

Usage:  python scripts/personalize.py --name "Your Full Name" --username your-github-name
"""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = ["README.md", "LICENSE", "CITATION.cff", "pyproject.toml", "CONTRIBUTING.md"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True, help="your full name")
    parser.add_argument("--username", required=True, help="your GitHub username")
    args = parser.parse_args()

    for relative in TARGETS:
        path = ROOT / relative
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        updated = text.replace("YOUR_GITHUB_USERNAME", args.username).replace("YOUR_NAME", args.name)
        if updated != text:
            path.write_text(updated, encoding="utf-8")
            print(f"Updated {relative}")
    print("Done. Placeholders replaced.")


if __name__ == "__main__":
    main()
