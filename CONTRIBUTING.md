# Contributing

Thanks for your interest in improving this project, by YOUR_NAME.

## Set up
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/spam-detector.git
cd spam-detector
python3 -m venv venv && source venv/bin/activate
pip install -e ".[app,dev]"
```

## Before you open a pull request
```bash
ruff check .     # style
pytest           # tests
```
- Keep changes small and focused, and add a test for new behaviour.
- Use clear commit messages, for example `Add threshold option to CLI`.
- Open an issue first for larger changes.
