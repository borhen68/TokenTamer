# Contributing to TokenTamer

Thanks for your interest in contributing! 🎉

## Getting Started

1. Fork the repo on GitHub
2. Clone your fork locally
3. Create a virtual environment and install in editable mode:

```bash
git clone https://github.com/borhen68/TokenTamer.git
cd TokenTamer
python3 -m venv venv
source venv/bin/activate
pip install -e ".[dev]"
```

## Development Workflow

### Running Tests

```bash
pytest tests/ -v
```

### Code Style

- Follow PEP 8
- Use type hints where reasonable
- Keep functions focused and small
- Add docstrings for public APIs

### Adding a New Language Skeletonizer

1. Add the language to `C_STYLE_LANGUAGES` in `token_tamer/skeletonizer.py`
2. If needed, add a language-specific regex pattern
3. Add a test case in `tests/test_skeletonizer.py`

### Adding a New Feature

1. Open an issue to discuss the feature
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Write tests first, then implementation
4. Ensure all tests pass
5. Submit a pull request

## Release Process

1. Update `CHANGELOG.md`
2. Bump version in `pyproject.toml`
3. Tag: `git tag vX.Y.Z`
4. Push: `git push origin vX.Y.Z`

## Code of Conduct

Be respectful. Constructive criticism welcome. No harassment.
