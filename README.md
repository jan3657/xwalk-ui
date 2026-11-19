# xwalk UI

Independent browser workbench, classic explorer and NER demonstration site for the
[xwalk matching engine](https://github.com/jan3657/xwalk). Extracted from that repository;
the Python package is `xwalk_ui` and the command is `xwalk-ui`.

Install alongside the matching engine checkout during development:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
# Optional: use your local engine checkout for development.
pip install --no-deps -e ../xwalk
xwalk-ui --root /path/to/workspace --offline-only
```

The committed browser build runs without Node. Rebuild or develop the React interface:

```sh
cd web
npm ci
npm test
npm run build
```

Run Python regression tests with `python -m pytest -q -m 'not integration'`.
`ruff check src tests scripts ner-site` checks Python style.
See [workbench usage](docs/guide/ui.md), [hosting](docs/guide/hosting.md),
and [NER deployment](ner-site/README.md). NER is a separate Vercel deployment
with its own pinned engine dependency in `ner-site/requirements.txt`.

The workbench, CI and NER deployment pin the renamed `xwalk-match` engine commit
`62b8d592515ade0d59c39462eca4f079954b7856` from the backend repository.
Installing from Git requires Git on the machine. The Python import and CLI remain `xwalk`.
Use a fresh environment when migrating from the former Git-installed `xwalk` distribution.
