# xwalk UI

Independent browser workbench, classic explorer and NER demonstration site for the
[xwalk matching engine](https://github.com/jan3657/xwalk). Extracted from that repository;
the Python package is `xwalk_ui` and the command is `xwalk-ui`.

Install alongside the matching engine checkout during development:

```sh
python3 -m venv .venv
. .venv/bin/activate
pip install -e '../xwalk[dev]' -e '.[dev]'
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

The workbench and CI pin the matching engine extraction commit
`ef3180b4d0d5a95b81c3e96f5307b48ad1d7d912` from the backend repository.
Installing from Git requires Git on the machine. NER deployment retains its separately
tested engine pin in `ner-site/requirements.txt`.
