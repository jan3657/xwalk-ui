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

The extracted workbench requires the engine revision accompanying this split,
including `xwalk.llm.chatgpt_protocol`. Until that revision is released, install
both sibling checkouts as above. Docker/release installs require that engine revision
published or supplied as a wheel; do not deploy against an older engine wheel.

CI checks out the engine main branch. Publish the accompanying engine split before
enabling CI, or dispatch the workflow with `engine_ref` set to its published branch.
