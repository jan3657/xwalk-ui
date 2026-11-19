#!/usr/bin/env node
// Smoke check of the NER demo page: drives the page through its states at a phone and a
// desktop width and fails on anything a visitor would see as broken.
//
//   node ner-site/tools/page_smoke.mjs BASE_URL [--only NAME[,NAME]] [--widths 390,1280]
//        [--shots DIR] [--dark] [--list]
//
// BASE_URL is a running tools/serve_local.py (any mode; --offline is enough). This
// script starts nothing itself. Only the "happy" state uses that server's own API; every
// other state answers /api/health and /api/ner from tools/fixtures/ through
// page.route, so one server covers them all and no state costs a model call.
//
// In every state it asserts: the expected result or message is visible, no page error,
// no console error (the missing /_vercel/insights/script.js and an API status the state
// expects are ignored), no horizontal scroll, and the Run button can be reached.
//
// The result states (cards, no-entities, cap, stale, keeps-result, page-hidden) use a
// response built here in the shape production answers: a pick per entity, one where the
// model overruled the search, one it declined, one that failed, one not looked up.
//
// The arrival states (example-*) cover the recorded example of public/example.json:
// shown on a full-catalog deployment, removed on any other, never over typed text, and
// correct whether /api/health answers before it, after it, with an error, or never.
// closing-panel covers the panel after the cards.
//
// Playwright is resolved from the local node_modules, NODE_PATH, or `npm root -g`, as in
// screenshot.mjs. PLAYWRIGHT_CHANNEL=chrome uses the installed Chrome.

import { createRequire } from "node:module";
import { execFileSync } from "node:child_process";
import { mkdirSync, readFileSync } from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const HERE = path.dirname(fileURLToPath(import.meta.url));
const FIXTURES = path.join(HERE, "fixtures");
// Not a real key: 20+ printable characters is all the page and the canned API ask for.
const FAKE_KEY = "sk-smoke-not-a-real-key-0000";
const OWN_KEY_ENTRY = "xwalk-ner-own-key";
const PAYER_ENTRY = "xwalk-ner-payer";
const EXAMPLE_FULL_START = "Aspirin relieved her fever";
const EXAMPLE_SAMPLE_START = "Clinic note";
const EXAMPLE_FILE = JSON.parse(readFileSync(path.join(HERE, "..", "public", "example.json"), "utf8"));
const EXAMPLE_NOTE = /^Recorded from a real run on 19 Nov 2026\.$/;
const EXAMPLE_STALE = /^This result is for the example text\. Press "Map this text" to map yours\.$/;

function loadPlaywright() {
  const local = createRequire(import.meta.url);
  try {
    return local("playwright");
  } catch (_) {
    const root = execFileSync("npm", ["root", "-g"], { encoding: "utf8" }).trim();
    return createRequire(path.join(root, "noop.js"))("playwright");
  }
}

function fixture(name) {
  return JSON.parse(readFileSync(path.join(FIXTURES, `${name}.json`), "utf8"));
}

// What production answers: the full catalog, three ontologies, no genes.
function productionHealth() {
  const health = fixture("health");
  health.backend = "neon";
  health.neon = { schema: "xwalk_full", catalog: "full" };
  if (!health.selectors.includes("retrieval")) health.selectors.unshift("retrieval");
  health.ontologies = [
    { slug: "chebi", name: "ChEBI", type: "chemical", count: 219011 },
    { slug: "ctd-disease", name: "CTD diseases / MeSH", type: "disease", count: 13323 },
    { slug: "foodon", name: "FoodOn", type: "food", count: 28684 },
  ];
  return health;
}

// ---------- a production-shaped response ----------

const SHAPED_TEXT = "Aspirin and metformin with oily fish; BRCA1, warfarin, garlic and cumin.";
const ONTOLOGY = { chemical: ["chebi", "CHEBI:"], disease: ["ctd-disease", "MESH:D"], food: ["foodon", "FOODON:"] };

function candidates(prefix, labels, from) {
  return labels.map((label, i) => ({
    id: `${prefix}${from + i}`,
    label,
    score: 0.0328 - i * 0.0004,
    retrievers: { bm25: i + 1, "dense-qwen3": i + 2 },
    url: `https://example.org/term/${prefix}${from + i}`,
  }));
}

// spec: [mention, type, status, pickIndex]. status null: no pick (search only);
// type "other": not looked up.
function shapedEntity(text, n, [mention, type, status, pickIndex]) {
  const start = text.indexOf(mention);
  const base = { id: `e${n}`, text: mention, type, start, end: start + mention.length };
  if (type === "other") return { ...base, ontology: null, matches: [], more: [], notes: ["no ontology for type 'other'"], pick: null };
  const [slug, prefix] = ONTOLOGY[type];
  const all = candidates(prefix, [`${mention} (salt)`, mention.toLowerCase(), `${mention} acid`, `${mention} ester`, `${mention} hydrate`, `${mention} oxide`, `${mention} dimer`], n * 100);
  let pick = null;
  if (status) {
    const chosen = Number.isInteger(pickIndex) ? all[pickIndex] : null;
    pick = {
      selector: "claude-opus-5-5",
      status,
      id: chosen ? chosen.id : null,
      label: chosen ? chosen.label : null,
      url: chosen ? chosen.url : null,
      confidence: status === "failed" ? null : 0.93,
      explanation: status === "failed" ? "cut off by the request deadline" : `The reason for ${mention}.`,
    };
  }
  return { ...base, ontology: slug, matches: all.slice(0, 5), more: all.slice(5), notes: [], pick };
}

function shapedResponse(text, specs, extra = {}) {
  return {
    ok: true,
    model: "claude-opus-5-5",
    mode: "live",
    selector: "claude-opus-5-5",
    entities: specs.map((spec, i) => shapedEntity(text, i + 1, spec)),
    retrieval: { bm25: { status: "ok" }, dense: { status: "ok" } },
    warnings: [],
    usage: { llm_calls: 1, prompt_tokens: 1134, completion_tokens: 187, embedding_requests: 1, selector_calls: specs.length, selector_prompt_tokens: 43198, selector_completion_tokens: 729, selector_cost_usd: null },
    timing_ms: { extract: 3289, map: 2339, select: 3197, total: 8825 },
    requested_model: "claude-opus-5-5",
    requested_selector: "claude-opus-5-5",
    fallbacks: [],
    notices: [],
    provider: "site",
    credential_source: "site",
    ...extra,
  };
}

// Every row a card can be: matched (the top hit), override, declined, not looked up,
// failed, needs a check, and matched again.
const SHAPED_SPECS = [
  ["Aspirin", "chemical", "match", 0],
  ["metformin", "chemical", "match", 1],
  ["oily fish", "food", "no_match", null],
  ["BRCA1", "other", null, null],
  ["warfarin", "chemical", "failed", null],
  ["garlic", "food", "review", 0],
  ["cumin", "food", "match", 0],
];

function shaped(extra) {
  return shapedResponse(SHAPED_TEXT, SHAPED_SPECS, extra);
}

function apiError(code, message) {
  return { ok: false, error: { code, message } };
}

function parseArgs(argv) {
  const opts = { only: null, widths: [390, 1280], shots: null, dark: false, list: false, positional: [] };
  for (let i = 0; i < argv.length; i += 1) {
    const a = argv[i];
    if (a === "--only") opts.only = argv[++i].split(",");
    else if (a === "--widths") opts.widths = argv[++i].split(",").map(Number);
    else if (a === "--shots") opts.shots = argv[++i];
    else if (a === "--dark") opts.dark = true;
    else if (a === "--list") opts.list = true;
    else if (a.startsWith("--")) throw new Error(`unknown option ${a}`);
    else opts.positional.push(a);
  }
  if (!opts.list && opts.positional.length !== 1) throw new Error("usage: page_smoke.mjs BASE_URL [--only NAME] [--widths 390,1280] [--shots DIR] [--dark]");
  if (opts.widths.some((w) => !Number.isInteger(w) || w < 200)) throw new Error("--widths takes integers >= 200");
  return opts;
}

// ---------- one page under test ----------

class Session {
  constructor(page, width, base) {
    this.page = page;
    this.width = width;
    this.base = base;
    this.phone = width < 700;
    this.failures = [];
    this.expectedStatuses = new Set();
    this.posts = []; // every POST /api/ner the page sent: { body, authorization }
    this.strayAuthorization = []; // any other request that carried an Authorization header
    this.health = "pass";
    this.healthGate = null; // a promise /api/health waits for before it answers
    this.ner = "pass";
    this.example = "pass"; // example.json: "pass", "missing" (404), "garbage", or { gate }
    this.failedLoads = false; // a state that cuts the connection expects net::ERR_FAILED
  }

  check(ok, message) {
    if (!ok) this.failures.push(message);
    return ok;
  }

  // health: "pass", "fail", or a JSON body. ner: "pass" or { status, body, headers,
  // raw (a body that is not JSON), gate (a promise the answer waits for), abort (the
  // connection is cut instead of answered) }. Read when the request arrives and again
  // after the gate, so a state can decide the answer while the run is under way.
  async install() {
    const { page } = this;
    page.on("pageerror", (err) => this.failures.push(`page error: ${err.message}`));
    page.on("console", (msg) => {
      if (msg.type() !== "error") return;
      const where = msg.location().url || "";
      const text = msg.text();
      if (where.includes("/_vercel/insights/") || text.includes("/_vercel/insights/")) return;
      const status = /status of (\d{3})/.exec(text);
      if (status && where.includes("/api/") && this.expectedStatuses.has(Number(status[1]))) return;
      if (this.failedLoads && /net::ERR_FAILED/.test(text) && where.includes("/api/ner")) return;
      if (this.example === "missing" && where.includes("/example.json")) return;
      this.failures.push(`console error: ${text} (${where})`);
    });
    page.on("request", (request) => {
      const url = new URL(request.url());
      const authorization = request.headers().authorization || "";
      if (url.pathname === "/api/ner" && request.method() === "POST") {
        let body = null;
        try {
          body = JSON.parse(request.postData() || "null");
        } catch (_) {
          /* left as null */
        }
        this.posts.push({ body, authorization });
      } else if (authorization) {
        this.strayAuthorization.push(`${request.method()} ${url.pathname}`);
      }
    });
    await page.route("**/example.json", async (route) => {
      const example = this.example;
      if (example === "missing") return route.fulfill({ status: 404, contentType: "text/plain", body: "not found" });
      if (example === "garbage") return route.fulfill({ status: 200, contentType: "application/json", body: "<html>not json" });
      if (example && example.gate) await example.gate;
      return route.continue();
    });
    await page.route("**/api/health", async (route) => {
      if (this.healthGate) await this.healthGate;
      if (this.health === "pass") return route.continue();
      if (this.health === "fail") return route.fulfill({ status: 500, contentType: "text/plain", body: "down" });
      return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(this.health) });
    });
    await page.route("**/api/ner", async (route) => {
      if (this.ner === "pass") return route.continue();
      if (this.ner.gate) await this.ner.gate;
      const ner = this.ner;
      if (ner.abort) return route.abort("failed");
      if (ner.raw !== undefined) return route.fulfill({ status: ner.status || 200, contentType: "text/html", body: ner.raw });
      let body = ner.body;
      if (typeof body === "function") body = body(JSON.parse(route.request().postData() || "{}"));
      return route.fulfill({ status: ner.status || 200, contentType: "application/json", headers: ner.headers || {}, body: JSON.stringify(body) });
    });
  }

  async open() {
    await this.page.goto(this.base, { waitUntil: "networkidle" });
  }

  // For a state that holds a request back: "networkidle" would wait for it.
  async openHeld() {
    await this.page.goto(this.base, { waitUntil: "load" });
    await this.page.waitForTimeout(400);
  }

  // What the results section is showing, read in one go.
  shown() {
    return this.page.evaluate(() => {
      const $ = (id) => document.getElementById(id);
      const seen = (id) => !$(id).hidden && $(id).getClientRects().length > 0;
      const results = $("results");
      return {
        visible: seen("results"),
        kind: results.getAttribute("data-kind"),
        heading: $("results-heading").textContent,
        note: seen("example-note") ? $("example-note").textContent : null,
        stale: seen("stale") ? $("stale").textContent : null,
        status: seen("status") ? $("status").textContent : null,
        details: $("run-details-summary").textContent,
        cards: document.querySelectorAll("#cards .card").length,
        legend: $("legend").textContent,
        next: seen("next") ? $("next-body").textContent : null,
        text: $("text").value,
        counter: $("text-counter").textContent,
        banner: seen("banner") ? $("banner").textContent : null,
        announce: $("announce").textContent,
        focus: document.activeElement ? document.activeElement.id || document.activeElement.tagName : null,
        scrollY: Math.round(window.scrollY),
        top: Math.round(results.getBoundingClientRect().top + window.scrollY),
        screen: window.innerHeight,
        stored: Object.keys(window.sessionStorage).concat(Object.keys(window.localStorage)),
      };
    });
  }

  // The recorded example as it must look on arrival: labelled, complete, and quiet.
  async expectExample(what, { text = EXAMPLE_FILE.text, counts = true } = {}) {
    await this.expectVisible('#results[data-kind="example"] .card', `the example result (${what})`);
    const seen = await this.shown();
    this.check(seen.heading === "Example result", `${what}: the heading is ${JSON.stringify(seen.heading)}`);
    this.check(EXAMPLE_NOTE.test(seen.note || ""), `${what}: the label is ${JSON.stringify(seen.note)}`);
    this.check(seen.status === null, `${what}: a timing sentence is shown for the example: ${JSON.stringify(seen.status)}`);
    this.check(seen.details === "Details of the recorded run", `${what}: the details are titled ${JSON.stringify(seen.details)}`);
    this.check(seen.cards === EXAMPLE_FILE.response.entities.length, `${what}: ${seen.cards} cards`);
    this.check(seen.text === text, `${what}: the text box holds ${JSON.stringify(seen.text.slice(0, 40))}`);
    this.check(seen.announce === "", `${what}: the example was announced: ${JSON.stringify(seen.announce)}`);
    this.check(/ChEBI/.test(seen.legend) && /FoodOn/.test(seen.legend), `${what}: the legend has no ontology names: ${JSON.stringify(seen.legend)}`);
    if (counts) this.check(/searched 261,018 ontology terms by keyword and by meaning, and the model chose among real candidates/.test(seen.next || ""), `${what}: the closing panel says ${JSON.stringify(seen.next)}`);
    this.check(this.posts.length === 0, `${what}: the example cost a request`);
    this.check(seen.stored.length === 0, `${what}: the page stored something: ${seen.stored.join(", ")}`);
    return seen;
  }

  // No example on show: the page as it was before there was one.
  async expectNoExample(what, { text = "" } = {}) {
    await this.page.waitForTimeout(300);
    const seen = await this.shown();
    this.check(!seen.visible, `${what}: results are shown (${seen.heading})`);
    this.check(seen.text === text, `${what}: the text box holds ${JSON.stringify(seen.text.slice(0, 40))}`);
    this.check(seen.counter.startsWith(`${text.length} /`), `${what}: the counter says ${JSON.stringify(seen.counter)}`);
    return seen;
  }

  visible(selector) {
    return this.page.locator(selector).first().isVisible();
  }

  text(selector) {
    return this.page.locator(selector).first().innerText();
  }

  async expectVisible(selector, what) {
    try {
      await this.page.waitForSelector(selector, { state: "visible", timeout: 15000 });
      return true;
    } catch (_) {
      this.failures.push(`${what || selector} is not visible`);
      return false;
    }
  }

  // Waits a little: several things are hidden only once a request has answered.
  async expectHidden(selector, what) {
    try {
      await this.page.waitForSelector(selector, { state: "hidden", timeout: 5000 });
    } catch (_) {
      this.failures.push(`${what || selector} should not be visible`);
    }
  }

  async expectText(selector, pattern, what) {
    const text = (await this.visible(selector)) ? await this.text(selector) : "";
    this.check(pattern.test(text), `${what || selector} should match ${pattern}, got ${JSON.stringify(text.slice(0, 160))}`);
  }

  async optionsOpen() {
    return this.page.evaluate(() => document.getElementById("options").open);
  }

  async loadExample() {
    await this.page.click("#load-example");
  }

  async run() {
    await this.page.click("#run");
  }

  async runExample() {
    await this.loadExample();
    await this.run();
  }

  async expectResults() {
    const outcome = await Promise.race([
      // The recorded example has cards too: only a run's own result counts.
      this.page.waitForSelector('#results[data-kind="run"]:not([hidden]) .card', { state: "visible", timeout: 130000 }).then(() => "results"),
      this.page.waitForSelector("#error:not([hidden])", { state: "visible", timeout: 130000 }).then(() => "error"),
    ]).catch(() => "nothing");
    if (outcome === "error") this.failures.push(`run failed: ${await this.text("#error-title")} ${await this.text("#error-message")}`);
    else if (outcome !== "results") this.failures.push("no results and no error after a run");
    await this.settled();
    return outcome === "results";
  }

  // A run has ended and the page is idle again.
  async settled() {
    await this.page.waitForFunction(() => document.getElementById("run").getAttribute("aria-busy") === "false", null, { timeout: 130000 }).catch(() => {
      this.failures.push("the run never ended (Run is still busy)");
    });
    await this.page.waitForLoadState("networkidle").catch(() => {});
  }

  // Asserted in every state, last.
  async common() {
    const { page } = this;
    const overflow = await page.evaluate(() => ({ scroll: document.documentElement.scrollWidth, client: document.documentElement.clientWidth }));
    this.check(overflow.scroll <= overflow.client, `horizontal scroll: scrollWidth ${overflow.scroll} > clientWidth ${overflow.client}`);

    if (!(await this.visible("#run"))) {
      this.failures.push("Run is not visible");
      return;
    }
    await page.locator("#run").scrollIntoViewIfNeeded();
    const run = await page.evaluate(() => {
      const button = document.getElementById("run");
      const r = button.getBoundingClientRect();
      const top = document.elementFromPoint(r.left + r.width / 2, r.top + r.height / 2);
      return {
        inView: r.top >= 0 && r.left >= 0 && r.bottom <= window.innerHeight && r.right <= window.innerWidth,
        onTop: Boolean(top) && button.contains(top),
        disabled: button.disabled,
        height: r.height,
        label: button.querySelector(".run-label").textContent,
      };
    });
    this.check(run.inView, "Run cannot be scrolled fully into view");
    this.check(run.onTop, "Run is covered by another element");
    this.check(run.height >= 44, `Run is ${run.height}px tall, under the 44px touch target`);
    this.check(!run.disabled, "Run is disabled at the end of the state");
    this.check(run.label === "Map this text", `Run label is ${JSON.stringify(run.label)}`);
    this.check(this.strayAuthorization.length === 0, `a key left the page outside POST /api/ner: ${this.strayAuthorization.join(", ")}`);
  }

  // A gate for the next answers of /api/ner: they wait until release() is called.
  gate() {
    let release;
    const promise = new Promise((resolve) => {
      release = resolve;
    });
    return { promise, release };
  }

  // What the browser tells a page when the phone locks or the app changes.
  async setVisibility(value) {
    await this.page.evaluate((v) => {
      Object.defineProperty(document, "visibilityState", { configurable: true, get: () => v });
      Object.defineProperty(document, "hidden", { configurable: true, get: () => v === "hidden" });
      document.dispatchEvent(new Event("visibilitychange"));
    }, value);
  }

  // Runs the shaped text against a production health answer and waits for the cards.
  async runShaped(response) {
    this.health = productionHealth();
    this.ner = { body: response || shaped() };
    await this.open();
    await this.page.fill("#text", SHAPED_TEXT);
    await this.run();
    return this.expectResults();
  }

  // The error box after a run that failed: its title, look and action.
  async expectError(title, { look = "alert", action = null, ownKey = false, body = null } = {}) {
    await this.settled();
    await this.expectVisible("#error", `the error box (${title})`);
    const seen = await this.page.evaluate(() => ({
      title: document.getElementById("error-title").textContent,
      body: document.getElementById("error-message").textContent,
      notice: document.getElementById("error").classList.contains("notice"),
      action: document.getElementById("error-action").hidden ? null : document.getElementById("error-action").textContent,
      ownKey: !document.getElementById("error-own-key").hidden,
      text: document.getElementById("text").value.length,
      status: document.getElementById("status").textContent,
    }));
    this.check(seen.title === title, `error title is ${JSON.stringify(seen.title)}, expected ${JSON.stringify(title)}`);
    this.check(seen.notice === (look === "notice"), `"${title}" has the wrong look (notice: ${seen.notice})`);
    this.check(seen.action === action, `"${title}" offers ${JSON.stringify(seen.action)}, expected ${JSON.stringify(action)}`);
    this.check(seen.ownKey === ownKey, `"${title}": 'Use my own key' shown is ${seen.ownKey}`);
    this.check(/\S/.test(seen.body) && !/^Something went wrong/.test(seen.body), `"${title}" has no body`);
    if (body) this.check(body.test(seen.body), `"${title}" body should match ${body}, got ${JSON.stringify(seen.body)}`);
    this.check(seen.text > 0, `"${title}" cleared the text box`);
    this.check(!/Failed after/.test(seen.status), `"${title}" still prints "Failed after"`);
    this.check(!/you have used|your allowance|\byou used\b/i.test(seen.title + seen.body), `"${title}" blames the visitor for a per-network limit`);
  }

  async chooseOwnKey(providerId) {
    const { page } = this;
    if (!(await this.optionsOpen())) await page.click("#options > summary");
    await page.selectOption("#provider", providerId);
    await page.fill("#own-key", FAKE_KEY);
  }
}

// ---------- the states ----------

const STATES = {
  // The server's own API, whatever mode it runs in.
  async happy(s) {
    await s.open();
    s.check(!(await s.optionsOpen()), "Options should be closed on arrival");
    await s.expectHidden("#progress", "the progress bar");
    await s.runExample();
    await s.expectResults();
    s.check((await s.page.locator("#cards .card").count()) > 0, "no result cards");
    s.check(/\S/.test(await s.text("#results-summary")), "the results summary is empty");
    await s.expectText("#status", /^Done in /, "the one-sentence status");
    await s.expectHidden("#error", "the error box");
    await s.expectHidden("#progress", "the progress bar after the run");
  },

  // What a visitor of the live site gets: the static copy, the exact count, the short
  // example, and Run on the first screen of a phone.
  async production(s) {
    const html = await (await fetch(s.base)).text();
    s.check(!/200-term|genes|NCBI Gene/i.test(html), "index.html carries sample wording (200-term, genes)");
    s.check(html.includes("<title>xwalk: biomedical text to ontology IDs</title>"), "index.html has another <title>");
    s.health = productionHealth();
    s.ner = { body: fixture("ner") };
    await s.open();
    await s.expectText("h1", /^Turn biomedical text into ontology IDs$/, "the headline");
    await s.expectText("#lede", /^Paste a sentence\. Claude finds the chemicals, diseases and foods in it, and xwalk links each one to a term in ChEBI, MeSH or FoodOn\.$/, "the lede");
    await s.expectText("#caveat", /^Searches 261,018 real ontology terms\. Every ID comes from the ontology; the model never writes one\.$/, "the trust line");
    await s.expectText("#options-summary", /^Options: Claude Opus 5\.5, free runs$/, "the Options summary");
    await s.expectText("#run-hint", /^Free, no sign-up, usually 10 to 30 seconds\./, "the hint under Run");
    await s.expectHidden("#banner", "the banner");
    await s.expectHidden("#allowance", "the free-runs line");
    s.check(!(await s.optionsOpen()), "Options should be closed on arrival");
    await s.expectHidden("#provider", "the payer select inside closed Options");
    const fold = await s.page.evaluate(() => ({ bottom: document.getElementById("run").getBoundingClientRect().bottom, height: window.innerHeight, y: window.scrollY }));
    s.check(fold.y === 0 && fold.bottom <= fold.height, `Run is below the first screen: bottom ${Math.round(fold.bottom)} of ${fold.height}`);
    await s.loadExample();
    const value = await s.page.inputValue("#text");
    s.check(value.startsWith(EXAMPLE_FULL_START), `Example loaded ${JSON.stringify(value.slice(0, 40))}`);
    const focus = await s.page.evaluate(() => document.activeElement && document.activeElement.id);
    // On a phone the text box is not focused, so no keyboard covers the page.
    s.check(focus === (s.phone ? "run" : "text"), `after Example the focus is on #${focus}`);
    await s.page.click("#clear-text");
    s.check((await s.page.inputValue("#text")) === "", "Clear left text behind");
  },

  async empty(s) {
    s.health = fixture("health");
    await s.open();
    await s.run();
    await s.expectVisible("#text-error", "the message at the text box");
    await s.expectText("#text-error", /^Paste some text first, or tap Example\.$/);
    await s.expectHidden("#error", "the error box");
    s.check(s.posts.length === 0, "an empty text was submitted");
    const seen = await s.page.evaluate(() => {
      const r = document.getElementById("text-error").getBoundingClientRect();
      return r.top >= 0 && r.bottom <= window.innerHeight;
    });
    s.check(seen, "the empty-text message is off screen");
    await s.page.fill("#text", "aspirin");
    await s.expectHidden("#text-error", "the message after typing");
  },

  async "too-long"(s) {
    s.health = fixture("health");
    await s.open();
    await s.page.fill("#text", "a".repeat(5001));
    await s.expectVisible("#text-error", "the message at the text box, before Run");
    await s.expectText("#text-error", /^This text is 5,001 characters; the demo takes up to 5,000\. Shorten it by 1\.$/);
    await s.run();
    await s.expectVisible("#text-error", "the message at the text box, after Run");
    await s.expectHidden("#error", "the error box");
    s.check(s.posts.length === 0, "a too-long text was submitted");
    await s.page.fill("#text", "a".repeat(5000));
    await s.expectHidden("#text-error", "the message at the limit");
  },

  async "limit-hour"(s) {
    s.health = fixture("health-blocked");
    s.ner = { status: 429, headers: { "Retry-After": "1380" }, body: fixture("error-blocked") };
    s.expectedStatuses.add(429);
    await s.open();
    await s.expectVisible("#allowance", "the free-runs line");
    await s.expectText("#allowance", /^The free runs for your network this hour are used\. Next one in \d+ min\.$/);
    await s.expectHidden("#run-hint", "the 'Free, no sign-up' hint");
    await s.expectVisible("#use-own-key", "'Use my own key'");
    await s.runExample();
    await s.expectError("The free runs for your network this hour are used.", { look: "notice", ownKey: true, body: /^You can run again in 23 minutes\.$/ });
    await s.expectHidden("#results", "results");
    s.check((await s.page.inputValue("#text")).length > 0, "the text was cleared by the error");
    // The limit is said once: while the notice is up, the line beside Run steps aside.
    await s.expectHidden("#allowance", "the free-runs line while the notice says the same");
    await s.expectHidden("#use-own-key", "the second 'Use my own key'");
    s.check((await s.page.locator("button:visible", { hasText: "Use my own key" }).count()) === 1, "'Use my own key' is not offered exactly once");
    await s.page.click("#error-dismiss");
    await s.expectVisible("#allowance", "the free-runs line after Dismiss");
    await s.page.click("#use-own-key");
    s.check(await s.optionsOpen(), "'Use my own key' did not open Options");
    await s.expectVisible("#provider", "the payer select");
    s.check((await s.page.inputValue("#provider")) === "site", "'Use my own key' changed who pays by itself");
  },

  async "limit-day"(s) {
    s.health = fixture("health-day");
    s.ner = { status: 429, body: fixture("error-day") };
    s.expectedStatuses.add(429);
    await s.open();
    await s.expectVisible("#allowance", "the free-runs line");
    await s.expectText("#allowance", /^The free runs for your network today are used\. They come back at .+\.$/);
    await s.expectVisible("#use-own-key", "'Use my own key'");
    await s.runExample();
    await s.expectError("The free runs for your network today are used.", { look: "notice", ownKey: true, body: /^They come back at (.+ \()?midnight UTC\)?\.$/ });
    await s.expectHidden("#allowance", "the free-runs line while the notice says the same");
    await s.page.click("#error-own-key");
    s.check(await s.optionsOpen(), "'Use my own key' in the error box did not open Options");
    await s.expectHidden("#error", "the error box after 'Use my own key'");
  },

  async "limits-unavailable"(s) {
    s.health = fixture("health-unavailable");
    s.ner = { status: 503, body: fixture("error-unavailable") };
    s.expectedStatuses.add(503);
    await s.open();
    await s.expectText("#allowance", /^The demo is briefly unavailable\. Try again in a minute\.$/);
    await s.runExample();
    await s.expectError("The demo is briefly unavailable.", { look: "notice", action: "Try again", ownKey: true });
  },

  async sonnet(s) {
    s.health = fixture("health-sonnet");
    s.ner = { body: fixture("ner-sonnet") };
    await s.open();
    await s.expectVisible("#allowance", "the free-runs line");
    await s.expectText("#allowance", /^The free runs on Claude Opus for your network this hour are used\. Runs now start on Claude Sonnet/);
    s.check(!(await s.page.evaluate(() => document.getElementById("allowance").classList.contains("blocked"))), "the Sonnet line is styled as blocked");
    await s.expectHidden("#use-own-key", "'Use my own key' while runs are still free");
    await s.expectText("#options-summary", /^Options: Claude Sonnet 5\.5, free runs$/);
    await s.runExample();
    await s.expectResults();
    await s.expectText("#notices", /The free runs on Claude Opus for your network are used for now, so this ran on Claude Sonnet 5\.5\./, "the step-down note");
    s.check(!/allowance/i.test(await s.text("#notices")), "the step-down note says 'allowance'");
  },

  async degraded(s) {
    s.health = fixture("health-degraded");
    s.ner = { body: fixture("ner-degraded") };
    await s.open();
    await s.expectVisible("#banner", "the banner");
    await s.expectText("#banner", /Offline mode/);
    await s.expectText("#caveat", /200-term sample/, "the sample wording");
    await s.loadExample();
    s.check((await s.page.inputValue("#text")).startsWith(EXAMPLE_SAMPLE_START), "offline mode did not load the sample example");
    await s.run();
    await s.expectResults();
    await s.expectVisible("#banner", "the banner after the run");
    await s.expectText("#notices", /Part of the search \(matching by meaning\) was unavailable for this run, so terms were found by keyword only\./, "the weaker-search note");
    const shown = await s.page.evaluate(() => ["banner", "notices", "warnings-box", "results-summary", "results-takeaway"].map((id) => document.getElementById(id).innerText).join(" "));
    s.check(!/BM25|retrieval|build_indexes|no index at/i.test(shown), `search jargon or the operator's reason is shown outside Run details: ${JSON.stringify(shown.slice(0, 200))}`);
    await s.expectText("#results-takeaway", /^No model chose here: each mention shows its top search result\.$/, "the search-only line");
    s.check((await s.page.locator("#cards .card.is-search .card-id").count()) > 0, "search-only cards show no top search result");
    s.check((await s.page.locator("#cards .card-conf").count()) === 0, "a search-only card shows a confidence");
  },

  async fallback(s) {
    s.health = fixture("health");
    s.ner = { body: fixture("ner-fallback") };
    await s.open();
    await s.runExample();
    await s.expectResults();
    await s.expectVisible("#notices", "the notice about the model that answered");
    await s.expectText("#notices", /Claude Opus 5\.5 was busy, so Claude Sonnet 5\.5 read your text instead\./, "the extraction step-down");
    await s.expectText("#notices", /Claude Sonnet 5\.5 was busy part of the way through, so Gemini 3\.1 Flash Lite chose the terms for the remaining mentions\./, "the selection step-down");
    await s.expectText("#warnings-box", /The model also named "Hydroxykynurenine", which does not appear in your text word for word, so it was left out\./, "the rewritten warning");
    s.check(!/you asked for/.test(await s.text("#results")), "the results still say 'you asked for'");
  },

  async "own-key-restored"(s) {
    s.health = fixture("health");
    s.ner = { body: (request) => ({ ...fixture("ner-own"), provider: request.provider }) };
    await s.open();
    await s.chooseOwnKey("anthropic");
    await s.expectText("#options-summary", /^Options: your Anthropic key$/);
    await s.page.reload({ waitUntil: "networkidle" });
    s.check(await s.optionsOpen(), "Options is closed after a reload with an own key chosen");
    await s.expectText("#options-summary", /^Options: your Anthropic key$/, "the Options summary after the reload");
    await s.expectVisible("#own-key", "the key field after the reload");
    s.check((await s.page.inputValue("#provider")) === "anthropic", "the payer choice was not restored");
    s.check((await s.page.inputValue("#own-key")) === FAKE_KEY, "the key was not restored");
    await s.expectText("#run-hint", /^Runs on your Anthropic key\. Use public or de-identified text: it is sent to Anthropic and not saved by this site\.$/);
    await s.expectHidden("#payer", "the payer hold");
    const stored = await s.page.evaluate(() => Object.keys(window.sessionStorage).concat(Object.keys(window.localStorage)));
    s.check(stored.every((k) => k === "xwalk-ner-own-key" || k === "xwalk-ner-payer"), `unexpected storage entries: ${stored.join(", ")}`);
    await s.runExample();
    await s.expectResults();
    s.check(s.posts.length === 1, `expected one POST, saw ${s.posts.length}`);
    const post = s.posts[0] || { body: {}, authorization: "" };
    s.check(post.authorization === `Bearer ${FAKE_KEY}`, "the own-key run did not carry the key");
    s.check(post.body && post.body.provider === "anthropic", "the own-key run did not name its provider");
  },

  // A saved own-key choice the site no longer offers: nothing runs until the visitor chooses.
  async "payer-hold"(s) {
    s.health = fixture("health-own-off");
    s.ner = { body: fixture("ner") };
    await s.page.addInitScript(
      ([keyEntry, payerEntry, key]) => {
        if (window.sessionStorage.getItem("smoke-seeded")) return;
        window.sessionStorage.setItem(keyEntry, JSON.stringify({ provider: "anthropic", key }));
        window.sessionStorage.setItem(payerEntry, "anthropic");
        window.sessionStorage.setItem("smoke-seeded", "1");
      },
      [OWN_KEY_ENTRY, PAYER_ENTRY, FAKE_KEY],
    );
    await s.open();
    await s.expectVisible("#payer", "the payer hold");
    await s.expectText("#payer-message", /no longer accepts your own Anthropic key/);
    await s.expectHidden("#run-hint", "the 'Free, no sign-up' hint during a hold");
    s.check(await s.page.evaluate(() => document.getElementById("run").disabled), "Run is enabled during a payer hold");
    s.check((await s.page.evaluate((k) => window.sessionStorage.getItem(k), OWN_KEY_ENTRY)) === null, "the key of a lost provider was kept");
    await s.page.fill("#text", "aspirin");
    await s.page.keyboard.press("Control+Enter");
    await s.page.waitForTimeout(300);
    s.check(s.posts.length === 0, "a run was sent during a payer hold");
    await s.page.click("#payer-site");
    await s.expectHidden("#payer", "the payer hold after choosing the site");
    await s.expectText("#options-summary", /free runs$/);
    await s.loadExample();
    await s.run();
    await s.expectResults();
    const post = s.posts[0] || { body: {}, authorization: "x" };
    s.check(post.authorization === "" && !(post.body && post.body.provider), "the run on the site's models carried a key or a provider");
  },

  // The same choice while the status cannot be read: held, with a way to try again.
  async "payer-unknown"(s) {
    s.health = "fail";
    s.expectedStatuses.add(500);
    await s.page.addInitScript(([payerEntry]) => window.sessionStorage.setItem(payerEntry, "anthropic"), [PAYER_ENTRY]);
    await s.open();
    await s.expectVisible("#payer", "the payer hold");
    await s.expectVisible("#payer-retry", "'Try again' in the hold");
    await s.expectVisible("#banner", "the banner");
    s.check(await s.page.evaluate(() => document.getElementById("run").disabled), "Run is enabled during a payer hold");
    s.health = fixture("health");
    await s.page.click("#payer-retry");
    await s.expectHidden("#payer", "the payer hold once health answers");
    await s.page.waitForLoadState("networkidle");
    await s.expectHidden("#banner", "the banner once health answers");
    s.check(await s.optionsOpen(), "Options is closed with a restored own-key choice");
    await s.expectText("#options-summary", /^Options: your Anthropic key$/);
  },

  async "own-key-refused"(s) {
    s.health = fixture("health");
    s.ner = { status: 401, body: fixture("error-own-refused") };
    s.expectedStatuses.add(401);
    await s.open();
    await s.chooseOwnKey("anthropic");
    await s.expectText("#own-key-note", /Sent only with your runs.*Kept in this tab.*Anthropic bills you/s, "the three facts about the key");
    await s.expectText("#selector-own", /^With your own key the top search result is used unless you choose a model here\.$/, "the line under the selector");
    // Closed again by the visitor: the error must bring the key field back.
    await s.page.click("#options > summary");
    await s.runExample();
    await s.expectVisible("#own-key-error", "the message at the key field");
    await s.expectText("#own-key-error", /^Anthropic did not accept this key\. Check that it was pasted whole, is still active, has credit, and may use Claude Opus 5\.5\.$/);
    await s.settled();
    await s.expectHidden("#error", "the error box for a key error");
    if (s.shot) await s.shot("refused");
    s.check(await s.optionsOpen(), "a refused key did not open Options");
    await s.expectVisible("#own-key", "the key field");
    s.check((await s.page.inputValue("#own-key")) === FAKE_KEY, "the refused key was removed from the field");
    s.check((s.posts[0] || {}).authorization === `Bearer ${FAKE_KEY}`, "the own-key run did not carry the key");
    s.check(!(await s.text("#own-key-error")).includes(FAKE_KEY), "the key is echoed in the message");
    s.ner = { status: 429, body: apiError("provider_rate_limited", "Anthropic is rate limiting this key; try again shortly.") };
    s.expectedStatuses.add(429);
    await s.run();
    await s.settled();
    await s.expectText("#own-key-error", /^Anthropic is rate limiting your key\. Wait a minute and try again, or choose a smaller model\.$/);
    s.ner = { status: 400, body: fixture("error-own-bad") };
    s.expectedStatuses.add(400);
    await s.run();
    await s.settled();
    await s.expectText("#own-key-error", /^This does not look like an API key\. Paste the whole key, with no spaces\.$/);
    await s.expectHidden("#error", "the error box for a key error");
    s.ner = { status: 429, headers: { "Retry-After": "600" }, body: apiError("visitor_limit_reached", "too many requests with your own key; try again in 10 minutes") };
    await s.run();
    await s.expectError("Too many runs with your own key.", { look: "notice", body: /Try again in 10 minutes\.$/ });
    await s.page.fill("#own-key", "short");
    await s.expectHidden("#own-key-error", "the key message after typing");
    await s.run();
    await s.expectText("#own-key-error", /^This does not look like an API key/);
    s.check(s.posts.length === 4, `a malformed key was sent (${s.posts.length} POSTs)`);
    await s.page.fill("#own-key", FAKE_KEY);
  },

  async "own-key-missing"(s) {
    s.health = fixture("health");
    await s.open();
    await s.page.click("#options > summary");
    await s.page.selectOption("#provider", "openrouter");
    await s.page.click("#options > summary");
    await s.runExample();
    await s.expectVisible("#own-key-error", "the message at the key field");
    await s.expectText("#own-key-error", /^Paste your OpenRouter API key, or switch back to the free runs\.$/);
    await s.expectHidden("#error", "the error box for a missing key");
    s.check(await s.optionsOpen(), "a missing key did not open Options");
    s.check(s.posts.length === 0, "a run without a key was sent");
    await s.page.click("#own-key-error button");
    s.check((await s.page.inputValue("#provider")) === "site", "'switch back to the free runs' did not choose the site");
    await s.expectHidden("#own-key-error", "the key message after switching back");
    await s.expectText("#options-summary", /free runs$/);
  },

  async "health-failing"(s) {
    s.health = "fail";
    s.expectedStatuses.add(500);
    s.ner = { body: fixture("ner") };
    await s.open();
    await s.expectVisible("#banner", "the banner");
    await s.expectText("#banner", /Could not reach the service\./);
    await s.expectVisible("#health-retry", "'Try again' in the banner");
    await s.expectHidden("#error", "the error box");
    await s.expectHidden("#caveat", "the claim about the catalog size");
    const top = await s.page.evaluate(() => document.getElementById("banner").getBoundingClientRect().top);
    s.check(top < (await s.page.evaluate(() => window.innerHeight)), "the banner is below the first screen");
    await s.page.fill("#text", "Clinic note: the patient has Alkaptonuria.");
    await s.run();
    await s.expectResults();
    s.health = "fail";
    await s.page.reload({ waitUntil: "networkidle" });
    s.health = fixture("health");
    await s.page.click("#health-retry");
    await s.expectHidden("#banner", "the banner after a retry that worked");
    await s.expectVisible("#caveat", "the trust line after a retry that worked");
  },

  // While a run is under way: the three steps say what they do, with a bar and a clock.
  async progress(s) {
    s.health = fixture("health");
    let release;
    const gate = new Promise((resolve) => {
      release = resolve;
    });
    const body = fixture("ner");
    await s.page.route("**/api/ner", async (route) => {
      await gate;
      await route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(body) });
    });
    await s.open();
    await s.expectHidden("#pipe .pipe-desc", "the step descriptions while idle");
    await s.runExample();
    await s.expectVisible("#progress", "the progress bar");
    await s.expectVisible("#pipe-find", "what the first step does");
    await s.expectText("#run .run-label", /^Mapping…$/, "the Run label while running");
    s.check(!/Running/.test(await s.page.locator("#status").innerText()), "the status line repeats 'Running'");
    await s.page.waitForTimeout(800); // the page scrolls the steps into view smoothly
    const seen = await s.page.evaluate(() => {
      const r = document.getElementById("progress").getBoundingClientRect();
      return r.top >= 0 && r.bottom <= window.innerHeight;
    });
    s.check(seen, "the progress bar is off screen during a run");
    if (s.shot) await s.shot("running");
    release();
    await s.expectResults();
    await s.expectHidden("#progress", "the progress bar after the run");
  },

  async "options-open"(s) {
    s.health = productionHealth();
    await s.open();
    await s.page.click("#options > summary");
    for (const id of ["provider", "model", "selector", "top-k"]) await s.expectVisible(`#${id}`, `#${id} inside Options`);
    const marker = () => s.page.evaluate(() => getComputedStyle(document.querySelector("#prompt-editor > summary"), "::before").transform);
    const closed = await marker();
    await s.page.click("#prompt-editor > summary");
    await s.expectVisible("#prompt", "the prompt editor");
    await s.page.waitForTimeout(400); // the marker turns in 150 ms
    s.check((await marker()) !== closed, "the prompt editor's marker does not turn when it opens");
    const size = await s.page.evaluate(() => parseFloat(getComputedStyle(document.getElementById("prompt")).fontSize));
    s.check(size >= 16, `the prompt box is ${size}px: iOS zooms the page on focus under 16px`);
    const options = await s.page.evaluate(() => [...document.getElementById("selector").options].map((o) => o.textContent));
    s.check(options.includes("Top search result, no model") && options.includes("Claude Opus 5.5 picks"), `selector options read ${options.join(" | ")}`);
    s.check(options.every((o) => !o.includes("/")), "a raw vendor/model id is shown in the selector list");
    await s.page.fill("#prompt", "x".repeat(4001));
    await s.page.fill("#text", "aspirin");
    await s.run();
    await s.expectVisible("#prompt-error", "the message at the prompt");
    s.check(s.posts.length === 0, "a too-long prompt was submitted");
    await s.page.click("#prompt-reset");
  },

  // Every kind of row, in the shape production answers.
  async cards(s) {
    await s.runShaped();
    const cards = await s.page.evaluate(() =>
      [...document.querySelectorAll("#cards .card")].map((c) => ({
        id: c.id,
        cls: c.className,
        text: c.innerText.replace(/\s+/g, " ").trim(),
        open: c.classList.contains("is-open"),
        expanded: c.querySelector(".card-more") ? c.querySelector(".card-more").getAttribute("aria-expanded") : null,
        height: Math.round(c.getBoundingClientRect().height),
        link: c.querySelector("a.card-id") ? { href: c.querySelector("a.card-id").href, target: c.querySelector("a.card-id").target, h: c.querySelector("a.card-id").getBoundingClientRect().height } : null,
        more: c.querySelector(".card-more") ? c.querySelector(".card-more").getBoundingClientRect().height : null,
      })),
    );
    s.check(cards.length === 7 && cards.every((c, i) => c.id === `card-e${i + 1}`), `cards are ${cards.map((c) => c.id).join(",")}`);
    const [aspirin, metformin, fish, brca, warfarin, garlic] = cards;
    const has = (card, pattern, what) => s.check(pattern.test(card.text), `${what}: ${JSON.stringify(card.text.slice(0, 140))}`);
    // matched: type and ontology in words, the ID as a link, the readable label, a quiet confidence
    has(aspirin, /^Chemical · ChEBI 0\.93 (The model's own confidence, not a measured accuracy: 0\.93 )?Aspirin → (is )?CHEBI:100 Aspirin \(salt\) Why, and 7 candidates$/, "the matched card");
    s.check(aspirin.link && aspirin.link.href === "https://example.org/term/CHEBI:100" && aspirin.link.target === "_blank", "the matched card's ID is not an external link");
    s.check(!/%/.test(cards.map((c) => c.text).join(" ")), "a confidence is shown as a percentage");
    s.check(!aspirin.open && aspirin.expanded === "false", "the matched card is open on arrival");
    // override: the tag, and it opens itself as the first piece of reasoning
    has(metformin, /not the top search hit/, "the override tag");
    s.check(metformin.open && metformin.expanded === "true", "the first override card did not open itself");
    has(metformin, /Search ranked metformin \(salt\) first; Claude Opus 5\.5 chose metformin instead\./, "the override sentence");
    has(metformin, /The reason for metformin\..*Chosen by Claude Opus 5\.5 from 7 candidates found by xwalk\./, "the reason and who chose");
    s.check(!/not the top search hit/.test(aspirin.text), "a top-hit pick is tagged as an override");
    // declined, not looked up, failed, to check
    has(fish, /^Food · FoodOn oily fish → no term fits Why, and 7 candidates$/, "the declined card");
    s.check(!fish.open, "more than one card opened itself");
    has(brca, /^Other BRCA1 → not looked up The model marked this mention, but this demo has no ontology for its type\. Genes, for example, are found but not mapped here\.$/, "the not-looked-up card");
    s.check(brca.more === null, "the not-looked-up card has something to expand");
    s.check(!/no ontology for type/.test(brca.text), "the raw server note is shown");
    has(warfarin, /^Chemical · ChEBI warfarin → could not be matched 7 candidates$/, "the failed card");
    has(garlic, /needs a check/, "the to-check tag");
    for (const c of cards) {
      if (c.more !== null) s.check(c.more >= 44, `${c.id}: the disclosure is ${c.more}px tall, under the 44px touch target`);
      if (c.link) s.check(c.link.h >= 24, `${c.id}: the ID link is ${c.link.h}px tall, under 24px`);
      if (!c.open && s.phone) s.check(c.height <= 150, `${c.id}: collapsed it is ${c.height}px tall on a phone`);
    }

    await s.expectText("#results-summary", /^7 mentions found: 3 matched, 1 to check, 1 where no term fits, 1 that could not be matched, 1 not looked up\.$/, "the summary");
    await s.expectText("#results-takeaway", /^In 1 of 4 matches Claude Opus 5\.5 chose a different term than the top search hit, and for 1 mention it found no fitting term rather than guess\.$/, "the takeaway");
    await s.expectText("#status", /^Done in 8\.8 s: 3\.3 s reading the text, 2\.3 s searching, 3\.2 s choosing terms\.$/, "the one-sentence status");
    await s.expectHidden("#results-cap", "the 25-cap line");
    await s.expectHidden("#stale", "the earlier-text line");
    await s.expectHidden("#warnings-box", "the notes box");
    const outside = await s.page.evaluate(() => ["results-summary", "results-takeaway", "status", "legend"].map((id) => document.getElementById(id).innerText).join(" "));
    s.check(!/candidate|retrieval|allowance|BM25/i.test(outside), `expert words above the cards: ${JSON.stringify(outside)}`);

    // the type is not said by colour alone
    const mention = await s.page.evaluate(() => {
      const m = document.querySelector('#highlighted a[href="#card-e2"]');
      const other = document.querySelector('#highlighted a[href="#card-e4"]');
      return { name: m.textContent, title: m.title, tags: document.querySelectorAll("#highlighted .mention-tag").length, other: other.textContent, count: document.querySelectorAll("#highlighted a.mention").length };
    });
    s.check(mention.name === "metformin (chemical)" && mention.title === "metformin: chemical", `the mention's accessible name is ${JSON.stringify(mention.name)}`);
    s.check(mention.other === "BRCA1 (other, not looked up)", `the unmapped mention's accessible name is ${JSON.stringify(mention.other)}`);
    s.check(mention.tags === 0 && mention.count === 7, "inline type pills are back, or a mention is not highlighted");
    await s.expectText("#legend", /Chemical\s*ChEBI · 219k terms/, "the legend");
    await s.expectText("#legend", /Other\s*not looked up/, "the legend for the unmapped type");
    const conf = await s.page.evaluate(() => {
      const c = document.querySelector("#card-e1 .card-conf");
      return { title: c.title, spoken: c.querySelector(".visually-hidden").textContent };
    });
    s.check(conf.title === "The model's own confidence, not a measured accuracy" && conf.spoken.startsWith(conf.title), "the confidence has no explanation");

    // after the run: one announcement, focus on the heading, nothing technical read out
    const after = await s.page.evaluate(() => ({ focus: document.activeElement && document.activeElement.id, said: document.getElementById("announce").textContent, live: document.getElementById("status").getAttribute("aria-live"), summaryLive: document.getElementById("results-summary").getAttribute("aria-live") }));
    s.check(after.focus === "results-heading", `after a run the focus is on #${after.focus}`);
    s.check(/^Done\. 7 mentions found/.test(after.said), `the announcement is ${JSON.stringify(after.said)}`);
    s.check(after.live === null && after.summaryLive === null, "the status or the summary is still a live region of its own");

    // expand and collapse
    await s.page.click("#card-e1 .card-more");
    await s.expectVisible("#card-detail-e1", "the opened card's detail");
    await s.expectText("#card-detail-e1", /The reason for Aspirin\./);
    s.check((await s.page.locator("#card-detail-e1 tbody tr:visible").count()) === 5, "the opened card does not list the top 5 candidates");
    s.check((await s.page.locator("#card-detail-e1 tr.is-picked").innerText()).includes("Aspirin (salt)"), "the chosen candidate is not marked");
    const wide = await s.page.locator("#card-detail-e1 td.cand-score").first().isVisible();
    s.check(wide === !s.phone, `scores and ranks are ${wide ? "shown" : "hidden"} at ${s.width}px`);
    await s.page.click("#card-detail-e1 .cands-toggle");
    s.check((await s.page.locator("#card-detail-e1 tbody tr:visible").count()) === 7, "'Show all' does not list every candidate");
    await s.page.click("#card-e1 .card-more");
    await s.expectHidden("#card-detail-e1", "the collapsed card's detail");
    await s.expectText("#expand-all", /^Expand all$/);
    await s.page.click("#expand-all");
    s.check((await s.page.locator("#cards .card-detail:visible").count()) === 6, "'Expand all' did not open every card that has detail");
    await s.expectText("#card-detail-e3", /Claude Opus 5\.5 looked at 7 candidates and declined to pick one\./, "the declined card's sentence");
    await s.expectText("#card-detail-e5", /The model did not finish choosing a term\./, "the failed card's sentence");
    s.check(!/cut off by the request deadline/.test(await s.text("#card-detail-e5")), "the failed card prints the raw reason");
    s.check(!/\b(null|undefined|NaN)\b/.test(await s.text("#results")), "the results print null, undefined or NaN");
    await s.expectText("#expand-all", /^Collapse all$/);
    if (s.shot) await s.shot("expanded");
    await s.page.click("#expand-all");
    s.check((await s.page.locator("#cards .card-detail:visible").count()) === 0, "'Collapse all' left a card open");

    // the numbers are one tap away, after the cards
    s.check(!(await s.page.evaluate(() => document.getElementById("run-details").open)), "Run details is open on arrival");
    await s.page.click("#run-details > summary");
    await s.expectText("#run-details", /Extracted by Claude Opus 5\.5\. Terms picked by Claude Opus 5\.5\./);
    await s.expectText("#run-details", /keyword \(BM25\) ok, embeddings \(dense\) ok/);
    await s.expectText("#run-details", /1,134 in, 187 out \(extraction\)/);
    s.check(!/cost unknown|\(live\)/.test(await s.text("#run-details")), "Run details still says 'cost unknown' or '(live)'");
    const order = await s.page.evaluate(() => document.getElementById("cards").compareDocumentPosition(document.getElementById("run-details")) & Node.DOCUMENT_POSITION_FOLLOWING);
    s.check(order !== 0, "Run details is not after the cards");
  },

  async "no-entities"(s) {
    s.health = productionHealth();
    s.ner = { body: shapedResponse("hello, testing", [], { timing_ms: { extract: 900, map: 0, total: 900 } }) };
    await s.open();
    await s.page.fill("#text", "hello, testing");
    await s.run();
    await s.expectVisible("#no-entities", "the no-entities panel");
    await s.settled();
    await s.expectText("#no-entities-title", /^No chemicals, diseases or foods found in this text\.$/);
    await s.expectText("#no-entities-body", /The demo looks for those kinds of terms/);
    for (const id of ["text-panel", "legend", "results-summary", "cards-tools", "error", "results-takeaway"]) await s.expectHidden(`#${id}`, `#${id} with no entities`);
    s.check((await s.page.locator("#cards .card").count()) === 0, "cards with no entities");
    s.check(/^Done\. No chemicals, diseases or foods found/.test(await s.page.evaluate(() => document.getElementById("announce").textContent)), "no entities is not announced");
    await s.page.click("#no-entities-example");
    s.check((await s.page.inputValue("#text")).startsWith(EXAMPLE_FULL_START), "'Load example' in the panel did not fill the text box");
    await s.expectVisible("#stale", "the earlier-text line once the example is loaded");
  },

  // Each failure has its own words and one way on. None clears the text box.
  async errors(s) {
    s.health = productionHealth();
    s.failedLoads = true;
    for (const status of [400, 429, 500, 502, 503, 504]) s.expectedStatuses.add(status);
    await s.open();
    await s.page.fill("#text", "aspirin for a fever");
    const cases = [
      [{ status: 429, headers: { "Retry-After": "5" }, body: apiError("visitor_limit_reached", "too many requests at once; try again in a few seconds") }, "One run at a time.", { look: "notice", action: "Try again", body: /from your network/ }],
      [{ status: 429, body: apiError("visitor_limit_reached", "you have used this hour's requests; try again in 23 minutes") }, "The free runs for your network this hour are used.", { look: "notice", ownKey: true, body: /^You can run again later this hour\.$/ }],
      [{ status: 502, body: apiError("provider_error", "the model provider failed (OverloadedError: overloaded)") }, "The model is busy right now.", { action: "Try again" }],
      [{ status: 502, body: apiError("provider_error", "the model did not answer within 60 seconds") }, "The model took too long.", { action: "Try again" }],
      [{ status: 502, body: apiError("provider_error", "the model provider failed (LLMError: bad answer)") }, "The model could not finish this run.", { action: "Try again" }],
      [{ status: 429, body: apiError("call_budget_exhausted", "the limit of 40 model calls per request was reached before the model answered; try again later") }, "The model is busy right now.", { action: "Try again" }],
      [{ status: 504, raw: "<html>Gateway Timeout</html>" }, "The run took too long and was stopped.", { action: "Try again" }],
      [{ status: 500, body: apiError("internal", "internal error; see the function log") }, "The demo hit an error on its side.", { action: "Try again", body: /open an issue/ }],
      [{ status: 502, raw: "Bad Gateway" }, "The demo hit an error on its side.", { action: "Try again" }],
      [{ status: 503, body: apiError("not_configured", "selector 'jev' has no client") }, "The demo hit an error on its side.", { action: "Try again" }],
      [{ abort: true }, "You seem to be offline.", { action: "Try again" }],
      [{ status: 400, body: apiError("bad_top_k", "'top_k' must be an integer") }, "Something went wrong.", { action: "Try again", body: /^'top_k' must be an integer$/ }],
    ];
    let first = true;
    for (const [ner, title, expected] of cases) {
      s.ner = ner;
      // After the first, the box's own action is the way to run again.
      if (first || !(await s.visible("#error-action"))) await s.run();
      else await s.page.click("#error-action");
      first = false;
      await s.expectError(title, expected);
    }
    s.check(!/function log/.test(await s.text("#error")), "the visitor is sent to a function log");
    const issue = await s.page.evaluate(() => {
      const a = [...document.querySelectorAll("#error a")];
      return a.length;
    });
    s.check(issue === 0, "a link from an earlier error is left in the box");
    // What the server says about the text is said at the text box.
    s.ner = { status: 400, body: fixture("error") };
    await s.page.click("#error-action");
    await s.settled();
    await s.expectVisible("#text-error", "the message at the text box for a server refusal");
    await s.expectHidden("#error", "the error box for a text error");
    await s.page.fill("#text", "aspirin");
    await s.expectHidden("#text-error", "the text message after typing");
    // Dismiss does what it says.
    s.ner = { status: 500, body: apiError("internal", "internal error; see the function log") };
    await s.run();
    await s.expectError("The demo hit an error on its side.", { action: "Try again" });
    s.check((await s.page.evaluate(() => document.activeElement && document.activeElement.id)) === "error-action", "the error's action does not have the focus");
    if (s.shot) await s.shot("site-fault");
    await s.page.click("#error-dismiss");
    await s.expectHidden("#error", "the error box after Dismiss");
  },

  // A failed run after a good one: the good result stays, the text stays.
  async "keeps-result"(s) {
    await s.runShaped();
    s.expectedStatuses.add(500);
    s.ner = { status: 500, body: apiError("internal", "internal error; see the function log") };
    await s.run();
    await s.expectError("The demo hit an error on its side.", { action: "Try again" });
    await s.expectVisible("#results", "the last good result after a failed run");
    s.check((await s.page.locator("#cards .card").count()) === 7, "the cards of the last good result are gone");
    await s.expectHidden("#stale", "the earlier-text line (the text has not changed)");
    s.check((await s.page.inputValue("#text")) === SHAPED_TEXT, "the failed run changed the text box");
    const above = await s.page.evaluate(() => document.getElementById("error").getBoundingClientRect().top < document.getElementById("results").getBoundingClientRect().top);
    s.check(above, "the error box is below the old result");
    s.ner = { body: shaped() };
    await s.page.click("#error-action");
    await s.expectResults();
    await s.expectHidden("#error", "the error box after a retry that worked");
  },

  // A result belongs to the text it was made from.
  async stale(s) {
    await s.runShaped();
    await s.expectHidden("#stale", "the earlier-text line on a fresh result");
    await s.page.fill("#text", `${SHAPED_TEXT} And more.`);
    await s.expectVisible("#stale", "the earlier-text line after an edit");
    await s.expectText("#stale", /^These results are for the earlier text\. Run again to map the text that is in the box now\.$/);
    s.check(await s.page.evaluate(() => document.getElementById("results").classList.contains("is-stale") && Number(getComputedStyle(document.querySelector(".results-body")).opacity) < 1), "the earlier result is not dimmed");
    if (s.shot) await s.shot("edited");
    await s.page.fill("#text", SHAPED_TEXT);
    await s.expectHidden("#stale", "the earlier-text line once the text is the same again");
    await s.page.click("#clear-text");
    await s.expectVisible("#stale", "the earlier-text line after Clear");

    // Typing while a request is under way: the answer does not take the caret or the scroll.
    const gate = s.gate();
    s.ner = { gate: gate.promise, body: shaped() };
    await s.page.fill("#text", SHAPED_TEXT);
    await s.run();
    await s.expectVisible("#progress", "the progress bar");
    await s.page.waitForTimeout(700); // the smooth scroll to the steps ends
    await s.page.locator("#text").scrollIntoViewIfNeeded();
    await s.page.focus("#text");
    // (The End key would scroll the page on macOS instead of moving the caret.)
    await s.page.evaluate(() => {
      const box = document.getElementById("text");
      box.setSelectionRange(box.value.length, box.value.length);
    });
    await s.page.keyboard.type(" Also cumin.");
    await s.page.waitForTimeout(300);
    const before = await s.page.evaluate(() => window.scrollY);
    gate.release();
    await s.settled();
    await s.page.waitForTimeout(700);
    const after = await s.page.evaluate(() => ({ y: window.scrollY, focus: document.activeElement && document.activeElement.id, said: document.getElementById("announce").textContent }));
    s.check(after.focus === "text", `the answer took the focus from the text box (now #${after.focus})`);
    s.check(Math.abs(after.y - before) <= 2, `the answer scrolled the page while the visitor typed (${before} -> ${after.y})`);
    s.check(/^Done\./.test(after.said) && /as it was when you ran it/.test(after.said), `the announcement is ${JSON.stringify(after.said)}`);
    await s.expectVisible("#stale", "the earlier-text line on a result for a text edited meanwhile");
    s.check((await s.page.inputValue("#text")).endsWith("Also cumin."), "the typed text was lost");
    const stored = await s.page.evaluate(() => Object.keys(window.sessionStorage).concat(Object.keys(window.localStorage)));
    s.check(stored.length === 0, `the page stored something: ${stored.join(", ")}`);
    s.check(!/aspirin|metformin/i.test(s.page.url()), "the text is in the URL");
  },

  // ---------- the recorded example on arrival ----------

  // Health first (the usual order once the function is warm): the example is there on
  // arrival, labelled, with no scroll, no focus and no request. Editing the text marks
  // it as the example's; a real run replaces it entirely.
  async "example-arrival"(s) {
    s.health = productionHealth();
    s.ner = { body: shaped() };
    await s.open();
    const seen = await s.expectExample("on arrival");
    s.check(seen.scrollY === 0, `the page scrolled on arrival (${seen.scrollY})`);
    s.check(seen.focus === "BODY", `something took the focus on arrival (#${seen.focus})`);
    s.check(seen.stale === null, "the example is marked as for another text on arrival");
    s.check(seen.banner === null, `a banner is shown for the example: ${JSON.stringify(seen.banner)}`);
    s.check(/ChEBI · 219k terms/.test(seen.legend), `the legend has no term counts: ${JSON.stringify(seen.legend)}`);
    await s.expectText("#results-summary", /^7 mentions found: 7 matched\.$/);
    await s.expectHidden("#notices", "'About this run' for the example");
    s.check(!(await s.page.evaluate(() => document.getElementById("run-details").open)), "the recorded run's details are open");
    if (s.width === 390) {
      // One short scroll: the result starts within a screen and a half.
      s.check(seen.top <= seen.screen * 1.5, `the example result starts at ${seen.top}px of a ${seen.screen}px screen`);
      const run = await s.page.evaluate(() => document.getElementById("run").getBoundingClientRect().bottom);
      s.check(run <= seen.screen, `Run is below the first screen (${Math.round(run)})`);
    }
    if (s.shot) await s.shot("arrival");

    // The visitor's own text: the result below is said to be the example's.
    await s.page.fill("#text", "Warfarin and grapefruit juice.");
    await s.expectVisible("#stale", "the line that says whose result this is");
    await s.expectText("#stale", EXAMPLE_STALE);
    const dim = await s.page.evaluate(() => Number(getComputedStyle(document.querySelector(".results-body")).opacity));
    s.check(dim >= 0.85 && dim < 1, `an earlier result is dimmed to ${dim}; a light dim (0.85) is wanted`);
    if (s.shot) await s.shot("edited");
    await s.page.fill("#text", EXAMPLE_FILE.text);
    await s.expectHidden("#stale", "the line once the box holds the example again");
    await s.page.click("#clear-text");
    await s.expectVisible("#stale", "the line after Clear");
    s.check((await s.shown()).heading === "Example result", "Clear changed the heading");

    // A failed run leaves the example where it is, still labelled.
    s.ner = { status: 500, body: apiError("internal", "boom") };
    s.expectedStatuses.add(500);
    await s.page.fill("#text", SHAPED_TEXT);
    await s.run();
    await s.expectError("The demo hit an error on its side.", { action: "Try again" });
    const kept = await s.shown();
    s.check(kept.visible && kept.kind === "example" && EXAMPLE_NOTE.test(kept.note || "") && EXAMPLE_STALE.test(kept.stale || ""), "a failed run changed how the example is labelled");

    // A real run replaces it: nothing of the example is left.
    s.ner = { body: shaped() };
    await s.page.click("#error-action");
    await s.expectResults();
    const real = await s.shown();
    s.check(real.kind === "run" && real.heading === "Results", `after a run the heading is ${JSON.stringify(real.heading)}`);
    s.check(real.note === null && real.stale === null, "the example's label is still shown after a run");
    s.check(/^Done in /.test(real.status || ""), `after a run the status is ${JSON.stringify(real.status)}`);
    s.check(real.details === "Run details", `after a run the details are titled ${JSON.stringify(real.details)}`);
    s.check(real.cards === SHAPED_SPECS.length, `after a run there are ${real.cards} cards`);
    s.check(!/Recorded|Example result/.test(await s.text("#results")), "the result of a run still mentions the example");
    // A later health answer does not bring the example back.
    await s.page.waitForTimeout(400);
    s.check((await s.shown()).kind === "run", "the example came back after the run");
  },

  // example.json first, health later (a cold start): the example does not wait. When
  // health then says full catalog it stays; a sample catalog or offline mode removes it.
  async "example-health-late"(s) {
    let gate = s.gate();
    s.healthGate = gate.promise;
    s.health = productionHealth();
    await s.openHeld();
    // The names and counts come from example.json until health answers.
    await s.expectExample("before health answers");
    await s.expectText("#caveat", /^Searches more than 260,000 real ontology terms\./, "the static trust line");
    gate.release();
    await s.page.waitForLoadState("networkidle");
    await s.expectText("#caveat", /^Searches 261,018 real ontology terms\./, "the trust line after health");
    const after = await s.expectExample("after health said full catalog");
    s.check(after.scrollY === 0 && after.focus === "BODY", "health arriving moved the page or the focus");

    for (const [what, health, start] of [
      ["a sample catalog", fixture("health"), EXAMPLE_SAMPLE_START],
      ["offline mode", { ...productionHealth(), mode: "offline" }, EXAMPLE_SAMPLE_START],
    ]) {
      gate = s.gate();
      s.healthGate = gate.promise;
      s.health = health;
      await s.openHeld();
      await s.expectExample(`before health says ${what}`);
      gate.release();
      await s.page.waitForLoadState("networkidle");
      // Gone, and the text it put in the box with it: the old behaviour.
      await s.expectNoExample(`after health said ${what}`);
      await s.loadExample();
      s.check((await s.page.inputValue("#text")).startsWith(start), `Example on ${what} loaded ${JSON.stringify((await s.page.inputValue("#text")).slice(0, 30))}`);
    }

    // Text the visitor typed over the example is theirs: it stays when the example goes.
    gate = s.gate();
    s.healthGate = gate.promise;
    s.health = fixture("health");
    await s.openHeld();
    await s.expectExample("before health, about to type");
    await s.page.fill("#text", "my own words");
    gate.release();
    await s.page.waitForLoadState("networkidle");
    await s.expectNoExample("after health said sample, with typed text", { text: "my own words" });
    s.healthGate = null;
  },

  // Health first, example.json later (a slow connection): typed text is never
  // overwritten, nothing takes the focus or scrolls, and a deployment that is not the
  // full catalog never shows it. A run started meanwhile is not covered by it either.
  async "example-late"(s) {
    let gate = s.gate();
    s.example = { gate: gate.promise };
    s.health = productionHealth();
    s.ner = { body: shaped() };
    await s.openHeld();
    await s.expectNoExample("before example.json arrives");
    await s.page.locator("#text").click();
    await s.page.keyboard.type("Warfarin and");
    const before = await s.shown();
    gate.release();
    await s.expectVisible('#results[data-kind="example"] .card', "the example once it arrives");
    const late = await s.shown();
    s.check(late.text === "Warfarin and", `typed text was overwritten: ${JSON.stringify(late.text.slice(0, 40))}`);
    s.check(late.focus === "text" && late.scrollY === before.scrollY, `the example took the focus or scrolled (#${late.focus}, ${before.scrollY} -> ${late.scrollY})`);
    s.check(EXAMPLE_STALE.test(late.stale || ""), `beside typed text the example is not marked as the example's: ${JSON.stringify(late.stale)}`);
    s.check(late.heading === "Example result" && EXAMPLE_NOTE.test(late.note || ""), "the late example is not labelled");
    await s.page.keyboard.type(" grapefruit");
    s.check((await s.page.inputValue("#text")) === "Warfarin and grapefruit", "typing after the example arrived went wrong");

    // Not the full catalog: it never appears, however late it arrives.
    gate = s.gate();
    s.example = { gate: gate.promise };
    s.health = fixture("health");
    await s.openHeld();
    gate.release();
    await s.page.waitForLoadState("networkidle");
    await s.expectNoExample("on a sample catalog");

    // A run started before it arrived: the run's result is not replaced or relabelled.
    gate = s.gate();
    s.example = { gate: gate.promise };
    s.health = productionHealth();
    await s.openHeld();
    await s.page.fill("#text", SHAPED_TEXT);
    await s.run();
    // (expectResults would wait for the network to go quiet, and example.json is held.)
    await s.expectVisible('#results[data-kind="run"] .card', "the result of a run made before the example arrived");
    gate.release();
    await s.settled();
    await s.page.waitForTimeout(300);
    const ran = await s.shown();
    s.check(ran.kind === "run" && ran.heading === "Results" && ran.note === null && ran.cards === SHAPED_SPECS.length, `the example replaced a run's result (${ran.heading}, ${ran.cards} cards)`);
    s.example = "pass";
  },

  // Health fails, or never answers: nothing is known about the deployment, the static
  // page is the production copy, and the example stays, labelled, with a working form.
  async "example-no-health"(s) {
    s.health = "fail";
    s.expectedStatuses.add(500);
    await s.open();
    await s.expectVisible("#banner", "the banner about the status");
    await s.expectExample("with health failing");
    // The retry succeeds: the example is still there and now has health's counts.
    s.health = productionHealth();
    await s.page.click("#health-retry");
    await s.page.waitForLoadState("networkidle");
    await s.expectHidden("#banner", "the banner after the retry");
    await s.expectExample("after the retry");

    // Never: the request hangs for as long as the page lives.
    s.healthGate = new Promise(() => {});
    await s.openHeld();
    const seen = await s.expectExample("with health never answering");
    s.check(seen.banner === null, "a banner is shown while health is still loading");
    s.ner = { body: shaped() };
    await s.run();
    await s.expectVisible('#results[data-kind="run"] .card', "the result of a run without health");
    s.check(s.posts.length === 1 && s.posts[0].body.text === EXAMPLE_FILE.text, "Run did not send the example text");
    s.check((await s.shown()).heading === "Results", "a run without health kept the example's heading");
  },

  // example.json missing or broken: the page is as it is without it.
  async "example-missing"(s) {
    for (const kind of ["missing", "garbage"]) {
      s.example = kind;
      s.health = productionHealth();
      s.ner = { body: shaped() };
      await s.open();
      await s.expectNoExample(`example.json ${kind}`);
      await s.expectText("#caveat", /^Searches 261,018 real ontology terms\./, "the trust line");
      await s.loadExample();
      s.check((await s.page.inputValue("#text")).startsWith(EXAMPLE_FULL_START), `Example did not load with example.json ${kind}`);
      await s.page.fill("#text", SHAPED_TEXT);
      await s.run();
      await s.expectResults();
      s.check((await s.shown()).heading === "Results", "the heading after a run");
    }
  },

  // The panel after the cards: what that was and where the library is.
  async "closing-panel"(s) {
    await s.runShaped();
    await s.expectVisible("#next", "the closing panel");
    await s.expectText("#next-title", /^That lookup is xwalk$/);
    await s.expectText("#next-body", /^The model never writes an ID\. xwalk searched 261,018 ontology terms by keyword and by meaning, and the model chose among real candidates\. xwalk is an open-source Python library; you can run the same lookup on your own vocabulary\.$/);
    const link = await s.page.evaluate(() => {
      const a = document.getElementById("next-github");
      const order = document.getElementById("cards").compareDocumentPosition(document.getElementById("next")) & Node.DOCUMENT_POSITION_FOLLOWING;
      return { text: a.textContent, href: a.getAttribute("href"), target: a.target, rel: a.rel, height: a.getBoundingClientRect().height, order, panels: document.querySelectorAll("#next").length };
    });
    s.check(link.text === "xwalk on GitHub" && link.href === "https://github.com/jan3657/xwalk?utm_source=xwalk.si", `the primary link is ${JSON.stringify(link)}`);
    s.check(link.target === "_blank" && /noopener/.test(link.rel), "the GitHub link does not open in a new tab with rel noopener");
    s.check(link.height >= 44, `the GitHub link is ${link.height}px tall`);
    s.check(link.order !== 0, "the closing panel is not after the cards");
    s.check(!/pip install|Built by/.test(await s.text("#next")), "the closing panel has an author or a pip line");
    if (s.shot) {
      await s.page.locator("#next").scrollIntoViewIfNeeded();
      await s.shot("panel");
    }
    await s.page.click("#next-try");
    await s.page.waitForTimeout(900);
    const tried = await s.page.evaluate(() => {
      const r = document.getElementById("text").getBoundingClientRect();
      return { focus: document.activeElement && document.activeElement.id, inView: r.top >= 0 && r.top < window.innerHeight };
    });
    s.check(tried.focus === "text" && tried.inView, `'Try your own text' left the focus on #${tried.focus}, text box in view: ${tried.inView}`);

    // Only what is true of the result: no model chose, and half the search was off.
    s.ner = { body: shaped({ entities: shaped().entities.map((e) => ({ ...e, pick: null })), retrieval: { bm25: { status: "ok" }, dense: { status: "unavailable", reason: "no index" } } }) };
    await s.run();
    await s.settled();
    await s.expectText("#next-body", /xwalk searched 261,018 ontology terms by keyword, and each mention shows the top search result\./, "the closing panel for a keyword-only run with no model");

    // Nothing was looked up: there is no lookup to explain.
    s.ner = { body: shaped({ entities: [] }) };
    await s.run();
    await s.settled();
    await s.expectVisible("#no-entities", "the no-entities panel");
    await s.expectHidden("#next", "the closing panel with no mentions");
  },

  // 25 entities is the per-run cap, and the page says so beside the summary.
  async cap(s) {
    const words = Array.from({ length: 25 }, (_, i) => `compound${String.fromCharCode(97 + i)}`);
    const text = `${words.join(", ")}.`;
    s.health = productionHealth();
    s.ner = { body: shapedResponse(text, words.map((w) => [w, "chemical", "match", 0]), { warnings: ["27 entities found; only the first 25 are kept"] }) };
    await s.open();
    await s.page.fill("#text", text);
    await s.run();
    await s.expectResults();
    s.check((await s.page.locator("#cards .card").count()) === 25, "not 25 cards");
    await s.expectVisible("#results-cap", "the line about the 25-entity cap");
    await s.expectText("#results-cap", /^25 is the most one run returns, so a long text may hold more mentions than are shown\.$/);
    await s.expectText("#warnings-box", /The model found 27 mentions; only the first 25 are shown\./, "the rewritten truncation warning");
    await s.expectHidden("#results-takeaway", "the takeaway when the model overruled nothing");
    s.check((await s.page.locator("#cards .card.is-open").count()) === 0, "a card opened itself with no override and nothing declined");
  },

  // The phone locks or the app changes during a run.
  async "page-hidden"(s) {
    s.health = productionHealth();
    s.failedLoads = true;
    let gate = s.gate();
    s.ner = { gate: gate.promise, abort: true };
    await s.open();
    await s.page.fill("#text", SHAPED_TEXT);
    await s.run();
    await s.expectVisible("#progress", "the progress bar");
    await s.setVisibility("hidden");
    await s.setVisibility("visible");
    gate.release();
    await s.expectError("The connection dropped before the result arrived.", { action: "Run again", body: /Your text is still here\./ });
    s.check((await s.page.inputValue("#text")) === SHAPED_TEXT, "the dropped run changed the text box");
    // Hidden and shown again, and the request survives: the result arrives as usual.
    gate = s.gate();
    s.ner = { gate: gate.promise, body: shaped() };
    await s.page.click("#error-action");
    await s.expectVisible("#progress", "the progress bar of the second run");
    await s.setVisibility("hidden");
    await s.setVisibility("visible");
    gate.release();
    await s.expectResults();
    await s.expectHidden("#error", "the error box after the run that worked");
    await s.expectHidden("#progress", "the progress bar after the run");
  },
};


// ---------- runner ----------

async function main() {
  const opts = parseArgs(process.argv.slice(2));
  const names = Object.keys(STATES).filter((n) => !opts.only || opts.only.includes(n));
  if (opts.list) {
    console.log(Object.keys(STATES).join("\n"));
    return;
  }
  if (opts.only && names.length !== opts.only.length) throw new Error(`unknown state in --only; known: ${Object.keys(STATES).join(", ")}`);
  const base = opts.positional[0].replace(/\/+$/, "") + "/";
  if (opts.shots) mkdirSync(opts.shots, { recursive: true });

  const { chromium } = loadPlaywright();
  const channel = process.env.PLAYWRIGHT_CHANNEL || undefined;
  const browser = await chromium.launch(channel ? { channel } : {});
  let failed = 0;
  try {
    for (const width of opts.widths) {
      const phone = width < 700;
      for (const name of names) {
        const context = await browser.newContext({
          viewport: { width, height: phone ? 844 : 800 },
          deviceScaleFactor: 1,
          colorScheme: opts.dark ? "dark" : "light",
          // A phone: touch, no hover, so the page's "no keyboard on tap" path is the one tested.
          hasTouch: phone,
          isMobile: phone,
        });
        const page = await context.newPage();
        const s = new Session(page, width, base);
        const shotPath = (label) => path.join(opts.shots, `${name}${label ? `-${label}` : ""}-${width}${opts.dark ? "-dark" : ""}.png`);
        if (opts.shots) s.shot = (label) => page.screenshot({ path: shotPath(label), fullPage: true });
        await s.install();
        try {
          await STATES[name](s);
          if (opts.shots) await page.screenshot({ path: shotPath(""), fullPage: true });
          await s.common();
        } catch (err) {
          s.failures.push(`threw: ${err && err.message ? err.message.split("\n")[0] : String(err)}`);
        }
        await context.close();
        const label = `${name} @ ${width}px`;
        if (s.failures.length) {
          failed += 1;
          console.log(`FAIL  ${label}`);
          for (const f of s.failures) console.log(`        ${f}`);
        } else {
          console.log(`ok    ${label}`);
        }
      }
    }
  } finally {
    await browser.close();
  }
  const total = names.length * opts.widths.length;
  console.log(failed ? `\n${failed} of ${total} failed` : `\nall ${total} passed`);
  if (failed) process.exit(1);
}

main().catch((err) => {
  console.error(err && err.stack ? err.stack : String(err));
  process.exit(1);
});
