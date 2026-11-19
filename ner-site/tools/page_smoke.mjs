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
// The review states (no-script, bfcache-payer, run-locks-payer, health-out-of-order,
// late-health-keeps-text, stalled-body, steps-and-allowance) each hold one failure that a
// review found and that must not come back: a native form submission without the script,
// a page restored from the back-forward cache after its key was forgotten, the payer or
// model changed during a run, health answers arriving out of order or late, and an
// answer whose body stalls. The states of the second review (restore-storage-fails,
// restore-models, health-during-run, restore-stale-health, run-claims, payer-entry-clean)
// do the same for: a restored page whose session storage cannot be written or read, model
// ids changed in another copy of the page, a health answer released during a run, a
// restored page with an older health answer than the tab's saved payer, what a result
// claims about picks and search hits, and what is written back to the payer entry.
//
// The guide-* states cover the figure that #pipe expands into: collapsed on arrival;
// played to the end for a mention the selector overruled and one that was the top hit;
// a declined pick; a mention with no ontology; retrieval alone and Jev; one retriever;
// the dashed boxes and a page with no result; a run started while it is open; a new
// result resetting it; and reduced motion. They answer the API from the recorded example
// (public/example.json) and from fixtures/guide-run.json, a recorded production run.
//
// The cache states cover what the server's caches and the recorded example save
// (ner_site/cache.py). cache-hit: an answer that came whole from the cache says so in
// place of the timing sentence, and "Run details" shows what was reused, also for a run
// that reused only some searches or picks. cache-live does the same against the server's
// own API (the second identical request). example-pressed, example-live and
// example-early: "Map this text" on the untouched example sends nothing and shows the
// recorded result with "Run it live"; an edited text, a changed option, an own key, a
// second press after a live run, and a press before health has answered all run for real.
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
// A real run recorded from production on 19 Nov 2026, with the text it was made from:
// ten mentions, two the selector declined, one of a type with no ontology, and one
// that only one of the two searches found candidates for. The guide-* states use it.
const GUIDE_RUN = JSON.parse(readFileSync(path.join(FIXTURES, "guide-run.json"), "utf8"));
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
    // Answers for the next reads of /api/health, one each, in the order the reads are
    // made: { gate, health }. When it is empty, `health` and `healthGate` answer.
    this.healthQueue = [];
    this.noScript = false; // app.js is not served: the page as it is without it
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
      // Without app.js: the load fails, and the policy refuses the form's submission.
      if (this.noScript && (where.includes("/app.js") || /app\.js|form-action/.test(text))) return;
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
    await page.route("**/app.js", (route) => (this.noScript ? route.abort("failed") : route.continue()));
    await page.route("**/api/health", async (route) => {
      const queued = this.healthQueue.shift();
      if (queued && queued.gate) await queued.gate;
      else if (!queued && this.healthGate) await this.healthGate;
      const health = queued ? queued.health : this.health;
      if (health === "pass") return route.continue();
      if (health === "fail") return route.fulfill({ status: 500, contentType: "text/plain", body: "down" });
      return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(health) });
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
    if (counts) this.check(/^The model never writes an ID\. xwalk looked up each mention by keyword and by meaning, and the model chose among real candidates\. The catalog holds 261,018 terms\. xwalk is an open-source Python library; you can run the same lookup on your own vocabulary\.$/.test(seen.next || ""), `${what}: the closing panel says ${JSON.stringify(seen.next)}`);
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
    // Without the script Run stays as index.html ships it: disabled.
    this.check(run.disabled === this.noScript, `Run is ${run.disabled ? "disabled" : "enabled"} at the end of the state`);
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

  // What a browser fires when it brings the page back from the back-forward cache.
  async restoredFromCache() {
    await this.page.evaluate(() => window.dispatchEvent(new PageTransitionEvent("pageshow", { persisted: true })));
    await this.page.waitForLoadState("networkidle").catch(() => {});
  }

  // The same, without waiting for the network: for a state that holds the health read
  // the restored page makes.
  async restoredHeld() {
    await this.page.evaluate(() => window.dispatchEvent(new PageTransitionEvent("pageshow", { persisted: true })));
  }

  // What a screen reader says after the Run button's name: its aria-describedby.
  runDescription() {
    return this.page.evaluate(() =>
      (document.getElementById("run").getAttribute("aria-describedby") || "")
        .split(/\s+/)
        .map((id) => (document.getElementById(id) ? document.getElementById(id).textContent.trim() : ""))
        .filter(Boolean)
        .join(" "),
    );
  }

  // The contrast ratio of an element's text (or of its ::placeholder) against what is
  // painted behind it, with the opacity of every ancestor taken into account.
  contrast(selector, pseudo) {
    return this.page.evaluate(
      ([sel, pseudoName]) => {
        const canvas = document.createElement("canvas");
        canvas.width = 1;
        canvas.height = 1;
        const cx = canvas.getContext("2d", { willReadFrequently: true });
        const rgba = (css) => {
          cx.clearRect(0, 0, 1, 1);
          cx.fillStyle = css;
          cx.fillRect(0, 0, 1, 1);
          const d = cx.getImageData(0, 0, 1, 1).data;
          return [d[0], d[1], d[2], d[3] / 255];
        };
        const over = (top, under) => [0, 1, 2].map((i) => top[i] * top[3] + under[i] * (1 - top[3])).concat(1);
        const mix = (inside, outside, alpha) => [0, 1, 2].map((i) => inside[i] * alpha + outside[i] * (1 - alpha)).concat(1);
        const chain = [];
        for (let n = document.querySelector(sel); n; n = n.parentElement) chain.unshift(n);
        // Each ancestor paints its background on what is behind it; an ancestor with an
        // opacity is then blended, with everything inside it, into what was behind.
        const paint = (i, behind) => {
          const style = getComputedStyle(chain[i]);
          const bg = over(rgba(style.backgroundColor), behind);
          let pair;
          if (i === chain.length - 1) {
            const own = pseudoName ? getComputedStyle(chain[i], pseudoName) : style;
            const ink = rgba(own.color);
            if (pseudoName) ink[3] *= Number(own.opacity);
            pair = [bg, over(ink, bg)];
          } else pair = paint(i + 1, bg);
          const alpha = Number(style.opacity);
          return alpha < 1 ? pair.map((c) => mix(c, behind, alpha)) : pair;
        };
        const [bg, fg] = paint(0, [255, 255, 255, 1]);
        const lum = (c) => {
          const [r, g, b] = c.slice(0, 3).map((v) => (v / 255 <= 0.03928 ? v / 255 / 12.92 : ((v / 255 + 0.055) / 1.055) ** 2.4));
          return 0.2126 * r + 0.7152 * g + 0.0722 * b;
        };
        const [hi, lo] = [lum(bg), lum(fg)].sort((a, b) => b - a);
        return Math.round(((hi + 0.05) / (lo + 0.05)) * 100) / 100;
      },
      [selector, pseudo || null],
    );
  }

  // ---------- the guide (the figure of the pipeline in #pipe) ----------

  // What the figure shows, read in one go.
  guide() {
    return this.page.evaluate(() => {
      const pipe = document.getElementById("pipe");
      const panel = document.getElementById("guide-panel");
      const seen = (n) => Boolean(n) && n.getClientRects().length > 0;
      const words = (n) => n.innerText.replace(/\s+/g, " ").trim();
      const nodes = {};
      for (const n of pipe.querySelectorAll(".guide-node")) {
        nodes[n.dataset.k] = {
          tag: n.tagName,
          seen: seen(n),
          name: words(n),
          label: n.querySelector(".guide-nn").textContent,
          sub: n.querySelector(".guide-ns").textContent,
          skipped: n.classList.contains("is-skipped"),
          pressed: n.getAttribute("aria-pressed"),
          done: n.hasAttribute("data-status"),
        };
      }
      const token = document.getElementById("guide-token");
      return {
        view: pipe.dataset.view,
        play: pipe.dataset.play,
        at: pipe.dataset.at || null,
        height: Math.round(pipe.getBoundingClientRect().height),
        toggle: document.getElementById("guide-toggle").textContent,
        expanded: document.getElementById("guide-toggle").getAttribute("aria-expanded"),
        nodes,
        active: [...pipe.querySelectorAll(".guide-node.is-active")].map((n) => n.dataset.k),
        facesHidden: [...pipe.querySelectorAll(".guide-face")].every((f) => f.getAttribute("aria-hidden") === "true"),
        chipsSeen: seen(document.getElementById("guide-chips")),
        chips: [...pipe.querySelectorAll(".guide-men")].map((c) => ({ text: c.firstChild.textContent, name: words(c), tag: c.tagName, current: c.classList.contains("is-current") && c.getAttribute("aria-pressed") === "true" })),
        playSeen: seen(document.getElementById("guide-play")),
        playDisabled: document.getElementById("guide-play").disabled,
        playLabel: document.getElementById("guide-play-label").textContent,
        hint: document.getElementById("guide-hint").textContent,
        panelSeen: seen(panel),
        panel: seen(panel) ? words(panel) : "",
        live: panel.getAttribute("aria-live"),
        lanes: [...panel.querySelectorAll(".guide-lanes h4")].map((n) => n.textContent),
        laneItems: [...panel.querySelectorAll(".guide-lanes .guide-lane")].map((lane) => [...lane.querySelectorAll(".guide-c")].map((c) => c.textContent)),
        ids: [...panel.querySelectorAll(".guide-id")].map((n) => n.textContent),
        bubbles: panel.querySelectorAll(".guide-bubble").length,
        // Parts of the panel that a play has still to reveal.
        hiddenParts: panel.querySelectorAll(".guide-c:not(.is-on), .guide-w:not(.is-on)").length,
        token: seen(token) && token.classList.contains("is-on"),
      };
    });
  }

  // Starts a record of what the figure does from now on: which boxes become active, in
  // order; which candidates are considered, picked or revealed; how often the panel (a
  // live region) is rewritten, and whether any of its text is retyped; what the
  // travelling label reads, and whether it ever leaves the page or widens it.
  guideWatch() {
    return this.page.evaluate(() => {
      if (window.__guide) window.__guide.stop();
      const pipe = document.getElementById("pipe");
      const panel = document.getElementById("guide-panel");
      const token = document.getElementById("guide-token");
      const log = { active: [], chips: [], writes: 0, retyped: 0, live: [], tokens: [], overflow: 0, tokenOut: 0 };
      const had = (r, cls) => (r.oldValue || "").split(/\s+/).includes(cls);
      const observer = new MutationObserver((records) => {
        let wrote = false;
        for (const r of records) {
          const t = r.target;
          if (r.type === "characterData" && panel.contains(t)) log.retyped += 1;
          else if (r.type === "childList" && panel.contains(t)) wrote = true;
          else if (r.type === "childList" && t === token && token.textContent && log.tokens[log.tokens.length - 1] !== token.textContent) log.tokens.push(token.textContent);
          else if (r.type === "attributes" && t.classList.contains("guide-node")) {
            // Several records of one step all see the class as it is now: said once.
            if (t.classList.contains("is-active") && !had(r, "is-active") && log.active[log.active.length - 1] !== t.dataset.k) log.active.push(t.dataset.k);
          } else if (r.type === "attributes" && t.classList.contains("guide-c") && panel.contains(t)) {
            const text = t.firstChild ? t.firstChild.textContent : "";
            for (const [cls, name] of [["is-consider", "consider"], ["is-picked", "picked"], ["is-on", "on"]]) {
              if (t.classList.contains(cls) && !had(r, cls) && log.chips[log.chips.length - 1] !== `${name}:${text}`) log.chips.push(`${name}:${text}`);
            }
          }
        }
        if (wrote) {
          log.writes += 1;
          log.live.push(panel.getAttribute("aria-live"));
        }
      });
      observer.observe(pipe, { subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: ["class"], attributeOldValue: true });
      const timer = setInterval(() => {
        if (document.documentElement.scrollWidth > document.documentElement.clientWidth) log.overflow += 1;
        if (!token.classList.contains("is-on")) return;
        const r = token.getBoundingClientRect();
        const frame = pipe.getBoundingClientRect();
        // The label eases a little past its place; it may not leave the figure by more.
        if (r.left < frame.left - 16 || r.right > Math.min(frame.right + 16, window.innerWidth) || r.top < frame.top - 16 || r.bottom > frame.bottom + 16) log.tokenOut += 1;
      }, 50);
      log.stop = () => {
        observer.disconnect();
        clearInterval(timer);
      };
      window.__guide = log;
    });
  }

  guideLog() {
    return this.page.evaluate(() => {
      const { stop, ...log } = window.__guide;
      stop();
      return log;
    });
  }

  // A play has run to its end.
  async guidePlayed(what) {
    await this.page.waitForSelector('#pipe[data-play="done"]', { timeout: 45000 }).catch(() => {
      this.failures.push(`the play of ${what} never ended`);
    });
  }

  // Chooses a mention to follow by its chip, which plays it.
  async follow(mention) {
    const chips = await this.page.locator("#guide-chips .guide-men").all();
    for (const chip of chips) {
      if ((await chip.evaluate((c) => c.firstChild.textContent)) === mention) return chip.click();
    }
    this.failures.push(`no chip for ${mention}`);
    return null;
  }

  // Runs the recorded production text against the recorded production answer.
  async runGuide() {
    this.health = productionHealth();
    this.ner = { body: GUIDE_RUN.response };
    await this.open();
    await this.page.fill("#text", GUIDE_RUN.text);
    await this.run();
    return this.expectResults();
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
    // The server remembers the example from the width or the scheme before this one
    // (cache-live holds the two sentences apart).
    await s.expectText("#status", /^(Done in |Answered from a recent identical run \()/, "the one-sentence status");
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
    await s.expectText("#run-hint", /^Free, no sign-up, usually 10 to 30 seconds\. Use public or de-identified text: it is sent to the model provider\.$/, "the hint under Run");
    const footer = await s.text("#privacy");
    s.check(!/not saved|stores no|kept for one day|writes no text|Each run is recorded/i.test(footer), "the footer still promises more than the site does");
    s.check(footer.includes("When embedding search is used, the mentions found in your text are sent to OpenRouter to compute search embeddings. The server may keep recent mentions, search results and answers in memory for a few hours so that it does not compute them twice; none of that is written to its database."), "the footer's line about what the server keeps in memory is not the agreed one");
    s.check(footer.includes("This site is built not to keep your text, prompt or results: its logs and its database are designed to hold run metadata only, for example time, model, token counts and country, and a visitor hash that changes every day."), "the footer's line about logs and the database is not the agreed one");
    s.check(/When run recording is enabled, this site attempts to save/.test(footer), "the footer says runs are recorded without its condition");
    s.check(/Runs on this site's own models are limited per network/.test(footer), "the footer's line about limits has no condition");
    const described = await s.runDescription();
    s.check(/^Runs on this site's models, free\.$/.test(described), `Run is described as ${JSON.stringify(described)}`);
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
    await s.page.click("#options > summary");
    // The lede names Claude while Claude reads the text, and no model by name otherwise.
    await s.page.selectOption("#model", "google/gemini-3.1-flash-lite");
    await s.expectText("#lede", /^Paste a sentence\. A language model finds the chemicals, diseases and foods in it, and xwalk links each one to a term in ChEBI, MeSH or FoodOn\.$/, "the lede with Gemini reading the text");
    await s.page.selectOption("#model", "claude-sonnet-5-5");
    await s.expectText("#lede", /^Paste a sentence\. Claude finds the chemicals/, "the lede with Claude Sonnet reading the text");
    await s.page.selectOption("#model", "claude-opus-5-5");
  },

  async empty(s) {
    s.health = fixture("health");
    await s.open();
    await s.run();
    await s.expectVisible("#text-error", "the message at the text box");
    await s.expectText("#text-error", /^Paste some text first, or tap Example\.$/);
    await s.expectHidden("#error", "the error box");
    s.check(s.posts.length === 0, "an empty text was submitted");
    const faint = await s.contrast("#text", "::placeholder");
    s.check(faint >= 4.5, `the placeholder's contrast is ${faint}:1`);
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
    await s.expectText("#text-error", /^This text is over the 5,000 character limit\. Shorten it to map it\.$/);
    await s.expectText("#text-counter", /^5,001 \/ 5,000 characters, 1 over$/, "the counter over the limit");
    // The alert is said when the limit is passed, not again on every keystroke after.
    await s.page.evaluate(() => {
      window.smokeAlerts = 0;
      new MutationObserver((records) => {
        window.smokeAlerts += records.length;
      }).observe(document.getElementById("text-error"), { childList: true, characterData: true, subtree: true, attributes: true });
    });
    await s.page.focus("#text");
    await s.page.evaluate(() => {
      const box = document.getElementById("text");
      box.setSelectionRange(box.value.length, box.value.length);
    });
    await s.page.keyboard.type("bcd");
    await s.expectText("#text-counter", /^5,004 \/ 5,000 characters, 4 over$/, "the counter after more typing");
    s.check((await s.page.evaluate(() => window.smokeAlerts)) === 0, "the too-long alert was rewritten while the visitor typed");
    s.check(/\btext-error\b/.test(await s.page.getAttribute("#text", "aria-describedby")), "the text box is not described by its error");
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
    // The tab is left open past midnight UTC and looked at again while this hour's runs
    // are used: yesterday's notice goes, and the line beside Run gives the wait that
    // holds now.
    s.health = fixture("health-blocked");
    await s.setVisibility("hidden");
    await s.setVisibility("visible");
    await s.expectHidden("#error", "the daily-limit notice once it is the hourly limit that blocks");
    await s.expectVisible("#allowance", "the hourly line after the daily notice");
    await s.expectText("#allowance", /^The free runs for your network this hour are used\. Next one in \d+ min\.$/, "the wait that holds now");
    // Looked at again later: the limit is over, and neither the notice nor the line
    // beside Run goes on saying that it is not.
    s.health = fixture("health");
    await s.setVisibility("hidden");
    await s.setVisibility("visible");
    await s.expectHidden("#error", "the daily-limit notice once the limit is over");
    await s.expectHidden("#allowance", "the daily-limit line once the limit is over");
    await s.expectVisible("#run-hint", "the 'Free, no sign-up' hint once the limit is over");
    s.health = fixture("health-day");
    await s.run();
    await s.expectError("The free runs for your network today are used.", { look: "notice", ownKey: true });
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
    // Counted from the result: two of its eleven mentions have no term to show.
    await s.expectText("#results-takeaway", /^No model chose here: 9 of the 11 mentions show their top search result\.$/, "the search-only line");
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
    // A cheaper model chosen for the key comes back with it: a reload must not move the
    // visitor's bill to the provider's default model.
    await s.page.selectOption("#model", "claude-haiku-5-5");
    await s.page.selectOption("#selector", "claude-haiku-5-5");
    await s.page.reload({ waitUntil: "networkidle" });
    s.check((await s.page.inputValue("#model")) === "claude-haiku-5-5", `after the reload the model is ${await s.page.inputValue("#model")}`);
    s.check((await s.page.inputValue("#selector")) === "claude-haiku-5-5", `after the reload the selector is ${await s.page.inputValue("#selector")}`);
    const described = await s.runDescription();
    s.check(/^Runs on your Anthropic key; Anthropic bills you\.$/.test(described), `Run is described as ${JSON.stringify(described)}`);
    s.check(await s.optionsOpen(), "Options is closed after a reload with an own key chosen");
    await s.expectText("#options-summary", /^Options: your Anthropic key$/, "the Options summary after the reload");
    await s.expectVisible("#own-key", "the key field after the reload");
    s.check((await s.page.inputValue("#provider")) === "anthropic", "the payer choice was not restored");
    s.check((await s.page.inputValue("#own-key")) === FAKE_KEY, "the key was not restored");
    await s.expectText("#run-hint", /^Runs on your Anthropic key\. Use public or de-identified text: it is sent to Anthropic\.$/);
    await s.expectHidden("#payer", "the payer hold");
    const stored = await s.page.evaluate(() => Object.keys(window.sessionStorage).concat(Object.keys(window.localStorage)));
    s.check(stored.every((k) => k === "xwalk-ner-own-key" || k === "xwalk-ner-payer"), `unexpected storage entries: ${stored.join(", ")}`);
    await s.runExample();
    await s.expectResults();
    s.check(s.posts.length === 1, `expected one POST, saw ${s.posts.length}`);
    const post = s.posts[0] || { body: {}, authorization: "" };
    s.check(post.authorization === `Bearer ${FAKE_KEY}`, "the own-key run did not carry the key");
    s.check(post.body && post.body.provider === "anthropic", "the own-key run did not name its provider");
    s.check(post.body && post.body.model === "claude-haiku-5-5" && post.body.selector === "claude-haiku-5-5", "the own-key run did not use the restored models");
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
    await s.expectText("#banner", /^Could not reach the service\. Runs are on hold: see the note under the button\./, "the banner during a hold");
    s.check(!/You can still try a run/.test(await s.text("#banner")), "the banner offers a run that the hold refuses");
    s.check(/on hold until you choose who pays/.test(await s.runDescription()), "Run's description does not say that runs are held");
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
    await s.expectText("#status", /^Done in 8\.8 s: 3\.3 s reading the text, 2\.3 s searching, 3\.2 s choosing terms\. Read by Claude Opus 5\.5; terms chosen by Claude Opus 5\.5\.$/, "the status line, with the models that answered");
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
    s.check(mention.name === "metformin (chemical)" && mention.title === "chemical", `the mention's accessible name is ${JSON.stringify(mention.name)}, its tooltip ${JSON.stringify(mention.title)}`);
    // The visitor's words are text on the page, never the value of an attribute.
    const copied = await s.page.evaluate(() => {
      const found = [];
      for (const node of document.querySelectorAll("#results *")) {
        for (const attr of node.attributes) if (/metformin|aspirin|warfarin|oily fish/i.test(attr.value)) found.push(`${node.tagName.toLowerCase()}[${attr.name}]`);
      }
      return found;
    });
    s.check(copied.length === 0, `text of the submission is copied into attributes: ${copied.join(", ")}`);
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
    // What 0.93 means is in the opened card: a phone has no hover for the tooltip.
    await s.expectText("#card-detail-e1 .card-conf-note", /^0\.93 is the model's own confidence, not a measured accuracy\.$/, "the confidence line in the opened card");
    s.check((await s.page.locator("#card-detail-e3 .card-conf-note, #card-detail-e5 .card-conf-note").count()) === 0, "a card with no term explains a confidence");
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
    const dimmed = await s.contrast("#no-entities-example");
    s.check(dimmed >= 4.5, `"Load example" in the dimmed result has a contrast of ${dimmed}:1`);
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

    // Clear and Example during a run are edits too: the answer leaves the focus alone.
    for (const button of ["clear-text", "load-example"]) {
      const held = s.gate();
      s.ner = { gate: held.promise, body: shaped() };
      await s.page.fill("#text", SHAPED_TEXT);
      await s.run();
      await s.expectVisible("#progress", "the progress bar");
      await s.page.waitForTimeout(700);
      await s.page.click(`#${button}`);
      const y = await s.page.evaluate(() => window.scrollY);
      held.release();
      await s.settled();
      await s.page.waitForTimeout(700);
      const then = await s.page.evaluate(() => ({ y: window.scrollY, focus: document.activeElement && document.activeElement.id }));
      s.check(then.focus !== "results-heading", `after #${button} during a run the answer took the focus`);
      s.check(Math.abs(then.y - y) <= 2, `after #${button} during a run the answer scrolled the page (${y} -> ${then.y})`);
      await s.expectVisible("#stale", `the earlier-text line after #${button} during a run`);
    }
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
    const nEx = EXAMPLE_FILE.response.entities.length;
    await s.expectText("#results-summary", new RegExp(`^${nEx} mentions found: ${nEx} matched\\.$`));
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
    // An empty box: there is no "yours" to map, so the line is not shown.
    await s.page.click("#clear-text");
    await s.expectHidden("#stale", "the line about the example's text while the box is empty");
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
    // Counted from the result: BRCA1 was not looked up, and of the six mentions with
    // candidates the model chose a term for four (it declined one; one pick failed).
    await s.expectText("#next-body", /^The model never writes an ID\. xwalk's search found terms for 6 of the 7 mentions by keyword and by meaning, and the model chose a term for 4 of those\. The catalog holds 261,018 terms\. xwalk is an open-source Python library; you can run the same lookup on your own vocabulary\.$/);
    s.check(!/searched [\d,]+/.test(await s.text("#next-body")), "the closing panel says the run searched the whole catalog");
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
    const keywordOnly = (m) => ({ ...m, retrievers: { bm25: m.retrievers.bm25 } });
    const searchOnly = shaped().entities.map((e) => ({ ...e, pick: null, matches: e.matches.map(keywordOnly), more: e.more.map(keywordOnly) }));
    s.ner = { body: shaped({ entities: searchOnly, retrieval: { bm25: { status: "ok" }, dense: { status: "unavailable", reason: "no index" } } }) };
    await s.run();
    await s.settled();
    await s.expectText("#next-body", /^The model never writes an ID\. xwalk's search found terms for 6 of the 7 mentions by keyword, and each of those shows its top search result\. The catalog holds 261,018 terms\./, "the closing panel for a keyword-only run with no model");
    await s.expectText("#status", /Read by Claude Opus 5\.5; no model chose the terms\.$/, "the status line of a run where no model chose");

    // Every pick failed: a selector is named on each, but no model chose anything, and
    // the cards do not show the top search result either. Nothing is said about choosing.
    const failed = shaped().entities.map((e) => (e.pick ? { ...e, pick: { ...e.pick, status: "failed", id: null, label: null, url: null, confidence: null, explanation: "cut off by the request deadline" } } : e));
    s.ner = { body: shaped({ entities: failed }) };
    await s.run();
    await s.settled();
    await s.expectText("#next-body", /^The model never writes an ID\. xwalk's search found terms for 6 of the 7 mentions by keyword and by meaning\. The catalog holds 261,018 terms\./, "the closing panel when every pick failed");
    s.check(!/chose|top search result/.test(await s.text("#next-body")), "the closing panel claims a choice that was not made");
    await s.expectText("#status", /Read by Claude Opus 5\.5\.$/, "the status line when every pick failed");
    await s.expectHidden("#results-takeaway", "the takeaway when every pick failed");
    await s.page.click("#run-details > summary");
    await s.expectText("#results-models", /^Extracted by Claude Opus 5\.5\. The model did not finish choosing the terms\.$/, "Run details when every pick failed");

    // Both halves of the search were off: one sentence says so, not "keyword only" and
    // "meaning only" at once; no mention has a term, and there is no lookup to point at.
    const nothing = shaped().entities.map((e) => ({ ...e, pick: null, matches: [], more: [], notes: e.ontology ? ["every retriever failed; no matches"] : e.notes }));
    s.ner = { body: shaped({ entities: nothing, retrieval: { bm25: { status: "unavailable", reason: "x" }, dense: { status: "unavailable", reason: "y" } } }) };
    await s.run();
    await s.settled();
    await s.expectText("#notices", /^About this run\s*The search was unavailable for this run, so the mentions could not be looked up\.$/, "the notice when the whole search was off");
    s.check(!/keyword only|meaning only/.test(await s.text("#results")), "the result says 'keyword only' and 'meaning only' at once");
    s.check(!/top search result/.test(await s.text("#results")), "a result with no candidate claims a top search result");
    await s.expectHidden("#next", "the closing panel when nothing could be looked up");

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

  // ---------- what the reviews found ----------

  // app.js never loads. The page then has a form and no handler: nothing may be
  // submitted, and above all no text may end up in the address. Three things stand in
  // the way, and each is checked by itself: Run ships disabled, the policy refuses any
  // form submission, and no field has a name, so a submission would carry nothing.
  async "no-script"(s) {
    const SECRET = "Patient Jane Roe takes warfarin";
    s.noScript = true;
    const response = await fetch(s.base);
    const policy = response.headers.get("content-security-policy") || "";
    s.check(/(^|;)\s*form-action 'none'/.test(policy), `the Content-Security-Policy does not set form-action 'none': ${JSON.stringify(policy)}`);
    const html = await response.text();
    s.check(/<button type="submit" id="run"[^>]*\sdisabled>/.test(html), "Run is not disabled in index.html");
    s.check(/<noscript><p class="banner">This demo needs JavaScript\.<\/p><\/noscript>/.test(html), "the noscript line is gone");
    await s.page.goto(s.base, { waitUntil: "load" });
    await s.page.waitForTimeout(300);
    s.check(await s.page.evaluate(() => document.getElementById("run").disabled), "without the script Run is enabled");
    const named = await s.page.evaluate(() => [...document.getElementById("ner-form").elements].filter((e) => e.name).map((e) => `#${e.id}[name=${e.name}]`));
    s.check(named.length === 0, `form fields have names, so a native submission would carry them: ${named.join(", ")}`);
    await s.page.fill("#text", SECRET);
    // What a visitor can do: tap the button, or press Enter in a field.
    const box = await s.page.locator("#run").boundingBox();
    await s.page.mouse.click(box.x + box.width / 2, box.y + box.height / 2);
    await s.page.evaluate(() => {
      document.getElementById("passcode-field").hidden = false;
    });
    await s.page.fill("#passcode", "hunter2");
    await s.page.press("#passcode", "Enter");
    await s.page.waitForTimeout(400);
    const clean = () => {
      const url = new URL(s.page.url());
      return url.search === "" && !/warfarin|Jane|hunter2/i.test(decodeURIComponent(url.href));
    };
    s.check(clean(), `a tap on Run without the script changed the address to ${s.page.url()}`);
    // The button's own guard taken away: the policy and the missing names still hold.
    await s.page.evaluate(() => {
      document.getElementById("run").disabled = false;
    });
    await s.page.locator("#run").scrollIntoViewIfNeeded();
    const again = await s.page.locator("#run").boundingBox();
    await s.page.mouse.click(again.x + again.width / 2, again.y + again.height / 2);
    await s.page.waitForTimeout(600);
    s.check(clean(), `a native submission put the form in the address: ${s.page.url()}`);
    s.check((await s.page.inputValue("#text").catch(() => "")) === SECRET, "a native submission reloaded the page");
    s.check(s.posts.length === 0, "something was posted without the script");
    // (Playwright goes on waiting for the navigation the policy refused; a fresh load
    // of the page ends that wait, and the checks every state ends with run on it.)
    await s.page.goto(s.base, { waitUntil: "load" });
  },

  // A page brought back from the back-forward cache is the old document, key field and
  // all. If the key was forgotten, or the payer changed, in another document of the tab
  // meanwhile, the restored page must not send the old key or write it back.
  async "bfcache-payer"(s) {
    s.health = fixture("health");
    s.ner = { body: (request) => ({ ...fixture("ner-own"), provider: request.provider || "site" }) };
    const stored = (entry) => s.page.evaluate((k) => window.sessionStorage.getItem(k), entry);
    const drop = (...entries) => s.page.evaluate((keys) => keys.forEach((k) => window.sessionStorage.removeItem(k)), entries);
    await s.open();
    await s.chooseOwnKey("anthropic");
    s.check(((await stored(OWN_KEY_ENTRY)) || "").includes(FAKE_KEY), "the key was not kept for the tab");

    // "Forget key" pressed in another document of this tab, then Back to this one.
    await drop(OWN_KEY_ENTRY);
    await s.restoredFromCache();
    s.check((await s.page.inputValue("#own-key")) === "", "a restored page kept a key that was forgotten");
    s.check((await s.page.inputValue("#provider")) === "anthropic", "forgetting the key changed who pays");
    await s.runExample();
    await s.expectVisible("#own-key-error", "the request for a key");
    await s.expectText("#own-key-error", /^Paste your Anthropic API key/);
    s.check(s.posts.length === 0, "a restored page ran after its key was forgotten");
    s.check((await stored(OWN_KEY_ENTRY)) === null, "a restored page wrote the forgotten key back");

    // The payer switched to the site there: this page says so, and runs without a key.
    await s.page.fill("#own-key", FAKE_KEY);
    await drop(OWN_KEY_ENTRY, PAYER_ENTRY);
    await s.restoredFromCache();
    s.check((await s.page.inputValue("#provider")) === "site", "a restored page kept a payer that was changed");
    s.check((await s.page.inputValue("#own-key")) === "", "a restored page kept the key of a payer that was changed");
    await s.expectText("#run-hint", /^Free, no sign-up/, "the hint after the payer changed");
    await s.run();
    await s.expectResults();
    s.check(s.posts.length === 1 && s.posts[0].authorization === "" && !s.posts[0].body.provider, "the run after the payer changed carried a key or a provider");
    s.check((await stored(OWN_KEY_ENTRY)) === null && (await stored(PAYER_ENTRY)) === null, "a restored page wrote the old choice back");

    // A choice that cannot be settled (its provider is not offered here) holds the run.
    await s.page.evaluate((k) => window.sessionStorage.setItem(k, "openai-eu"), PAYER_ENTRY);
    await s.restoredFromCache();
    await s.expectVisible("#payer", "the payer hold on a restored page");
    s.check(await s.page.evaluate(() => document.getElementById("run").disabled), "Run is enabled on a restored page whose payer is unsettled");
    await s.page.keyboard.press("Control+Enter");
    await s.page.waitForTimeout(300);
    s.check(s.posts.length === 1, "a run was sent while the payer was unsettled");
    await s.page.click("#payer-site");
    await s.expectHidden("#payer", "the payer hold after choosing the site");

    // The same through the browser's own history: a second copy of the page in this
    // tab, "Forget key" there, then Back. Whether the browser restores the first copy
    // from its cache or loads it again, the forgotten key is not in it and is not sent.
    await s.chooseOwnKey("anthropic");
    await s.page.goto(`${s.base}?copy=2`, { waitUntil: "networkidle" });
    s.check((await s.page.inputValue("#own-key")) === FAKE_KEY, "the second copy did not restore the key");
    await s.page.click("#forget-key");
    await s.page.goBack({ waitUntil: "load" });
    await s.page.waitForLoadState("networkidle").catch(() => {});
    await s.page.waitForTimeout(300);
    s.check(!new URL(s.page.url()).search, `Back did not return to the first copy: ${s.page.url()}`);
    s.check((await s.page.inputValue("#own-key")) === "", "after Back the page holds the forgotten key");
    await s.runExample();
    await s.page.waitForTimeout(400);
    s.check(s.posts.length === 1, "after Back a run went out with the forgotten key");
    s.check(s.posts.every((p) => !p.authorization), "the forgotten key was sent");
    s.check((await stored(OWN_KEY_ENTRY)) === null, "after Back the forgotten key was written back");
  },

  // During a run nothing can change who pays or which models run, and what the page
  // says about the run when it ends is about the provider and the model it was
  // submitted with.
  async "run-locks-payer"(s) {
    s.health = fixture("health");
    s.expectedStatuses.add(401);
    s.expectedStatuses.add(429);
    const gate = s.gate();
    s.ner = { gate: gate.promise, status: 401, body: fixture("error-own-refused") };
    await s.open();
    await s.chooseOwnKey("anthropic");
    await s.runExample();
    await s.expectVisible("#progress", "the progress bar");
    const locked = await s.page.evaluate(() => ["provider", "own-key", "forget-key", "model", "selector"].filter((id) => !document.getElementById(id).disabled));
    s.check(locked.length === 0, `during a run these can still be changed: ${locked.join(", ")}`);
    await s.expectText("#run-hint", /^Runs on your Anthropic key\./, "the hint during the run");
    // Past the disabled controls, as a script or a restored form might: still no change.
    await s.page.evaluate(() => {
      const provider = document.getElementById("provider");
      provider.value = "openrouter";
      provider.dispatchEvent(new Event("change", { bubbles: true }));
      document.getElementById("forget-key").click();
      document.getElementById("model").value = "claude-haiku-5-5";
    });
    s.check((await s.page.inputValue("#provider")) === "anthropic", "the payer changed during a run");
    s.check((await s.page.inputValue("#own-key")) === FAKE_KEY, "the key was forgotten during a run");
    await s.expectText("#options-summary", /^Options: your Anthropic key$/, "the Options summary during the run");
    gate.release();
    await s.settled();
    await s.expectVisible("#own-key-error", "the message at the key field");
    // The run was submitted with Claude Opus 5.5; the list was moved to Haiku meanwhile.
    await s.expectText("#own-key-error", /^Anthropic did not accept this key\. .* may use Claude Opus 5\.5\.$/, "the key error, bound to the submitted provider and model");
    s.check((s.posts[0] || { body: {} }).body.provider === "anthropic" && s.posts[0].body.model === "claude-opus-5-5", "the run was not the Anthropic run on Opus");
    const open = await s.page.evaluate(() => ["provider", "own-key", "forget-key", "model", "selector"].filter((id) => document.getElementById(id).disabled));
    s.check(open.length === 0, `after the run these are still disabled: ${open.join(", ")}`);

    // A notice about the site's free runs does not stay up over another payer.
    await s.page.selectOption("#provider", "site");
    s.ner = { status: 429, headers: { "Retry-After": "1380" }, body: fixture("error-blocked") };
    await s.run();
    await s.expectError("The free runs for your network this hour are used.", { look: "notice", ownKey: true });
    await s.page.selectOption("#provider", "openrouter");
    await s.expectHidden("#error", "the free-runs notice after choosing an own key");
    await s.page.selectOption("#provider", "site");
  },

  // The first health read of a cold function can answer after a run and the read that
  // follows it. The late answer, or its failure, must not undo what the visitor chose
  // in between.
  async "health-out-of-order"(s) {
    const settings = () => s.page.evaluate(() => ({ provider: document.getElementById("provider").value, model: document.getElementById("model").value, selector: document.getElementById("selector").value, key: document.getElementById("own-key").value }));
    for (const late of ["answer", "failure"]) {
      const first = s.gate();
      s.healthQueue = [{ gate: first.promise, health: late === "answer" ? productionHealth() : "fail" }];
      s.health = productionHealth();
      s.ner = { body: shaped() };
      if (late === "failure") s.expectedStatuses.add(500);
      await s.openHeld();
      await s.page.fill("#text", SHAPED_TEXT);
      await s.run();
      await s.expectVisible('#results[data-kind="run"] .card', "the result of a run made before health answered");
      // The read after the run answers at once and fills the form.
      await s.expectVisible("#options-summary", "the Options summary");
      await s.page.waitForFunction(() => document.getElementById("provider").options.length > 1, null, { timeout: 5000 }).catch(() => s.failures.push("the read after the run did not fill the form"));
      await s.chooseOwnKey("anthropic");
      await s.page.selectOption("#model", "claude-haiku-5-5");
      await s.page.selectOption("#selector", "claude-haiku-5-5");
      const chosen = await settings();
      first.release();
      await s.page.waitForLoadState("networkidle").catch(() => {});
      await s.page.waitForTimeout(400);
      const kept = await settings();
      s.check(JSON.stringify(kept) === JSON.stringify(chosen), `a late ${late} of the first health read changed the choices: ${JSON.stringify(chosen)} -> ${JSON.stringify(kept)}`);
      await s.expectHidden("#payer", `the payer hold after a late ${late}`);
      await s.expectHidden("#banner", `the banner after a late ${late}`);
      await s.expectVisible("#model", `the model list after a late ${late}`);
      await s.page.evaluate(() => window.sessionStorage.clear());
    }
    s.healthQueue = [];
  },

  // Map is pressed on the prefilled example before health answers; health then says
  // this deployment is not the one the recorded example belongs to. The recording goes,
  // the text that is being mapped stays, and its result is not called an earlier text's.
  async "late-health-keeps-text"(s) {
    const health = s.gate();
    const answer = s.gate();
    s.healthGate = health.promise;
    s.health = fixture("health");
    s.ner = { gate: answer.promise, body: shaped() };
    await s.openHeld();
    await s.expectExample("before health answers");
    await s.run();
    await s.expectVisible("#progress", "the progress bar");
    health.release();
    // The answer arrives during the run and is kept until the run ends: the form the
    // run was submitted with does not change under it.
    await s.page.waitForTimeout(500);
    s.check((await s.page.evaluate(() => document.getElementById("model").options.length)) === 0, "a health answer filled the model list during a run");
    s.check((await s.page.inputValue("#text")) === EXAMPLE_FILE.text, `health arriving during the run changed the text box to ${JSON.stringify((await s.page.inputValue("#text")).slice(0, 40))}`);
    s.healthGate = null;
    answer.release();
    await s.expectVisible('#results[data-kind="run"] .card', "the result of the run");
    await s.settled();
    await s.page.waitForFunction(() => /200-term sample/.test(document.getElementById("caveat").textContent), null, { timeout: 5000 }).catch(() => s.failures.push("the health answer kept during the run was not applied after it"));
    await s.expectHidden('#results[data-kind="example"]', "the recorded example on a sample catalog");
    s.check(s.posts.length === 1 && s.posts[0].body.text === EXAMPLE_FILE.text, "the run did not send the example text");
    s.check((await s.page.inputValue("#text")) === EXAMPLE_FILE.text, "the text box does not hold the text that was mapped");
    await s.expectHidden("#stale", "the earlier-text line on a result for the text in the box");
  },

  // The answer's headers arrive and its body does not finish. The request's timer has
  // to stay armed until the body is read; when it runs out, or the connection is cut,
  // Run comes back and the visitor is told that the run may have been counted. The
  // browser's fetch is replaced for POST /api/ner (a route cannot stall a body), and
  // the 130-second timer is shortened to under a second.
  async "stalled-body"(s) {
    s.health = productionHealth();
    await s.page.addInitScript(() => {
      const realFetch = window.fetch.bind(window);
      const realTimeout = window.setTimeout.bind(window);
      window.setTimeout = (fn, ms, ...rest) => realTimeout(fn, ms === 130000 ? 800 : ms, ...rest);
      window.fetch = (input, init = {}) => {
        const url = typeof input === "string" ? input : input.url;
        if (!window.smokeBody || !/\/api\/ner$/.test(url)) return realFetch(input, init);
        const body = new ReadableStream({
          start(controller) {
            controller.enqueue(new TextEncoder().encode('{"ok": true, "entities": ['));
            if (window.smokeBody === "cut") realTimeout(() => controller.error(new TypeError("network error")), 200);
            else if (init.signal) init.signal.addEventListener("abort", () => controller.error(new DOMException("The operation was aborted.", "AbortError")));
          },
        });
        return Promise.resolve(new Response(body, { status: 200, headers: { "Content-Type": "application/json" } }));
      };
    });
    await s.open();
    await s.page.fill("#text", SHAPED_TEXT);
    const ended = (what) =>
      s.page.waitForFunction(() => document.getElementById("run").getAttribute("aria-busy") === "false", null, { timeout: 8000 }).catch(() => {
        s.failures.push(`${what}: Run is still disabled 8 seconds after a 0.8-second timeout`);
      });

    // Stalled: the phone is locked and unlocked meanwhile, as in the report.
    await s.page.evaluate(() => {
      window.smokeBody = "stall";
    });
    await s.run();
    await s.expectVisible("#progress", "the progress bar");
    await s.setVisibility("hidden");
    await s.setVisibility("visible");
    await ended("a stalled body");
    await s.expectError("The result did not finish arriving.", { action: "Run again", body: /Your text is still here\. The interrupted run may still have been counted\.$/ });

    // Cut off part of the way through.
    await s.page.evaluate(() => {
      window.smokeBody = "cut";
    });
    await s.page.click("#error-action");
    await ended("a body that was cut off");
    await s.expectError("The connection dropped before the result arrived.", { action: "Run again", body: /The interrupted run may still have been counted\.$/ });
    s.check((await s.page.inputValue("#text")) === SHAPED_TEXT, "an interrupted answer changed the text box");

    // And the page is whole again: the next run, with a real answer, shows its result.
    await s.page.evaluate(() => {
      window.smokeBody = null;
    });
    s.ner = { body: shaped() };
    await s.page.click("#error-action");
    await s.expectResults();
  },

  // What the page promises before and during a run follows what will be submitted: the
  // free-runs line names Claude Sonnet only for the step that uses Claude, the Decide
  // step does not promise a model's pick when no model is asked, and the result names
  // the models that answered.
  async "steps-and-allowance"(s) {
    const GEMINI = "google/gemini-3.1-flash-lite";
    s.health = fixture("health-sonnet");
    await s.open();
    await s.expectText("#allowance", /Runs now start on Claude Sonnet, a faster model\.$/, "the free-runs line when Claude reads the text");
    await s.page.click("#options > summary");
    await s.page.selectOption("#model", GEMINI);
    await s.expectText("#allowance", /^The free runs on Claude Opus for your network this hour are used\. The choice of terms now starts on Claude Sonnet, a faster model\.$/, "the free-runs line when only the choice of terms uses Claude");
    await s.expectText("#options-summary", /^Options: Gemini 3\.1 Flash Lite, free runs$/, "the Options summary with Gemini reading");
    await s.page.selectOption("#selector", GEMINI);
    await s.expectHidden("#allowance", "the free-runs line when no step uses Claude");
    await s.page.selectOption("#selector", "claude-opus-5-5");

    // A model chooses: the Decide step says so, and the result names who answered.
    let gate = s.gate();
    s.ner = { gate: gate.promise, body: shaped({ model: GEMINI, selector: "claude-sonnet-5-5", entities: shaped().entities.map((e) => (e.pick ? { ...e, pick: { ...e.pick, selector: "claude-sonnet-5-5" } } : e)) }) };
    await s.page.fill("#text", SHAPED_TEXT);
    await s.run();
    await s.expectVisible("#pipe-decide", "what the Decide step does");
    await s.expectText("#pipe-decide", /^The model picks the best term from the shortlist/, "the Decide step when a model is asked to choose");
    gate.release();
    await s.expectResults();
    await s.expectText("#status", /Read by Gemini 3\.1 Flash Lite; terms chosen by Claude Sonnet 5\.5\.$/, "the models that answered");
    s.check(s.posts[0].body.model === GEMINI && s.posts[0].body.selector === "claude-opus-5-5", "the run was not submitted with the chosen models");

    // The top search result, no model: nothing about a pick, a confidence or a reason.
    await s.page.selectOption("#selector", "retrieval").catch(() => {});
    if ((await s.page.inputValue("#selector")) !== "retrieval") {
      // This fixture's site list has no "retrieval": an own key's default is that.
      await s.chooseOwnKey("anthropic");
    }
    s.check((await s.page.inputValue("#selector")) === "retrieval", "the selector is not the top search result");
    gate = s.gate();
    s.ner = { gate: gate.promise, body: shaped({ entities: shaped().entities.map((e) => ({ ...e, pick: null })) }) };
    await s.run();
    await s.expectVisible("#pipe-decide", "what the Decide step does");
    await s.expectText("#pipe-decide", /^No model chooses in this run: each mention gets its top search result\.$/, "the Decide step when no model is asked");
    s.check(!/confidence|reason/.test(await s.text("#pipe-decide")), "the Decide step promises a confidence and a reason that will not come");
    gate.release();
    await s.expectResults();
    await s.expectText("#status", /no model chose the terms\.$/, "the status line when no model chose");
  },

  // ---------- what the second review found ----------

  // A restored page whose session storage is full (writes fail, reads and removals
  // work) or cannot be read at all. The key that was forgotten in another copy of the
  // page must not be sent: what is on record is read directly, whether or not anything
  // can be written, and where it cannot be read the key is cleared from the field and
  // runs are held until the visitor enters a key again or picks who pays.
  async "restore-storage-fails"(s) {
    const OTHER_KEY = "sk-smoke-typed-again-11111111";
    s.health = fixture("health");
    s.ner = { body: (request) => ({ ...fixture("ner-own"), provider: request.provider || "site" }) };
    await s.open();
    await s.chooseOwnKey("anthropic");
    // The tab's storage is full from here on.
    await s.page.evaluate(() => {
      window.smokeSetItem = Storage.prototype.setItem;
      Storage.prototype.setItem = function setItem() {
        throw new DOMException("The quota has been exceeded.", "QuotaExceededError");
      };
    });
    // "Forget key" in another copy of the page (a removal still works), then Back.
    await s.page.evaluate((k) => window.sessionStorage.removeItem(k), OWN_KEY_ENTRY);
    await s.restoredFromCache();
    s.check((await s.page.inputValue("#own-key")) === "", "with storage full, a restored page kept a key that was forgotten");
    s.check((await s.page.inputValue("#provider")) === "anthropic", "with storage full, forgetting the key changed who pays");
    await s.runExample();
    await s.expectVisible("#own-key-error", "the request for a key");
    await s.expectText("#own-key-error", /^Paste your Anthropic API key/);
    await s.page.keyboard.press("Control+Enter");
    await s.page.waitForTimeout(300);
    s.check(s.posts.length === 0, "with storage full, a restored page ran after its key was forgotten");

    // Storage that cannot be read: nothing tells the restored page whether its key is
    // still the tab's. The key leaves the field and nothing is submitted.
    await s.page.fill("#own-key", FAKE_KEY);
    await s.page.evaluate(() => {
      window.smokeGetItem = Storage.prototype.getItem;
      Storage.prototype.getItem = function getItem() {
        throw new DOMException("The operation is insecure.", "SecurityError");
      };
    });
    await s.restoredFromCache();
    s.check((await s.page.inputValue("#own-key")) === "", "a restored page that cannot read its storage kept the key in its field");
    await s.expectVisible("#payer", "the hold on a restored page that cannot read its storage");
    await s.expectText("#payer-message", /cannot read what this tab saved about who pays, so your Anthropic key was cleared from the page\. Nothing has run\./, "the hold's message");
    s.check(await s.page.evaluate(() => document.getElementById("run").disabled), "Run is enabled on a restored page that cannot read its storage");
    s.check(/on hold until you choose who pays/.test(await s.runDescription()), "Run's description does not say that runs are held");
    await s.page.focus("#text");
    await s.page.keyboard.press("Control+Enter");
    await s.page.waitForTimeout(300);
    s.check(s.posts.length === 0, "a run was sent from a restored page that cannot read its storage");
    // A key entered again settles it, and that key (no older one) is the one sent.
    await s.page.fill("#own-key", OTHER_KEY);
    await s.expectHidden("#payer", "the hold after the key was entered again");
    await s.run();
    await s.expectResults();
    s.check(s.posts.length === 1 && s.posts[0].authorization === `Bearer ${OTHER_KEY}` && s.posts[0].body.provider === "anthropic", "the run after the key was entered again did not carry that key");
    // Or the visitor picks the site's models: no key goes with that run.
    await s.restoredFromCache();
    await s.expectVisible("#payer", "the hold after a second restore");
    s.check((await s.page.inputValue("#own-key")) === "", "the second restore kept the key in its field");
    await s.page.click("#payer-site");
    await s.expectHidden("#payer", "the hold after choosing the site");
    s.check((await s.page.inputValue("#provider")) === "site", "'Use the site's models' did not change who pays");
    await s.run();
    await s.expectResults();
    s.check(s.posts.length === 2 && s.posts[1].authorization === "" && !s.posts[1].body.provider, "the run on the site's models carried a key or a provider");
    await s.page.evaluate(() => {
      Storage.prototype.setItem = window.smokeSetItem;
      Storage.prototype.getItem = window.smokeGetItem;
    });
  },

  // The payer is the same on the restored page and on record, and the models are not:
  // another copy of the page chose a cheaper model for both steps. The next run is
  // billed to the visitor, so it uses the models on record, not the ones the old
  // document still shows.
  async "restore-models"(s) {
    s.health = fixture("health");
    s.ner = { body: (request) => ({ ...fixture("ner-own"), provider: request.provider || "site" }) };
    const record = (choice) => s.page.evaluate(([k, v]) => window.sessionStorage.setItem(k, JSON.stringify(v)), [PAYER_ENTRY, choice]);
    const models = () => s.page.evaluate(() => `${document.getElementById("model").value} / ${document.getElementById("selector").value}`);
    await s.open();
    await s.chooseOwnKey("anthropic");
    s.check((await models()) === "claude-opus-5-5 / retrieval", `an Anthropic key starts on ${await models()}`);
    await record({ provider: "anthropic", model: "claude-haiku-5-5", selector: "claude-haiku-5-5" });
    await s.restoredFromCache();
    s.check((await models()) === "claude-haiku-5-5 / claude-haiku-5-5", `a restored page kept ${await models()} over the models on record`);
    s.check((await s.page.inputValue("#own-key")) === FAKE_KEY, "the restore lost the key that is on record");
    await s.runExample();
    await s.expectResults();
    const post = s.posts[0] || { body: {}, authorization: "" };
    s.check(post.body.provider === "anthropic" && post.body.model === "claude-haiku-5-5" && post.body.selector === "claude-haiku-5-5", `the run after the restore was submitted with ${post.body.model} and ${post.body.selector}`);
    s.check(post.authorization === `Bearer ${FAKE_KEY}`, "the run after the restore did not carry the key");
    // The payer was chosen afresh in the other copy, with no models picked: its defaults.
    await record({ provider: "anthropic", model: "", selector: "" });
    await s.restoredFromCache();
    s.check((await models()) === "claude-opus-5-5 / retrieval", `with no models on record a restored page shows ${await models()}`);
  },

  // A health read is under way when Run is pressed, and it answers during the run with
  // the run's provider turned off. Nothing on the form changes under the run; when the
  // run is refused, the refusal names the provider and the model it was submitted with,
  // in a place that is on show; only then is the form put right for the next run.
  async "health-during-run"(s) {
    s.health = fixture("health");
    s.expectedStatuses.add(401);
    await s.open();
    await s.chooseOwnKey("anthropic");
    await s.page.selectOption("#model", "claude-haiku-5-5");
    const controls = () =>
      s.page.evaluate(() => {
        const $ = (id) => document.getElementById(id);
        const options = (id) => [...$(id).options].map((o) => o.value).join(",");
        return {
          provider: $("provider").value,
          providers: options("provider"),
          key: $("own-key").value,
          keyShown: $("own-key").getClientRects().length > 0,
          model: $("model").value,
          models: options("model"),
          selector: $("selector").value,
          selectors: options("selector"),
          hold: !$("payer").hidden,
          summary: $("options-summary").textContent,
        };
      });
    const before = await controls();
    const late = s.gate();
    const answer = s.gate();
    s.healthQueue = [{ gate: late.promise, health: fixture("health-own-off") }];
    s.health = fixture("health-own-off");
    s.ner = { gate: answer.promise, status: 401, body: fixture("error-own-refused") };
    await s.restoredHeld();
    await s.runExample();
    await s.expectVisible("#progress", "the progress bar");
    late.release();
    await s.page.waitForTimeout(700);
    const during = await controls();
    s.check(JSON.stringify(during) === JSON.stringify(before), `a health answer changed the form during a run: ${JSON.stringify(before)} -> ${JSON.stringify(during)}`);
    await s.expectText("#run-hint", /^Runs on your Anthropic key\./, "the hint during the run");
    answer.release();
    await s.settled();
    s.check((s.posts[0] || { body: {} }).body.provider === "anthropic" && s.posts[0].body.model === "claude-haiku-5-5" && s.posts[0].authorization === `Bearer ${FAKE_KEY}`, "the run was not the Anthropic run on Haiku");
    // The refusal is on show, against the submitted provider and model.
    await s.expectVisible("#error", "the refusal, in a place that is on show");
    await s.expectText("#error-title", /^The run on your Anthropic key did not go through\.$/, "the title of the refusal");
    await s.expectText("#error-message", /^Anthropic did not accept this key\. .* may use Claude Haiku 5\.5\.$/, "the refusal, bound to the submitted provider and model");
    const hiddenAlerts = await s.page.evaluate(() => [...document.querySelectorAll('[role="alert"]')].filter((n) => n.textContent.trim() && n.getClientRects().length === 0).map((n) => n.id));
    s.check(hiddenAlerts.length === 0, `a message is written into an element that is not on show: ${hiddenAlerts.join(", ")}`);
    // And the form is put right for the next run: the provider is gone, its key with it.
    s.check((await s.page.inputValue("#provider")) === "site", "the provider that was turned off is still chosen after the run");
    await s.expectVisible("#payer", "the payer hold after the run");
    await s.expectText("#payer-message", /no longer accepts your own Anthropic key, so the key has been forgotten/);
    s.check((await s.page.evaluate((k) => window.sessionStorage.getItem(k), OWN_KEY_ENTRY)) === null, "the page says the key was forgotten while it is still in session storage");
    await s.page.keyboard.press("Control+Enter");
    await s.page.waitForTimeout(300);
    s.check(s.posts.length === 1, "a run was sent while the payer was unsettled");
    await s.page.click("#payer-site");
    await s.expectHidden("#payer", "the payer hold after choosing the site");
    await s.page.click("#error-dismiss");
  },

  // A restored page holds an older health answer than the one another copy of the page
  // saved its payer under. The page does not say that the key was forgotten while the
  // key is still in session storage: it waits for the health read it makes, and then
  // either brings the payer back with its key, or removes the key and says so.
  async "restore-stale-health"(s) {
    const stored = (entry) => s.page.evaluate((k) => window.sessionStorage.getItem(k), entry);
    const save = (provider, model) =>
      s.page.evaluate(
        ([keyEntry, payerEntry, key, p, m]) => {
          window.sessionStorage.setItem(keyEntry, JSON.stringify({ provider: p, key }));
          window.sessionStorage.setItem(payerEntry, JSON.stringify({ provider: p, model: m, selector: "retrieval" }));
        },
        [OWN_KEY_ENTRY, PAYER_ENTRY, FAKE_KEY, provider, model],
      );
    const forgottenWhileStored = async (when) => {
      const message = (await s.visible("#payer-message")) ? await s.text("#payer-message") : "";
      s.check(!(/forgotten/.test(message) && (await stored(OWN_KEY_ENTRY)) !== null), `${when}: the page says the key was forgotten while it is still in session storage`);
      return message;
    };
    s.health = fixture("health-own-off");
    s.ner = { body: (request) => ({ ...fixture("ner-own"), provider: request.provider || "site" }) };
    s.expectedStatuses.add(500);
    await s.open();
    s.check(!(await s.visible("#run-with")) && (await s.page.evaluate(() => document.getElementById("provider").options.length)) === 1, "this page's health answer offers own keys");

    // Another copy, loaded when own keys were on offer, saved an OpenAI key. Back.
    await save("openai", "gpt-6-luna");
    const fresh = s.gate();
    s.healthQueue = [{ gate: fresh.promise, health: fixture("health") }];
    s.health = fixture("health");
    await s.restoredHeld();
    await s.expectVisible("#payer", "the hold while the restored page waits for health");
    const waiting = await forgottenWhileStored("before fresh health");
    s.check(/Checking whether it can still be used/.test(waiting), `while it waits the page says ${JSON.stringify(waiting)}`);
    s.check(await s.page.evaluate(() => document.getElementById("run").disabled), "Run is enabled while the saved payer is unsettled");
    fresh.release();
    await s.expectHidden("#payer", "the hold once fresh health offers the saved payer");
    s.check((await s.page.inputValue("#provider")) === "openai", `fresh health offers OpenAI, and the payer is ${await s.page.inputValue("#provider")}`);
    s.check((await s.page.inputValue("#own-key")) === FAKE_KEY, "the key on record did not come back with its payer");
    s.check(!(await s.page.evaluate(() => document.getElementById("run").disabled)), "Run stayed held although the saved payer is offered");
    await s.runExample();
    await s.expectResults();
    s.check(s.posts.length === 1 && s.posts[0].body.provider === "openai" && s.posts[0].authorization === `Bearer ${FAKE_KEY}`, "the run after the restore was not the OpenAI run on the saved key");

    // A saved payer that fresh health cannot be asked about: no word of "forgotten".
    await save("mistral", "mistral-large");
    s.healthQueue = [{ health: "fail" }];
    await s.restoredFromCache();
    await s.expectVisible("#payer-retry", "'Try again' when the health read of a restored page fails");
    const unknown = await forgottenWhileStored("after a failed health read");
    s.check(/could not be loaded/.test(unknown), `after a failed read the page says ${JSON.stringify(unknown)}`);
    s.check((await stored(OWN_KEY_ENTRY)) !== null, "a failed health read removed the key");
    // Health answers and does not offer it: the key is removed, and then the page says so.
    await s.page.click("#payer-retry");
    await s.page.waitForLoadState("networkidle").catch(() => {});
    await s.expectHidden("#payer-retry", "'Try again' once health answered");
    const lost = await forgottenWhileStored("after health answered without the payer");
    s.check(/so the key has been forgotten/.test(lost), `once health does not offer the payer the page says ${JSON.stringify(lost)}`);
    s.check((await stored(OWN_KEY_ENTRY)) === null, "the key of a payer that is not offered is still in session storage");
    await s.page.keyboard.press("Control+Enter");
    await s.page.waitForTimeout(300);
    s.check(s.posts.length === 1, "a run was sent while the payer was unsettled");
    await s.page.click("#payer-site");
    await s.expectHidden("#payer", "the payer hold after choosing the site");
  },

  // What a result says about its own run is counted from its entities: a selector that
  // declined every candidate reviewed them and chose nothing; "each mention" is said
  // only when it is each one; and the catalog-wide search flags do not overrule a term
  // that this run did find.
  async "run-claims"(s) {
    const text = () => s.page.evaluate(() => document.getElementById("results").innerText);
    const rerun = async (response) => {
      s.ner = { body: response };
      await s.run();
      await s.settled();
    };
    // One mention, and the model declined all its candidates.
    await s.runShaped(shapedResponse(SHAPED_TEXT, [["oily fish", "food", "no_match", null]]));
    await s.expectText("#status", /Read by Claude Opus 5\.5; candidates reviewed by Claude Opus 5\.5, which chose no term\.$/, "the status line when the model declined every candidate");
    s.check(!/terms chosen by|chose among|Terms picked by/.test(await text()), "a model that declined every candidate is credited with choosing terms");
    await s.expectText("#next-body", /^The model never writes an ID\. xwalk looked up each mention by keyword and by meaning\. The catalog holds/, "the closing panel when the model declined every candidate");
    await s.page.click("#run-details > summary");
    await s.expectText("#results-models", /^Extracted by Claude Opus 5\.5\. Candidates reviewed by Claude Opus 5\.5; no term was picked\.$/, "Run details when the model declined every candidate");

    // One mention with a top search result, one that was not looked up; no model.
    await rerun(shapedResponse(SHAPED_TEXT, [["Aspirin", "chemical", null, null], ["BRCA1", "other", null, null]], { selector: "retrieval" }));
    await s.expectText("#next-body", /^The model never writes an ID\. xwalk's search found terms for 1 of the 2 mentions by keyword and by meaning, and that one shows its top search result\. The catalog holds/, "the closing panel with one hit and one mention not looked up");
    await s.expectText("#results-takeaway", /^No model chose here: 1 of the 2 mentions shows its top search result\.$/, "the search-only line with one hit of two");
    s.check(!/each mention|each entity/.test(await text()), "the result credits each mention with a search result while one was not looked up");
    s.check((await s.page.locator("#cards .card.is-unsearched").count()) === 1, "no card is marked as not looked up");

    // Both catalog-wide flags say "unavailable" (one ontology lacks an index) while this
    // run's mentions were found, by both kinds of search.
    const off = { bm25: { status: "unavailable", reason: "x" }, dense: { status: "unavailable", reason: "y" } };
    await rerun(shaped({ retrieval: off }));
    await s.expectText("#notices", /Parts of the search were unavailable for this run, so some mentions may have been found by one kind of search only, or not at all\./, "the notice when both flags are off and terms were found");
    s.check(!/could not be looked up|search was unavailable for this run/.test(await text()), "the result says the search was unavailable beside a term it found");
    // One flag off, and candidates ranked by that kind of search all the same.
    await rerun(shaped({ retrieval: { bm25: { status: "ok" }, dense: { status: "unavailable", reason: "y" } } }));
    await s.expectText("#notices", /Matching by meaning was unavailable for part of this run, so some terms were found by keyword only\./, "the notice when matching by meaning was off for part of the run");
    s.check(!/Part of the search \(matching by meaning\)/.test(await text()), "the result says every term was found by keyword only beside terms ranked by meaning");
  },

  // The payer entry is rewritten on arrival. Only ids the page can name go back into
  // it: strings that were put into the entry and are no offered model, selector or
  // known provider are not preserved.
  async "payer-entry-clean"(s) {
    s.health = fixture("health");
    const entry = () => s.page.evaluate((k) => window.sessionStorage.getItem(k), PAYER_ENTRY);
    const seed = (value) => s.page.evaluate(([k, v]) => window.sessionStorage.setItem(k, JSON.stringify(v)), [PAYER_ENTRY, value]);
    await s.open();
    await seed({ provider: "anthropic", model: "Patient Jane Roe takes warfarin", selector: "sk-not-an-offered-selector" });
    await s.page.reload({ waitUntil: "networkidle" });
    s.check((await entry()) === JSON.stringify({ provider: "anthropic", model: "", selector: "" }), `strings that are no offered ids were written back: ${await entry()}`);
    s.check((await s.page.inputValue("#model")) === "claude-opus-5-5" && (await s.page.inputValue("#selector")) === "retrieval", "ids that are not offered changed the lists");
    // Ids that are offered are kept as they are.
    await seed({ provider: "anthropic", model: "claude-haiku-5-5", selector: "claude-sonnet-5-5" });
    await s.page.reload({ waitUntil: "networkidle" });
    s.check((await entry()) === JSON.stringify({ provider: "anthropic", model: "claude-haiku-5-5", selector: "claude-sonnet-5-5" }), `offered ids were not kept: ${await entry()}`);
    // A provider this page cannot name: the hold stays (after every reload), the string does not.
    await seed({ provider: "Patient Jane Roe", model: "claude-haiku-5-5", selector: "x" });
    await s.page.reload({ waitUntil: "networkidle" });
    await s.expectVisible("#payer", "the payer hold for a provider that is not offered");
    s.check((await entry()) === JSON.stringify({ provider: "other", model: "", selector: "" }), `an unknown provider string was written back: ${await entry()}`);
    await s.page.reload({ waitUntil: "networkidle" });
    await s.expectVisible("#payer", "the payer hold after another reload");
    await s.page.click("#payer-site");
    await s.expectHidden("#payer", "the payer hold after choosing the site");
    s.check((await entry()) === null, "choosing the site left a payer entry");
  },

  // ---------- the guide: the figure of the pipeline ----------

  // On arrival with the recorded example: one compact row, nothing playing, nothing
  // moved, and the example result no lower on a phone than the row allows.
  async "guide-arrival"(s) {
    s.health = productionHealth();
    await s.open();
    await s.expectExample("on arrival");
    await s.page.waitForTimeout(1200); // anything that played by itself would have started
    const g = await s.guide();
    s.check(g.view === "collapsed" && g.toggle === "How it works" && g.expanded === "false", `on arrival the figure is ${g.view}, its button ${JSON.stringify(g.toggle)} (${g.expanded})`);
    s.check(g.play === "idle" && g.active.length === 0 && !g.token, `something plays on arrival: ${g.play}, ${g.active.join(",")}`);
    for (const k of ["reader", "retriever", "selector"]) s.check(g.nodes[k].seen && g.nodes[k].tag === "BUTTON" && g.nodes[k].pressed === null, `the ${k} agent is not a plain button in the collapsed row`);
    s.check(/^Reader finds the mentions/.test(g.nodes.reader.name) && /^Retriever /.test(g.nodes.retriever.name) && /^Selector /.test(g.nodes.selector.name), `the agents are named ${JSON.stringify([g.nodes.reader.name, g.nodes.retriever.name, g.nodes.selector.name])}`);
    s.check(!g.nodes.result.seen && !g.nodes.scorer.seen && !g.panelSeen && !g.playSeen, "collapsed, the row shows more than the three agents");
    s.check(g.facesHidden, "a face is exposed to assistive technology");
    s.check(g.height <= (s.phone ? 60 : 48), `the collapsed row is ${g.height}px tall`);
    const seen = await s.shown();
    s.check(seen.scrollY === 0 && seen.focus === "BODY", `the page moved on arrival: scrollY ${seen.scrollY}, focus ${seen.focus}`);
    // 787px before the figure replaced the three steps (390x844).
    if (s.width === 390) s.check(seen.top <= 787 + 40, `the example result starts at ${seen.top}px on a phone, more than 40px lower than before`);
    await s.expectHidden("#progress", "the progress bar");
    await s.expectHidden("#pipe .pipe-desc", "what each agent does, while idle");
  },

  // Expand, and follow three mentions of the example to the end: one where the selector
  // overruled the search, one that was the top hit, and what the boxes say afterwards.
  async "guide-play"(s) {
    s.health = productionHealth();
    await s.open();
    await s.expectExample("before the figure is opened");
    const entity = (text) => EXAMPLE_FILE.response.entities.find((e) => e.text === text);
    await s.page.focus("#guide-toggle");
    const before = await s.shown();
    await s.page.keyboard.press("Enter");
    await s.page.waitForTimeout(700);
    let g = await s.guide();
    s.check(g.view === "expanded" && g.toggle === "Close" && g.expanded === "true", `after "How it works" the figure is ${g.view}, its button ${JSON.stringify(g.toggle)}`);
    // The visitor asked to see it: it opens, comes into view and plays the chosen mention once.
    s.check(g.play === "playing" || g.play === "done", `opening the figure did not start a play: ${g.play}`);
    const opened = await s.shown();
    s.check(opened.scrollY >= before.scrollY && opened.focus === "guide-toggle", `opening the figure lost the place: scrollY ${opened.scrollY}, focus ${opened.focus}`);
    await s.page.waitForFunction(() => document.getElementById("pipe").getAttribute("data-play") === "done", null, { timeout: 25000 });
    await s.page.waitForTimeout(300);
    g = await s.guide();
    s.check(g.chips.length === EXAMPLE_FILE.response.entities.length && g.chips.filter((c) => c.current).map((c) => c.text).join() === "metformin", `the mentions to follow are ${JSON.stringify(g.chips)}`);
    s.check(g.chips.every((c) => c.tag === "BUTTON" && /\((chemical|disease|food)\)$/.test(c.name)), "a mention chip is not a button named with its type");
    // At rest: the finished figure for the mention where the selector overruled the search.
    s.check(g.nodes.result.label === "CHEBI:6801" && /^Result/.test(g.panel), `at rest the figure shows ${g.nodes.result.label}: ${JSON.stringify(g.panel.slice(0, 60))}`);
    s.check(g.hint === "Following “metformin”. Tap a box to see what it does.", `the hint says ${JSON.stringify(g.hint)}`);
    for (const k of Object.keys(g.nodes)) s.check(g.nodes[k].seen && g.nodes[k].tag === "BUTTON" && /\S/.test(g.nodes[k].name), `the ${k} box is not a visible, named button`);
    s.check(g.live === "off" || g.live === "polite", `the panel's aria-live is ${g.live}`);
    await s.expectText(".guide-rowlabel", /^The xwalk library continues the loop \(not run by this demo\)$/, "the label of the dashed rows");
    await s.expectText(".guide-loop", /^The full loop is described in the FoodOntoRAG preprint\.$/, "the line under the dashed rows");
    const link = await s.page.evaluate(() => {
      const a = document.querySelector(".guide-loop a");
      return [a.href, a.target, a.rel];
    });
    s.check(link.join() === "https://arxiv.org/abs/2603.09758,_blank,noopener", `the preprint link is ${link.join(" ")}`);

    // Overruled: the one above is considered first, then the choice; the reason is
    // revealed, not retyped, so the panel is rewritten once per box and no more.
    await s.guideWatch();
    await s.page.click("#guide-play");
    await s.guidePlayed("metformin");
    let log = await s.guideLog();
    g = await s.guide();
    s.check(log.active.join() === "reader,retriever,selector,result", `the mention visited ${log.active.join(",")}`);
    const first = entity("metformin").matches[0].label;
    s.check(log.chips.indexOf(`consider:1. ${first}`) >= 0 && log.chips.indexOf(`consider:1. ${first}`) < log.chips.indexOf("picked:2. metformin"), `the selector did not consider the top hit before its choice: ${log.chips.join(" | ")}`);
    s.check(log.writes <= 4 && log.retyped === 0, `the panel was rewritten ${log.writes} times and retyped ${log.retyped} times during one play`);
    s.check(log.live.every((v) => v === "polite"), `the panel was ${log.live.join(",")} while it played`);
    s.check(log.overflow === 0 && log.tokenOut === 0, `the travelling label widened the page (${log.overflow}) or left the figure (${log.tokenOut})`);
    s.check(log.tokens.includes("metformin") && log.tokens.includes("CHEBI:6801"), `the travelling label read ${log.tokens.join(", ")}`);
    s.check(g.play === "done" && g.playLabel === "Play again" && g.active.length === 0 && !g.token, `after the play: ${g.play}, ${JSON.stringify(g.playLabel)}, label shown ${g.token}`);
    s.check(g.nodes.result.label === "CHEBI:6801" && g.at === "result", `the journey ended at ${g.at} with ${g.nodes.result.label}`);
    s.check(/metformin → CHEBI:6801 metformin/.test(g.panel) && /Confidence 0\.93: the model's own, not a measured accuracy\. Not the top search hit: the selector overruled the ranking\./.test(g.panel), `the result panel says ${JSON.stringify(g.panel)}`);
    s.check(g.panel.includes(entity("metformin").pick.explanation), "the result panel does not carry the model's reason as the response gives it");

    // The boxes afterwards: the finished state of each.
    await s.page.click('.guide-node[data-k="reader"]');
    g = await s.guide();
    s.check(/^Reader · Claude Opus 5\.5 reads the whole text and marks the mentions/.test(g.panel) && /…of hummus\. She takes metformin for type 2 diabetes,…/.test(g.panel) && /“metformin” is a chemical, so it goes to ChEBI/.test(g.panel), `the Reader panel says ${JSON.stringify(g.panel)}`);
    s.check(g.nodes.reader.pressed === "true" && g.nodes.result.pressed === "false" && g.live === "polite", "the Reader box is not marked as the one on show");
    await s.page.click('.guide-node[data-k="retriever"]');
    g = await s.guide();
    s.check(/^Retriever · xwalk searches ChEBI two ways and merges the rankings/.test(g.panel), `the Retriever panel says ${JSON.stringify(g.panel.slice(0, 90))}`);
    s.check(g.lanes.join(" | ") === "Keyword search | Meaning search | Merged, 21 candidates", `the lanes are ${g.lanes.join(" | ")}`);
    const ranks = (key) => [...entity("metformin").matches, ...entity("metformin").more].filter((m) => key in m.retrievers).sort((a, b) => a.retrievers[key] - b.retrievers[key]).slice(0, 4).map((m) => `${m.retrievers[key]}. ${m.label}`);
    s.check(g.laneItems[0].join("|") === ranks("bm25").join("|") && g.laneItems[1].join("|") === ranks("dense-qwen3").join("|"), `the lanes do not hold each search's own top four: ${JSON.stringify(g.laneItems)}`);
    s.check(g.laneItems[2].join("|") === entity("metformin").matches.slice(0, 4).map((m, i) => `${i + 1}. ${m.label}`).join("|"), `the merged lane is ${JSON.stringify(g.laneItems[2])}`);
    await s.page.click('.guide-node[data-k="selector"]');
    g = await s.guide();
    s.check(/^Selector · Claude Opus 5\.5 reads the shortlist against the sentence/.test(g.panel) && /2\. metformin \(chosen\)/.test(g.panel) && /These are 4 of the 21 candidates it was given\./.test(g.panel), `the Selector panel says ${JSON.stringify(g.panel)}`);
    if (s.shot) await s.shot("selector");

    // The top hit: settled at once, nothing above it to consider.
    await s.guideWatch();
    await s.follow("Aspirin");
    await s.guidePlayed("Aspirin");
    log = await s.guideLog();
    g = await s.guide();
    s.check(!log.chips.some((c) => c.startsWith("consider:")) && log.chips.includes(`picked:1. ${entity("Aspirin").matches[0].label}`), `the top hit was not settled at once: ${log.chips.join(" | ")}`);
    s.check(g.nodes.result.label === "CHEBI:15365" && /Also the top search hit\./.test(g.panel) && !/overruled/.test(g.panel), `the top-hit result says ${JSON.stringify(g.panel)}`);
    s.check(g.chips.filter((c) => c.current).map((c) => c.text).join() === "Aspirin" && g.hint.startsWith("Following “Aspirin”."), "the figure does not say that it follows Aspirin");
    s.check(log.overflow === 0 && log.tokenOut === 0, `the travelling label widened the page (${log.overflow}) or left the figure (${log.tokenOut})`);

    // A disease goes to the disease ontology, by the name health gives it.
    await s.follow("fever");
    await s.page.click('.guide-node[data-k="retriever"]');
    g = await s.guide();
    s.check(g.play !== "playing" && g.active.length === 0, "a tap on a box did not end the play");
    s.check(/^Retriever · xwalk searches CTD diseases \/ MeSH two ways/.test(g.panel), `the Retriever panel for a disease says ${JSON.stringify(g.panel.slice(0, 80))}`);

    // Keyboard focus is visible on a box, a chip and Play.
    await s.page.keyboard.press("Tab");
    for (const sel of ['.guide-node[data-k="scorer"]', "#guide-chips .guide-men", "#guide-play"]) {
      await s.page.focus(sel);
      const ring = await s.page.evaluate((q) => {
        const style = getComputedStyle(document.querySelector(q));
        return `${style.outlineStyle} ${style.outlineWidth}`;
      }, sel);
      s.check(ring === "solid 3px", `${sel} shows ${ring} as its focus ring`);
    }
    if (s.shot) await s.shot("finished");
    await s.page.click("#guide-toggle");
    g = await s.guide();
    s.check(g.view === "collapsed" && g.toggle === "How it works" && !g.panelSeen, "Close did not collapse the figure");
    const end = await s.shown();
    s.check(end.stored.length === 0 && s.posts.length === 0, `the figure stored something or cost a request: ${end.stored.join(",")}, ${s.posts.length} posts`);
  },

  // A declined pick: the selector considers the shortlist and settles on none. And a
  // mention that only one of the two searches found candidates for has one lane.
  async "guide-declined"(s) {
    const entity = (text) => GUIDE_RUN.response.entities.find((e) => e.text === text);
    await s.runGuide();
    await s.page.click("#guide-toggle");
    await s.guideWatch();
    await s.follow("oily fish");
    await s.page.waitForSelector(".guide-node.is-active[data-k='selector']", { timeout: 15000 }).catch(() => s.failures.push("the declined mention never reached the Selector"));
    if (s.shot) await s.shot("considering");
    await s.guidePlayed("oily fish");
    const log = await s.guideLog();
    let g = await s.guide();
    const short = entity("oily fish").matches.slice(0, 4).map((m, i) => `${i + 1}. ${m.label}`);
    s.check(short.every((c) => log.chips.includes(`consider:${c}`)) && !log.chips.some((c) => c.startsWith("picked:")), `the selector did not consider the four and pick none: ${log.chips.join(" | ")}`);
    s.check(log.chips.includes("on:None of these fits"), "the verdict 'None of these fits' was never shown");
    s.check(log.active.join() === "reader,retriever,selector,result", `the declined mention visited ${log.active.join(",")}`);
    s.check(g.nodes.result.label === "no term fits" && g.at === "result", `the journey ended at ${g.at} with ${JSON.stringify(g.nodes.result.label)}`);
    s.check(/oily fish → no term fits/.test(g.panel) && /Claude Opus 5\.5 looked at 30 candidates and declined to pick one\./.test(g.panel) && g.panel.includes(entity("oily fish").pick.explanation), `the result panel says ${JSON.stringify(g.panel)}`);
    s.check(g.ids.length === 0 && !/Confidence/.test(g.panel), `a declined pick shows an ID or a confidence: ${g.ids.join(",")}`);
    await s.page.click('.guide-node[data-k="selector"]');
    g = await s.guide();
    s.check(/None of these fits/.test(g.panel) && !/\(chosen\)/.test(g.panel) && g.panel.includes(entity("oily fish").pick.explanation), `the Selector panel says ${JSON.stringify(g.panel)}`);

    // HbA1c: keyword search ranked none of its candidates. One lane, labelled.
    await s.follow("HbA1c");
    await s.page.click('.guide-node[data-k="retriever"]');
    g = await s.guide();
    s.check(g.lanes.join(" | ") === "Meaning search, 20 candidates", `the lanes for a mention one search found are ${g.lanes.join(" | ")}`);
    s.check(/only meaning search ranked candidates for this mention/.test(g.panel) && !/two ways|Merged/.test(g.panel), `the Retriever panel says ${JSON.stringify(g.panel.slice(0, 120))}`);
    s.check(g.nodes.retriever.sub === "meaning search only", `the Retriever box says ${JSON.stringify(g.nodes.retriever.sub)}`);
  },

  // A mention of a type with no ontology: the journey ends after the Reader.
  async "guide-unmapped"(s) {
    await s.runGuide();
    await s.page.click("#guide-toggle");
    await s.guideWatch();
    await s.follow("BRCA1");
    await s.guidePlayed("BRCA1");
    const log = await s.guideLog();
    let g = await s.guide();
    s.check(log.active.join() === "reader", `a mention with no ontology visited ${log.active.join(",")}`);
    s.check(g.at === "reader" && /Recognised, but this demo has no ontology for it\./.test(g.panel) && /“BRCA1” is marked as another kind of mention/.test(g.panel), `the journey ended at ${g.at}: ${JSON.stringify(g.panel)}`);
    s.check(!/goes to/.test(g.panel), "the Reader sends a mention with no ontology to one");
    s.check(["retriever", "selector", "result"].every((k) => g.nodes[k].skipped) && !g.nodes.reader.skipped, "the boxes the mention did not reach are not marked");
    s.check(g.nodes.result.label === "not looked up" && g.ids.length === 0, `the result box says ${JSON.stringify(g.nodes.result.label)}`);
    if (s.shot) await s.shot("ended");
    for (const k of ["retriever", "selector", "result"]) {
      await s.page.click(`.guide-node[data-k="${k}"]`);
      g = await s.guide();
      s.check(/“BRCA1” did not get this far: this demo has no ontology for its type, so it was not looked up\./.test(g.panel), `the ${k} panel for a mention that never reached it says ${JSON.stringify(g.panel)}`);
      s.check(g.ids.length === 0 && !/Claude|candidates/.test(g.panel), `the ${k} panel invents a step for a mention that never reached it`);
    }
  },

  // Retrieval alone ranked, and Jev: the Selector box says what actually chose, and no
  // reason is shown where the response has none.
  async "guide-retrieval-only"(s) {
    await s.runShaped(shapedResponse(SHAPED_TEXT, [["Aspirin", "chemical", null, null], ["BRCA1", "other", null, null]], { selector: "retrieval", requested_selector: "retrieval", model: "claude-sonnet-5-5" }));
    await s.page.click("#guide-toggle");
    await s.guideWatch();
    await s.page.click("#guide-play");
    await s.guidePlayed("Aspirin");
    const log = await s.guideLog();
    let g = await s.guide();
    s.check(!log.chips.some((c) => c.startsWith("consider:")) && log.chips.includes("picked:1. Aspirin (salt)"), `with no model the top result was not simply taken: ${log.chips.join(" | ")}`);
    s.check(g.nodes.selector.sub === "top search result, no model" && g.nodes.result.sub === "the top search result", `the boxes say ${JSON.stringify([g.nodes.selector.sub, g.nodes.result.sub])}`);
    s.check(/Aspirin → CHEBI:100 Aspirin \(salt\)/.test(g.panel) && /The top search result\. No model chose, so there is no reason and no confidence\./.test(g.panel), `the result panel says ${JSON.stringify(g.panel)}`);
    s.check(g.bubbles === 0 && !/Confidence \d/.test(g.panel), "a reason or a confidence is shown where no model chose");
    await s.page.click('.guide-node[data-k="selector"]');
    g = await s.guide();
    s.check(/^Selector · no model chose here: the top search result is taken as it is/.test(g.panel) && g.bubbles === 0 && !/Claude|Opus|Sonnet|reads the shortlist/.test(g.panel), `the Selector panel says ${JSON.stringify(g.panel)}`);
    // Who read is named from the response, not from the page's defaults.
    await s.page.click('.guide-node[data-k="reader"]');
    g = await s.guide();
    s.check(/^Reader · Claude Sonnet 5\.5 reads the whole text/.test(g.panel), `the Reader panel says ${JSON.stringify(g.panel.slice(0, 80))}`);

    // Jev chose, and gave no reason for one of them.
    const jev = shapedResponse(SHAPED_TEXT, [["Aspirin", "chemical", "match", 1], ["metformin", "chemical", "no_match", null]]);
    for (const e of jev.entities) e.pick = { ...e.pick, selector: "jev", explanation: e.text === "Aspirin" ? "" : "same 0.10" };
    s.ner = { body: jev };
    await s.run();
    await s.settled();
    await s.page.click('.guide-node[data-k="selector"]');
    g = await s.guide();
    s.check(/^Selector · Jev, a small trained ranking model, picks from the shortlist/.test(g.panel) && /2\. aspirin \(chosen\)/.test(g.panel), `the Selector panel for Jev says ${JSON.stringify(g.panel)}`);
    s.check(g.bubbles === 0 && g.nodes.selector.sub === "Jev picks one, or none" && g.nodes.result.sub === "chosen by Jev", `Jev's pick without a reason shows one: ${g.bubbles} bubbles, ${JSON.stringify([g.nodes.selector.sub, g.nodes.result.sub])}`);
    s.check(!/Claude|Opus|Sonnet/.test(g.panel), "a pick by Jev is credited to a language model");
    await s.follow("metformin");
    await s.page.click('.guide-node[data-k="selector"]');
    g = await s.guide();
    s.check(/None of these fits/.test(g.panel) && /same 0\.10/.test(g.panel) && g.bubbles === 1, `Jev's declined pick says ${JSON.stringify(g.panel)}`);
  },

  // Only one retriever available: one lane, labelled, with the server's note in plain
  // words. And a mention the search found nothing for stops at the Retriever.
  async "guide-one-retriever"(s) {
    s.health = fixture("health-degraded");
    s.ner = { body: fixture("ner-degraded") };
    await s.open();
    await s.runExample();
    await s.expectResults();
    await s.page.click('.guide-node[data-k="retriever"]');
    let g = await s.guide();
    s.check(g.view === "expanded", "a tap on an agent did not open the figure");
    s.check(g.lanes.length === 1 && /^Keyword search, 1 candidate$/.test(g.lanes[0]), `with one retriever the lanes are ${g.lanes.join(" | ")}`);
    s.check(/only keyword search ranked candidates for this mention/.test(g.panel) && /Found by keyword search only; matching by meaning was unavailable\./.test(g.panel) && !/Meaning search|Merged|two ways/.test(g.panel), `the Retriever panel says ${JSON.stringify(g.panel)}`);
    s.check(g.nodes.retriever.sub === "keyword search only", `the Retriever box says ${JSON.stringify(g.nodes.retriever.sub)}`);
    await s.page.click('.guide-node[data-k="reader"]');
    g = await s.guide();
    s.check(/^Reader · the offline stand-in reads the whole text/.test(g.panel) && !/Claude|Opus/.test(g.panel), `the Reader panel offline says ${JSON.stringify(g.panel.slice(0, 80))}`);
    if (s.shot) await s.shot("one-lane");

    await s.guideWatch();
    await s.follow("aspirin");
    await s.guidePlayed("aspirin");
    const log = await s.guideLog();
    g = await s.guide();
    s.check(log.active.join() === "reader,retriever", `a mention the search found nothing for visited ${log.active.join(",")}`);
    s.check(g.at === "retriever" && /found no term for this mention/.test(g.panel) && g.lanes.length === 0, `its journey ended at ${g.at}: ${JSON.stringify(g.panel)}`);
    s.check(g.nodes.result.label === "no term found" && g.nodes.selector.skipped && g.nodes.result.skipped, `the result box says ${JSON.stringify(g.nodes.result.label)}`);
  },

  // No result and no example (a sample deployment before a run): Play is unavailable
  // and every box still explains itself. The dashed boxes carry their tag.
  async "guide-dashed"(s) {
    s.health = fixture("health");
    await s.open();
    await s.expectNoExample("a sample deployment");
    await s.page.click('.guide-node[data-k="reader"]');
    let g = await s.guide();
    s.check(g.view === "expanded" && g.playDisabled && g.chips.length === 0 && !g.chipsSeen, `with no result: ${g.view}, Play disabled ${g.playDisabled}, ${g.chips.length} chips`);
    s.check(g.hint === "Map a text to follow one of its mentions here. Tap a box to see what it does.", `the hint says ${JSON.stringify(g.hint)}`);
    const find = await s.page.evaluate(() => document.getElementById("pipe-find").textContent);
    s.check(g.panel.startsWith("Reader") && g.panel.includes(find) && /There is no result on the page yet/.test(g.panel) && !/Claude|Opus|Sonnet/.test(g.panel), `the Reader panel with no result says ${JSON.stringify(g.panel)}`);
    s.check(g.nodes.result.label === "Ontology ID" && !Object.values(g.nodes).some((n) => n.skipped), "with no result a box is marked as not reached");
    for (const [k, name] of [["scorer", "Scorer"], ["verifier", "Verifier"], ["policy", "Policy"], ["rewriter", "Rewriter"], ["ledger", "Ledger"]]) {
      await s.page.click(`.guide-node[data-k="${k}"]`);
      g = await s.guide();
      s.check(g.panel.startsWith(name) && /In the xwalk library\. Not run by this demo\.$/.test(g.panel) && g.panel.length > name.length + 80, `the ${name} panel says ${JSON.stringify(g.panel)}`);
      s.check(g.nodes[k].pressed === "true" && g.at === k && g.play === "idle" && g.active.length === 0, `a tap on ${name} did more than show its panel`);
      if (k === "policy") {
        s.check(/matched\s*needs review\s*unmatched\s*failed/.test(g.panel), `the Policy panel lists ${JSON.stringify(g.panel)}`);
        if (s.shot) await s.shot("policy");
      }
    }
    for (const k of ["retriever", "selector", "result"]) {
      await s.page.click(`.guide-node[data-k="${k}"]`);
      g = await s.guide();
      s.check(g.panel.length > 60 && !/Claude|Opus|Sonnet|undefined|null/.test(g.panel), `the ${k} panel with no result says ${JSON.stringify(g.panel)}`);
    }
    await s.page.click("#guide-play", { force: true }).catch(() => {});
    g = await s.guide();
    s.check(g.play === "idle" && g.active.length === 0, "Play did something with no result on the page");
  },

  // A run started while the figure is open: the bar and the clock show, the three
  // agents are awake, the figure can still be played, and none of it touches the run.
  async "guide-running"(s) {
    s.health = productionHealth();
    const gate = s.gate();
    s.ner = { gate: gate.promise, body: GUIDE_RUN.response };
    await s.open();
    await s.expectExample("before the run");
    await s.page.click("#guide-toggle");
    await s.page.fill("#text", GUIDE_RUN.text);
    await s.run();
    await s.expectVisible("#progress", "the progress bar with the figure open");
    await s.page.waitForTimeout(800); // the page scrolls the bar into view smoothly
    const during = await s.page.evaluate(() => {
      const r = document.getElementById("progress").getBoundingClientRect();
      const name = (q) => getComputedStyle(document.querySelector(q)).animationName;
      return {
        seen: r.top >= 0 && r.bottom <= window.innerHeight,
        awake: ["reader", "retriever", "selector"].map((k) => name(`.guide-node[data-k="${k}"] .guide-face`)),
        asleep: name('.guide-node[data-k="scorer"] .guide-face'),
        time: document.getElementById("progress-time").textContent,
      };
    });
    s.check(during.seen, "the progress bar is off screen during a run with the figure open");
    s.check(during.awake.every((a) => a === "guide-bob") && during.asleep === "none", `during a run the faces are ${during.awake.join(",")}; a library box is ${during.asleep}`);
    s.check(/^\d+ s$/.test(during.time), `the clock says ${JSON.stringify(during.time)}`);
    let g = await s.guide();
    s.check(g.view === "expanded" && g.chips.length === EXAMPLE_FILE.response.entities.length && !g.playDisabled, "the figure of the example is not there during the run");
    s.check(!Object.values(g.nodes).some((n) => n.done), "a box is ticked during a run");
    // The example can be played while waiting, and boxes tapped: the run goes on.
    await s.page.click("#guide-play");
    await s.page.waitForSelector(".guide-node.is-active", { timeout: 5000 }).catch(() => s.failures.push("Play did nothing during a run"));
    await s.page.click('.guide-node[data-k="policy"]');
    await s.follow("hummus");
    if (s.shot) await s.shot("running");
    s.check((await s.page.getAttribute("#run", "aria-busy")) === "true" && s.posts.length === 1, "the figure interrupted the run");
    await s.expectText("#run .run-label", /^Mapping…$/, "the Run label while the figure plays");
    gate.release();
    await s.expectResults();
    await s.expectHidden("#progress", "the progress bar after the run");
    await s.page.waitForTimeout(1500); // a play left running would move on in this time
    g = await s.guide();
    s.check(g.view === "expanded" && g.play === "idle" && g.active.length === 0 && !g.token, `after the run the figure is ${g.view}, ${g.play}, active ${g.active.join(",")}`);
    s.check(g.chips.length === 10 && g.chips.filter((c) => c.current).map((c) => c.text).join() === "Metformin" && g.playLabel === "Play", `the figure does not follow the new result: ${JSON.stringify(g.chips.map((c) => c.text))}`);
    s.check(s.posts.length === 1, `${s.posts.length} runs were sent`);
  },

  // A new result resets the figure to that result, whatever it was doing; and a pick
  // that failed ends at the Selector with no ID.
  async "guide-new-result"(s) {
    s.health = productionHealth();
    const gate = s.gate();
    const sonnet = shaped({ model: "claude-sonnet-5-5" });
    for (const e of sonnet.entities) if (e.pick) e.pick.selector = "claude-sonnet-5-5";
    s.ner = { gate: gate.promise, body: sonnet };
    await s.open();
    await s.expectExample("before the run");
    await s.page.click("#guide-toggle");
    await s.page.fill("#text", SHAPED_TEXT);
    await s.run();
    await s.follow("green tea");
    await s.page.waitForSelector(".guide-node.is-active[data-k='retriever']", { timeout: 10000 }).catch(() => s.failures.push("the play never reached the Retriever"));
    gate.release();
    await s.expectResults();
    let g = await s.guide();
    s.check(g.play === "idle" && g.active.length === 0 && !g.token, `the play of the old result went on: ${g.play}, ${g.active.join(",")}`);
    s.check(g.chips.map((c) => c.text).join() === "Aspirin,metformin,oily fish,BRCA1,warfarin,garlic,cumin", `the chips are ${g.chips.map((c) => c.text).join()}`);
    s.check(g.chips.filter((c) => c.current).map((c) => c.text).join() === "metformin" && g.nodes.result.label === "CHEBI:201" && /metformin → CHEBI:201/.test(g.panel), `the figure rests on ${g.nodes.result.label}: ${JSON.stringify(g.panel.slice(0, 80))}`);
    s.check(g.live === "off", "the page announced the figure of a new result by itself");
    s.check(!/green tea|FOODON:00005515/.test(g.panel + g.hint), "the figure still shows the old result");
    await s.page.waitForTimeout(2500); // timers of the old play would fire in this time
    const later = await s.guide();
    s.check(later.panel === g.panel && later.active.length === 0 && later.play === "idle", "a timer of the old play changed the figure of the new result");
    const focus = await s.shown();
    s.check(focus.focus === "results-heading", `after the run the focus is on ${focus.focus}`);
    // Who read and who picked are the ones this response names.
    await s.page.click('.guide-node[data-k="reader"]');
    g = await s.guide();
    s.check(/^Reader · Claude Sonnet 5\.5 reads/.test(g.panel) && g.live === "polite", `the Reader panel says ${JSON.stringify(g.panel.slice(0, 60))}`);
    await s.page.click('.guide-node[data-k="selector"]');
    g = await s.guide();
    s.check(/^Selector · Claude Sonnet 5\.5 reads the shortlist/.test(g.panel) && !/Opus/.test(g.panel), `the Selector panel says ${JSON.stringify(g.panel.slice(0, 80))}`);

    // A pick that failed: the selector could not answer; no ID, and no raw reason.
    await s.guideWatch();
    await s.follow("warfarin");
    await s.guidePlayed("warfarin");
    const log = await s.guideLog();
    g = await s.guide();
    s.check(log.active.join() === "reader,retriever,selector", `a failed pick visited ${log.active.join(",")}`);
    s.check(g.at === "selector" && /Claude Sonnet 5\.5 could not answer for this mention, so no term was chosen/.test(g.panel), `the journey of a failed pick ended at ${g.at}: ${JSON.stringify(g.panel)}`);
    s.check(g.ids.length === 0 && g.bubbles === 0 && !/deadline|\(chosen\)/.test(g.panel) && g.nodes.result.label === "could not be matched" && g.nodes.result.skipped, `a failed pick shows an ID, a reason or a choice: ${JSON.stringify(g.nodes.result.label)}`);
    // "Needs a check" is said where the selector marked it.
    await s.follow("garlic");
    await s.page.click('.guide-node[data-k="result"]');
    g = await s.guide();
    s.check(/garlic → FOODON:600/.test(g.panel) && /Claude Sonnet 5\.5 marked it as needing a check\./.test(g.panel), `the result panel of a pick to check says ${JSON.stringify(g.panel)}`);
  },

  // Reduced motion: no travel, no bobbing; Play shows the finished state at once.
  async "guide-reduced-motion"(s) {
    await s.page.emulateMedia({ reducedMotion: "reduce" });
    s.health = productionHealth();
    const gate = s.gate();
    s.ner = { gate: gate.promise, body: GUIDE_RUN.response };
    await s.open();
    await s.expectExample("with reduced motion");
    await s.page.click("#guide-toggle");
    await s.guideWatch();
    await s.page.click("#guide-play");
    await s.page.waitForTimeout(250);
    let g = await s.guide();
    let log = await s.guideLog();
    s.check(g.play === "done" && g.playLabel === "Play again" && g.at === "result" && g.nodes.result.label === "CHEBI:6801", `with reduced motion Play shows ${g.play} at ${g.at}: ${g.nodes.result.label}`);
    s.check(log.active.length === 0 && log.tokens.length === 0 && !g.token && log.writes === 1, `with reduced motion something travelled: ${log.active.join(",")} ${log.tokens.join(",")}, ${log.writes} writes`);
    s.check(/metformin → CHEBI:6801 metformin/.test(g.panel) && g.hiddenParts === 0, "the finished state is not all on show");
    await s.guideWatch();
    await s.follow("green tea");
    await s.page.waitForTimeout(250);
    g = await s.guide();
    log = await s.guideLog();
    s.check(g.play === "done" && g.nodes.result.label === "FOODON:00005515" && log.active.length === 0, `a chip with reduced motion shows ${g.play}: ${g.nodes.result.label}`);
    await s.page.click('.guide-node[data-k="selector"]');
    g = await s.guide();
    s.check(g.hiddenParts === 0 && /\(chosen\)/.test(g.panel), "a tapped box does not show its finished state");
    // During a run nothing bobs either.
    await s.page.fill("#text", GUIDE_RUN.text);
    await s.run();
    await s.expectVisible("#progress", "the progress bar");
    const still = await s.page.evaluate(() => ["reader", "retriever", "selector"].map((k) => getComputedStyle(document.querySelector(`.guide-node[data-k="${k}"] .guide-face`)).animationName));
    s.check(still.every((a) => a === "none"), `with reduced motion the faces are ${still.join(",")} during a run`);
    gate.release();
    await s.expectResults();
  },

  // An answer the server already had, and a run that reused part of its work.
  async "cache-hit"(s) {
    const original = { timing_ms: { extract: 3289, map: 2339, select: 3197, total: 8825 }, usage: { llm_calls: 1, prompt_tokens: 1134, completion_tokens: 187, embedding_requests: 1, selector_calls: 7 } };
    const noCalls = { llm_calls: 0, prompt_tokens: null, completion_tokens: null, embedding_requests: 0, selector_calls: 0, selector_prompt_tokens: null, selector_completion_tokens: null, selector_cost_usd: null };
    const hit = (age) => shaped({ usage: noCalls, timing_ms: { total: 1 }, cache: { response: "hit", retrieval_hits: 0, pick_hits: 0, age_s: age, original } });
    s.health = productionHealth();
    s.ner = { body: hit(245) };
    await s.open();
    await s.page.fill("#text", SHAPED_TEXT);
    await s.run();
    await s.expectResults();
    await s.expectText("#status", /^Answered from a recent identical run \(4 minutes ago\)\. No model was called\. That run was read by Claude Opus 5\.5; terms chosen by Claude Opus 5\.5\.$/, "the status of a cached answer");
    let seen = await s.shown();
    s.check(seen.kind === "run" && seen.heading === "Results" && seen.cards === SHAPED_SPECS.length, `a cached answer is shown as ${seen.kind}, ${JSON.stringify(seen.heading)}, ${seen.cards} cards`);
    s.check(!/Done in/.test(await s.text("#results")), "a cached answer still shows a timing sentence");
    await s.page.click("#run-details > summary");
    const rows = () => s.page.evaluate(() => Object.fromEntries([...document.querySelectorAll("#run-details-list dt")].map((dt) => [dt.textContent, dt.nextElementSibling.textContent])));
    let details = await rows();
    s.check(details.Cache === "the whole answer was reused from a recent identical run (4 minutes ago); no model call and no search for this request", `Run details say of the cache: ${JSON.stringify(details.Cache)}`);
    s.check(details["That run"] === "extract 3.29 s, search 2.34 s, pick 3.20 s, total 8.82 s; 1 model call for extraction, 7 for picking" || details["That run"] === "extract 3.29 s, search 2.34 s, pick 3.20 s, total 8.83 s; 1 model call for extraction, 7 for picking", `Run details say of the original run: ${JSON.stringify(details["That run"])}`);
    s.check(details["Model calls"] === "0 for extraction" && details.Timing === "total 1 ms" && details.Tokens === undefined, `a cached answer's own figures are ${JSON.stringify([details["Model calls"], details.Timing, details.Tokens])}`);
    if (s.shot) await s.shot("hit");

    // How old the answer is, in the words a visitor reads.
    for (const [age, words] of [[20, "less than a minute ago"], [60, "1 minute ago"], [3599, "59 minutes ago"], [7300, "2 hours ago"]]) {
      s.ner = { body: hit(age) };
      await s.run();
      await s.settled();
      const status = await s.text("#status");
      s.check(status.startsWith(`Answered from a recent identical run (${words}). No model was called.`), `an answer ${age} s old is said as ${JSON.stringify(status.slice(0, 80))}`);
    }

    // A run that reused some searches and one pick: an ordinary result, and only "Run
    // details" says what was reused.
    s.ner = { body: shaped({ cache: { response: "miss", retrieval_hits: 3, pick_hits: 1 } }) };
    await s.run();
    await s.settled();
    await s.expectText("#status", /^Done in 8\.8 s: /, "the status of a run that reused part of its work");
    await s.page.evaluate(() => (document.getElementById("run-details").open = false));
    s.check(!/reused|recent identical/i.test(await s.text("#results")), "a partial reuse is said outside Run details");
    await s.page.click("#run-details > summary");
    details = await rows();
    s.check(details.Cache === "search results reused for 3 mentions, 1 pick reused (from recent runs, at no model cost)" && details["That run"] === undefined, `Run details say of a partial reuse: ${JSON.stringify(details.Cache)}`);
    s.ner = { body: shaped({ cache: { response: "miss", retrieval_hits: 0, pick_hits: 0 } }) };
    await s.run();
    await s.settled();
    s.check((await rows()).Cache === "nothing reused from recent runs", `Run details say of a run that reused nothing: ${JSON.stringify((await rows()).Cache)}`);
    // A server without caches says nothing about one.
    s.ner = { body: shaped() };
    await s.run();
    await s.settled();
    s.check((await rows()).Cache === undefined, "Run details have a cache row for a response without one");
    seen = await s.shown();
    s.check(/^Done in /.test(seen.status || ""), `after a plain run the status is ${JSON.stringify(seen.status)}`);
  },

  // The same, end to end, against the server's own API: the second identical request is
  // answered from its cache. A server that reports no cache (the canned one, or
  // NER_SITE_CACHE=off) has nothing to show here.
  async "cache-live"(s) {
    await s.open();
    const text = `Note ${Date.now()}-${s.width}: she took haloperidol, later ketamine.`;
    const answers = [];
    s.page.on("response", async (response) => {
      if (new URL(response.url()).pathname !== "/api/ner") return;
      try {
        answers.push(await response.json());
      } catch (_) {
        answers.push(null);
      }
    });
    // The n-th answer, once this listener has read its body (the page may be done first).
    const answer = async (n) => {
      for (let i = 0; i < 100 && answers.length < n; i += 1) await s.page.waitForTimeout(50);
      return answers[n - 1];
    };
    await s.page.fill("#text", text);
    await s.run();
    await s.expectResults();
    const first = await answer(1);
    if (!first || !first.cache) {
      console.log(`        (cache-live @ ${s.width}px: this server reports no cache; nothing checked)`);
      return;
    }
    s.check(first.cache.response === "miss", `the first run of a new text was a ${first.cache.response}`);
    await s.expectText("#status", /^Done in /, "the status of the first run");
    await s.run();
    await s.expectResults();
    const second = await answer(2);
    s.check(second && second.cache && second.cache.response === "hit" && second.usage.llm_calls === 0, `the second identical run answered ${JSON.stringify(second && second.cache)}`);
    await s.expectText("#status", /^Answered from a recent identical run \(less than a minute ago\)\. No model was called\./, "the status of the second identical run");
    s.check(s.posts.length === 2, `${s.posts.length} requests were sent for two presses`);
    // One more character: a run again, which reuses what was looked up for the mentions.
    await s.page.fill("#text", text + " ");
    await s.run();
    await s.expectResults();
    const third = await answer(3);
    s.check(third && third.cache && third.cache.response === "miss" && third.cache.retrieval_hits >= 1, `a text sharing its mentions answered ${JSON.stringify(third && third.cache)}`);
    await s.expectText("#status", /^Done in /, "the status of a run that reused part of its work");
    await s.page.click("#run-details > summary");
    await s.expectText("#run-details", /Cache\s*search results reused for \d+ mentions?/, "the cache row of Run details");
  },

  // "Map this text" on the untouched example: nothing is sent.
  async "example-pressed"(s) {
    s.health = productionHealth();
    s.ner = { body: shaped() };
    await s.open();
    await s.expectExample("on arrival");
    await s.expectHidden("#recorded-note", "the line about the recorded result, before the button is pressed");
    await s.run();
    await s.expectVisible("#recorded-note", "the line about the recorded result");
    await s.page.waitForTimeout(700); // the scroll
    await s.expectText("#recorded-note", /^This is the recorded result for the example, so nothing was sent\.\s*Run it live$/, "the line about the recorded result");
    let seen = await s.shown();
    s.check(s.posts.length === 0, `pressing the button on the untouched example sent ${s.posts.length} request(s)`);
    s.check(seen.kind === "example" && seen.heading === "Example result" && EXAMPLE_NOTE.test(seen.note || ""), `the recorded result is labelled ${JSON.stringify([seen.kind, seen.heading, seen.note])}`);
    s.check(seen.status === null && seen.stale === null && seen.details === "Details of the recorded run", `beside the recorded result: status ${JSON.stringify(seen.status)}, stale ${JSON.stringify(seen.stale)}, details ${JSON.stringify(seen.details)}`);
    s.check(seen.cards === EXAMPLE_FILE.response.entities.length, `${seen.cards} cards after the press`);
    s.check(seen.focus === "results-heading", `after the press the focus is on #${seen.focus}`);
    s.check(seen.announce === "This is the recorded result for the example, so nothing was sent.", `the press was announced as ${JSON.stringify(seen.announce)}`);
    // Brought into view as a finished run is: the result starts at the top of the screen.
    const place = await s.page.evaluate(() => Math.round(document.getElementById("results").getBoundingClientRect().top));
    s.check(place >= -4 && place <= seen.screen / 2, `after the press the result starts ${place}px from the top of a ${seen.screen}px screen`);
    const button = await s.page.evaluate(() => {
      const b = document.getElementById("run-live");
      const r = b.getBoundingClientRect();
      return { tag: b.tagName, type: b.type, height: r.height, seen: r.top >= 0 && r.bottom <= window.innerHeight };
    });
    s.check(button.tag === "BUTTON" && button.type === "button" && button.seen, `"Run it live" is ${JSON.stringify(button)}`);
    s.check(button.height >= (s.phone ? 44 : 32), `"Run it live" is ${button.height}px tall`);
    const contrast = await s.contrast("#recorded-text");
    s.check(contrast >= 4.5, `the line about the recorded result has a contrast of ${contrast}:1`);
    s.check(seen.stored.length === 0, `the press stored something: ${seen.stored.join(", ")}`);
    if (s.shot) await s.shot("pressed");

    // Ctrl+Enter is the same button.
    await s.page.focus("#text");
    await s.page.keyboard.press("Control+Enter");
    await s.page.waitForTimeout(300);
    s.check(s.posts.length === 0, "Ctrl+Enter on the untouched example sent a request");

    // The figure still follows the recorded result.
    await s.page.click("#guide-toggle");
    await s.page.waitForFunction(() => document.getElementById("pipe").getAttribute("data-play") === "done", null, { timeout: 25000 });
    const g = await s.guide();
    s.check(g.view === "expanded" && g.chips.length === EXAMPLE_FILE.response.entities.length && g.nodes.result.label === "CHEBI:6801", `after the press the figure shows ${g.view}, ${g.chips.length} mentions, ${g.nodes.result.label}`);
    await s.page.click("#guide-toggle");

    // Another text: the line goes, and the result is said to be the example's.
    await s.page.fill("#text", EXAMPLE_FILE.text + " And more.");
    await s.expectHidden("#recorded-note", "the line about the recorded result once the text was edited");
    await s.expectText("#stale", EXAMPLE_STALE);
    await s.page.fill("#text", EXAMPLE_FILE.text);
    await s.expectVisible("#recorded-note", "the line about the recorded result once the box holds the example again");

    // A changed option: the button runs for real, with that option.
    await s.page.click("#options > summary");
    await s.page.selectOption("#top-k", "3");
    await s.run();
    await s.expectResults();
    s.check(s.posts.length === 1 && s.posts[0].body.text === EXAMPLE_FILE.text && s.posts[0].body.top_k === 3 && !s.posts[0].authorization, `with another option the page sent ${JSON.stringify(s.posts.map((p) => p.body && p.body.top_k))}`);
    seen = await s.shown();
    s.check(seen.kind === "run" && seen.heading === "Results" && seen.note === null, `after a real run the result is labelled ${JSON.stringify([seen.kind, seen.heading, seen.note])}`);
    await s.expectHidden("#recorded-note", "the line about the recorded result after a real run");
  },

  // "Run it live", and the other ways the example is run for real.
  async "example-live"(s) {
    s.health = productionHealth();
    s.ner = { status: 500, body: apiError("internal", "boom") };
    s.expectedStatuses.add(500);
    await s.open();
    await s.expectExample("on arrival");
    await s.run();
    await s.expectVisible("#recorded-note", "the line about the recorded result");
    s.check(s.posts.length === 0, "the first press sent a request");

    // A live run that fails: the recorded result stays, and "Try again" is a live run
    // again, not the recording.
    await s.page.click("#run-live");
    await s.expectError("The demo hit an error on its side.", { action: "Try again" });
    let seen = await s.shown();
    s.check(s.posts.length === 1 && seen.kind === "example" && seen.heading === "Example result", `after a failed live run: ${s.posts.length} request(s), ${seen.kind}, ${JSON.stringify(seen.heading)}`);
    const gate = s.gate();
    s.ner = { gate: gate.promise, body: shaped() };
    await s.page.click("#error-action");
    await s.expectVisible("#progress", "the progress bar of the live run");
    await s.expectHidden("#recorded-note", "the line about the recorded result while the live run is under way");
    s.check(s.posts.length === 2, `"Try again" after a failed live run sent ${s.posts.length - 1} more request(s)`);
    const sent = s.posts[1];
    s.check(sent.body.text === EXAMPLE_FILE.text && sent.body.prompt === null && sent.body.top_k === 5 && sent.body.model === "claude-opus-5-5" && !("provider" in sent.body) && !sent.authorization, `the live run sent ${JSON.stringify({ ...sent.body, text: sent.body.text.slice(0, 20) })}`);
    gate.release();
    await s.expectResults();
    seen = await s.shown();
    s.check(seen.kind === "run" && seen.heading === "Results" && seen.note === null && /^Done in /.test(seen.status || ""), `the live run is shown as ${JSON.stringify([seen.kind, seen.heading, seen.note, seen.status])}`);
    await s.expectHidden("#recorded-note", "the line about the recorded result after the live run");

    // Pressed again with the live result of this text on display: it runs again.
    s.ner = { body: shaped() };
    await s.run();
    await s.expectResults();
    s.check(s.posts.length === 3, `a second press after a live run sent ${s.posts.length - 2} more request(s)`);

    // Another text, run; then the example again from the Example button: recorded again.
    await s.page.fill("#text", SHAPED_TEXT);
    await s.run();
    await s.expectResults();
    s.check(s.posts.length === 4, "the visitor's own text was not run");
    await s.loadExample();
    await s.run();
    await s.expectVisible("#recorded-note", "the line about the recorded result, the example loaded again");
    seen = await s.shown();
    s.check(s.posts.length === 4 && seen.kind === "example" && seen.heading === "Example result" && EXAMPLE_NOTE.test(seen.note || ""), `the example loaded again was answered with ${JSON.stringify([s.posts.length, seen.kind, seen.heading])}`);
    s.check(seen.cards === EXAMPLE_FILE.response.entities.length && seen.stale === null, `the recorded result came back with ${seen.cards} cards, stale ${JSON.stringify(seen.stale)}`);

    // Another model that reads the text: a real run.
    if (!(await s.optionsOpen())) await s.page.click("#options > summary");
    await s.page.selectOption("#model", "claude-sonnet-5-5");
    await s.run();
    await s.expectResults();
    s.check(s.posts.length === 5 && s.posts[4].body.model === "claude-sonnet-5-5", "another model did not run the example for real");
    await s.page.selectOption("#model", "claude-opus-5-5");

    // An own key: a real run, on that key.
    await s.chooseOwnKey("anthropic");
    await s.run();
    await s.expectResults();
    s.check(s.posts.length === 6 && s.posts[5].body.provider === "anthropic" && s.posts[5].authorization === `Bearer ${FAKE_KEY}`, "an own key did not run the example for real");
  },

  // Before health has answered the page does not know that the defaults are the ones
  // the example was recorded with: the press is a run, as it was.
  async "example-early"(s) {
    const health = s.gate();
    s.health = productionHealth();
    s.healthGate = health.promise;
    s.ner = { body: shaped() };
    await s.openHeld();
    await s.expectExample("before health answers", { counts: false });
    await s.run();
    await s.expectResults();
    s.check(s.posts.length === 1 && s.posts[0].body.text === EXAMPLE_FILE.text, `a press before health answered sent ${s.posts.length} request(s)`);
    await s.expectHidden("#recorded-note", "the line about the recorded result");
    health.release();
    await s.page.waitForLoadState("networkidle").catch(() => {});
    // A prompt of the visitor's own: a real run too.
    await s.page.click("#options > summary");
    await s.page.click("#prompt-editor > summary");
    await s.page.fill("#prompt", "Find only the foods.");
    await s.run();
    await s.expectResults();
    s.check(s.posts.length === 2 && s.posts[1].body.prompt === "Find only the foods.", "an edited prompt did not run the example for real");
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
