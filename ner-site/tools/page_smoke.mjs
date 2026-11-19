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
    this.ner = "pass";
  }

  check(ok, message) {
    if (!ok) this.failures.push(message);
    return ok;
  }

  // health: "pass", "fail", or a JSON body. ner: "pass" or { status, body }.
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
    await page.route("**/api/health", async (route) => {
      if (this.health === "pass") return route.continue();
      if (this.health === "fail") return route.fulfill({ status: 500, contentType: "text/plain", body: "down" });
      return route.fulfill({ status: 200, contentType: "application/json", body: JSON.stringify(this.health) });
    });
    await page.route("**/api/ner", async (route) => {
      if (this.ner === "pass") return route.continue();
      let body = this.ner.body;
      if (typeof body === "function") body = body(JSON.parse(route.request().postData() || "{}"));
      return route.fulfill({ status: this.ner.status || 200, contentType: "application/json", body: JSON.stringify(body) });
    });
  }

  async open() {
    await this.page.goto(this.base, { waitUntil: "networkidle" });
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
      this.page.waitForSelector("#results:not([hidden]) .card", { state: "visible", timeout: 130000 }).then(() => "results"),
      this.page.waitForSelector("#error:not([hidden])", { state: "visible", timeout: 130000 }).then(() => "error"),
    ]).catch(() => "nothing");
    if (outcome === "error") this.failures.push(`run failed: ${await this.text("#error-message")}`);
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
    s.ner = { status: 429, body: fixture("error-blocked") };
    s.expectedStatuses.add(429);
    await s.open();
    await s.expectVisible("#allowance", "the free-runs line");
    await s.expectText("#allowance", /^The free runs for your network this hour are used\. Next one in \d+ min\.$/);
    await s.expectHidden("#run-hint", "the 'Free, no sign-up' hint");
    await s.expectVisible("#use-own-key", "'Use my own key'");
    await s.runExample();
    await s.expectVisible("#error", "the error box");
    await s.settled();
    await s.expectHidden("#results", "results");
    s.check((await s.page.inputValue("#text")).length > 0, "the text was cleared by the error");
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
    await s.expectVisible("#error", "the error box");
    await s.settled();
  },

  async "limits-unavailable"(s) {
    s.health = fixture("health-unavailable");
    s.ner = { status: 503, body: fixture("error-unavailable") };
    s.expectedStatuses.add(503);
    await s.open();
    await s.expectText("#allowance", /^The demo is briefly unavailable\. Try again in a minute\.$/);
    await s.runExample();
    await s.expectVisible("#error", "the error box");
    await s.settled();
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
  },

  async fallback(s) {
    s.health = fixture("health");
    s.ner = { body: fixture("ner-fallback") };
    await s.open();
    await s.runExample();
    await s.expectResults();
    await s.expectVisible("#notices", "the notice about the model that answered");
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
    // Closed again by the visitor: the error must bring the key field back.
    await s.page.click("#options > summary");
    await s.runExample();
    await s.expectVisible("#error", "the error box");
    await s.expectText("#error-message", /refused this key/);
    await s.settled();
    s.check(await s.optionsOpen(), "a refused key did not open Options");
    await s.expectVisible("#own-key", "the key field");
    s.check((await s.page.inputValue("#own-key")) === FAKE_KEY, "the refused key was removed from the field");
    s.check((s.posts[0] || {}).authorization === `Bearer ${FAKE_KEY}`, "the own-key run did not carry the key");
  },

  async "own-key-missing"(s) {
    s.health = fixture("health");
    await s.open();
    await s.page.click("#options > summary");
    await s.page.selectOption("#provider", "openrouter");
    await s.page.click("#options > summary");
    await s.runExample();
    await s.expectVisible("#error", "the error box");
    s.check(await s.optionsOpen(), "a missing key did not open Options");
    s.check(s.posts.length === 0, "a run without a key was sent");
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
    await s.page.click("#prompt-editor > summary");
    await s.expectVisible("#prompt", "the prompt editor");
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
