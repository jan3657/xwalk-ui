// xwalk NER demo page. ES2020, no dependencies, no build step.
// Every piece of server or model data is inserted with textContent, never innerHTML.
"use strict";

(() => {
  // Mentions picked from the bundled 200-term samples (ChEBI, CTD diseases, NCBI Gene,
  // FoodOn), plus "aspirin" (not in the ChEBI sample) and "fever" (type other). The
  // emoji is deliberate: it is one code point but two UTF-16 units, so it checks that
  // the highlighter converts the server's code-point offsets correctly.
  // tools/fixtures/ner.json is built for exactly this text.
  const EXAMPLE_TEXT =
    "Clinic note 🩺: the patient has Alkaptonuria and was screened for " +
    "Hydroxykynureninuria. She took haloperidol, later ketamine, and aspirin for a fever. " +
    "Genotyping covered NAT2 and ADA. Her diet: hummus, garlic and cumin.";
  // The example on a full-catalog deployment: every mention maps (checked on production,
  // 19 Nov 2026). The text above stays the example for offline and sample modes.
  const EXAMPLE_FULL =
    "Aspirin relieved her fever after a meal of hummus. She takes metformin for type 2 diabetes, " +
    "has high blood pressure, and drank green tea every morning.";

  const TYPES = ["chemical", "disease", "gene", "food", "other"];
  const DEFAULT_LIMITS = { max_chars: 5000, max_prompt_chars: 4000, max_entities: 25, top_k_max: 10 };
  // Names for the model ids the site knows; any other id is shown as it is.
  const MODEL_NAMES = {
    "claude-opus-5-5": "Claude Opus 5.5",
    "claude-sonnet-5-5": "Claude Sonnet 5.5",
    "claude-haiku-5-5": "Claude Haiku 5.5",
    "google/gemini-3.1-flash-lite": "Gemini 3.1 Flash Lite",
    "openai/gpt-6-luna": "GPT-6 Luna",
  };
  const PASSCODE_KEY = "xwalk-ner-passcode";
  // A visitor's own provider key: one session storage entry, {provider, key}. It is
  // removed on "Forget key" and whenever the provider changes, and it leaves the page
  // only as the Authorization header of POST /api/ner.
  const OWN_KEY_ENTRY = "xwalk-ner-own-key";
  // That the visitor chose to pay with an own key, and for which provider: the provider
  // id alone, never the key. It outlives the key (which is forgotten when its provider
  // is no longer offered), so that no number of reloads turns that choice into a run on
  // the site's keys. Only the visitor choosing the site's models removes it.
  const PAYER_ENTRY = "xwalk-ner-payer";
  const PROVIDER_NAMES = { anthropic: "Anthropic", openai: "OpenAI", openrouter: "OpenRouter" };
  // What the server accepts as a key: 20 to 512 printable ASCII characters, no spaces.
  const KEY_SHAPE = /^[\x21-\x7e]{20,512}$/;
  const REQUEST_TIMEOUT_MS = 130000;
  // Errors whose cause sits inside the closed "Options" section.
  const OPTIONS_CODES = ["bad_credential", "provider_key_refused", "provider_rate_limited", "prompt_too_long"];
  // On a phone, focusing the text box from a tap raises the keyboard over the page.
  const touch = typeof matchMedia === "function" && matchMedia("(hover: none) and (pointer: coarse)").matches;

  const $ = (id) => document.getElementById(id);
  const el = {
    form: $("ner-form"),
    text: $("text"),
    counter: $("text-counter"),
    textError: $("text-error"),
    textHelp: $("text-help"),
    example: $("load-example"),
    clear: $("clear-text"),
    prompt: $("prompt"),
    promptHint: $("prompt-hint"),
    promptReset: $("prompt-reset"),
    promptError: $("prompt-error"),
    promptEditor: $("prompt-editor"),
    options: $("options"),
    optionsSummary: $("options-summary"),
    runWith: $("run-with"),
    provider: $("provider"),
    ownKeyField: $("own-key-field"),
    ownKeyLabel: $("own-key-label"),
    ownKey: $("own-key"),
    forgetKey: $("forget-key"),
    ownKeyNote: $("own-key-note"),
    modelField: $("model-field"),
    model: $("model"),
    selectorField: $("selector-field"),
    selector: $("selector"),
    allowance: $("allowance"),
    allowanceRow: $("allowance-row"),
    useOwnKey: $("use-own-key"),
    runHint: $("run-hint"),
    topK: $("top-k"),
    passField: $("passcode-field"),
    passcode: $("passcode"),
    payer: $("payer"),
    payerMsg: $("payer-message"),
    payerRetry: $("payer-retry"),
    payerSite: $("payer-site"),
    run: $("run"),
    banner: $("banner"),
    error: $("error"),
    errorMsg: $("error-message"),
    status: $("status"),
    pipe: $("pipe"),
    pipeFind: $("pipe-find"),
    pipeSearch: $("pipe-search"),
    modelHint: $("model-hint"),
    progress: $("progress"),
    progressTime: $("progress-time"),
    progressHint: $("progress-hint"),
    results: $("results"),
    summary: $("results-summary"),
    ranOn: $("results-models"),
    notices: $("notices"),
    noticesList: $("notices-list"),
    highlighted: $("highlighted"),
    legend: $("legend"),
    lede: $("lede"),
    caveat: $("caveat"),
    warnings: $("warnings"),
    cards: $("cards"),
  };

  const state = {
    health: null,
    limits: { ...DEFAULT_LIMITS },
    defaultPrompt: "",
    ontologyNames: {},
    providers: [], // own-key providers the API offers; empty when it offers none
    running: false,
    healthFailed: false, // the status read on arrival failed; said in the top banner
    visitorAt: 0, // when health.visitor was read, for "next one in N min"
    textError: null, // "empty" or "long" while the line under the text box is shown
    // Set while it is not settled who pays for the next run: a key kept from earlier
    // in this tab has not been restored yet ("pending"), could not be because health
    // did not load ("unknown"), or is for a provider the site no longer offers
    // ("lost"). Nothing is submitted while it is set; see holdRuns().
    payer: null,
  };

  // ---------- small helpers ----------

  function h(tag, attrs, ...children) {
    const node = document.createElement(tag);
    if (attrs) {
      for (const [k, v] of Object.entries(attrs)) {
        if (v === null || v === undefined || v === false) continue;
        if (k === "class") node.className = v;
        else if (k === "text") node.textContent = v;
        else node.setAttribute(k, v === true ? "" : String(v));
      }
    }
    for (const c of children) {
      if (c === null || c === undefined) continue;
      node.append(typeof c === "string" ? document.createTextNode(c) : c);
    }
    return node;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
  }

  // Python len(): code points, not UTF-16 units.
  function codePointLength(s) {
    let n = 0;
    for (const _ of s) n += 1; // eslint-disable-line no-unused-vars
    return n;
  }

  function storageGet(key) {
    try {
      return window.sessionStorage.getItem(key) || "";
    } catch (_) {
      return "";
    }
  }

  function storageSet(key, value) {
    try {
      if (value) window.sessionStorage.setItem(key, value);
      else window.sessionStorage.removeItem(key);
    } catch (_) {
      /* storage blocked: the value simply is not remembered */
    }
  }

  function safeHref(url) {
    if (typeof url !== "string") return null;
    return /^https?:\/\//i.test(url) ? url : null;
  }

  function fmtSeconds(ms) {
    if (typeof ms !== "number") return "unknown";
    return ms < 1000 ? `${ms} ms` : `${(ms / 1000).toFixed(2)} s`;
  }

  function modelName(id) {
    if (typeof id !== "string" || !id) return "unknown";
    return Object.prototype.hasOwnProperty.call(MODEL_NAMES, id) ? MODEL_NAMES[id] : id;
  }

  function isClaude(id) {
    return typeof id === "string" && id.startsWith("claude-");
  }

  function typeLabel(t) {
    return TYPES.includes(t) ? t : "other";
  }

  // ---------- API ----------

  class ApiError extends Error {
    constructor(message, code, status) {
      super(message);
      this.code = code || null;
      this.status = status || 0;
    }
  }

  // `bearer` is a visitor's own provider key. Only the POST /api/ner call may carry it;
  // any other request that tried to would be a bug, so it is refused here.
  async function api(method, path, payload, bearer) {
    const headers = { "X-Requested-With": "xwalk-ner" };
    if (payload !== undefined) headers["Content-Type"] = "application/json";
    const pass = storageGet(PASSCODE_KEY);
    if (pass) headers["X-Passcode"] = pass;
    if (bearer) {
      if (method !== "POST" || path !== "/api/ner") throw new ApiError("Internal error: a key was attached to the wrong request.", "internal", 0);
      headers.Authorization = `Bearer ${bearer}`;
    }

    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    let response;
    try {
      response = await fetch(path, {
        method,
        headers,
        body: payload === undefined ? undefined : JSON.stringify(payload),
        signal: controller.signal,
        credentials: "same-origin",
        cache: "no-store",
        // A request that carries a key is never followed to another address.
        redirect: bearer ? "error" : "follow",
      });
    } catch (err) {
      const aborted = err && err.name === "AbortError";
      throw new ApiError(
        aborted
          ? "The request took too long and was cancelled. Try a shorter text."
          : "Network error: the server could not be reached.",
        aborted ? "timeout" : "network",
        0,
      );
    } finally {
      clearTimeout(timer);
    }

    let raw = "";
    try {
      raw = await response.text();
    } catch (_) {
      throw new ApiError(`The server answered ${response.status} but the body could not be read.`, "bad_response", response.status);
    }
    let data;
    try {
      data = JSON.parse(raw);
    } catch (_) {
      throw new ApiError(
        `The server answered ${response.status} ${response.statusText || ""}`.trim() +
          " with a response that is not JSON.",
        "bad_response",
        response.status,
      );
    }
    if (!data || typeof data !== "object") {
      throw new ApiError(`The server answered ${response.status} with an unexpected response.`, "bad_response", response.status);
    }
    if (data.ok !== true) {
      const e = data.error && typeof data.error === "object" ? data.error : {};
      const msg = typeof e.message === "string" && e.message ? e.message : `Request failed (HTTP ${response.status}).`;
      throw new ApiError(msg, typeof e.code === "string" ? e.code : null, response.status);
    }
    return data;
  }

  // ---------- messages ----------

  function showError(err) {
    const msg = err instanceof Error ? err.message : String(err);
    el.errorMsg.textContent = msg;
    el.error.hidden = false;
    if (err && OPTIONS_CODES.includes(err.code)) openOptions();
    if (err && (err.code === "passcode_required" || err.code === "passcode_invalid")) {
      el.passField.hidden = false;
      el.passcode.focus();
    }
    // A key the server or the provider refused stays in the field, to be corrected.
    if (err && (err.code === "provider_key_refused" || err.code === "bad_credential") && !el.ownKeyField.hidden && !el.runWith.hidden) {
      el.ownKey.focus();
    }
  }

  function hideError() {
    el.error.hidden = true;
    el.errorMsg.textContent = "";
  }

  // Problems with the text are said at the text box, not in the error box below the form.
  function showTextError(kind, message) {
    state.textError = kind;
    el.textError.textContent = message;
    el.textError.hidden = false;
    el.text.setAttribute("aria-invalid", "true");
  }

  function clearTextError() {
    state.textError = null;
    el.textError.hidden = true;
    el.textError.textContent = "";
    el.text.removeAttribute("aria-invalid");
  }

  function tooLongText(n) {
    const max = state.limits.max_chars;
    return `This text is ${n.toLocaleString("en")} characters; the demo takes up to ${max.toLocaleString("en")}. Shorten it by ${(n - max).toLocaleString("en")}.`;
  }

  function showPromptError(message) {
    el.promptError.textContent = message;
    el.promptError.hidden = !message;
    if (message) el.prompt.setAttribute("aria-invalid", "true");
    else el.prompt.removeAttribute("aria-invalid");
  }

  // Everything a first run does not need sits in one closed section. It opens itself
  // whenever something inside needs the visitor: an own-key choice, a hold, a key error.
  function openOptions() {
    el.options.open = true;
  }

  function setStatus(parts) {
    clear(el.status);
    parts.forEach((p, i) => {
      if (i) el.status.append(h("span", { class: "sep", "aria-hidden": "true", text: "·" }));
      el.status.append(h("span", null, p));
    });
  }

  function renderBanner(mode, retrieval) {
    clear(el.banner);
    const items = [];
    if (state.healthFailed) {
      const retry = h("button", { type: "button", class: "ghost small", id: "health-retry", text: "Try again" });
      retry.addEventListener("click", () => {
        hideError();
        loadHealth();
      });
      el.banner.append(
        h(
          "div",
          { class: "banner-row" },
          h("p", null, h("strong", { text: "Could not reach the service." }), " You can still try a run; some options are missing until the page reloads."),
          retry,
        ),
      );
    }
    if (mode === "offline") {
      items.push([
        "Offline mode.",
        " Entities come from a stand-in extractor, not a language model, so results only illustrate the flow.",
      ]);
    }
    const dense = retrieval && retrieval.dense;
    if (dense && dense.status !== "ok") {
      items.push([
        "Dense retrieval is unavailable.",
        " Matching uses BM25 keyword search only" + (dense.reason ? ` (${dense.reason}).` : "."),
      ]);
    }
    const bm25 = retrieval && retrieval.bm25;
    if (bm25 && bm25.status !== "ok") {
      items.push(["BM25 retrieval is unavailable.", bm25.reason ? ` ${bm25.reason}` : ""]);
    }
    for (const [title, rest] of items) {
      el.banner.append(h("p", null, h("strong", { text: title }), rest));
    }
    el.banner.hidden = !el.banner.firstChild;
  }

  // ---------- form setup ----------

  function updateCounter() {
    const n = codePointLength(el.text.value);
    const max = state.limits.max_chars;
    el.counter.textContent = `${n.toLocaleString()} / ${max.toLocaleString()} characters`;
    el.counter.classList.toggle("over", n > max);
    // Said as soon as the text passes the limit, not only after Run.
    if (n > max) showTextError("long", tooLongText(n));
    else if (state.textError === "long") clearTextError();
    el.textHelp.textContent = `Up to ${max.toLocaleString("en")} characters and ${state.limits.max_entities} mentions per run.`;
  }

  function fillTopK(max) {
    const current = Number(el.topK.value) || 5;
    clear(el.topK);
    for (let k = 1; k <= max; k += 1) el.topK.append(h("option", { value: k, text: String(k) }));
    el.topK.value = String(Math.min(current, max));
  }

  function applyHealth(health) {
    state.health = health;
    state.visitorAt = Date.now();
    state.healthFailed = false;
    if (health.limits && typeof health.limits === "object") {
      state.limits = { ...DEFAULT_LIMITS, ...health.limits };
    }
    state.defaultPrompt = typeof health.default_prompt === "string" ? health.default_prompt : "";
    if (!el.prompt.value) el.prompt.value = state.defaultPrompt;
    el.promptReset.disabled = !state.defaultPrompt;
    el.promptHint.textContent =
      "This is the text the model is given before yours. Change it to try other entity types or stricter rules. " +
      `Your edit is used for your runs in this tab only. Up to ${state.limits.max_prompt_chars.toLocaleString("en")} characters.`;

    fillProviders(health);

    fillTopK(state.limits.top_k_max || 10);

    el.passField.hidden = !health.passcode_required;
    if (health.passcode_required) el.passcode.value = storageGet(PASSCODE_KEY);

    state.ontologyNames = {};
    for (const o of health.ontologies || []) {
      state.ontologyNames[o.slug] = o.name;
    }
    renderIntro(health);
    renderLegend();
    renderBanner(health.mode, health.retrieval);
    applyProvider();
    updateCounter();
  }

  // ---------- own keys ----------

  // The own-key providers health offers, cleaned. An API from before own keys sends no
  // `own_keys`, and one with them turned off sends none: both give an empty list.
  function ownProviders(health) {
    const own = health && health.own_keys;
    if (!own || typeof own !== "object" || own.enabled !== true || !Array.isArray(own.providers)) return [];
    const found = [];
    for (const p of own.providers) {
      if (!p || typeof p !== "object" || typeof p.id !== "string" || !p.id || p.id === "site") continue;
      const models = (Array.isArray(p.models) ? p.models : []).filter((m) => typeof m === "string" && m);
      if (!models.length || found.some((f) => f.id === p.id)) continue;
      found.push({
        id: p.id,
        name: typeof p.name === "string" && p.name ? p.name : providerName(p.id),
        models,
        defaultModel: models.includes(p.default_model) ? p.default_model : models[0],
      });
    }
    return found;
  }

  function providerName(id) {
    const known = state.providers.find((p) => p.id === id);
    if (known) return known.name;
    return Object.prototype.hasOwnProperty.call(PROVIDER_NAMES, id) ? PROVIDER_NAMES[id] : id;
  }

  // The own-key provider chosen under "Who pays for the model" (the #run-with block,
  // inside Options), or null for the site's models.
  function currentProvider() {
    if (el.runWith.hidden) return null;
    return state.providers.find((p) => p.id === el.provider.value) || null;
  }

  function storedKey() {
    const raw = storageGet(OWN_KEY_ENTRY);
    if (!raw) return null;
    try {
      const entry = JSON.parse(raw);
      if (entry && typeof entry.provider === "string" && typeof entry.key === "string" && entry.key) return entry;
    } catch (_) {
      /* not ours: treated as no key */
    }
    return null;
  }

  function rememberKey() {
    const provider = currentProvider();
    const key = provider ? el.ownKey.value.trim() : "";
    storageSet(OWN_KEY_ENTRY, key ? JSON.stringify({ provider: provider.id, key }) : "");
  }

  function forgetKey() {
    el.ownKey.value = "";
    storageSet(OWN_KEY_ENTRY, "");
  }

  // The own-key provider the visitor last chose in this tab ("" for the site's models).
  // A key kept from before this entry existed counts as that choice.
  function savedPayer() {
    const chosen = storageGet(PAYER_ENTRY);
    if (chosen && chosen !== "site") return chosen;
    const kept = storedKey();
    return kept ? kept.provider : "";
  }

  function rememberPayer(providerId) {
    storageSet(PAYER_ENTRY, providerId && providerId !== "site" ? providerId : "");
  }

  // ---------- who pays ----------

  // A run goes to the site's models unless an own-key provider is chosen, so a choice
  // that silently falls back to "site" moves the bill from the visitor to the site
  // (and the visitor's text to a model they did not pick). While a saved own-key
  // choice is unsettled nothing is submitted, by the button or by the keyboard, and
  // the visitor is told why. Only an explicit choice lifts it: the saved choice coming
  // back, "Use the site's models", or another pick under "Who pays for the model". The choice is
  // kept in PAYER_ENTRY, so the hold is back after every reload until then.
  function holdRuns(kind, providerId) {
    // Only a name this page knows is shown; session storage is not trusted further.
    const known = Object.prototype.hasOwnProperty.call(PROVIDER_NAMES, providerId);
    const key = known ? `${PROVIDER_NAMES[providerId]} key` : "API key";
    state.payer = { kind, provider: providerId };
    if (kind === "pending") {
      el.payerMsg.textContent = `You chose to run on your own ${key}. Checking whether it can still be used on this site; runs wait for that.`;
    } else if (kind === "unknown") {
      el.payerMsg.textContent =
        `You chose to run on your own ${key}, but the service status could not be loaded, so this page cannot tell whether it can still be used. ` +
        'Nothing has run. Try again, or choose "Use the site\'s models" to forget the key and run on the site\'s models (free, limited).';
    } else {
      el.payerMsg.textContent =
        `This site no longer accepts your own ${key}, so the key has been forgotten. Nothing has run on the site's models. ` +
        'Choose "Use the site\'s models" to run on them (free, limited)' +
        (state.providers.length ? ', or pick another key under "Who pays for the model" in Options.' : ".");
    }
    el.payerRetry.hidden = kind !== "unknown";
    el.payer.hidden = false;
    openOptions();
    syncRun();
    renderPayerCopy();
  }

  function releaseRuns() {
    state.payer = null;
    el.payer.hidden = true;
    el.payerMsg.textContent = "";
    syncRun();
    renderPayerCopy();
  }

  function syncRun() {
    el.run.disabled = state.running || Boolean(state.payer);
  }

  function fillProviderOptions() {
    clear(el.provider);
    el.provider.append(h("option", { value: "site", text: "This site (free, limited per hour)" }));
    for (const p of state.providers) el.provider.append(h("option", { value: p.id, text: `My ${p.name} key` }));
    el.runWith.hidden = state.providers.length === 0;
  }

  // The "Who pays for the model" choice. An own-key choice made earlier in this tab comes back after
  // a reload, with its key when one is kept. When its provider is no longer offered the
  // key is dropped, the choice stays on record, and the visitor is asked (again after
  // every reload) before anything runs on the site's models in its place.
  function fillProviders(health) {
    state.providers = ownProviders(health);
    fillProviderOptions();
    const chosen = savedPayer();
    const kept = storedKey();
    if (chosen && state.providers.some((p) => p.id === chosen)) {
      el.provider.value = chosen;
      el.ownKey.value = kept && kept.provider === chosen ? kept.key : "";
      if (!el.ownKey.value) forgetKey();
      rememberPayer(chosen);
      releaseRuns();
      // A restored own-key choice is never left inside a closed section.
      openOptions();
    } else {
      el.provider.value = "site";
      forgetKey();
      if (chosen) {
        rememberPayer(chosen);
        holdRuns("lost", chosen);
      } else releaseRuns();
    }
  }

  function sameProviders(a, b) {
    return JSON.stringify(a) === JSON.stringify(b);
  }

  // After a later health read: the own-key providers may have changed while the tab
  // was open. The visitor's choices are kept where they still exist. A provider that
  // was turned off stops being offered, its key is forgotten (field and session
  // storage), and nothing runs on the site's models until the visitor chooses that.
  function reconcileProviders(health) {
    const next = ownProviders(health);
    if (sameProviders(next, state.providers)) return;
    const chosen = currentProvider();
    const model = el.model.value;
    const selector = el.selector.value;
    state.providers = next;
    fillProviderOptions();
    if (!chosen) {
      el.provider.value = "site";
      return;
    }
    if (next.some((p) => p.id === chosen.id)) {
      el.provider.value = chosen.id;
      applyProvider();
      if ([...el.model.options].some((o) => o.value === model)) el.model.value = model;
      if ([...el.selector.options].some((o) => o.value === selector)) el.selector.value = selector;
      renderAllowance();
      return;
    }
    forgetKey();
    el.provider.value = "site";
    applyProvider();
    rememberPayer(chosen.id); // still the visitor's choice, until they make another
    holdRuns("lost", chosen.id);
  }

  // Models, selectors and the key field for the current "Who pays" choice.
  function applyProvider() {
    const health = state.health || {};
    const provider = currentProvider();
    clear(el.model);
    clear(el.selector);
    if (provider) {
      for (const m of provider.models) el.model.append(h("option", { value: m, text: modelName(m) }));
      el.model.value = provider.defaultModel;
      // Picks run on the same key or not at all: Jev and the site's models bill the site.
      el.selector.append(h("option", { value: "retrieval", text: selectorOption("retrieval") }));
      for (const m of provider.models) el.selector.append(h("option", { value: m, text: selectorOption(m) }));
      el.selector.value = "retrieval";
      el.selectorField.hidden = false;
      el.ownKeyLabel.textContent = `Your ${provider.name} API key`;
      el.ownKeyNote.textContent =
        `Your key stays in this browser tab and is sent with each run to this site, which passes it to ${provider.name} ` +
        `for that run and does not store or log it. ${provider.name} bills you for the model calls. ` +
        "The site still pays for term lookup. Use a restricted key with a spending limit if you can.";
    } else {
      const selectors = Array.isArray(health.selectors) ? health.selectors : [];
      for (const s of selectors) el.selector.append(h("option", { value: s, text: selectorOption(s) }));
      if (selectors.includes(health.default_selector)) el.selector.value = health.default_selector;
      el.selectorField.hidden = selectors.length <= 1;

      const models = Array.isArray(health.models) ? health.models : [];
      for (const m of models) el.model.append(h("option", { value: m, text: modelName(m) }));
      if (health.default_model) el.model.value = health.default_model;
      el.ownKeyNote.textContent = "";
    }
    el.ownKeyField.hidden = !provider;
    el.ownKeyNote.hidden = !provider;
    renderAllowance();
  }

  // The selector a request would use: the chosen one, or the server's default when the
  // field is not offered.
  function effectiveSelector() {
    if (!el.selectorField.hidden) return el.selector.value;
    return state.health && typeof state.health.default_selector === "string" ? state.health.default_selector : "";
  }

  // "in N minutes", rounded up, or "later" when the server gave no usable time.
  function tryAgain(s) {
    if (typeof s !== "number" || !Number.isFinite(s)) return "try again later";
    const minutes = Math.max(1, Math.ceil(s / 60));
    return `try again in ${minutes} minute${minutes === 1 ? "" : "s"}`;
  }

  // Seconds still to go of a wait the server reported when health was read.
  function secondsLeft(total) {
    if (typeof total !== "number" || !Number.isFinite(total)) return null;
    return Math.max(0, total - (Date.now() - state.visitorAt) / 1000);
  }

  // What the next request will meet, from health.visitor, for the chosen payer. An
  // older API sends no `visitor`, and then nothing is shown. The limits are counted per
  // IP address, which phones share, so the wording says "your network", never "you".
  function allowanceMessage() {
    const health = state.health;
    const visitor = health && health.visitor && typeof health.visitor === "object" ? health.visitor : null;
    if (!visitor) return null;
    if (currentProvider()) {
      // An own key is outside the site's free runs; only its own flood limit can stop it.
      if (visitor.own_left_today === 0) {
        return { blocked: true, text: "Too many runs with your own key today. They come back at midnight UTC." };
      }
      if (visitor.own_left === 0) {
        return { blocked: true, wait: secondsLeft(visitor.resets_in_s), text: `Too many runs with your own key; ${tryAgain(secondsLeft(visitor.resets_in_s))}.` };
      }
      return null;
    }
    if (visitor.tier === "blocked") {
      if (visitor.limits === "unavailable") {
        return { blocked: true, text: "The demo is briefly unavailable. Try again in a minute." };
      }
      if (visitor.window === "day") {
        const left = secondsLeft(visitor.day_resets_in_s);
        const at = left === null ? "midnight UTC" : new Date(Date.now() + left * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
        return { blocked: true, text: `The free runs for your network today are used. They come back at ${at}.` };
      }
      const left = secondsLeft(visitor.resets_in_s);
      return {
        blocked: true,
        wait: left,
        text: "The free runs for your network this hour are used." + (left === null ? " Try again later." : ` Next one in ${Math.max(1, Math.ceil(left / 60))} min.`),
      };
    }
    // The Sonnet sentences are only true when the free runs are counted, Claude can be
    // reached and the request would use Claude at all.
    const counted = visitor.limits === undefined || visitor.limits === "on";
    const claudeDown = health.claude && health.claude.status === "unavailable";
    const wantsClaude = isClaude(el.model.value) || isClaude(effectiveSelector());
    if (!counted || claudeDown || !wantsClaude) return null;
    if (visitor.tier === "sonnet") {
      const span = visitor.opus_left_today === 0 ? "today" : "this hour";
      return { blocked: false, sonnet: true, text: `The free runs on Claude Opus for your network ${span} are used. Runs now start on Claude Sonnet, a faster model.` };
    }
    if (visitor.tier === "opus" && visitor.opus_left === 1) {
      return { blocked: false, text: "1 free run on Claude Opus is left for your network this hour; after that, runs start on Claude Sonnet." };
    }
    return null;
  }

  // The line beside Run. Run itself stays enabled: a tap re-checks with the server,
  // and a stale count must never lock a visitor out.
  let allowanceTimer = null;

  function renderAllowance() {
    const message = allowanceMessage();
    const blocked = Boolean(message && message.blocked);
    el.allowance.textContent = message ? message.text : "";
    el.allowance.classList.toggle("blocked", blocked);
    el.allowance.hidden = !message;
    el.allowanceRow.classList.toggle("blocked", blocked);
    el.allowanceRow.hidden = !message;
    // The own-key way out is only offered where it exists and is not already taken.
    el.useOwnKey.hidden = !(blocked && !currentProvider() && state.providers.length > 0);
    // A single model needs no dropdown, unless the visitor is paying for it and should
    // see which model that is.
    const count = el.model.options.length;
    el.modelField.hidden = count === 0 || (count === 1 && !currentProvider());

    // While a timed wait is shown, the minutes are recounted, and health is read again
    // when the wait is over.
    const timed = Boolean(message && typeof message.wait === "number");
    if (timed && allowanceTimer === null) {
      allowanceTimer = setInterval(() => {
        const now = allowanceMessage();
        if (now && typeof now.wait === "number" && now.wait <= 0) refreshHealth();
        else renderAllowance();
      }, 30000);
    } else if (!timed && allowanceTimer !== null) {
      clearInterval(allowanceTimer);
      allowanceTimer = null;
    }
    renderPayerCopy();
  }

  // The two places that say who pays for the next run: the summary of the closed
  // "Options" section, and the line under Run.
  function renderPayerCopy() {
    const provider = currentProvider();
    const message = allowanceMessage();
    let summary = "Options";
    if (state.payer) {
      const known = Object.prototype.hasOwnProperty.call(PROVIDER_NAMES, state.payer.provider);
      const key = known ? `${PROVIDER_NAMES[state.payer.provider]} key` : "API key";
      summary = state.payer.kind === "lost" ? "Options: choose who pays" : `Options: your ${key}, waiting`;
    } else if (provider) {
      summary = `Options: your ${provider.name} key`;
    } else if (state.health && state.health.mode === "offline") {
      summary = "Options: offline stand-in, no model";
    } else if (state.health && el.model.value) {
      const stepped = message && message.sonnet && isClaude(el.model.value);
      // "free runs" is not said while there are none left.
      summary = `Options: ${stepped ? MODEL_NAMES["claude-sonnet-5-5"] : modelName(el.model.value)}` + (message && message.blocked ? "" : ", free runs");
    }
    el.optionsSummary.textContent = summary;
    el.runHint.textContent = provider
      ? `Runs on your ${provider.name} key. Use public or de-identified text: it is sent to ${provider.name} and not saved by this site.`
      : "Free, no sign-up, usually 10 to 30 seconds. Use public or de-identified text: it is sent to the model provider and not saved by this site.";
    // Not while it would be untrue: a hold, or no free run left.
    el.runHint.hidden = Boolean(state.payer) || Boolean(message && message.blocked);
  }

  // "Use my own key": show where that is chosen. Nothing changes who pays until the
  // visitor picks a provider there.
  function useOwnKey() {
    openOptions();
    reveal(el.runWith, "center");
    el.provider.focus({ preventScroll: true });
  }

  // Health is read again after each request so the allowance line stays current, and
  // so that an open tab learns when a provider was turned off. Only the parts that
  // change between requests are taken: the form keeps the visitor's choices where they
  // still exist (reconcileProviders). A failed read keeps what was shown.
  async function refreshHealth() {
    let health;
    try {
      health = await api("GET", "/api/health");
    } catch (_) {
      return;
    }
    if (!state.health) {
      applyHealth(health);
      return;
    }
    state.health = { ...state.health, visitor: health.visitor, claude: health.claude, own_keys: health.own_keys };
    state.visitorAt = Date.now();
    reconcileProviders(health);
    renderAllowance();
  }

  // index.html carries the production copy. A full catalog only gets its exact term
  // count; a sample catalog or offline mode gets wording that is true for it.
  function renderIntro(health) {
    const full = Boolean(health.neon && health.neon.catalog === "full");
    const offline = health.mode === "offline";
    const ontologies = Array.isArray(health.ontologies) ? health.ontologies : [];
    el.caveat.hidden = false;
    if (full) {
      const total = ontologies.reduce((sum, o) => sum + (Number(o.count) || 0), 0);
      if (total > 0) {
        el.caveat.textContent = `Searches ${total.toLocaleString("en")} real ontology terms. Every ID comes from the ontology; the model never writes one.`;
      }
      if (!offline) return;
    }
    const plural = { chemical: "chemicals", disease: "diseases", gene: "genes", food: "foods" };
    const list = (items, last) => (items.length < 2 ? items.join("") : `${items.slice(0, -1).join(", ")} ${last} ${items[items.length - 1]}`);
    const types = ontologies.map((o) => String(o.type));
    if (ontologies.length) {
      el.lede.textContent =
        `Paste a sentence. ${offline ? "A stand-in extractor" : "A language model"} finds the ${list(types.map((t) => plural[t] || t), "and")} in it, ` +
        `and xwalk links each one to a term in ${list(ontologies.map((o) => String(o.name)), "or")}.`;
      el.pipeFind.textContent = `${offline ? "A stand-in extractor" : "The language model"} reads your text and marks every ${list(types, "and")}.`;
      el.modelHint.textContent = `Finds the ${list(types.map((t) => plural[t] || t), "and")} in your text.`;
    }
    if (full) return;
    clear(el.caveat);
    el.caveat.append(
      "Each ontology here is a ",
      h("strong", { text: "200-term sample" }),
      ", not the full vocabulary, so many real-world terms will not be found. Every ID comes from the sample; the model never writes one.",
    );
    el.pipeSearch.textContent =
      "xwalk looks each one up in the sample ontology with keyword search (BM25) and embedding search, and merges the two rankings into a shortlist.";
  }

  // "Example" loads a sentence the deployment can map: the production one unless
  // health says this is a sample catalog or offline mode.
  function exampleText() {
    const health = state.health;
    if (!health) return EXAMPLE_FULL;
    return health.neon && health.neon.catalog === "full" && health.mode === "live" ? EXAMPLE_FULL : EXAMPLE_TEXT;
  }

  function renderLegend() {
    clear(el.legend);
    const byType = {};
    for (const o of (state.health && state.health.ontologies) || []) byType[o.type] = o;
    for (const t of TYPES) {
      const o = byType[t];
      if (!o && t !== "other" && state.health && state.health.neon && state.health.neon.catalog === "full") continue;
      const target = o ? `${o.name}, ${Number(o.count).toLocaleString("en")} terms` : t === "other" ? "not mapped" : "no ontology";
      el.legend.append(
        h("li", null, h("span", { class: `swatch type-${t}`, "aria-hidden": "true" }), h("span", { class: "legend-type", text: t }), h("span", { class: "legend-target", text: target })),
      );
    }
  }

  // Until health has answered, an own-key choice made earlier in this tab has no "Run
  // with" field to show it, and a run would go to the site's models: runs are held.
  async function loadHealth() {
    const chosen = savedPayer();
    if (chosen) holdRuns("pending", chosen);
    try {
      applyHealth(await api("GET", "/api/health"));
    } catch (err) {
      state.healthFailed = true;
      renderBanner(null, null);
      // Nothing is known about the catalog, so nothing is claimed about its size.
      el.caveat.hidden = true;
      fillTopK(DEFAULT_LIMITS.top_k_max);
      el.modelField.hidden = true;
      renderLegend();
      updateCounter();
      // Still chosen (the visitor may have picked the site's models while waiting).
      const still = savedPayer();
      if (still) holdRuns("unknown", still);
      else releaseRuns();
    }
  }

  // ---------- results ----------

  // Server offsets are Python str indices (code points). JS strings are UTF-16, so
  // index into an array of code points instead.
  function renderHighlighted(text, entities) {
    clear(el.highlighted);
    const cps = Array.from(text);
    const valid = entities
      .filter((e) => Number.isInteger(e.start) && Number.isInteger(e.end) && e.start >= 0 && e.end <= cps.length && e.start < e.end)
      .filter((e) => cps.slice(e.start, e.end).join("") === e.text)
      .sort((a, b) => a.start - b.start);
    let pos = 0;
    let skipped = entities.length - valid.length;
    for (const e of valid) {
      if (e.start < pos) {
        skipped += 1;
        continue;
      }
      if (e.start > pos) el.highlighted.append(cps.slice(pos, e.start).join(""));
      const t = typeLabel(e.type);
      el.highlighted.append(
        h(
          "a",
          { class: `mention type-${t}`, href: `#card-${e.id}`, title: `${e.text}: ${t}` },
          h("span", { class: "mention-text", text: cps.slice(e.start, e.end).join("") }),
          h("span", { class: "visually-hidden", text: " (" }),
          h("span", { class: "mention-tag", text: t }),
          h("span", { class: "visually-hidden", text: ")" }),
        ),
      );
      pos = e.end;
    }
    if (pos < cps.length) el.highlighted.append(cps.slice(pos).join(""));
    return skipped;
  }

  function selectorName(s) {
    if (s === "retrieval") return "Retrieval ranking only";
    if (s === "jev") return "Jev (TypeSafe decision model)";
    return modelName(s);
  }

  // The same choices as they read in the "How the final term is chosen" list.
  function selectorOption(s) {
    if (s === "retrieval") return "Top search result, no model";
    if (s === "jev") return "Jev picks (a small trained ranking model)";
    return `${modelName(s)} picks`;
  }

  const PICK_TEXT = {
    match: "Picked",
    review: "Picked, needs review",
    no_match: "No candidate fits",
    failed: "Selection failed",
  };

  function renderPick(p) {
    const status = PICK_TEXT[p.status] ? p.status : "failed";
    const href = safeHref(p.url);
    const term = p.id
      ? h(
          "span",
          { class: "pick-term" },
          href ? h("a", { href, target: "_blank", rel: "noopener noreferrer", text: p.id }) : h("span", { text: p.id }),
          h("span", { text: " " + (p.label || "") }),
        )
      : null;
    const confidence = typeof p.confidence === "number" ? h("span", { class: "pick-conf", text: "confidence " + p.confidence.toFixed(2) }) : null;
    return h(
      "div",
      { class: "pick pick-" + status },
      h("span", { class: "pick-who", text: selectorName(p.selector) + ": " }),
      h("strong", { class: "pick-status", text: status === "review" && !p.id ? "Unsure, needs review (no clear candidate)" : PICK_TEXT[status] }),
      term ? h("span", { text: " " }) : null,
      term,
      confidence ? h("span", { text: " · " }) : null,
      confidence,
      p.explanation ? h("p", { class: "pick-why", text: String(p.explanation) }) : null,
    );
  }

  function retrieverText(retrievers) {
    const parts = Object.entries(retrievers || {}).map(([name, rank]) => `${name} #${rank}`);
    return parts.length ? parts.join(", ") : "-";
  }

  function renderCard(e) {
    const t = typeLabel(e.type);
    const ontologyName = e.ontology ? state.ontologyNames[e.ontology] || e.ontology : null;
    const head = h(
      "header",
      { class: "card-head" },
      h("h3", { class: "card-title", id: `card-title-${e.id}` }, h("span", { class: `dot type-${t}`, "aria-hidden": "true" }), h("span", { text: e.text })),
      h("span", { class: `badge type-${t}`, text: t }),
      h("span", { class: "card-onto", text: ontologyName ? `→ ${ontologyName}` : "not mapped" }),
    );
    const card = h("article", { class: "card", id: `card-${e.id}`, "aria-labelledby": `card-title-${e.id}`, tabindex: "-1" }, head);
    const picked = e.pick && e.pick.id ? e.pick.id : null;
    if (e.pick) card.append(renderPick(e.pick));

    const matches = Array.isArray(e.matches) ? e.matches : [];
    const more = Array.isArray(e.more) ? e.more : [];
    if (matches.length) {
      const all = matches.concat(more);
      const tbody = h("tbody");
      const hiddenRows = [];
      all.forEach((m, i) => {
        const href = safeHref(m.url);
        const idCell = href
          ? h("a", { href, target: "_blank", rel: "noopener noreferrer", text: m.id })
          : h("span", { text: m.id });
        const row = h(
          "tr",
          { class: [i === 0 ? "top" : "", m.id === picked ? "picked" : ""].join(" ").trim() || null },
          h("td", { "data-label": "Rank", class: "c-rank num", text: String(i + 1) }),
          h("td", { "data-label": "ID", class: "c-id" }, idCell),
          h("td", { "data-label": "Label", class: "c-label", text: m.label || "-" }),
          h("td", { "data-label": "Score", class: "c-score num", text: typeof m.score === "number" ? m.score.toFixed(4) : "-" }),
          h("td", { "data-label": "Ranks", class: "c-ranks", text: retrieverText(m.retrievers) }),
        );
        if (i >= matches.length) {
          row.hidden = true;
          hiddenRows.push(row);
        }
        tbody.append(row);
      });
      const pickedRank = picked ? all.findIndex((m) => m.id === picked) + 1 : 0;
      const shown = h("span", { class: "cands-count", text: `showing ${matches.length} of ${all.length}` });
      card.append(
        h(
          "div",
          { class: "cands-head" },
          h("h4", { class: "cands-title", text: "Retrieved candidates" }),
          h(
            "p",
            { class: "cands-note" },
            "From retrieval (BM25 keyword search plus dense embeddings, fused by rank), before any selection",
            pickedRank ? ` · the pick is #${pickedRank}` : "",
            " · ",
            shown,
          ),
        ),
      );
      card.append(
        h(
          "table",
          { class: "matches" },
          h("caption", { class: "visually-hidden", text: `Retrieved candidate terms for ${e.text}, ranked` }),
          h(
            "thead",
            null,
            h("tr", null, h("th", { scope: "col", class: "num", text: "#" }), h("th", { scope: "col", text: "ID" }), h("th", { scope: "col", text: "Label" }), h("th", { scope: "col", class: "num", text: "Fused score" }), h("th", { scope: "col", text: "Retriever ranks" })),
          ),
          tbody,
        ),
      );
      if (hiddenRows.length) {
        const toggle = h("button", { type: "button", class: "ghost small cands-toggle", "aria-expanded": "false", text: `Show all ${all.length} retrieved candidates` });
        toggle.addEventListener("click", () => {
          const open = toggle.getAttribute("aria-expanded") !== "true";
          for (const r of hiddenRows) r.hidden = !open;
          toggle.setAttribute("aria-expanded", open ? "true" : "false");
          toggle.textContent = open ? `Show the top ${matches.length} only` : `Show all ${all.length} retrieved candidates`;
          shown.textContent = `showing ${open ? all.length : matches.length} of ${all.length}`;
        });
        card.append(toggle);
      }
    } else if (e.ontology) {
      const full = state.health && state.health.neon && state.health.neon.catalog === "full";
      card.append(h("p", { class: "empty", text: full ? "No term matched." : "No term in this 200-term sample matched. The real ontology may well contain it." }));
    }

    const notes = Array.isArray(e.notes) ? e.notes : [];
    if (notes.length) {
      card.append(h("ul", { class: "notes" }, ...notes.map((n) => h("li", { text: String(n) }))));
    }
    return card;
  }

  function usageText(usage) {
    if (!usage) return "usage unknown";
    const calls = `${usage.llm_calls} LLM call${usage.llm_calls === 1 ? "" : "s"}`;
    const p = usage.prompt_tokens;
    const c = usage.completion_tokens;
    const tokens =
      p === null || p === undefined || c === null || c === undefined
        ? "tokens unknown"
        : `${p.toLocaleString()} + ${c.toLocaleString()} tokens`;
    const emb = `${usage.embedding_requests} embedding request${usage.embedding_requests === 1 ? "" : "s"}`;
    const parts = [calls, tokens, emb];
    if (usage.selector_calls) {
      const cost = typeof usage.selector_cost_usd === "number" ? "$" + usage.selector_cost_usd.toFixed(5) : "cost unknown";
      parts.push(usage.selector_calls + " selector call" + (usage.selector_calls === 1 ? "" : "s") + " (" + cost + ")");
    }
    return parts.join(", ");
  }

  function retrievalText(retrieval) {
    const parts = Object.entries(retrieval || {}).map(([name, s]) => `${name} ${s && s.status === "ok" ? "ok" : "unavailable"}`);
    return parts.length ? `Retrieval: ${parts.join(", ")}` : "Retrieval: unknown";
  }

  function joinNames(names) {
    return names.length < 2 ? names.join("") : `${names.slice(0, -1).join(", ")} and ${names[names.length - 1]}`;
  }

  // The selectors that made the picks, in order of first appearance. After a step-down
  // this differs from the selector the request asked for.
  function pickersOf(entities) {
    const pickers = [];
    for (const e of entities) {
      const who = e && e.pick && typeof e.pick.selector === "string" ? e.pick.selector : "";
      if (who && !pickers.includes(who)) pickers.push(who);
    }
    return pickers;
  }

  // The results header names what actually ran: the extraction model, and the model
  // (or models, after a step-down) that made the picks.
  function renderRanOn(data, pickers) {
    clear(el.ranOn);
    let askedShown = null;
    if (data.mode === "offline") {
      el.ranOn.append("Extracted by the offline stand-in.");
    } else if (typeof data.model === "string" && data.model) {
      el.ranOn.append("Extracted by ", h("strong", { text: modelName(data.model) }));
      if (data.credential_source === "visitor" && typeof data.provider === "string" && data.provider && data.provider !== "site") {
        el.ranOn.append(` on your ${providerName(data.provider)} key`);
      }
      if (typeof data.requested_model === "string" && data.requested_model && data.requested_model !== data.model) {
        el.ranOn.append(h("span", { class: "asked", text: ` (you asked for ${modelName(data.requested_model)})` }));
        askedShown = data.requested_model;
      }
      el.ranOn.append(".");
    }
    if (pickers.length) {
      el.ranOn.append(" Matches picked by ");
      pickers.forEach((who, i) => {
        if (i) el.ranOn.append(i === pickers.length - 1 ? " and " : ", ");
        el.ranOn.append(h("strong", { text: selectorName(who) }));
      });
      const asked = data.requested_selector;
      // Not repeated when the extraction sentence already said the same thing.
      if (typeof asked === "string" && asked && !pickers.includes(asked) && asked !== askedShown) {
        el.ranOn.append(h("span", { class: "asked", text: ` (you asked for ${selectorName(asked)})` }));
      }
      el.ranOn.append(".");
    }
    el.ranOn.hidden = !el.ranOn.firstChild;
  }

  const FALLBACK_STAGE = { extract: "Extraction", select: "Selection" };

  // One sentence per step-down, for a response that lists `fallbacks` without `notices`.
  function fallbackSentence(f) {
    if (!f || typeof f !== "object" || typeof f.from !== "string" || typeof f.to !== "string") return null;
    const reason = typeof f.reason === "string" && f.reason ? ` (${f.reason.replace(/_/g, " ")})` : "";
    const stage = FALLBACK_STAGE[f.stage];
    return stage
      ? `${stage} moved from ${modelName(f.from)} to ${modelName(f.to)}${reason}.`
      : `This request started on ${modelName(f.to)} instead of ${modelName(f.from)}${reason}.`;
  }

  function renderNotices(data) {
    clear(el.noticesList);
    let lines = (Array.isArray(data.notices) ? data.notices : []).filter((n) => typeof n === "string" && n.trim());
    if (!lines.length && Array.isArray(data.fallbacks)) lines = data.fallbacks.map(fallbackSentence).filter(Boolean);
    for (const line of new Set(lines)) el.noticesList.append(h("li", { text: line }));
    el.notices.hidden = !el.noticesList.firstChild;
  }

  function renderResults(text, data) {
    const entities = Array.isArray(data.entities) ? data.entities : [];
    const skipped = renderHighlighted(text, entities);

    clear(el.cards);
    for (const e of entities) el.cards.append(renderCard(e));

    clear(el.warnings);
    const warnings = (Array.isArray(data.warnings) ? data.warnings : []).map(String);
    if (skipped) warnings.push(`${skipped} entit${skipped === 1 ? "y" : "ies"} could not be highlighted (offsets do not match the text).`);
    for (const w of warnings) el.warnings.append(h("li", { text: w }));
    el.warnings.hidden = warnings.length === 0;

    const mapped = entities.filter((e) => e.ontology && e.matches && e.matches.length).length;
    el.summary.textContent = entities.length
      ? `${entities.length} entit${entities.length === 1 ? "y" : "ies"} found, ${mapped} with candidate terms.`
      : "No entities found.";
    const pickers = pickersOf(entities);
    renderRanOn(data, pickers);
    renderNotices(data);

    const timing = data.timing_ms || {};
    setStatus([
      retrievalText(data.retrieval),
      `Model ${modelName(data.model)} (${data.mode || "unknown"}), picked by ${pickers.length ? joinNames(pickers.map(selectorName)) : selectorName(data.selector || "retrieval")}`,
      usageText(data.usage),
      `extract ${fmtSeconds(timing.extract)}, map ${fmtSeconds(timing.map)}` + (timing.select !== undefined ? `, select ${fmtSeconds(timing.select)}` : "") + `, total ${fmtSeconds(timing.total)}`,
    ]);
    renderBanner(data.mode, data.retrieval);
    el.results.hidden = false;
  }

  // ---------- run ----------

  function setRunning(on) {
    state.running = on;
    syncRun();
    el.run.setAttribute("aria-busy", on ? "true" : "false");
    el.results.setAttribute("aria-busy", on ? "true" : "false");
    el.run.querySelector(".run-label").textContent = on ? "Mapping…" : "Map this text";
    showProgress(on);
  }

  // What a visitor sees while a run is under way: the three steps with what each one
  // does, a moving bar and a clock, so half a minute of waiting does not look like a
  // page that hung. No step is marked done by the clock: the API answers once, at the
  // end. The clock is hidden from screen readers; the status line announces the start.
  const PROGRESS_USUAL = "This usually takes 10 to 30 seconds. Keep this page open.";
  const PROGRESS_SLOW = "Taking longer than usual. Long texts with many mentions can need up to two minutes.";
  let progressTimer = null;

  function showProgress(on) {
    if (progressTimer !== null) {
      clearInterval(progressTimer);
      progressTimer = null;
    }
    el.progress.hidden = !on;
    el.pipe.setAttribute("data-state", on ? "running" : "idle");
    if (!on) return;
    const started = performance.now();
    const tick = () => {
      const s = Math.floor((performance.now() - started) / 1000);
      el.progressTime.textContent = `${s} s`;
      el.progressHint.textContent = s >= 45 ? PROGRESS_SLOW : PROGRESS_USUAL;
    };
    tick();
    progressTimer = setInterval(tick, 1000);
    // On a phone the steps can be below the screen's edge.
    reveal(el.pipe, "nearest");
  }

  function reveal(node, block) {
    if (typeof node.scrollIntoView !== "function") return;
    const calm = typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
    node.scrollIntoView({ block, behavior: calm ? "auto" : "smooth" });
  }

  function revealResults() {
    reveal(el.results, "start");
  }

  async function run() {
    if (state.running) return;
    // Both ways to submit end here (the button and Ctrl/Cmd+Enter).
    if (state.payer) {
      (el.payerRetry.hidden ? el.payerSite : el.payerRetry).focus();
      return;
    }
    hideError();
    const text = el.text.value;
    const n = codePointLength(text);
    if (!text.trim() || n > state.limits.max_chars) {
      if (text.trim()) showTextError("long", tooLongText(n));
      else showTextError("empty", "Paste some text first, or tap Example.");
      // On a phone the keyboard would cover the message, so the box is not focused.
      if (touch) reveal(el.textError, "center");
      else el.text.focus();
      return;
    }
    const promptValue = el.prompt.value;
    const promptLength = codePointLength(promptValue);
    if (promptLength > state.limits.max_prompt_chars) {
      showPromptError(`The prompt is ${promptLength.toLocaleString("en")} characters; the limit is ${state.limits.max_prompt_chars.toLocaleString("en")}.`);
      openOptions();
      el.promptEditor.open = true;
      el.prompt.focus();
      return;
    }
    showPromptError("");
    const provider = currentProvider();
    if (provider) {
      const typed = el.ownKey.value.trim();
      if (!typed || !KEY_SHAPE.test(typed)) {
        showError(
          new ApiError(
            typed
              ? "That does not look like an API key: it should be 20 to 512 characters with no spaces."
              : `Enter your ${provider.name} API key first, or choose "This site" under "Who pays for the model".`,
            "bad_credential",
          ),
        );
        el.ownKey.focus();
        return;
      }
      rememberKey();
    }
    if (!el.passField.hidden) storageSet(PASSCODE_KEY, el.passcode.value);

    const payload = {
      text,
      prompt: !promptValue.trim() || promptValue === state.defaultPrompt ? null : promptValue,
      model: el.model.value || null,
      top_k: Number(el.topK.value) || 5,
      selector: el.selectorField.hidden ? null : el.selector.value || null,
    };
    // An API from before own keys never sees the field.
    if (provider) payload.provider = provider.id;

    setRunning(true);
    // Said once for screen readers; sighted visitors have the steps and the button.
    clear(el.status);
    el.status.append(h("span", { class: "visually-hidden", text: "Mapping your text. This usually takes 10 to 30 seconds." }));
    const started = performance.now();
    try {
      // The key is read here, for this call, and held nowhere else.
      const data = await api("POST", "/api/ner", payload, provider ? el.ownKey.value.trim() : undefined);
      renderResults(text, data);
      setRunning(false);
      revealResults();
    } catch (err) {
      el.results.hidden = true;
      renderBanner(state.health && state.health.mode, state.health && state.health.retrieval);
      showError(err);
      setStatus([`Failed after ${fmtSeconds(Math.round(performance.now() - started))}.`]);
    } finally {
      setRunning(false);
    }
    refreshHealth();
  }

  // ---------- wiring ----------

  el.form.addEventListener("submit", (ev) => {
    ev.preventDefault();
    run();
  });
  el.form.addEventListener("keydown", (ev) => {
    if (ev.key === "Enter" && (ev.ctrlKey || ev.metaKey)) {
      ev.preventDefault();
      run();
    }
  });
  el.text.addEventListener("input", () => {
    if (state.textError === "empty") clearTextError();
    updateCounter();
  });
  el.prompt.addEventListener("input", () => showPromptError(""));
  el.useOwnKey.addEventListener("click", useOwnKey);
  // A tab that slept through a timed wait learns that the wait is over.
  document.addEventListener("visibilitychange", () => {
    if (document.visibilityState === "visible" && allowanceTimer !== null && !state.running) refreshHealth();
  });
  el.model.addEventListener("change", renderAllowance);
  el.selector.addEventListener("change", renderAllowance);
  el.example.addEventListener("click", () => {
    el.text.value = exampleText();
    clearTextError();
    updateCounter();
    if (touch) {
      // No keyboard: the next step is the button, so that is what comes into view.
      el.run.focus({ preventScroll: true });
      reveal(el.run, "nearest");
    } else el.text.focus();
  });
  el.clear.addEventListener("click", () => {
    el.text.value = "";
    clearTextError();
    updateCounter();
    if (!touch) el.text.focus();
  });
  el.promptReset.addEventListener("click", () => {
    el.prompt.value = state.defaultPrompt;
    el.prompt.focus();
  });
  el.passcode.addEventListener("change", () => storageSet(PASSCODE_KEY, el.passcode.value));
  // A key belongs to one provider: changing the choice forgets it, in the field and in
  // session storage, before anything else happens.
  el.provider.addEventListener("change", () => {
    forgetKey();
    rememberPayer(el.provider.value);
    applyProvider();
    releaseRuns(); // the visitor has chosen who pays
  });
  el.payerSite.addEventListener("click", () => {
    forgetKey();
    rememberPayer("site");
    el.provider.value = "site";
    if (state.health) applyProvider();
    releaseRuns();
    el.run.focus();
  });
  el.payerRetry.addEventListener("click", () => {
    hideError();
    loadHealth();
  });
  el.ownKey.addEventListener("input", rememberKey);
  el.forgetKey.addEventListener("click", () => {
    forgetKey();
    el.ownKey.focus();
  });
  $("error-dismiss").addEventListener("click", () => {
    hideError();
    if (!touch) el.text.focus();
  });

  const isMac = /Mac|iPhone|iPad/.test(navigator.platform || navigator.userAgent || "");
  $("shortcut-key").textContent = isMac ? "⌘" : "Ctrl";

  fillTopK(DEFAULT_LIMITS.top_k_max);
  el.topK.value = "5";
  updateCounter();
  loadHealth();
})();
