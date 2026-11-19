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
  // Errors about a visitor's own key: said at the key field, not in the error box.
  const KEY_CODES = ["bad_credential", "provider_key_refused", "provider_rate_limited"];
  const ISSUES_URL = "https://github.com/jan3657/xwalk/issues";
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
    ownKeyError: $("own-key-error"),
    selectorOwn: $("selector-own"),
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
    errorTitle: $("error-title"),
    errorMsg: $("error-message"),
    errorAction: $("error-action"),
    errorOwnKey: $("error-own-key"),
    announce: $("announce"),
    status: $("status"),
    pipe: $("pipe"),
    pipeFind: $("pipe-find"),
    pipeSearch: $("pipe-search"),
    modelHint: $("model-hint"),
    progress: $("progress"),
    progressTime: $("progress-time"),
    progressHint: $("progress-hint"),
    results: $("results"),
    resultsHeading: $("results-heading"),
    exampleNote: $("example-note"),
    stale: $("stale"),
    summary: $("results-summary"),
    cap: $("results-cap"),
    takeaway: $("results-takeaway"),
    ranOn: $("results-models"),
    noEntities: $("no-entities"),
    noEntitiesTitle: $("no-entities-title"),
    noEntitiesExample: $("no-entities-example"),
    textPanel: $("text-panel"),
    notices: $("notices"),
    noticesList: $("notices-list"),
    highlighted: $("highlighted"),
    legend: $("legend"),
    lede: $("lede"),
    caveat: $("caveat"),
    warningsBox: $("warnings-box"),
    warnings: $("warnings"),
    cardsTools: $("cards-tools"),
    expandAll: $("expand-all"),
    cards: $("cards"),
    runDetails: $("run-details"),
    runDetailsList: $("run-details-list"),
    runDetailsSummary: $("run-details-summary"),
    next: $("next"),
    nextBody: $("next-body"),
    nextTry: $("next-try"),
  };

  const state = {
    health: null,
    limits: { ...DEFAULT_LIMITS },
    defaultPrompt: "",
    ontologyNames: {},
    ontologies: [], // { slug, name, type, count }: from health, or from example.json until health answers
    providers: [], // own-key providers the API offers; empty when it offers none
    running: false,
    healthFailed: false, // the status read on arrival failed; said in the top banner
    visitorAt: 0, // when health.visitor was read, for "next one in N min"
    textError: null, // "empty", "long" or "server" while the line under the text box is shown
    // The result on display: { text, data, example }. A result belongs to the text it
    // was made from: once the box holds something else the result is marked as being for
    // the earlier text. `example` is true for the recorded example, which is never the
    // visitor's own run.
    shown: null,
    // example.json once it has loaded: { text, recorded, data, ontologies }.
    example: null,
    touched: false, // the visitor typed, or used Example or Clear
    prefilled: false, // the text box holds the example because this page put it there
    ran: false, // a run was started: the recorded example is not shown after that
    limitNotice: false, // the error box is saying that a limit is used up
    // The run under way: { provider, started, hidden, edited }. `hidden` once the page
    // went to the background during it, `edited` once the visitor typed during it.
    runCtx: null,
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
    constructor(message, code, status, retryAfter) {
      super(message);
      this.code = code || null;
      this.status = status || 0;
      // Seconds from the Retry-After header of a refusal, or null.
      this.retryAfter = typeof retryAfter === "number" && Number.isFinite(retryAfter) && retryAfter > 0 ? retryAfter : null;
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
      const after = response.headers.get("Retry-After");
      throw new ApiError(msg, typeof e.code === "string" ? e.code : null, response.status, after ? Number(after) : null);
    }
    return data;
  }

  // ---------- messages ----------

  function minutesText(seconds) {
    const minutes = Math.max(1, Math.ceil(seconds / 60));
    return `${minutes} minute${minutes === 1 ? "" : "s"}`;
  }

  // What the error box says for a failed run: a title, a body, one action. The error
  // `code` decides; the server's sentence is read only where one code covers several
  // cases (its wording is held by the API's tests), and is shown as it is only for a
  // code this page does not know. `ctx` is the run that failed.
  function describeError(err, ctx) {
    const message = err.message || "";
    const own = Boolean(ctx && ctx.provider);
    const offerKey = !own && state.providers.length > 0;
    const wait = err.retryAfter;
    switch (err.code) {
      case "visitor_limit_reached":
        if (/at once/.test(message)) {
          return {
            look: "notice",
            title: "One run at a time.",
            body: ["Another run from your network is still going. Wait a few seconds and try again."],
            action: "Try again",
          };
        }
        if (own || /own key/.test(message)) {
          return {
            look: "notice",
            limit: true,
            title: "Too many runs with your own key.",
            body: [`Runs with an own key are limited per network too. ${wait ? `Try again in ${minutesText(wait)}.` : "Try again later."}`],
          };
        }
        if (/today/.test(message)) {
          const at = wait ? new Date(Date.now() + wait * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : null;
          return {
            look: "notice",
            limit: true,
            title: "The free runs for your network today are used.",
            body: [at ? `They come back at ${at} (midnight UTC).` : "They come back at midnight UTC."],
            ownKey: offerKey,
          };
        }
        return {
          look: "notice",
          limit: true,
          title: "The free runs for your network this hour are used.",
          body: [wait ? `You can run again in ${minutesText(wait)}.` : "You can run again later this hour."],
          ownKey: offerKey,
        };
      case "limits_unavailable":
        return {
          look: "notice",
          limit: true,
          title: "The demo is briefly unavailable.",
          body: ["The site cannot count free runs right now, so it is not starting any. This usually clears within a minute."],
          action: "Try again",
          ownKey: offerKey,
        };
      case "call_budget_exhausted":
        return { look: "alert", title: "The model is busy right now.", body: ["The run could not finish. Try again in a minute."], action: "Try again" };
      case "provider_error":
        if (/did not answer|timeout/i.test(message)) {
          return { look: "alert", title: "The model took too long.", body: ["It did not answer in time. Try again, or try a shorter text."], action: "Try again" };
        }
        if (/overloaded|rate.?limit/i.test(message)) {
          return { look: "alert", title: "The model is busy right now.", body: ["The run could not finish. Try again in a minute."], action: "Try again" };
        }
        return { look: "alert", title: "The model could not finish this run.", body: ["The model provider returned an error. Try again in a minute."], action: "Try again" };
      case "timeout":
        return {
          look: "alert",
          title: "The run took too long and was stopped.",
          body: ["Long texts with many mentions can run out of time. Try again, or try a shorter text."],
          action: "Try again",
        };
      case "network": {
        // A request that died late, or while the page was in the background, was most
        // likely cut off by the phone, not refused by the network.
        const late = Boolean(ctx) && (ctx.hidden || performance.now() - ctx.started > 3000);
        if (late) {
          return {
            look: "alert",
            title: "The connection dropped before the result arrived.",
            body: ["This usually happens when the phone locks or the app goes to the background. Your text is still here. The interrupted run may still have been counted."],
            action: "Run again",
          };
        }
        return { look: "alert", title: "You seem to be offline.", body: ["The site could not be reached. Check your connection and try again."], action: "Try again" };
      }
      case "passcode_required":
        return { look: "notice", title: "This demo needs a passcode.", body: ["Enter it under the Run button and run again."] };
      case "passcode_invalid":
        return { look: "alert", title: "The passcode is not correct.", body: ["Check it and run again."] };
      default:
    }
    if (err.code === "bad_response" && err.status === 504) {
      return {
        look: "alert",
        title: "The run took too long and was stopped.",
        body: ["Long texts with many mentions can run out of time. Try again, or try a shorter text."],
        action: "Try again",
      };
    }
    if (err.code === "internal" || err.code === "not_configured" || err.code === "bad_response" || (!err.code && err.status >= 500)) {
      return {
        look: "alert",
        title: "The demo hit an error on its side.",
        body: [
          "This is a fault in the site, not in your text. Try again; if it keeps happening, please ",
          h("a", { href: ISSUES_URL, target: "_blank", rel: "noopener", text: "open an issue" }),
          ".",
        ],
        action: "Try again",
      };
    }
    // A code this page does not know: the server's own sentence.
    return { look: "alert", title: "Something went wrong.", body: [message], action: "Try again" };
  }

  // What is said at the key field for an error about the visitor's own key.
  function keyErrorText(err, provider) {
    if (err.code === "provider_key_refused") {
      return `${provider.name} did not accept this key. Check that it was pasted whole, is still active, has credit, and may use ${modelName(el.model.value)}.`;
    }
    if (err.code === "provider_rate_limited") {
      return `${provider.name} is rate limiting your key. Wait a minute and try again, or choose a smaller model.`;
    }
    return "This does not look like an API key. Paste the whole key, with no spaces.";
  }

  function showKeyError(...parts) {
    clear(el.ownKeyError);
    el.ownKeyError.append(...parts);
    el.ownKeyError.hidden = false;
    el.ownKey.setAttribute("aria-invalid", "true");
    openOptions();
  }

  function clearKeyError() {
    clear(el.ownKeyError);
    el.ownKeyError.hidden = true;
    el.ownKey.removeAttribute("aria-invalid");
  }

  // A failed run. Problems with a field are said at that field; everything else goes to
  // the error box. Nothing here touches the text box or a result already on display.
  function showError(err, ctx) {
    const e = err instanceof ApiError ? err : new ApiError(err instanceof Error ? err.message : String(err), null, 0);
    const calm = Boolean(ctx && ctx.edited); // the visitor is typing: no focus, no scroll
    if (OPTIONS_CODES.includes(e.code)) openOptions();
    if (e.code === "missing_text" || e.code === "text_too_long") {
      const n = codePointLength(el.text.value);
      if (e.code === "text_too_long" && n > state.limits.max_chars) showTextError("long", tooLongText(n));
      else showTextError("server", e.code === "missing_text" ? "Paste some text first, or tap Example." : "This text is too long for the demo. Shorten it and run again.");
      if (!calm) reveal(el.textError, "center");
      return;
    }
    if (e.code === "prompt_too_long") {
      showPromptError("The instructions are too long. Shorten them, or reset them to the default.");
      el.promptEditor.open = true;
      if (!calm) el.prompt.focus();
      return;
    }
    const provider = currentProvider();
    if (KEY_CODES.includes(e.code) && provider) {
      // A key the server or the provider refused stays in the field, to be corrected.
      showKeyError(keyErrorText(e, provider));
      if (!calm) el.ownKey.focus();
      return;
    }
    const d = describeError(e, ctx);
    el.error.classList.toggle("notice", d.look === "notice");
    el.errorTitle.textContent = d.title;
    clear(el.errorMsg);
    el.errorMsg.append(...d.body);
    el.errorAction.textContent = d.action || "";
    el.errorAction.hidden = !d.action;
    el.errorOwnKey.hidden = !d.ownKey;
    el.error.hidden = false;
    // The box now says what the line beside Run says; the line steps aside for it.
    state.limitNotice = Boolean(d.limit);
    renderAllowance();
    if (e.code === "passcode_required" || e.code === "passcode_invalid") {
      el.passField.hidden = false;
      if (!calm) el.passcode.focus();
      return;
    }
    if (calm) return;
    const target = d.action ? el.errorAction : d.ownKey ? el.errorOwnKey : $("error-dismiss");
    target.focus({ preventScroll: true });
    reveal(el.error, "nearest");
  }

  function hideError() {
    el.error.hidden = true;
    clear(el.errorMsg);
    if (state.limitNotice) {
      state.limitNotice = false;
      renderAllowance();
    }
  }

  // One sentence for screen readers, in the one live region kept for that.
  function announce(text) {
    el.announce.textContent = text;
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
        " Mentions come from a stand-in extractor, not a language model, so results only illustrate the flow.",
      ]);
    }
    // The operator's reason is never printed here; it is in "Run details" after a run.
    const dense = retrieval && retrieval.dense;
    if (dense && dense.status !== "ok") {
      items.push(["Part of the search is unavailable.", " Matching by meaning is off right now, so terms are found by keyword only. Synonyms and paraphrases may be missed."]);
    }
    const bm25 = retrieval && retrieval.bm25;
    if (bm25 && bm25.status !== "ok") {
      items.push(["Part of the search is unavailable.", " Matching by keyword is off right now, so terms are found by meaning only."]);
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
      "This is the text the model is given before yours. Change it to look for other kinds of mention or to set stricter rules. " +
      `Your edit is used for your runs in this tab only. Up to ${state.limits.max_prompt_chars.toLocaleString("en")} characters.`;

    fillProviders(health);

    fillTopK(state.limits.top_k_max || 10);

    el.passField.hidden = !health.passcode_required;
    if (health.passcode_required) el.passcode.value = storageGet(PASSCODE_KEY);

    state.ontologyNames = {};
    state.ontologies = Array.isArray(health.ontologies) ? health.ontologies : [];
    for (const o of state.ontologies) {
      state.ontologyNames[o.slug] = o.name;
    }
    renderIntro(health);
    renderBanner(health.mode, health.retrieval);
    applyProvider();
    updateCounter();
    syncExample();
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
      clear(el.ownKeyNote);
      el.ownKeyNote.append(
        h("li", { text: `Sent only with your runs, to this site and from there to ${provider.name}. This site does not store or log it.` }),
        h("li", { text: "Kept in this tab until you close it or press Forget key." }),
        h("li", { text: `${provider.name} bills you for the model calls. A key with a spending limit is safest.` }),
      );
    } else {
      const selectors = Array.isArray(health.selectors) ? health.selectors : [];
      for (const s of selectors) el.selector.append(h("option", { value: s, text: selectorOption(s) }));
      if (selectors.includes(health.default_selector)) el.selector.value = health.default_selector;
      el.selectorField.hidden = selectors.length <= 1;

      const models = Array.isArray(health.models) ? health.models : [];
      for (const m of models) el.model.append(h("option", { value: m, text: modelName(m) }));
      if (health.default_model) el.model.value = health.default_model;
      clear(el.ownKeyNote);
    }
    clearKeyError();
    el.ownKeyField.hidden = !provider;
    el.ownKeyNote.hidden = !provider;
    el.selectorOwn.hidden = !provider;
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
    // While the error box says the same thing (a used-up limit, with the same way out),
    // it is not said a second time here.
    const said = blocked && state.limitNotice;
    el.allowance.hidden = !message || said;
    el.allowanceRow.classList.toggle("blocked", blocked);
    el.allowanceRow.hidden = !message || said;
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
    return exampleFits() === false ? EXAMPLE_TEXT : EXAMPLE_FULL;
  }

  // Whether this is the kind of deployment the production example belongs to: the full
  // catalog and a real model. null until health has answered.
  function exampleFits() {
    const health = state.health;
    if (!health) return null;
    return Boolean(health.neon && health.neon.catalog === "full" && health.mode === "live");
  }

  // ---------- the recorded example ----------

  // On arrival the page shows a finished result under the form: one real run of the
  // example sentence, recorded from production and served as a static file. It is
  // labelled as an example with its date and is never presented as the visitor's run.
  // It does not wait for health (a function that can take seconds to wake up): it is
  // shown as soon as the file arrives and taken away again if health then says this is
  // not a full-catalog, live deployment. If the file does not load, nothing happens.
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];

  // "2026-11-19" as "19 Nov 2026"; "" for anything else.
  function recordedDate(iso) {
    const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(typeof iso === "string" ? iso : "");
    const month = m ? MONTHS[Number(m[2]) - 1] : null;
    return month ? `${Number(m[3])} ${month} ${m[1]}` : "";
  }

  // A static file: fetched with a plain fetch, never api(), which is the function that
  // carries the passcode and a visitor's key.
  async function loadExampleResult() {
    let file;
    try {
      const response = await fetch("example.json", { credentials: "same-origin" });
      if (!response.ok) return;
      file = await response.json();
    } catch (_) {
      return;
    }
    const data = file && typeof file === "object" ? file.response : null;
    const recorded = file && typeof file === "object" ? recordedDate(file.recorded) : "";
    // Without its date it could not be labelled, so it is not shown.
    if (!data || typeof data !== "object" || !Array.isArray(data.entities) || !data.entities.length) return;
    if (typeof file.text !== "string" || !file.text || !recorded) return;
    const ontologies = (Array.isArray(file.ontologies) ? file.ontologies : []).filter((o) => o && typeof o.slug === "string" && typeof o.name === "string");
    state.example = { text: file.text, recorded, data, ontologies };
    showExampleResult();
  }

  function showExampleResult() {
    const example = state.example;
    // Not on a sample catalog or offline, and not once the visitor has run something.
    if (!example || exampleFits() === false || state.ran || state.shown) return;
    if (!state.health) {
      // The names and sizes as they were when the example was recorded, until health answers.
      state.ontologies = example.ontologies;
      for (const o of example.ontologies) state.ontologyNames[o.slug] = o.name;
    }
    // Text the visitor typed, or the browser restored, is never overwritten.
    if (!state.touched && el.text.value === "") {
      el.text.value = example.text;
      state.prefilled = true;
      updateCounter();
    }
    // No scroll, no focus, no announcement: nothing has happened yet.
    renderResults(example.text, example.data, { example: true });
  }

  // After every health answer. A deployment the example does not belong to loses it,
  // and the text box goes back to empty if the example in it was put there by this page.
  function syncExample() {
    const shown = state.shown;
    if (!shown) return;
    if (shown.example && exampleFits() === false) {
      el.results.hidden = true;
      state.shown = null;
      syncStale();
      if (state.prefilled && !state.touched && el.text.value === state.example.text) {
        el.text.value = "";
        updateCounter();
      }
      state.prefilled = false;
      return;
    }
    // The legend and the closing panel carry term counts, which come from health.
    const entities = Array.isArray(shown.data.entities) ? shown.data.entities : [];
    renderLegend(entities);
    renderNext(shown.data, entities, pickersOf(entities));
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
      updateCounter();
      // Still chosen (the visitor may have picked the site's models while waiting).
      const still = savedPayer();
      if (still) holdRuns("unknown", still);
      else releaseRuns();
    }
  }

  // ---------- results ----------

  const TYPE_NAMES = { chemical: "Chemical", disease: "Disease", gene: "Gene", food: "Food", other: "Other" };
  const CONFIDENCE_HELP = "The model's own confidence, not a measured accuracy";

  // Server offsets are Python str indices (code points). JS strings are UTF-16, so
  // index into an array of code points instead. A mention is told apart by its tint and
  // underline; its type is also in its accessible name and its tooltip.
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
      const kind = e.ontology ? t : `${t}, not looked up`;
      el.highlighted.append(
        h(
          "a",
          { class: `mention type-${t}`, href: `#card-${e.id}`, title: `${e.text}: ${kind}` },
          cps.slice(e.start, e.end).join(""),
          h("span", { class: "visually-hidden", text: ` (${kind})` }),
        ),
      );
      pos = e.end;
    }
    if (pos < cps.length) el.highlighted.append(cps.slice(pos).join(""));
    return skipped;
  }

  // The key to the tints: only the types this result holds.
  function renderLegend(entities) {
    clear(el.legend);
    const byType = {};
    for (const o of state.ontologies) byType[o.type] = o;
    for (const t of TYPES) {
      const mine = entities.filter((e) => typeLabel(e.type) === t);
      if (!mine.length) continue;
      const o = mine.some((e) => e.ontology) ? byType[t] : null;
      let target = "not looked up";
      if (o) {
        const count = Number(o.count) || 0;
        target = `${o.name} · ${count >= 1000 ? `${Math.round(count / 1000)}k` : count} terms`;
      } else if (mine.some((e) => e.ontology)) {
        target = state.ontologyNames[mine.find((e) => e.ontology).ontology] || "";
      }
      el.legend.append(
        h("li", { class: `type-${t}` }, h("span", { class: "swatch", "aria-hidden": "true" }), h("span", { class: "legend-type", text: TYPE_NAMES[t] }), target ? h("span", { class: "legend-target", text: target }) : null),
      );
    }
  }

  // Who made a pick, as it reads in a sentence.
  function selectorName(s) {
    if (s === "retrieval") return "the top search result";
    if (s === "jev") return "Jev";
    return modelName(s);
  }

  // The same choices as they read in the "How the final term is chosen" list.
  function selectorOption(s) {
    if (s === "retrieval") return "Top search result, no model";
    if (s === "jev") return "Jev picks (a small trained ranking model)";
    return `${modelName(s)} picks`;
  }

  // How one entity came out. "matched" and "check" carry a term the model chose;
  // "search" carries the top search result because no model was asked to choose.
  function outcomeOf(e) {
    if (!e.ontology) return "unsearched";
    const p = e.pick && typeof e.pick === "object" ? e.pick : null;
    if (p) {
      if (p.status === "match" && p.id) return "matched";
      if (p.status === "review") return p.id ? "check" : "unsure";
      if (p.status === "no_match") return "declined";
      return "failed";
    }
    return Array.isArray(e.matches) && e.matches.length ? "search" : "none";
  }

  // The model chose a term that the search had not ranked first.
  function isOverride(e) {
    const top = Array.isArray(e.matches) && e.matches[0];
    return Boolean(e.pick && e.pick.id && top && top.id && e.pick.id !== top.id);
  }

  const OUTCOME_TEXT = {
    unsure: "unsure, no term chosen",
    declined: "no term fits",
    failed: "could not be matched",
    none: "no term found",
    unsearched: "not looked up",
  };

  const RETRIEVER_NAMES = { bm25: "keyword", dense: "meaning" };

  function retrieverText(retrievers) {
    const parts = Object.entries(retrievers || {}).map(([name, rank]) => `${RETRIEVER_NAMES[name.split("-")[0]] || name} #${rank}`);
    return parts.length ? parts.join(", ") : "-";
  }

  // The server's notes on one entity, in plain words. An unknown note is shown as it is.
  function entityNote(note) {
    const n = String(note);
    if (/^no ontology for type/.test(n)) return null; // said by the card itself
    if (/^dense retrieval unavailable; BM25 only$/.test(n)) return "Found by keyword search only; matching by meaning was unavailable.";
    if (/^bm25 retrieval unavailable; dense only$/.test(n)) return "Found by meaning only; keyword search was unavailable.";
    if (n === "every retriever failed; no matches") return "The search failed for this one.";
    if (n === "cut off by the request deadline") return "The run ran out of time before this one was finished.";
    return n;
  }

  function plural(n, one, many) {
    return `${n.toLocaleString("en")} ${n === 1 ? one : many || `${one}s`}`;
  }

  function setCardOpen(card, open) {
    const toggle = card.querySelector(".card-more");
    const detail = card.querySelector(".card-detail");
    if (!toggle || !detail) return;
    detail.hidden = !open;
    toggle.setAttribute("aria-expanded", open ? "true" : "false");
    card.classList.toggle("is-open", open);
  }

  function syncExpandAll() {
    const toggles = [...el.cards.querySelectorAll(".card-more")];
    const allOpen = toggles.length > 0 && toggles.every((t) => t.getAttribute("aria-expanded") === "true");
    el.cardsTools.hidden = toggles.length === 0;
    el.expandAll.textContent = allOpen ? "Collapse all" : "Expand all";
    el.expandAll.setAttribute("aria-expanded", allOpen ? "true" : "false");
  }

  // What is folded away in a card: why this term, who chose it, and the candidates the
  // search found. "Candidate" and the search's own numbers appear only here.
  function renderDetail(e, outcome) {
    const p = e.pick && typeof e.pick === "object" ? e.pick : null;
    const matches = Array.isArray(e.matches) ? e.matches : [];
    const all = matches.concat(Array.isArray(e.more) ? e.more : []);
    const picked = p && p.id ? p.id : null;
    const who = p ? selectorName(p.selector) : "";
    const count = plural(all.length, "candidate");
    const detail = h("div", { class: "card-detail", id: `card-detail-${e.id}` });

    if (isOverride(e)) {
      detail.append(h("p", { class: "card-over", text: `Search ranked ${matches[0].label || matches[0].id} first; ${who} chose ${p.label || p.id} instead.` }));
    }
    const why = p && p.explanation && outcome !== "failed" ? h("p", { class: "card-why", text: String(p.explanation) }) : null;
    let by = null;
    if (all.length) {
      if (outcome === "matched" || outcome === "check") by = `Chosen by ${who} from ${count} found by xwalk.`;
      else if (outcome === "declined") by = `${who} looked at ${count} and declined to pick one.`;
      else if (outcome === "unsure") by = `${who} looked at ${count} and was not sure that any of them fits.`;
      else if (outcome === "failed") by = `The model did not finish choosing a term. xwalk found ${count}.`;
      else if (outcome === "search") by = `The top of ${count} found by xwalk. No model chose.`;
    }
    const byLine = by ? h("p", { class: "card-by", text: by }) : null;
    const lines = outcome === "declined" || outcome === "unsure" ? [byLine, why] : [why, byLine];
    for (const line of lines) if (line) detail.append(line);

    if (all.length) {
      const tbody = h("tbody");
      const extra = [];
      all.forEach((m, i) => {
        const href = safeHref(m.url);
        const chosen = m.id === picked;
        const row = h(
          "tr",
          { class: chosen ? "is-picked" : null },
          h("td", { class: "cand-rank", text: String(i + 1) }),
          h("td", { class: "cand-label" }, chosen ? h("span", { class: "cand-check", "aria-hidden": "true", text: "✓ " }) : null, m.label || "-", chosen ? h("span", { class: "visually-hidden", text: " (chosen)" }) : null),
          h("td", { class: "cand-id" }, href ? h("a", { href, target: "_blank", rel: "noopener noreferrer", text: m.id }) : h("span", { text: m.id })),
          h("td", { class: "cand-score", text: typeof m.score === "number" ? m.score.toFixed(4) : "-" }),
          h("td", { class: "cand-ranks", text: retrieverText(m.retrievers) }),
        );
        // Beyond the first few only the chosen one is listed until asked for.
        if (i >= matches.length && !chosen) {
          row.hidden = true;
          extra.push(row);
        }
        tbody.append(row);
      });
      detail.append(
        h(
          "table",
          { class: "cands" },
          h("caption", { class: "visually-hidden", text: `Candidate terms for ${e.text}, as the search ranked them` }),
          h(
            "thead",
            null,
            h("tr", null, h("th", { scope: "col", class: "cand-rank", text: "#" }), h("th", { scope: "col", text: "Label" }), h("th", { scope: "col", text: "ID" }), h("th", { scope: "col", class: "cand-score", text: "Fused score" }), h("th", { scope: "col", class: "cand-ranks", text: "Search ranks" })),
          ),
          tbody,
        ),
      );
      if (extra.length) {
        const showAll = `Show all ${all.length} candidates`;
        const toggle = h("button", { type: "button", class: "ghost small cands-toggle", "aria-expanded": "false", text: showAll });
        toggle.addEventListener("click", () => {
          const open = toggle.getAttribute("aria-expanded") !== "true";
          for (const r of extra) r.hidden = !open;
          toggle.setAttribute("aria-expanded", open ? "true" : "false");
          toggle.textContent = open ? "Show fewer" : showAll;
        });
        detail.append(toggle);
      }
    }
    return detail;
  }

  // One entity: collapsed it is the answer (mention, term, ID); the evidence is behind
  // one button. The type is named in words on the first line, not by colour alone.
  function renderCard(e) {
    const t = typeLabel(e.type);
    const outcome = outcomeOf(e);
    const p = e.pick && typeof e.pick === "object" ? e.pick : null;
    const matches = Array.isArray(e.matches) ? e.matches : [];
    const all = matches.length + (Array.isArray(e.more) ? e.more.length : 0);
    const ontologyName = e.ontology ? state.ontologyNames[e.ontology] || e.ontology : null;
    const term = outcome === "matched" || outcome === "check" ? p : outcome === "search" ? matches[0] : null;

    const top = h("div", { class: "card-top" }, h("span", { class: "card-kind", text: ontologyName ? `${TYPE_NAMES[t]} · ${ontologyName}` : TYPE_NAMES[t] }));
    if (isOverride(e)) top.append(h("span", { class: "card-tag", text: "not the top search hit" }));
    if (outcome === "check") top.append(h("span", { class: "card-tag", text: "needs a check" }));
    if (term === p && p && typeof p.confidence === "number") {
      top.append(
        h(
          "span",
          { class: "card-conf", title: CONFIDENCE_HELP },
          h("span", { "aria-hidden": "true", text: p.confidence.toFixed(2) }),
          h("span", { class: "visually-hidden", text: `${CONFIDENCE_HELP}: ${p.confidence.toFixed(2)}` }),
        ),
      );
    }

    const main = h("div", { class: "card-main" }, h("h3", { class: "card-title", id: `card-title-${e.id}`, text: e.text }), h("span", { class: "card-arrow", "aria-hidden": "true", text: "→" }));
    if (term) {
      const href = safeHref(term.url);
      main.append(
        h("span", { class: "visually-hidden", text: " is " }),
        href ? h("a", { class: "card-id", href, target: "_blank", rel: "noopener noreferrer", text: term.id }) : h("span", { class: "card-id", text: term.id }),
      );
      if (term.label) main.append(h("span", { class: "card-label", text: term.label }));
    } else {
      main.append(h("span", { class: "card-none", text: OUTCOME_TEXT[outcome] }));
    }

    const card = h("article", { class: `card type-${t} is-${outcome}`, id: `card-${e.id}`, "aria-labelledby": `card-title-${e.id}`, tabindex: "-1" }, top, main);

    if (outcome === "unsearched") {
      // Only where it is true: a sample catalog does map genes.
      const ontologies = (state.health && state.health.ontologies) || [];
      const genes = ontologies.length > 0 && !ontologies.some((o) => o.type === "gene");
      card.append(h("p", { class: "card-note", text: "The model marked this mention, but this demo has no ontology for its type." + (genes ? " Genes, for example, are found but not mapped here." : "") }));
    } else if (outcome === "none") {
      const full = state.health && state.health.neon && state.health.neon.catalog === "full";
      card.append(h("p", { class: "card-note", text: full ? "The search found no term for this one." : "No term in this 200-term sample matched. The real ontology may well contain it." }));
    }
    const notes = (Array.isArray(e.notes) ? e.notes : []).map(entityNote).filter(Boolean);
    for (const n of new Set(notes)) card.append(h("p", { class: "card-note", text: n }));

    const hasWhy = Boolean(p && p.explanation && outcome !== "failed");
    if (all || hasWhy) {
      const label = hasWhy && all ? `Why, and ${plural(all, "candidate")}` : hasWhy ? "Why" : plural(all, "candidate");
      const toggle = h("button", { type: "button", class: "card-more", "aria-expanded": "false", "aria-controls": `card-detail-${e.id}` }, h("span", { class: "card-more-text", text: label }));
      const detail = renderDetail(e, outcome);
      detail.hidden = true;
      toggle.addEventListener("click", () => {
        setCardOpen(card, toggle.getAttribute("aria-expanded") !== "true");
        syncExpandAll();
      });
      card.append(toggle, detail);
    }
    return card;
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

  // "Run details" names what actually ran: the extraction model, and the model (or
  // models, after a step-down) that chose the terms.
  function renderRanOn(data, pickers) {
    clear(el.ranOn);
    if (data.mode === "offline") {
      el.ranOn.append("Extracted by the offline stand-in.");
    } else if (typeof data.model === "string" && data.model) {
      el.ranOn.append("Extracted by ", h("strong", { text: modelName(data.model) }));
      if (data.credential_source === "visitor" && typeof data.provider === "string" && data.provider && data.provider !== "site") {
        el.ranOn.append(` on your ${providerName(data.provider)} key`);
      }
      el.ranOn.append(".");
    }
    if (pickers.length) {
      el.ranOn.append(" Terms picked by ");
      pickers.forEach((who, i) => {
        if (i) el.ranOn.append(i === pickers.length - 1 ? " and " : ", ");
        el.ranOn.append(h("strong", { text: selectorName(who) }));
      });
      el.ranOn.append(".");
    } else if (data.mode !== "offline") {
      el.ranOn.append(" No model picked the terms: each entity shows its top search result.");
    }
    el.ranOn.hidden = !el.ranOn.firstChild;
  }

  function fmtShort(ms) {
    return `${(ms / 1000).toFixed(1)} s`;
  }

  // The engineering read-out, folded away after the cards. The search's own words
  // (BM25, embeddings, the operator's reason for a failure) belong here and nowhere else.
  function renderRunDetails(data, pickers) {
    renderRanOn(data, pickers);
    clear(el.runDetailsList);
    const row = (name, value) => el.runDetailsList.append(h("dt", { text: name }), h("dd", { text: value }));
    const names = { bm25: "keyword (BM25)", dense: "embeddings (dense)" };
    const search = Object.entries(data.retrieval || {}).map(([name, st]) => {
      const ok = st && st.status === "ok";
      return `${names[name] || name} ${ok ? "ok" : "unavailable"}${!ok && st && st.reason ? ` (${st.reason})` : ""}`;
    });
    if (search.length) row("Search", search.join(", "));
    const usage = data.usage;
    if (usage && typeof usage === "object") {
      if (typeof usage.llm_calls === "number") {
        row("Model calls", `${usage.llm_calls} for extraction` + (usage.selector_calls ? `, ${usage.selector_calls} for picking` : ""));
      }
      const tokens = [];
      const num = (v) => typeof v === "number";
      if (num(usage.prompt_tokens) && num(usage.completion_tokens)) tokens.push(`${usage.prompt_tokens.toLocaleString("en")} in, ${usage.completion_tokens.toLocaleString("en")} out (extraction)`);
      if (num(usage.selector_prompt_tokens) && num(usage.selector_completion_tokens)) tokens.push(`${usage.selector_prompt_tokens.toLocaleString("en")} in, ${usage.selector_completion_tokens.toLocaleString("en")} out (picking)`);
      if (tokens.length) row("Tokens", tokens.join("; "));
      if (num(usage.embedding_requests)) row("Embedding requests", String(usage.embedding_requests));
      if (num(usage.selector_cost_usd)) row("Picking cost", "$" + usage.selector_cost_usd.toFixed(5));
    }
    const timing = data.timing_ms || {};
    const parts = [];
    if (typeof timing.extract === "number") parts.push(`extract ${fmtSeconds(timing.extract)}`);
    if (typeof timing.map === "number") parts.push(`search ${fmtSeconds(timing.map)}`);
    if (typeof timing.select === "number") parts.push(`pick ${fmtSeconds(timing.select)}`);
    if (typeof timing.total === "number") parts.push(`total ${fmtSeconds(timing.total)}`);
    if (parts.length) row("Timing", parts.join(", "));
    el.runDetails.hidden = !el.ranOn.firstChild && !el.runDetailsList.firstChild;
  }

  // One step-down, in the page's own words. null for a reason this page does not know.
  function fallbackSentence(f) {
    if (!f || typeof f !== "object" || typeof f.from !== "string" || typeof f.to !== "string") return null;
    const from = modelName(f.from);
    const to = modelName(f.to);
    const then = f.stage === "select" ? `${to} chose the terms for the remaining mentions` : `${to} read your text instead`;
    switch (f.reason) {
      case "opus_allowance_used":
        return `The free runs on Claude Opus for your network are used for now, so this ran on ${to}.`;
      case "claude_unavailable":
        return `Claude is not available right now, so this ran on ${to}.`;
      case "limits_unavailable":
        return `The site could not count free runs on ${from}, so this ran on ${to}.`;
      case "rate_limited":
      case "overloaded":
      case "call_limit":
        return f.stage === "select" ? `${from} was busy part of the way through, so ${then}.` : `${from} was busy, so ${then}.`;
      case "transient":
      case "timeout":
        return `${from} did not answer in time, so ${then}.`;
      case "refused":
        return `${from} declined this text, so ${then}.`;
      default:
        return null;
    }
  }

  // "About this run": a different model answered, or part of the search was missing.
  // The sentences are written here from `fallbacks`; the server's own `notices` are
  // used as they are when a step-down has a reason this page does not know.
  function renderNotices(data, searched) {
    clear(el.noticesList);
    const served = (Array.isArray(data.notices) ? data.notices : []).filter((n) => typeof n === "string" && n.trim());
    const mine = (Array.isArray(data.fallbacks) ? data.fallbacks : []).map(fallbackSentence);
    const lines = mine.length && mine.every(Boolean) ? mine : served.length ? served : mine.filter(Boolean);
    const retrieval = data.retrieval || {};
    if (searched && retrieval.dense && retrieval.dense.status !== "ok") {
      lines.push("Part of the search (matching by meaning) was unavailable for this run, so terms were found by keyword only. Synonyms and paraphrases may be missed.");
    }
    if (searched && retrieval.bm25 && retrieval.bm25.status !== "ok") {
      lines.push("Part of the search (matching by keyword) was unavailable for this run, so terms were found by meaning only.");
    }
    for (const line of new Set(lines)) el.noticesList.append(h("li", { text: line }));
    el.notices.hidden = !el.noticesList.firstChild;
  }

  // The server's warnings in plain words. Their wording is held by the API's tests; one
  // this page does not know is shown as it is. null: said elsewhere on the page.
  function warningText(w) {
    let m = /^entity '(.*)' dropped: not found in text$/.exec(w);
    if (m) return `The model also named "${m[1]}", which does not appear in your text word for word, so it was left out.`;
    m = /^entity '(.*)' dropped: overlaps '(.*)'$/.exec(w);
    if (m) return `The model also named "${m[1]}", which overlaps "${m[2]}", so it was left out.`;
    m = /^(\d+) entities found; only the first (\d+) are kept$/.exec(w);
    if (m) return `The model found ${m[1]} mentions; only the first ${m[2]} are shown.`;
    m = /^dense search failed for (\d+) of (\d+) entities/.exec(w);
    if (m) return `Matching by meaning failed for ${m[1]} of ${m[2]} mentions; those were found by keyword only.`;
    if (/^(dense|bm25) retrieval unavailable/.test(w)) return null; // "About this run" says it
    return w;
  }

  const COUNT_TEXT = [
    ["matched", (n) => `${n} matched`],
    ["check", (n) => `${n} to check`],
    ["declined", (n) => `${n} where no term fits`],
    ["failed", (n) => `${n} that could not be matched`],
    ["search", (n) => `${n} with a top search result`],
    ["none", (n) => `${n} with no term found`],
    ["unsearched", (n) => `${n} not looked up`],
  ];

  // Counts that add up to the number found, in the words the cards use.
  function summaryText(outcomes) {
    const count = (k) => outcomes.filter((o) => o === k).length;
    const parts = [];
    for (const [key, text] of COUNT_TEXT) {
      const n = key === "check" ? count("check") + count("unsure") : count(key);
      if (n) parts.push(text(n));
    }
    return `${plural(outcomes.length, "mention")} found: ${parts.join(", ")}.`;
  }

  // What the model added to the search, said once and only when it is true.
  function takeawayText(entities, outcomes, pickers) {
    const who = pickers.length === 1 ? selectorName(pickers[0]) : "the model";
    const chosen = outcomes.filter((o) => o === "matched" || o === "check").length;
    const overrides = entities.filter(isOverride).length;
    const declined = outcomes.filter((o) => o === "declined").length;
    const over = overrides ? `in ${overrides} of ${plural(chosen, "match", "matches")} ${who} chose a different term than the top search hit` : "";
    const none = declined ? `for ${plural(declined, "mention")} ${overrides ? "it" : who} found no fitting term rather than guess` : "";
    if (over || none) {
      const s = over && none ? `${over}, and ${none}.` : `${over || none}.`;
      return s.charAt(0).toUpperCase() + s.slice(1);
    }
    if (!pickers.length && outcomes.includes("search")) return "No model chose here: each mention shows its top search result.";
    return "";
  }

  function entityTypesText() {
    const names = { chemical: "chemicals", disease: "diseases", gene: "genes", food: "foods" };
    const types = ((state.health && state.health.ontologies) || []).map((o) => names[o.type] || String(o.type));
    if (!types.length) return "chemicals, diseases or foods";
    return types.length < 2 ? types[0] : `${types.slice(0, -1).join(", ")} or ${types[types.length - 1]}`;
  }

  // The closing panel: what the visitor just watched, and where the library is. Only
  // what is true of this result is said: how many terms were searched (when known), by
  // which kinds of search, and whether a model chose.
  function renderNext(data, entities, pickers) {
    const searched = entities.some((e) => e.ontology);
    el.next.hidden = !searched;
    if (!searched) return;
    const total = state.ontologies.reduce((sum, o) => sum + (Number(o.count) || 0), 0);
    const terms = total > 0 ? `${total.toLocaleString("en")} ontology terms` : "the ontology terms";
    const retrieval = data.retrieval || {};
    const off = (name) => Boolean(retrieval[name]) && retrieval[name].status !== "ok";
    const how = off("dense") ? "by keyword" : off("bm25") ? "by meaning" : "by keyword and by meaning";
    const chose = pickers.length ? "and the model chose among real candidates" : "and each mention shows the top search result";
    el.nextBody.textContent =
      `The model never writes an ID. xwalk searched ${terms} ${how}, ${chose}. ` +
      "xwalk is an open-source Python library; you can run the same lookup on your own vocabulary.";
  }

  // `opts.example`: this is the recorded example, not a run. It gets its own heading and
  // label, no timing sentence, and leaves the banner alone.
  function renderResults(text, data, opts) {
    const example = Boolean(opts && opts.example);
    const entities = Array.isArray(data.entities) ? data.entities : [];
    const outcomes = entities.map(outcomeOf);
    const pickers = pickersOf(entities);
    const none = entities.length === 0;
    const skipped = renderHighlighted(text, entities);
    renderLegend(entities);

    clear(el.cards);
    for (const e of entities) el.cards.append(renderCard(e));
    // One piece of reasoning is on show without a tap: the first card where the model
    // overruled the search or declined every candidate.
    const first = entities.findIndex((e, i) => isOverride(e) || outcomes[i] === "declined" || outcomes[i] === "unsure");
    if (first >= 0) setCardOpen(el.cards.children[first], true);
    syncExpandAll();

    clear(el.warnings);
    const warnings = (Array.isArray(data.warnings) ? data.warnings : []).map((w) => warningText(String(w))).filter(Boolean);
    if (skipped) warnings.push(`${plural(skipped, "mention")} could not be highlighted in the text.`);
    for (const w of new Set(warnings)) el.warnings.append(h("li", { text: w }));
    el.warningsBox.hidden = !el.warnings.firstChild;

    // No entities: one panel that says what the demo looks for, in place of an empty
    // text panel and a legend for colours that are not used.
    el.noEntities.hidden = !none;
    el.noEntitiesTitle.textContent = `No ${entityTypesText()} found in this text.`;
    el.textPanel.hidden = none;
    el.summary.hidden = none;
    el.status.hidden = none || example;
    el.summary.textContent = none ? "" : summaryText(outcomes);

    // The model is asked to stop at the cap, so a full list may not be the whole text.
    const cap = state.limits.max_entities;
    const capped = typeof cap === "number" && entities.length >= cap;
    el.cap.textContent = capped ? `${cap} is the most one run returns, so a long text may hold more mentions than are shown.` : "";
    el.cap.hidden = !capped;

    const takeaway = takeawayText(entities, outcomes, pickers);
    el.takeaway.textContent = takeaway;
    el.takeaway.hidden = !takeaway;

    const timing = data.timing_ms || {};
    let done = "";
    if (typeof timing.total === "number") {
      const steps = [];
      if (typeof timing.extract === "number") steps.push(`${fmtShort(timing.extract)} reading the text`);
      if (typeof timing.map === "number") steps.push(`${fmtShort(timing.map)} searching`);
      if (typeof timing.select === "number") steps.push(`${fmtShort(timing.select)} choosing terms`);
      done = `Done in ${fmtShort(timing.total)}` + (steps.length ? `: ${steps.join(", ")}.` : ".");
    }
    // "Done in 8.8 s" would read as the visitor's own wait.
    el.status.textContent = example ? "" : done;

    el.results.setAttribute("data-kind", example ? "example" : "run");
    el.resultsHeading.textContent = example ? "Example result" : "Results";
    el.exampleNote.textContent = example ? `Recorded from a real run on ${state.example.recorded}.` : "";
    el.exampleNote.hidden = !example;

    renderNotices(data, entities.some((e) => e.ontology));
    renderRunDetails(data, pickers);
    el.runDetailsSummary.textContent = example ? "Details of the recorded run" : "Run details";
    el.runDetails.open = false;
    renderNext(data, entities, pickers);
    // Before a run the banner speaks for the service; after one, the run's own search
    // status is said beside its results ("About this run"). The example is not a run.
    if (!example) renderBanner(data.mode, state.health && state.health.retrieval);
    state.shown = { text, data, example };
    el.results.hidden = false;
    syncStale();
  }

  // A result belongs to the text it was made from. Once the box holds another text the
  // result stays, dimmed, under one line that says so.
  function syncStale() {
    const stale = Boolean(state.shown) && !el.results.hidden && el.text.value !== state.shown.text;
    el.results.classList.toggle("is-stale", stale);
    el.stale.textContent =
      state.shown && state.shown.example
        ? 'This result is for the example text. Press "Map this text" to map yours.'
        : "These results are for the earlier text. Run again to map the text that is in the box now.";
    el.stale.hidden = !stale;
  }

  // ---------- run ----------

  function setRunning(on) {
    if (state.running === on) return;
    state.running = on;
    syncRun();
    el.run.setAttribute("aria-busy", on ? "true" : "false");
    el.results.setAttribute("aria-busy", on ? "true" : "false");
    el.run.querySelector(".run-label").textContent = on ? "Mapping…" : "Map this text";
    showProgress(on);
    if (on) holdScreen();
    else releaseScreen();
  }

  // A phone that locks its screen during a run suspends the page and can drop the
  // request. Where the browser allows it the screen is kept on for the run; where it
  // does not, nothing happens here and a dropped run is explained when it fails.
  let wakeLock = null;

  async function holdScreen() {
    if (!navigator.wakeLock || typeof navigator.wakeLock.request !== "function" || document.visibilityState !== "visible") return;
    try {
      const lock = await navigator.wakeLock.request("screen");
      if (state.running && !wakeLock) wakeLock = lock;
      else lock.release().catch(() => {});
    } catch (_) {
      /* not allowed here: the run goes on without it */
    }
  }

  function releaseScreen() {
    if (!wakeLock) return;
    wakeLock.release().catch(() => {});
    wakeLock = null;
  }

  // What a visitor sees while a run is under way: the three steps with what each one
  // does, a moving bar and a clock, so half a minute of waiting does not look like a
  // page that hung. No step is marked done by the clock: the API answers once, at the
  // end. The clock is hidden from screen readers; they are told when a run starts, that
  // it is still going after 20 seconds, and when it is slow.
  const PROGRESS_USUAL = "This usually takes 10 to 30 seconds. Keep this page open" + (touch ? " and the screen on." : ".");
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
    let said = 0;
    const tick = () => {
      const s = Math.floor((performance.now() - started) / 1000);
      el.progressTime.textContent = `${s} s`;
      el.progressHint.textContent = s >= 45 ? PROGRESS_SLOW : PROGRESS_USUAL;
      if (s >= 45 && said < 45) announce(PROGRESS_SLOW);
      else if (s >= 20 && said < 20) announce("Still mapping your text.");
      if (s >= 20) said = s >= 45 ? 45 : 20;
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

  function useSite() {
    forgetKey();
    rememberPayer("site");
    el.provider.value = "site";
    if (state.health) applyProvider();
    releaseRuns();
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
    if (state.textError === "server") clearTextError();
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
    clearKeyError();
    if (provider) {
      const typed = el.ownKey.value.trim();
      if (!typed) {
        const back = h("button", { type: "button", class: "linklike", text: "switch back to the free runs" });
        back.addEventListener("click", () => {
          useSite();
          el.provider.focus();
        });
        showKeyError(`Paste your ${provider.name} API key, or `, back, ".");
        el.ownKey.focus();
        return;
      }
      if (!KEY_SHAPE.test(typed)) {
        showKeyError("This does not look like an API key. Paste the whole key, with no spaces.");
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

    const ctx = { provider, started: performance.now(), hidden: document.visibilityState === "hidden", edited: false };
    state.runCtx = ctx;
    state.ran = true;
    setRunning(true);
    // Said once for screen readers; sighted visitors have the steps and the button.
    announce("Mapping your text. This usually takes 10 to 30 seconds.");
    try {
      // The key is read here, for this call, and held nowhere else.
      const data = await api("POST", "/api/ner", payload, provider ? el.ownKey.value.trim() : undefined);
      renderResults(text, data);
      setRunning(false);
      const said = el.noEntities.hidden ? el.summary.textContent : el.noEntitiesTitle.textContent;
      announce(`Done. ${said}` + (el.stale.hidden ? "" : " These results are for the text as it was when you ran it."));
      // A visitor who went on typing keeps the caret and the scroll position.
      if (!ctx.edited) {
        el.resultsHeading.focus({ preventScroll: true });
        revealResults();
      }
    } catch (err) {
      setRunning(false);
      announce("");
      // The text box and the last good result stay as they are.
      renderBanner(state.health && state.health.mode, state.health && state.health.retrieval);
      showError(err, ctx);
    } finally {
      setRunning(false);
      state.runCtx = null;
    }
    refreshHealth();
  }

  // ---------- wiring ----------

  function loadExample() {
    state.touched = true;
    el.text.value = exampleText();
    clearTextError();
    updateCounter();
    syncStale();
    if (touch) {
      // No keyboard: the next step is the button, so that is what comes into view.
      el.run.focus({ preventScroll: true });
      reveal(el.run, "nearest");
    } else el.text.focus();
  }

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
    state.touched = true;
    if (state.textError === "empty" || state.textError === "server") clearTextError();
    if (state.runCtx) state.runCtx.edited = true;
    updateCounter();
    syncStale();
  });
  el.prompt.addEventListener("input", () => showPromptError(""));
  el.useOwnKey.addEventListener("click", useOwnKey);
  document.addEventListener("visibilitychange", () => {
    const visible = document.visibilityState === "visible";
    if (state.running) {
      // The request may not survive the background; if it fails, the message says so.
      if (!visible && state.runCtx) state.runCtx.hidden = true;
      // The browser lets go of the screen lock when the page is hidden.
      if (visible) {
        wakeLock = null;
        holdScreen();
      }
      return;
    }
    // A tab that slept through a timed wait learns that the wait is over.
    if (visible && allowanceTimer !== null) refreshHealth();
  });
  el.model.addEventListener("change", renderAllowance);
  el.selector.addEventListener("change", renderAllowance);
  el.example.addEventListener("click", loadExample);
  el.noEntitiesExample.addEventListener("click", loadExample);
  el.clear.addEventListener("click", () => {
    state.touched = true;
    el.text.value = "";
    clearTextError();
    updateCounter();
    syncStale();
    if (!touch) el.text.focus();
  });
  el.expandAll.addEventListener("click", () => {
    const open = el.expandAll.getAttribute("aria-expanded") !== "true";
    for (const card of el.cards.children) setCardOpen(card, open);
    syncExpandAll();
  });
  // "Try your own text": to the text box, ready to type. An example still in the box is
  // selected, so typing or pasting replaces it.
  el.nextTry.addEventListener("click", () => {
    el.text.focus({ preventScroll: true });
    if (el.text.value === EXAMPLE_FULL || el.text.value === EXAMPLE_TEXT) el.text.select();
    reveal(el.form, "start");
  });
  // A browser that puts typed text back after the scripts ran (back and forward) fires
  // no input event; the counter and the "earlier text" line are brought up to date.
  window.addEventListener("pageshow", () => {
    updateCounter();
    syncStale();
  });
  el.promptReset.addEventListener("click", () => {
    el.prompt.value = state.defaultPrompt;
    showPromptError("");
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
    useSite();
    el.run.focus();
  });
  el.payerRetry.addEventListener("click", () => {
    hideError();
    loadHealth();
  });
  el.ownKey.addEventListener("input", () => {
    clearKeyError();
    rememberKey();
  });
  el.forgetKey.addEventListener("click", () => {
    forgetKey();
    clearKeyError();
    el.ownKey.focus();
  });
  el.errorAction.addEventListener("click", () => {
    hideError();
    run();
  });
  el.errorOwnKey.addEventListener("click", () => {
    hideError();
    useOwnKey();
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
  loadExampleResult();
})();
