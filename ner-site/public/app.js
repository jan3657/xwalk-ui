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
  // Beside the provider id it holds the model and the selector chosen for that key
  // ({provider, model, selector}; ids only), so that a reload does not quietly move the
  // visitor's bill to another model. A plain provider id, as written before, is read too.
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
    runPayer: $("run-payer"),
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
    pipeDecide: $("pipe-decide"),
    guideToggle: $("guide-toggle"),
    guideChips: $("guide-chips"),
    guidePlay: $("guide-play"),
    guidePlayIcon: $("guide-play-icon"),
    guidePlayLabel: $("guide-play-label"),
    guideHint: $("guide-hint"),
    guidePanel: $("guide-panel"),
    guideToken: $("guide-token"),
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
    // Which used-up limit the error box is telling of ("hour", "day", "own",
    // "unavailable"), or false while it tells of none.
    limitNotice: false,
    // A health answer that arrived during a run: { health, load, expire }. It is applied
    // when the run ends (settleHealth), so that who pays and which models run stay, on
    // the page, what the run was submitted with.
    heldHealth: null,
    // The refusal now shown at the key field: { err, provider, model } of the run it is
    // about. If health then takes that provider away, the refusal moves to the error box.
    keyError: null,
    // The intro is the production copy of index.html (a full catalog and a real model).
    ledeStatic: false,
    // The run under way: { provider, model, selector, started, startedAt, hidden,
    // edited, abort }. The first three are what was submitted: every message about the
    // run is written from them, never from what the form shows by then. `hidden` once
    // the page went to the background during it, `edited` once the visitor typed, or
    // used Example or Clear, during it. `abort` cancels the request.
    runCtx: null,
    // A page brought back from the back-forward cache during a run: who pays is checked
    // against session storage once that run has ended.
    payerCheckDue: false,
    // Set while it is not settled who pays for the next run: a key kept from earlier
    // in this tab has not been restored yet ("pending"), could not be because health
    // did not load ("unknown"), is for a provider the site no longer offers ("lost"),
    // or a page brought back from the browser's history could not read what the tab
    // saved ("unread"). Nothing is submitted while it is set; see holdRuns().
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
  // `hook`, when given, receives `abort`: a way to cancel this request from outside.
  async function api(method, path, payload, bearer, hook) {
    const headers = { "X-Requested-With": "xwalk-ner" };
    if (payload !== undefined) headers["Content-Type"] = "application/json";
    const pass = storageGet(PASSCODE_KEY);
    if (pass) headers["X-Passcode"] = pass;
    if (bearer) {
      if (method !== "POST" || path !== "/api/ner") throw new ApiError("Internal error: a key was attached to the wrong request.", "internal", 0);
      headers.Authorization = `Bearer ${bearer}`;
    }

    const controller = new AbortController();
    // Armed until the body has been read: an answer whose headers arrive and whose body
    // then stalls is cancelled like one that never answers.
    const timer = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);
    if (hook) hook.abort = () => controller.abort();
    let response;
    let raw = "";
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
      clearTimeout(timer);
      const aborted = err && err.name === "AbortError";
      throw new ApiError(
        aborted
          ? "The request took too long and was cancelled. Try a shorter text."
          : "Network error: the server could not be reached.",
        aborted ? "timeout" : "network",
        0,
      );
    }
    try {
      raw = await response.text();
    } catch (_) {
      // The answer had started and did not finish: cut off, or stalled until the timer
      // cancelled it. Either way the server may have done the work.
      const stalled = controller.signal.aborted;
      const cut = new ApiError(
        stalled ? "The answer stopped arriving and the request was cancelled." : "The connection was lost while the answer was arriving.",
        stalled ? "timeout" : "network",
        response.status,
      );
      cut.interrupted = true;
      throw cut;
    } finally {
      clearTimeout(timer);
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
            limit: "own",
            title: "Too many runs with your own key.",
            body: [`Runs with an own key are limited per network too. ${wait ? `Try again in ${minutesText(wait)}.` : "Try again later."}`],
          };
        }
        if (/today/.test(message)) {
          const at = wait ? new Date(Date.now() + wait * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }) : null;
          return {
            look: "notice",
            limit: "day",
            title: "The free runs for your network today are used.",
            body: [at ? `They come back at ${at} (midnight UTC).` : "They come back at midnight UTC."],
            ownKey: offerKey,
          };
        }
        return {
          look: "notice",
          limit: "hour",
          title: "The free runs for your network this hour are used.",
          body: [wait ? `You can run again in ${minutesText(wait)}.` : "You can run again later this hour."],
          ownKey: offerKey,
        };
      case "limits_unavailable":
        return {
          look: "notice",
          limit: "unavailable",
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
        if (err.interrupted) {
          return {
            look: "alert",
            title: "The result did not finish arriving.",
            body: ["The connection stalled part of the way through. Your text is still here. The interrupted run may still have been counted."],
            action: "Run again",
          };
        }
        return {
          look: "alert",
          title: "The run took too long and was stopped.",
          body: ["Long texts with many mentions can run out of time. Try again, or try a shorter text. The stopped run may still have been counted."],
          action: "Try again",
        };
      case "network": {
        // A request that died late, part of the way through its answer, or while the
        // page was in the background, was most likely cut off by the phone, not refused
        // by the network.
        const late = Boolean(err.interrupted) || (Boolean(ctx) && (ctx.hidden || performance.now() - ctx.started > 3000));
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

  // What is said at the key field for an error about the visitor's own key. `provider`
  // and `model` are the ones the failed run was submitted with.
  function keyErrorText(err, provider, model) {
    if (err.code === "provider_key_refused") {
      return `${provider.name} did not accept this key. Check that it was pasted whole, is still active, has credit, and may use ${modelName(model)}.`;
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
    state.keyError = null;
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
      if (e.code === "text_too_long" && n > state.limits.max_chars) showTextError("long", tooLongText());
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
    // The provider the run was submitted with, not the one the form shows now.
    const provider = ctx ? ctx.provider : currentProvider();
    if (KEY_CODES.includes(e.code) && provider) {
      const refused = { err: e, provider, model: ctx ? ctx.model : el.model.value };
      const shown = currentProvider();
      if (shown && shown.id === provider.id && !state.payer) {
        // A key the server or the provider refused stays in the field, to be corrected.
        showKeyError(keyErrorText(e, provider, refused.model));
        state.keyError = refused;
        if (!calm) el.ownKey.focus();
      } else {
        // The key field is not that provider's any more (or is not on show): the
        // refusal is said in the error box, with the provider's name.
        showRunKeyError(refused);
        if (!calm) reveal(el.error, "nearest");
      }
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
    state.limitNotice = d.limit || false;
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

  // A refusal of a visitor's own key, said in the error box: for a run whose provider
  // the form no longer shows, where the key field is hidden or belongs to another one.
  function showRunKeyError(refused) {
    el.error.classList.remove("notice");
    el.errorTitle.textContent = `The run on your ${refused.provider.name} key did not go through.`;
    clear(el.errorMsg);
    el.errorMsg.append(keyErrorText(refused.err, refused.provider, refused.model));
    el.errorAction.hidden = true;
    el.errorOwnKey.hidden = true;
    el.error.hidden = false;
    state.limitNotice = false;
    renderAllowance();
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

  // The same sentence whatever the length: the line is an alert, and a number in it
  // would be read out again on every keystroke. The counter beside it has the numbers.
  function tooLongText() {
    return `This text is over the ${state.limits.max_chars.toLocaleString("en")} character limit. Shorten it to map it.`;
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
          // Not "you can still try a run" while runs are held for an own-key choice.
          h("p", null, h("strong", { text: "Could not reach the service." }), state.payer ? " Runs are on hold: see the note under the button." : " You can still try a run; some options are missing until the page reloads."),
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
    const denseOff = Boolean(retrieval && retrieval.dense && retrieval.dense.status !== "ok");
    const bm25Off = Boolean(retrieval && retrieval.bm25 && retrieval.bm25.status !== "ok");
    if (denseOff && bm25Off) {
      items.push(["The search is unavailable.", " Mentions can be found in your text right now, but they cannot be looked up."]);
    } else if (denseOff) {
      items.push(["Part of the search is unavailable.", " Matching by meaning is off right now, so terms are found by keyword only. Synonyms and paraphrases may be missed."]);
    } else if (bm25Off) {
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
    el.counter.textContent = `${n.toLocaleString()} / ${max.toLocaleString()} characters` + (n > max ? `, ${(n - max).toLocaleString()} over` : "");
    el.counter.classList.toggle("over", n > max);
    // Said as soon as the text passes the limit, not only after Run, and said once: the
    // alert is not rewritten while the text stays over the limit.
    if (n > max) {
      if (state.textError !== "long") showTextError("long", tooLongText());
    } else if (state.textError === "long") clearTextError();
    el.textHelp.textContent = `Up to ${max.toLocaleString("en")} characters and ${state.limits.max_entities} mentions per run.`;
  }

  function fillTopK(max) {
    const current = Number(el.topK.value) || 5;
    clear(el.topK);
    for (let k = 1; k <= max; k += 1) el.topK.append(h("option", { value: k, text: String(k) }));
    el.topK.value = String(Math.min(current, max));
  }

  // The first answer fills the form, from session storage where a choice was saved. An
  // answer applied after that (a retry) keeps what the visitor has chosen since,
  // wherever the new answer still offers it. While a payer hold is up, storage decides
  // (except a hold put up because storage could not be read).
  function applyHealth(health) {
    const again = Boolean(state.health) && (!state.payer || state.payer.kind === "unread");
    const before = again ? { provider: el.provider.value, model: el.model.value, selector: el.selector.value } : null;
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

    if (again) reconcileProviders(health);
    else fillProviders(health);

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
    restoreModels(before || savedChoice());
    renderSteps(effectiveSelector());
    updateCounter();
    syncExample();
  }

  // Puts back a model and a selector chosen earlier (before this health answer, or
  // before a reload) when they belong to the payer now shown and are still offered.
  function restoreModels(choice) {
    if (!choice || choice.provider !== el.provider.value) return;
    const offers = (select, value) => Boolean(value) && [...select.options].some((o) => o.value === value);
    if (offers(el.model, choice.model)) el.model.value = choice.model;
    if (offers(el.selector, choice.selector)) el.selector.value = choice.selector;
    renderAllowance();
  }

  // On a page brought back from the browser's history: the model and the selector on
  // record for the own-key provider now shown are the ones the next run uses, also when
  // the provider itself did not change. An id that is not on record, or is no longer
  // offered, gives the provider's default (as a fresh load of the page would).
  function applyStoredModels(choice) {
    const provider = currentProvider();
    if (!provider || !choice || choice.provider !== provider.id) return;
    const offers = (select, value) => Boolean(value) && [...select.options].some((o) => o.value === value);
    el.model.value = offers(el.model, choice.model) ? choice.model : provider.defaultModel;
    el.selector.value = offers(el.selector, choice.selector) ? choice.selector : "retrieval";
    renderAllowance();
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
    return keyOf(storageGet(OWN_KEY_ENTRY));
  }

  // The key entry as it is in session storage, parsed: { provider, key } or null.
  function keyOf(raw) {
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

  // The own-key choice the visitor last made in this tab: { provider, model, selector },
  // or null for the site's models. A key kept from before this entry existed counts as
  // that choice.
  function savedChoice() {
    return choiceOf(storageGet(PAYER_ENTRY), storedKey());
  }

  // The same from the two entries as read: `raw` is the payer entry, `kept` the key.
  function choiceOf(raw, kept) {
    if (raw && raw !== "site") {
      if (raw.charAt(0) !== "{") return { provider: raw, model: "", selector: "" };
      try {
        const c = JSON.parse(raw);
        if (c && typeof c.provider === "string" && c.provider && c.provider !== "site") {
          return { provider: c.provider, model: typeof c.model === "string" ? c.model : "", selector: typeof c.selector === "string" ? c.selector : "" };
        }
      } catch (_) {
        /* not ours: treated as no choice */
      }
    }
    return kept ? { provider: kept.provider, model: "", selector: "" } : null;
  }

  // What may be written back to the payer entry. Session storage is not trusted: only
  // ids this page can name go back into it. For a provider on offer, a model or a
  // selector it offers; for one that is not on offer (a hold), a provider and models
  // this page knows by name. Anything else is written as "other" or left out.
  function cleanChoice(choice) {
    const offered = state.providers.find((p) => p.id === choice.provider);
    const named = (id) => Object.prototype.hasOwnProperty.call(MODEL_NAMES, id);
    if (!offered) {
      const known = Object.prototype.hasOwnProperty.call(PROVIDER_NAMES, choice.provider);
      const keep = (id) => (known && (id === "retrieval" || named(id)) ? id : "");
      return { provider: known ? choice.provider : "other", model: keep(choice.model), selector: keep(choice.selector) };
    }
    return {
      provider: offered.id,
      model: offered.models.includes(choice.model) ? choice.model : "",
      selector: choice.selector === "retrieval" || offered.models.includes(choice.selector) ? choice.selector : "",
    };
  }

  // The provider of that choice ("" for the site's models).
  function savedPayer() {
    const choice = savedChoice();
    return choice ? choice.provider : "";
  }

  // The models saved for a provider stay until the visitor picks others for it.
  function rememberPayer(providerId) {
    if (!providerId || providerId === "site") {
      storageSet(PAYER_ENTRY, "");
      return;
    }
    const before = savedChoice();
    const same = Boolean(before) && before.provider === providerId;
    storageSet(PAYER_ENTRY, JSON.stringify(cleanChoice({ provider: providerId, model: same ? before.model : "", selector: same ? before.selector : "" })));
  }

  // The visitor picked a model or a selector while paying with an own key.
  function rememberModels() {
    const provider = currentProvider();
    if (!provider || state.payer) return;
    storageSet(PAYER_ENTRY, JSON.stringify({ provider: provider.id, model: el.model.value, selector: el.selector.value }));
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
    } else if (kind === "unread") {
      el.payerMsg.textContent =
        `This page came back from the browser's history and cannot read what this tab saved about who pays, so your ${key} was cleared from the page. ` +
        'Nothing has run. Enter the key again, or choose "Use the site\'s models" to run on them (free, limited).';
    } else {
      // Said only once the key is gone from session storage: every caller removes it first.
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
    if (state.healthFailed) renderBanner(null, null);
  }

  function releaseRuns() {
    const held = Boolean(state.payer);
    state.payer = null;
    el.payer.hidden = true;
    el.payerMsg.textContent = "";
    syncRun();
    renderPayerCopy();
    if (held && state.healthFailed) renderBanner(null, null);
  }

  // Run ships disabled in index.html and is enabled from here, once the submit handler
  // is installed. While a run is under way, who pays and which models run cannot be
  // changed: the run was submitted with them, and what the page says about it (the hint,
  // the Options summary, an error at the key field) has to stay true of that run.
  function syncRun() {
    el.run.disabled = state.running || Boolean(state.payer);
    for (const control of [el.provider, el.ownKey, el.forgetKey, el.model, el.selector]) control.disabled = state.running;
  }

  // A page brought back from the back-forward cache is the old document, with the old
  // key still in its field. Meanwhile another document of this tab may have forgotten
  // that key or changed who pays or which models run; session storage, which the tab's
  // documents share, is where that is on record. So before the restored page can
  // submit, the payer and key entries are read, directly:
  // - a read that fails: the key is cleared from the field, and runs are held until the
  //   visitor enters a key again or picks who pays ("unread");
  // - no key on record: the field is emptied, and an own-key run asks for the key;
  // - a payer this page's (older) health answer does not offer: nothing is said to be
  //   forgotten; runs wait for the health read that follows ("pending"), which restores
  //   the choice or removes the key and then says so (settleHealth);
  // - otherwise the payer, its key and its models are put as they are on record.
  // Whether storage can be written plays no part: a tab whose storage is full can still
  // have had its key removed.
  function reconcilePayer() {
    let kept;
    let choice;
    try {
      kept = keyOf(window.sessionStorage.getItem(OWN_KEY_ENTRY));
      choice = choiceOf(window.sessionStorage.getItem(PAYER_ENTRY), kept);
    } catch (_) {
      const shown = currentProvider();
      el.ownKey.value = "";
      clearKeyError();
      if (shown && !state.payer) holdRuns("unread", shown.id);
      return;
    }
    const chosen = choice ? choice.provider : "";
    if (!state.health) {
      // Health is still loading or failed: the key field is not on show, and whether
      // the choice can be used is not known.
      el.ownKey.value = "";
      if (chosen) holdRuns(state.healthFailed ? "unknown" : "pending", chosen);
      else if (state.payer) releaseRuns();
      return;
    }
    if (chosen && !state.providers.some((p) => p.id === chosen)) {
      el.ownKey.value = "";
      if (el.provider.value !== "site") {
        el.provider.value = "site";
        applyProvider();
      }
      holdRuns("pending", chosen);
      return;
    }
    const want = chosen || "site";
    if (el.provider.value !== want) {
      el.provider.value = want;
      applyProvider();
    }
    applyStoredModels(choice);
    const key = chosen && kept && kept.provider === chosen ? kept.key : "";
    if (el.ownKey.value.trim() !== key) {
      el.ownKey.value = key;
      clearKeyError();
    }
    if (state.payer) releaseRuns();
    else renderPayerCopy();
    if (chosen) openOptions();
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
    // A refusal on show at the key field goes with the field; it is said again below.
    const refused = state.keyError;
    forgetKey();
    el.provider.value = "site";
    applyProvider();
    rememberPayer(chosen.id); // still the visitor's choice, until they make another
    holdRuns("lost", chosen.id);
    if (refused) showRunKeyError(refused);
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
        return { blocked: true, limit: "own", wait: secondsLeft(visitor.day_resets_in_s), text: "Too many runs with your own key today. They come back at midnight UTC." };
      }
      if (visitor.own_left === 0) {
        return { blocked: true, limit: "own", wait: secondsLeft(visitor.resets_in_s), text: `Too many runs with your own key; ${tryAgain(secondsLeft(visitor.resets_in_s))}.` };
      }
      return null;
    }
    if (visitor.tier === "blocked") {
      if (visitor.limits === "unavailable") {
        return { blocked: true, limit: "unavailable", text: "The demo is briefly unavailable. Try again in a minute." };
      }
      if (visitor.window === "day") {
        const left = secondsLeft(visitor.day_resets_in_s);
        const at = left === null ? "midnight UTC" : new Date(Date.now() + left * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
        // `wait`: the line is recounted and health read again when the day is over.
        return { blocked: true, limit: "day", wait: left, text: `The free runs for your network today are used. They come back at ${at}.` };
      }
      const left = secondsLeft(visitor.resets_in_s);
      return {
        blocked: true,
        limit: "hour",
        wait: left,
        text: "The free runs for your network this hour are used." + (left === null ? " Try again later." : ` Next one in ${Math.max(1, Math.ceil(left / 60))} min.`),
      };
    }
    // The Sonnet sentences are only true when the free runs are counted, Claude can be
    // reached and the request would use Claude at all.
    const counted = visitor.limits === undefined || visitor.limits === "on";
    const claudeDown = health.claude && health.claude.status === "unavailable";
    // Which step would use Claude: the one that reads the text, or only the choice of
    // terms. "Runs start on Claude Sonnet" is not said of a run another model starts.
    const reads = isClaude(el.model.value);
    const wantsClaude = reads || isClaude(effectiveSelector());
    if (!counted || claudeDown || !wantsClaude) return null;
    if (visitor.tier === "sonnet") {
      const span = visitor.opus_left_today === 0 ? "today" : "this hour";
      return {
        blocked: false,
        sonnet: true,
        text: `The free runs on Claude Opus for your network ${span} are used. ` + (reads ? "Runs now start on Claude Sonnet, a faster model." : "The choice of terms now starts on Claude Sonnet, a faster model."),
      };
    }
    if (visitor.tier === "opus" && visitor.opus_left === 1) {
      return {
        blocked: false,
        text: "1 free run on Claude Opus is left for your network this hour; after that, " + (reads ? "runs start on Claude Sonnet." : "the choice of terms starts on Claude Sonnet."),
      };
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
    // While the error box says the same thing (the same used-up limit, with the same way
    // out), it is not said a second time here. A box about another limit (yesterday's
    // daily one, say, while this hour's is used) does not stand in for this line.
    const said = blocked && state.limitNotice === message.limit;
    el.allowance.hidden = !message || said;
    el.allowanceRow.classList.toggle("blocked", blocked);
    el.allowanceRow.hidden = !message || said;
    // The own-key way out is only offered where it exists and is not already taken.
    el.useOwnKey.hidden = !(blocked && !currentProvider() && state.providers.length > 0);
    // A single model needs no dropdown, unless the visitor is paying for it and should
    // see which model that is.
    const count = el.model.options.length;
    el.modelField.hidden = count === 0 || (count === 1 && !currentProvider());
    syncLede();

    // While a timed wait is shown, the minutes are recounted, and health is read again
    // when the wait is over.
    const timed = Boolean(message && typeof message.wait === "number");
    if (timed && allowanceTimer === null) {
      allowanceTimer = setInterval(() => {
        const now = allowanceMessage();
        if (now && typeof now.wait === "number" && now.wait <= 0) refreshHealth({ expire: true });
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
      summary = state.payer.kind === "lost" || state.payer.kind === "unread" ? "Options: choose who pays" : `Options: your ${key}, waiting`;
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
      ? `Runs on your ${provider.name} key. Use public or de-identified text: it is sent to ${provider.name}.`
      : "Free, no sign-up, usually 10 to 30 seconds. Use public or de-identified text: it is sent to the model provider.";
    // Not while it would be untrue: a hold, or no free run left.
    el.runHint.hidden = Boolean(state.payer) || Boolean(message && message.blocked);
    // Who pays, in a few words, as part of the Run button's description: a screen
    // reader that goes straight to the button hears it without opening Options. The
    // allowance line, when there is one, follows it (aria-describedby in index.html).
    let payer = "Runs on this site's models, free.";
    if (state.payer) payer = "Runs are on hold until you choose who pays.";
    else if (provider) payer = `Runs on your ${provider.name} key; ${provider.name} bills you.`;
    else if (state.health && state.health.mode === "offline") payer = "Runs on an offline stand-in, not a model.";
    else if (message && message.blocked) payer = "Runs on this site's models.";
    el.runPayer.textContent = payer;
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
  //
  // `opts.expire`: this read was made because a wait may be over (a timer ran out, the
  // tab came back). If the error box still tells of a used-up limit and health no longer
  // reports that limit, the box is closed. Not after a run: there the box is that run's
  // answer.
  async function refreshHealth(opts) {
    const read = await readHealth();
    if (read.superseded) return;
    settleHealth({ health: read.health, load: false, expire: Boolean(opts && opts.expire) });
  }

  // What is done with a health answer (`health` null: the read failed). `load`: it was
  // read for the whole form (arrival, "Try again"); otherwise for what changes between
  // requests. While a run is under way nothing is done with it: the payer, the key and
  // the model lists stay what the run was submitted with, and the answer is kept and
  // applied when the run ends (run()). Of several kept answers the newest good one counts.
  function settleHealth(answer) {
    if (state.running) {
      const held = state.heldHealth;
      state.heldHealth = held
        ? { health: answer.health || held.health, load: held.load || answer.load, expire: held.expire || answer.expire }
        : answer;
      return;
    }
    const health = answer.health;
    if (answer.load) {
      if (health) {
        applyHealth(health);
        return;
      }
      // A failed read after another read has answered changes nothing: the form is filled.
      if (state.health) return;
      state.healthFailed = true;
      renderBanner(null, null);
      // Nothing is known about the catalog, so nothing is claimed about its size.
      el.caveat.hidden = true;
      for (const node of el.pipe.querySelectorAll(".guide-node")) {
    guide.nodes[node.getAttribute("data-k")] = node;
    node.addEventListener("click", () => guideTap(node.getAttribute("data-k")));
  }
  el.guideToggle.addEventListener("click", () => guideSetOpen(!guideOpen()));
  el.guidePlay.addEventListener("click", () => guidePlay(guide.at));
  guideReset();

  fillTopK(DEFAULT_LIMITS.top_k_max);
      el.modelField.hidden = true;
      updateCounter();
      // Still chosen (the visitor may have picked the site's models while waiting).
      const still = savedPayer();
      if (still) holdRuns("unknown", still);
      else releaseRuns();
      return;
    }
    // A restored page waiting to learn whether its saved payer is offered (reconcilePayer).
    const waiting = Boolean(state.health) && Boolean(state.payer) && state.payer.kind === "pending";
    if (!health) {
      if (waiting) holdRuns("unknown", state.payer.provider);
      return;
    }
    if (!state.health) {
      applyHealth(health);
      return;
    }
    state.health = { ...state.health, visitor: health.visitor, claude: health.claude, own_keys: health.own_keys };
    state.visitorAt = Date.now();
    if (waiting) {
      // As on arrival, from session storage: the payer comes back with its key and its
      // models, or, if it is not offered, its key is removed and the hold then says so.
      fillProviders(health);
      applyProvider();
      applyStoredModels(savedChoice());
    } else reconcileProviders(health);
    renderAllowance();
    if (answer.expire && state.limitNotice && health.visitor && typeof health.visitor === "object") {
      const now = allowanceMessage();
      if (!(now && now.blocked && now.limit === state.limitNotice)) hideError();
    }
  }

  // Every read of /api/health goes through here and takes a number. Answers can arrive
  // out of order (the first read of a cold function can take longer than a run and the
  // read after it): an answer older than one already applied is dropped, so it cannot
  // put the form back to an earlier state. A failed read never counts as newer.
  let healthAsked = 0;
  let healthApplied = 0;

  async function readHealth() {
    healthAsked += 1;
    const seq = healthAsked;
    let health = null;
    try {
      health = await api("GET", "/api/health");
    } catch (_) {
      /* a failed read: null */
    }
    if (seq < healthApplied) return { superseded: true, health: null };
    if (health) healthApplied = seq;
    return { superseded: false, health };
  }

  // index.html carries the production copy. A full catalog only gets its exact term
  // count; a sample catalog or offline mode gets wording that is true for it.
  function renderIntro(health) {
    const full = Boolean(health.neon && health.neon.catalog === "full");
    const offline = health.mode === "offline";
    const ontologies = Array.isArray(health.ontologies) ? health.ontologies : [];
    el.caveat.hidden = false;
    state.ledeStatic = full && !offline;
    syncLede();
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
  }

  // The production lede names Claude, the default reader. With another model chosen to
  // read the text it says "A language model". Only the production copy is touched; the
  // sample and offline ledes (renderIntro) name no model.
  const LEDE = el.lede.textContent;

  function syncLede() {
    if (!state.ledeStatic) return;
    const reader = el.model.value;
    const text = !reader || isClaude(reader) ? LEDE : LEDE.replace("Claude finds", "A language model finds");
    if (el.lede.textContent !== text) el.lede.textContent = text;
  }

  // What the Search and Decide steps say while a run is under way, written from what
  // that run was submitted with and from what health says of the search. `selector` is
  // the submitted selector ("" when the page does not know which one the server will
  // use). A step that will not happen is not promised, and where the page cannot know
  // it says less.
  function renderSteps(selector) {
    const health = state.health;
    const where = !health ? "its ontology" : health.neon && health.neon.catalog === "full" ? "the full ontology" : "the sample ontology";
    const retrieval = (health && health.retrieval) || {};
    const off = (name) => Boolean(retrieval[name]) && retrieval[name].status !== "ok";
    if (off("bm25") && off("dense")) {
      el.pipeSearch.textContent = "The search is unavailable right now, so the mentions cannot be looked up.";
    } else if (off("dense")) {
      el.pipeSearch.textContent = `xwalk looks each one up in ${where} with keyword search (BM25) and makes a shortlist. Embedding search is unavailable right now.`;
    } else if (off("bm25")) {
      el.pipeSearch.textContent = `xwalk looks each one up in ${where} with embedding search and makes a shortlist. Keyword search is unavailable right now.`;
    } else {
      el.pipeSearch.textContent = `xwalk looks each one up in ${where} with keyword search (BM25) and embedding search, and merges the two rankings into a shortlist.`;
    }
    if (selector === "retrieval") {
      el.pipeDecide.textContent = "No model chooses in this run: each mention gets its top search result.";
    } else if (selector === "jev") {
      el.pipeDecide.textContent = "Jev, a small trained ranking model, picks a term from the shortlist or says that none fits.";
    } else if (selector) {
      el.pipeDecide.textContent = "The model picks the best term from the shortlist, or says none fits, and gives a confidence and a reason.";
    } else {
      el.pipeDecide.textContent = "A term is chosen for each mention from its shortlist.";
    }
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
      guideReset();
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
    if (chosen && !state.health) holdRuns("pending", chosen);
    const read = await readHealth();
    if (read.superseded) return;
    settleHealth({ health: read.health, load: true, expire: false });
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
          // The tooltip names the type only: the mention itself is the link's text.
          { class: `mention type-${t}`, href: `#card-${e.id}`, title: kind },
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
    // The number on the card's first line, in words: a phone has no hover for its tooltip.
    if ((outcome === "matched" || outcome === "check") && typeof p.confidence === "number") {
      detail.append(h("p", { class: "card-by card-conf-note", text: `${p.confidence.toFixed(2)} is the model's own confidence, not a measured accuracy.` }));
    }

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
  // this differs from the selector the request asked for. A pick that failed names a
  // selector too, but nothing was chosen there, so it does not count.
  // Answering is not choosing: a selector that declined every candidate it was shown,
  // or was not sure of any, reviewed them and chose no term. `chose` lists the selectors
  // with at least one term chosen; the whole list is everyone who answered.
  const DECIDED = ["matched", "check", "unsure", "declined"];
  const CHOSEN = ["matched", "check"];

  function pickersOf(entities) {
    const pickers = [];
    pickers.chose = [];
    for (const e of entities) {
      const outcome = e ? outcomeOf(e) : "";
      if (!DECIDED.includes(outcome)) continue;
      const who = typeof e.pick.selector === "string" ? e.pick.selector : "";
      if (!who) continue;
      if (!pickers.includes(who)) pickers.push(who);
      if (CHOSEN.includes(outcome) && !pickers.chose.includes(who)) pickers.chose.push(who);
    }
    return pickers;
  }

  // What a result actually holds, counted from its entities: how many mentions there
  // are, for how many the search returned at least one term (`hits`), how many have a
  // term a model chose (`picked`), and how many show the top search result because no
  // model was asked (`top`). Every sentence about "this run" is written from these.
  function tally(entities) {
    const outcomes = entities.map(outcomeOf);
    const count = (...kinds) => outcomes.filter((o) => kinds.includes(o)).length;
    return {
      mentions: entities.length,
      hits: entities.filter((e) => Array.isArray(e.matches) && e.matches.length > 0).length,
      picked: count("matched", "check"),
      top: count("search"),
      failed: count("failed"),
    };
  }

  // The kinds of search that ranked at least one candidate of this result.
  function rankedBy(entities) {
    const ranked = new Set();
    for (const e of entities) {
      for (const m of (Array.isArray(e.matches) ? e.matches : []).concat(Array.isArray(e.more) ? e.more : [])) {
        for (const name of Object.keys((m && m.retrievers) || {})) ranked.add(name.split("-")[0]);
      }
    }
    return ranked;
  }

  // Who answered, in one sentence for the line under the summary: the model that read
  // the text and the models that chose terms, as the response names them. "" offline.
  function answeredText(data, entities, pickers) {
    if (data.mode === "offline" || typeof data.model !== "string" || !data.model) return "";
    const own = data.credential_source === "visitor" && typeof data.provider === "string" && data.provider && data.provider !== "site";
    const read = `Read by ${modelName(data.model)}` + (own ? ` on your ${providerName(data.provider)} key` : "");
    if (pickers.chose.length) return `${read}; terms chosen by ${joinNames(pickers.chose.map(selectorName))}.`;
    // It answered, and declined every candidate or was sure of none: no term was chosen.
    if (pickers.length) return `${read}; candidates reviewed by ${joinNames(pickers.map(selectorName))}, which chose no term.`;
    const outcomes = entities.map(outcomeOf);
    return outcomes.includes("search") && !outcomes.includes("failed") ? `${read}; no model chose the terms.` : `${read}.`;
  }

  // "Run details" names what actually ran: the extraction model, and the model (or
  // models, after a step-down) that chose the terms.
  function renderRanOn(data, pickers, outcomes) {
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
    const named = pickers.chose.length ? pickers.chose : pickers;
    if (named.length) {
      el.ranOn.append(pickers.chose.length ? " Terms picked by " : " Candidates reviewed by ");
      named.forEach((who, i) => {
        if (i) el.ranOn.append(i === named.length - 1 ? " and " : ", ");
        el.ranOn.append(h("strong", { text: selectorName(who) }));
      });
      el.ranOn.append(pickers.chose.length ? "." : "; no term was picked.");
    } else if (data.mode !== "offline" && outcomes.includes("failed")) {
      el.ranOn.append(" The model did not finish choosing the terms.");
    } else if (data.mode !== "offline" && outcomes.includes("search")) {
      el.ranOn.append(" No model picked the terms: where the search found any, the top result is shown.");
    }
    el.ranOn.hidden = !el.ranOn.firstChild;
  }

  function fmtShort(ms) {
    return `${(ms / 1000).toFixed(1)} s`;
  }

  // The engineering read-out, folded away after the cards. The search's own words
  // (BM25, embeddings, the operator's reason for a failure) belong here and nowhere else.
  function renderRunDetails(data, pickers, outcomes) {
    renderRanOn(data, pickers, outcomes);
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
  // The search status of a response is one flag per kind of search for the whole
  // catalog, and it reads "unavailable" as soon as one ontology lacks that kind. So what
  // is said of this run is checked against its own entities: a kind of search that
  // ranked a candidate here was not off for the whole run, and a run with a term found
  // was not without its search.
  function renderNotices(data, searched, entities) {
    const ranked = rankedBy(entities);
    const hits = tally(entities).hits;
    clear(el.noticesList);
    const served = (Array.isArray(data.notices) ? data.notices : []).filter((n) => typeof n === "string" && n.trim());
    const mine = (Array.isArray(data.fallbacks) ? data.fallbacks : []).map(fallbackSentence);
    const lines = mine.length && mine.every(Boolean) ? mine : served.length ? served : mine.filter(Boolean);
    const retrieval = data.retrieval || {};
    const off = (name) => Boolean(retrieval[name]) && retrieval[name].status !== "ok";
    if (searched && off("dense") && off("bm25")) {
      lines.push(
        hits
          ? "Parts of the search were unavailable for this run, so some mentions may have been found by one kind of search only, or not at all."
          : "The search was unavailable for this run, so the mentions could not be looked up.",
      );
    } else if (searched && off("dense")) {
      lines.push(
        ranked.has("dense")
          ? "Matching by meaning was unavailable for part of this run, so some terms were found by keyword only. Synonyms and paraphrases may be missed."
          : "Part of the search (matching by meaning) was unavailable for this run, so terms were found by keyword only. Synonyms and paraphrases may be missed.",
      );
    } else if (searched && off("bm25")) {
      lines.push(
        ranked.has("bm25")
          ? "Matching by keyword was unavailable for part of this run, so some terms were found by meaning only."
          : "Part of the search (matching by keyword) was unavailable for this run, so terms were found by meaning only.",
      );
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
    if (!pickers.length && outcomes.includes("search")) {
      const top = outcomes.filter((o) => o === "search").length;
      if (top === outcomes.length) return "No model chose here: each mention shows its top search result.";
      return `No model chose here: ${top} of the ${outcomes.length} mentions ${top === 1 ? "shows its" : "show their"} top search result.`;
    }
    return "";
  }

  function entityTypesText() {
    const names = { chemical: "chemicals", disease: "diseases", gene: "genes", food: "foods" };
    const types = ((state.health && state.health.ontologies) || []).map((o) => names[o.type] || String(o.type));
    if (!types.length) return "chemicals, diseases or foods";
    return types.length < 2 ? types[0] : `${types.slice(0, -1).join(", ")} or ${types[types.length - 1]}`;
  }

  // The closing panel: what the visitor just watched, and where the library is. Only
  // what this result shows is said, counted from its entities (tally): for how many
  // mentions the search returned a term, how many of those a model chose a term for, and
  // how many show the top search result. "Each mention" is said only when it is each
  // one. The kinds of search are read from the candidates themselves (which search
  // ranked them). The term count is the size of the catalog: a mention is looked up in
  // the one ontology of its type, so no run searches all of it. With no candidate at all
  // (the search failed, or found nothing) there is no lookup to point at, and no panel.
  function renderNext(data, entities, pickers) {
    const ranked = rankedBy(entities);
    const t = tally(entities);
    el.next.hidden = t.hits === 0;
    if (!t.hits) return;
    const total = state.ontologies.reduce((sum, o) => sum + (Number(o.count) || 0), 0);
    const kinds = [ranked.has("bm25") ? "by keyword" : "", ranked.has("dense") ? "by meaning" : ""].filter(Boolean);
    const how = kinds.length ? ` ${kinds.join(" and ")}` : "";
    const all = t.hits === t.mentions;
    const looked = all ? `xwalk looked up each mention${how}` : `xwalk's search found terms for ${t.hits} of the ${t.mentions} mentions${how}`;
    const then = [];
    if (t.picked && pickers.chose.length) {
      then.push(t.picked === t.hits ? "the model chose among real candidates" : `the model chose a term for ${t.picked} of ${all ? "them" : "those"}`);
    }
    if (t.top && !t.failed) {
      if (t.top === t.hits) then.push(all ? "each mention shows the top search result" : t.hits === 1 ? "that one shows its top search result" : "each of those shows its top search result");
      else then.push(`${t.top} ${t.top === 1 ? "shows" : "show"} the top search result`);
    }
    el.nextBody.textContent =
      `The model never writes an ID. ${looked}${then.map((part) => `, and ${part}`).join("")}. ` +
      (total > 0 ? `The catalog holds ${total.toLocaleString("en")} terms. ` : "") +
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
    // After a run the page names the models that answered, which need not be the ones
    // the intro names or the ones that were asked for. "Done in 8.8 s" on the example
    // would read as the visitor's own wait.
    const answered = answeredText(data, entities, pickers);
    el.status.textContent = example ? "" : [done, answered].filter(Boolean).join(" ");

    el.results.setAttribute("data-kind", example ? "example" : "run");
    el.resultsHeading.textContent = example ? "Example result" : "Results";
    el.exampleNote.textContent = example ? `Recorded from a real run on ${state.example.recorded}.` : "";
    el.exampleNote.hidden = !example;

    renderNotices(data, entities.some((e) => e.ontology), entities);
    renderRunDetails(data, pickers, outcomes);
    el.runDetailsSummary.textContent = example ? "Details of the recorded run" : "Run details";
    el.runDetails.open = false;
    renderNext(data, entities, pickers);
    // Before a run the banner speaks for the service; after one, the run's own search
    // status is said beside its results ("About this run"). The example is not a run.
    if (!example) renderBanner(data.mode, state.health && state.health.retrieval);
    state.shown = { text, data, example };
    el.results.hidden = false;
    syncStale();
    // The figure follows the result on display, from its first mention again.
    guideReset();
  }

  // A result belongs to the text it was made from. Once the box holds another text the
  // result stays, dimmed, under one line that says so.
  // The recorded example is not called out while the box is empty (there is no "yours"
  // to map) or while the visitor's own run is under way (the button is already pressed).
  function syncStale() {
    const other = Boolean(state.shown) && !el.results.hidden && el.text.value !== state.shown.text;
    const stale = other && !(state.shown.example && (state.running || el.text.value === ""));
    el.results.classList.toggle("is-stale", stale);
    el.stale.textContent =
      state.shown && state.shown.example
        ? 'This result is for the example text. Press "Map this text" to map yours.'
        : "These results are for the earlier text. Run again to map the text that is in the box now.";
    el.stale.hidden = !stale;
  }

  // ---------- the guide: the figure that plays ----------

  // #pipe, expanded, is a figure of the pipeline. It follows one mention of the result
  // on display (the recorded example, then the visitor's latest run) from box to box,
  // and shows in its panel what each box did with it. Everything it shows is read from
  // that response; nothing is requested and nothing is stored. A mention visits only
  // the boxes that ran for it: one with no ontology stops at the Reader, one the search
  // found nothing for stops at the Retriever, one the selector did not answer for stops
  // at the Selector. The dashed boxes are the rest of the xwalk loop, which this demo
  // does not run: they explain themselves and are never visited.
  const GUIDE_DEMO = ["reader", "retriever", "selector", "result"];
  const GUIDE_NAMES = { reader: "Reader", retriever: "Retriever", selector: "Selector", result: "Result" };
  const GUIDE_SUBS = { retriever: "two searches, merged", selector: "picks one, or none", result: "with the reason" };
  const GUIDE_LIB = {
    scorer: ["Scorer", "The selector answers “which of these is best”. The best of several wrong answers can still look confident. The scorer asks a different question, “is this choice actually correct for the whole sentence”, and gives its own confidence."],
    verifier: ["Verifier", "Runs only when the score lands in a doubtful band. It takes a second look at the choice and can object to it."],
    policy: ["Policy", "Turns the attempt into one outcome, or decides it is not good enough yet and asks for another attempt."],
    rewriter: ["Rewriter", "Proposes a new search query for the mention, as the preprint’s synonym generator does, and sends it back to the retriever for another round."],
    ledger: ["Ledger", "Records every attempt with its candidates, the choice and the reason, so a run can be inspected, reviewed and resumed."],
  };
  const GUIDE_LIB_NOTE = "In the xwalk library. Not run by this demo.";
  const GUIDE_OUTCOMES = ["matched", "needs review", "unmatched", "failed"];
  const GUIDE_LANES = { bm25: "Keyword search", dense: "Meaning search" };
  const GUIDE_LANE_WORDS = { bm25: "keyword", dense: "meaning" };
  const GUIDE_RESULT_HELP = "Each mention ends with the ID of a real ontology term, or with a plain statement that no term fits. The ID always comes from the ontology; the model never writes one.";
  const GUIDE_TAP = "Tap any box above, including the dashed ones, to see what it does.";
  const GUIDE_SHORT = 4; // candidates shown per lane

  const guide = {
    nodes: {}, // data-k -> the box's button
    at: -1, // index of the mention being followed in the shown result's entities; -1: none
    box: null, // the box whose panel is on show
    timers: [],
    pos: null, // where the travelling label is, relative to #pipe
  };

  function guideOpen() {
    return el.pipe.getAttribute("data-view") === "expanded";
  }

  function guideCalm() {
    return typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
  }

  function guideEntities() {
    const data = state.shown && state.shown.data;
    return data && Array.isArray(data.entities) ? data.entities.filter((e) => e && typeof e === "object") : [];
  }

  function guideLater(ms, fn) {
    guide.timers.push(setTimeout(fn, ms));
  }

  // Ends a play where it is. Nothing of the run under way, if there is one, is touched.
  function guideStop() {
    for (const t of guide.timers) clearTimeout(t);
    guide.timers = [];
    guide.pos = null;
    const token = el.guideToken;
    token.classList.remove("is-on");
    if (typeof token.getAnimations === "function") for (const a of token.getAnimations()) a.cancel();
    guideMark(null);
    if (el.pipe.getAttribute("data-play") === "playing") el.pipe.setAttribute("data-play", "idle");
  }

  function guideMark(active) {
    for (const [k, node] of Object.entries(guide.nodes)) node.classList.toggle("is-active", k === active);
  }

  // Which box the panel is about. Said to a screen reader only while the figure is
  // open: collapsed, the boxes are three ways to open it and nothing is "pressed".
  function guidePressed() {
    const open = guideOpen();
    for (const [k, node] of Object.entries(guide.nodes)) {
      node.classList.toggle("is-sel", k === guide.box);
      if (open) node.setAttribute("aria-pressed", k === guide.box ? "true" : "false");
      else node.removeAttribute("aria-pressed");
    }
  }

  function guideSetOpen(open) {
    if (guideOpen() === open) return;
    if (!open) guideStop();
    el.pipe.setAttribute("data-view", open ? "expanded" : "collapsed");
    el.guideToggle.setAttribute("aria-expanded", open ? "true" : "false");
    el.guideToggle.textContent = open ? "Close" : "How it works";
    guidePressed();
  }

  // The mention a visitor is shown first: one where the selector overruled the search,
  // else one it declined, else the first with a term, else the first.
  function guideDefault(entities) {
    const outcomes = entities.map(outcomeOf);
    const tries = [
      entities.findIndex((e) => isOverride(e)),
      outcomes.findIndex((o) => o === "declined" || o === "unsure"),
      outcomes.findIndex((o) => o === "matched" || o === "check" || o === "search"),
    ];
    const found = tries.find((i) => i >= 0);
    return found === undefined ? (entities.length ? 0 : -1) : found;
  }

  function guideWords(text, n, fromEnd) {
    const cut = text.length > 90;
    const piece = fromEnd ? text.slice(-90) : text.slice(0, 90);
    const words = piece.match(fromEnd ? /\S+\s*/g : /\s*\S+/g) || [];
    if (cut && words.length) {
      if (fromEnd) words.shift();
      else words.pop();
    }
    const kept = (fromEnd ? words.slice(-n) : words.slice(0, n)).join("").replace(/\s+/g, " ");
    const more = kept.trim().length < text.replace(/\s+/g, " ").trim().length;
    return fromEnd ? (more ? "…" : "") + kept : kept + (more ? "…" : "");
  }

  // The mention with a few words either side, cut from the submitted text by the
  // server's code-point offsets. If the offsets do not hold, the mention alone.
  function guideSnippet(text, e) {
    const mention = String(e.text);
    const cps = Array.from(typeof text === "string" ? text : "");
    const ok = Number.isInteger(e.start) && Number.isInteger(e.end) && e.start >= 0 && e.end <= cps.length && e.start < e.end && cps.slice(e.start, e.end).join("") === mention;
    if (!ok) return ["", mention, ""];
    return [guideWords(cps.slice(0, e.start).join(""), 4, true), mention, guideWords(cps.slice(e.end).join(""), 4, false)];
  }

  // Everything the figure says about one mention, read from the response.
  function guideJourney(e) {
    const outcome = outcomeOf(e);
    const p = e.pick && typeof e.pick === "object" ? e.pick : null;
    const all = (Array.isArray(e.matches) ? e.matches : []).concat(Array.isArray(e.more) ? e.more : []).filter((m) => m && typeof m === "object");
    const label = (m) => String(m.label || m.id || "-");
    const term = outcome === "matched" || outcome === "check" ? p : outcome === "search" ? all[0] : null;

    // One lane per retriever that ranked a candidate: its own top four, by its own rank.
    const keys = [];
    for (const m of all) for (const k of Object.keys(m.retrievers || {})) if (!keys.includes(k)) keys.push(k);
    const first = (k) => ["bm25", "dense"].indexOf(k.split("-")[0]);
    keys.sort((a, b) => (first(a) < 0 ? 9 : first(a)) - (first(b) < 0 ? 9 : first(b)));
    const lanes = keys.map((k) => ({
      base: k.split("-")[0],
      title: GUIDE_LANES[k.split("-")[0]] || `${k} search`,
      items: all
        .filter((m) => Number.isFinite(m.retrievers[k]))
        .sort((a, b) => a.retrievers[k] - b.retrievers[k])
        .slice(0, GUIDE_SHORT)
        .map((m) => ({ n: m.retrievers[k], id: m.id, label: label(m) })),
    }));
    const merged = all.slice(0, GUIDE_SHORT).map((m, i) => ({ n: i + 1, id: m.id, label: label(m) }));

    // What the selector panel lists: the head of the merged order, and the chosen term
    // in the last place when it sits lower.
    const short = merged.map((item) => ({ ...item }));
    let pickAt = -1;
    if (term) {
      const at = all.findIndex((m) => m.id === term.id);
      if (at >= 0 && at < short.length) pickAt = at;
      else {
        const item = { n: at >= 0 ? at + 1 : null, id: term.id, label: label(term), gap: true };
        if (short.length >= GUIDE_SHORT) short[short.length - 1] = item;
        else short.push(item);
        pickAt = short.length - 1;
      }
    }

    const stops = ["reader"];
    if (e.ontology) stops.push("retriever");
    if (e.ontology && (p || all.length)) stops.push("selector");
    if (["matched", "check", "search", "declined", "unsure"].includes(outcome)) stops.push("result");

    const mention = String(e.text);
    let ended = "";
    if (outcome === "unsearched") ended = `“${mention}” did not get this far: this demo has no ontology for its type, so it was not looked up.`;
    else if (outcome === "none") ended = `“${mention}” did not get this far: the search found no term for it.`;
    else if (outcome === "failed") ended = `“${mention}” did not get this far: the selector did not answer, so there is no ID.`;

    const who = p ? selectorName(p.selector) : "";
    const why = p && p.explanation && outcome !== "failed" ? String(p.explanation) : "";
    return {
      e,
      mention,
      type: typeLabel(e.type),
      onto: e.ontology ? state.ontologyNames[e.ontology] || String(e.ontology) : "",
      outcome,
      p,
      all,
      term,
      lanes,
      merged,
      short,
      pickAt,
      stops,
      ended,
      who,
      why,
      jev: Boolean(p) && p.selector === "jev",
      override: isOverride(e),
      notes: [...new Set((Array.isArray(e.notes) ? e.notes : []).map(entityNote).filter(Boolean))],
      // What the result box is named once the journey is over.
      resultName: term ? String(term.id) : OUTCOME_TEXT[outcome] || "Ontology ID",
    };
  }

  function guideCurrent() {
    const e = guideEntities()[guide.at];
    return e ? guideJourney(e) : null;
  }

  // The short lines under the box names, true for the mention being followed.
  function guideSubs(j, finished) {
    const sub = (k, text) => {
      guide.nodes[k].querySelector(".guide-ns").textContent = text;
    };
    for (const k of GUIDE_DEMO) guide.nodes[k].classList.toggle("is-skipped", Boolean(j) && !j.stops.includes(k));
    const result = guide.nodes.result;
    const name = result.querySelector(".guide-nn");
    if (!j) {
      sub("retriever", GUIDE_SUBS.retriever);
      sub("selector", GUIDE_SUBS.selector);
      sub("result", GUIDE_SUBS.result);
      name.textContent = "Ontology ID";
      result.removeAttribute("data-outcome");
      return;
    }
    const one = j.lanes.length === 1 ? GUIDE_LANE_WORDS[j.lanes[0].base] : null;
    if (!j.all.length) sub("retriever", j.e.ontology ? "found no term" : GUIDE_SUBS.retriever);
    else if (j.lanes.length > 1) sub("retriever", `${j.lanes.length === 2 ? "two" : j.lanes.length} searches, merged`);
    else sub("retriever", one ? `${one} search only` : "search results");
    if (j.outcome === "search") sub("selector", "top search result, no model");
    else if (j.jev) sub("selector", "Jev picks one, or none");
    else sub("selector", GUIDE_SUBS.selector);

    name.textContent = finished ? j.resultName : "Ontology ID";
    if (finished && !j.term) result.setAttribute("data-outcome", "none");
    else result.removeAttribute("data-outcome");
    let said = GUIDE_SUBS.result;
    if (finished) {
      if (j.outcome === "search") said = "the top search result";
      else if (j.outcome === "failed") said = "the selector did not answer";
      else if (j.outcome === "none") said = "the search found nothing";
      else if (j.outcome === "unsearched") said = "no ontology for its type";
      else if (!j.why) said = j.term ? `chosen by ${j.who}` : `said by ${j.who}`;
    }
    sub("result", said);
  }

  function guideChipsOf(lane, items, target, on) {
    return items.map((item) => {
      if (item.gap) lane.append(h("span", { class: "guide-gap", "aria-hidden": "true", text: "…" }));
      const chip = h("span", { class: `guide-c${item.id === target ? " is-same" : ""}${on ? " is-on" : ""}`, text: `${item.n === null ? "" : `${item.n}. `}${item.label}` });
      lane.append(chip);
      return chip;
    });
  }

  // The reason as the response gives it. Every word is in the panel from the start;
  // a play only reveals them, so the text is exposed to a screen reader once, whole.
  function guideBubble(text, on) {
    const bubble = h("p", { class: "guide-bubble" });
    const words = text.split(/\s+/).filter(Boolean).map((w) => h("span", { class: `guide-w${on ? " is-on" : ""}`, text: `${w} ` }));
    bubble.append(...words);
    return { bubble, words };
  }

  function guideSelectorTitle(j) {
    if (j.outcome === "search") return "Selector · no model chose here: the top search result is taken as it is";
    if (j.jev) return "Selector · Jev, a small trained ranking model, picks from the shortlist";
    return `Selector · ${j.who || "the model"} reads the shortlist against the sentence`;
  }

  function guideReaderName(data) {
    if (data.mode === "offline") return "the offline stand-in";
    return typeof data.model === "string" && data.model ? modelName(data.model) : "the model";
  }

  // Fills the panel for one box. `full` shows the finished state; otherwise the parts a
  // play reveals are returned. `quiet`: the page wrote this itself (a new result), so
  // it is not announced; the next thing a visitor asks for is.
  function guideShow(k, full, quiet) {
    const panel = el.guidePanel;
    const j = guideCurrent();
    const data = state.shown && state.shown.data;
    const v = {};
    guide.box = k;
    guidePressed();
    if (k) el.pipe.setAttribute("data-at", k);
    else el.pipe.removeAttribute("data-at");
    panel.setAttribute("aria-live", quiet ? "off" : "polite");
    clear(panel);

    if (!k) {
      panel.append(
        h("h3", { text: "What happens to your text" }),
        h("p", { text: "The three boxes with faces run in this demo: one finds the mentions, one searches the ontology, one picks a term. The dashed boxes are the rest of the xwalk loop." }),
        h("p", { class: "guide-meta", text: GUIDE_TAP }),
      );
      return v;
    }
    if (GUIDE_LIB[k]) {
      panel.append(h("h3", { text: GUIDE_LIB[k][0] }), h("p", { text: GUIDE_LIB[k][1] }));
      if (k === "policy") panel.append(h("div", { class: "guide-outcomes" }, ...GUIDE_OUTCOMES.map((o) => h("span", { text: o }))));
      panel.append(h("span", { class: "guide-libnote", text: GUIDE_LIB_NOTE }));
      return v;
    }
    // What the box does in general, in the words the page already uses for this
    // deployment: said when there is no mention to follow, or when the mention being
    // followed did not reach the box.
    const general = { reader: el.pipeFind.textContent, retriever: el.pipeSearch.textContent, selector: el.pipeDecide.textContent, result: GUIDE_RESULT_HELP }[k];
    if (!j || !data) {
      panel.append(h("h3", { text: GUIDE_NAMES[k] }), h("p", { text: general }), h("p", { class: "guide-meta", text: "There is no result on the page yet. Map a text, then press Play to follow one of its mentions through these boxes." }));
      return v;
    }
    if (!j.stops.includes(k)) {
      panel.append(h("h3", { text: GUIDE_NAMES[k] }), h("p", { text: j.ended }), h("p", { class: "guide-meta", text: general }));
      return v;
    }
    const on = full ? " is-on" : "";

    if (k === "reader") {
      panel.append(h("h3", { text: `Reader · ${guideReaderName(data)} reads the whole text and marks the mentions` }));
      const [before, mention, after] = guideSnippet(state.shown.text, j.e);
      v.snip = h("div", { class: "guide-snip" }, before, h("strong", { text: mention }), after);
      const kind = j.type === "other" ? "is marked as another kind of mention" : `is a ${j.type}`;
      v.out = h("span", { class: `guide-c guide-out${on}`, text: `“${j.mention}” ${kind}${j.onto ? `, so it goes to ${j.onto}` : ""}` });
      panel.append(v.snip, v.out);
      if (!j.e.ontology) {
        v.end = h("p", { class: `guide-c guide-out${on}`, text: "Recognised, but this demo has no ontology for it." });
        panel.append(v.end);
      }
    } else if (k === "retriever") {
      const count = plural(j.all.length, "candidate");
      v.lanes = [];
      if (!j.all.length) {
        panel.append(h("h3", { text: `Retriever · xwalk searched ${j.onto} and found no term for this mention` }));
      } else if (j.lanes.length > 1) {
        panel.append(h("h3", { text: `Retriever · xwalk searches ${j.onto} ${j.lanes.length === 2 ? "two" : "several"} ways and merges the rankings` }));
      } else {
        const only = j.lanes.length ? GUIDE_LANE_WORDS[j.lanes[0].base] : null;
        panel.append(h("h3", { text: `Retriever · xwalk searched ${j.onto}${only ? `; only ${only} search ranked candidates for this mention` : ""}` }));
      }
      if (j.all.length) {
        const lanes = h("div", { class: "guide-lanes" });
        // One retriever: its ranking is the shortlist. One lane, not two and a merge.
        const shown = j.lanes.length > 1 ? j.lanes.concat([{ title: `Merged, ${count}`, items: j.merged }]) : [{ title: `${j.lanes.length ? j.lanes[0].title : "Search"}, ${count}`, items: j.lanes.length ? j.lanes[0].items : j.merged }];
        for (const lane of shown) {
          const box = h("div", { class: "guide-lane" }, h("h4", { text: lane.title }));
          v.lanes.push(guideChipsOf(box, lane.items, j.term ? j.term.id : null, full));
          lanes.append(box);
        }
        panel.append(lanes);
      }
      for (const note of j.notes) panel.append(h("p", { class: "guide-meta", text: note }));
    } else if (k === "selector") {
      panel.append(h("h3", { text: guideSelectorTitle(j) }));
      const lane = h("div", { class: "guide-lane" });
      v.short = guideChipsOf(lane, j.short, null, true);
      const done = (chip, i) => {
        chip.classList.toggle("is-picked", i === j.pickAt);
        chip.classList.toggle("is-passed", j.pickAt < 0 ? j.outcome !== "failed" : i < j.pickAt);
      };
      v.done = () => v.short.forEach((chip, i) => {
        chip.classList.remove("is-consider");
        done(chip, i);
      });
      if (j.pickAt >= 0) v.short[j.pickAt].append(h("span", { class: "visually-hidden", text: " (chosen)" }));
      if (full) v.done();
      if (v.short.length) panel.append(lane);
      if (j.all.length > j.short.length) panel.append(h("p", { class: "guide-meta", text: `These are ${j.short.length} of the ${plural(j.all.length, "candidate")} it was given.` }));
      let verdict = "";
      if (j.outcome === "declined") verdict = "None of these fits";
      else if (j.outcome === "unsure") verdict = "Not sure that any of these fits";
      else if (j.outcome === "failed") verdict = `${j.jev ? "Jev" : j.who || "The selector"} could not answer for this mention, so no term was chosen`;
      if (verdict) {
        v.verdict = h("span", { class: `guide-c guide-verdict${on}`, text: verdict });
        panel.append(v.verdict);
      }
      if (j.why) {
        const made = guideBubble(j.why, full);
        v.words = made.words;
        panel.append(made.bubble);
      }
    } else {
      panel.append(h("h3", { text: "Result" }));
      const line = h("p", null, h("strong", { text: j.mention }), " → ");
      if (j.term) {
        const href = safeHref(j.term.url);
        line.append(href ? h("a", { class: "guide-id", href, target: "_blank", rel: "noopener noreferrer", text: String(j.term.id) }) : h("span", { class: "guide-id", text: String(j.term.id) }));
        if (j.term.label) line.append(` ${j.term.label}`);
      } else line.append(OUTCOME_TEXT[j.outcome] || "");
      panel.append(line);
      const said = [];
      if (j.term === j.p && j.p && typeof j.p.confidence === "number") said.push(`Confidence ${j.p.confidence.toFixed(2)}: the model's own, not a measured accuracy.`);
      if (j.outcome === "check") said.push(`${j.who} marked it as needing a check.`);
      if (j.outcome === "search") said.push("The top search result. No model chose, so there is no reason and no confidence.");
      else if (j.term) said.push(j.override ? "Not the top search hit: the selector overruled the ranking." : "Also the top search hit.");
      else if (j.outcome === "declined") said.push(`${j.who} looked at ${plural(j.all.length, "candidate")} and declined to pick one.`);
      else if (j.outcome === "unsure") said.push(`${j.who} looked at ${plural(j.all.length, "candidate")} and was not sure that any of them fits.`);
      if (said.length) panel.append(h("p", { class: "guide-meta", text: said.join(" ") }));
      if (j.why) panel.append(guideBubble(j.why, true).bubble);
      panel.append(h("p", { class: "guide-meta", text: GUIDE_TAP }));
    }
    return v;
  }

  // Moves the travelling label to a box. It is positioned from script (the CSSOM, which
  // the page's style policy allows) and kept inside #pipe, so it never widens the page.
  function guideFly(node, text) {
    const token = el.guideToken;
    if (text) token.textContent = text;
    const frame = el.pipe.getBoundingClientRect();
    const r = node.getBoundingClientRect();
    const x = Math.round(Math.max(0, Math.min(r.left - frame.left + 10, frame.width - token.offsetWidth - 2)));
    const y = Math.round(Math.max(0, r.top - frame.top - 17));
    const from = guide.pos || [x, y];
    token.classList.add("is-on");
    const to = `translate(${x}px, ${y}px)`;
    if (typeof token.animate === "function") {
      token.animate([{ transform: `translate(${from[0]}px, ${from[1]}px)` }, { transform: to }], { duration: guide.pos ? 520 : 1, easing: "cubic-bezier(.3,1.3,.5,1)", fill: "forwards" });
    } else token.style.transform = to;
    guide.pos = [x, y];
  }

  function guideChips() {
    const entities = guideEntities();
    clear(el.guideChips);
    el.guideChips.hidden = entities.length === 0;
    if (!entities.length) return;
    el.guideChips.append(h("span", { class: "guide-chips-label", id: "guide-chips-label", text: "Follow a mention:" }));
    entities.forEach((e, i) => {
      const t = typeLabel(e.type);
      const chip = h("button", { type: "button", class: `guide-men type-${t}`, "aria-pressed": i === guide.at ? "true" : "false" }, String(e.text), h("span", { class: "visually-hidden", text: ` (${t})` }));
      chip.classList.toggle("is-current", i === guide.at);
      chip.addEventListener("click", () => guidePlay(i));
      el.guideChips.append(chip);
    });
  }

  function guideChip(i) {
    return el.guideChips.querySelectorAll(".guide-men")[i] || null;
  }

  function guideFollow(i) {
    guide.at = i;
    el.guideChips.querySelectorAll(".guide-men").forEach((chip, n) => {
      chip.classList.toggle("is-current", n === i);
      chip.setAttribute("aria-pressed", n === i ? "true" : "false");
    });
  }

  function guidePlayLabel(again) {
    el.guidePlayIcon.textContent = again ? "↻" : "▶";
    el.guidePlayLabel.textContent = again ? "Play again" : "Play";
  }

  function guideHint(j) {
    if (j) el.guideHint.textContent = `Following “${j.mention}”. Tap a box to see what it does.`;
    else if (state.shown) el.guideHint.textContent = "This result has no mention to follow. Tap a box to see what it does.";
    else el.guideHint.textContent = "Map a text to follow one of its mentions here. Tap a box to see what it does.";
  }

  // The finished figure for the mention being followed: where its journey ended.
  function guideFinish(j, quiet) {
    guideMark(null);
    el.guideToken.classList.remove("is-on");
    guideSubs(j, true);
    guideShow(j.stops[j.stops.length - 1], true, quiet);
  }

  // A result arrived, or left: the figure is that result's, at rest, from its first
  // mention. Nothing plays, nothing is announced, nothing scrolls.
  function guideReset() {
    guideStop();
    const entities = guideEntities();
    guide.at = guideDefault(entities);
    guideChips();
    const j = guideCurrent();
    el.guidePlay.disabled = !j;
    guidePlayLabel(false);
    guideHint(j);
    el.pipe.setAttribute("data-play", "idle");
    if (j) guideFinish(j, true);
    else {
      guideSubs(null, false);
      guideShow(null, true, true);
    }
  }

  // A tap on a box: what it did with the mention, finished. A play under way ends.
  function guideTap(k) {
    guideStop();
    guideSetOpen(true);
    const j = guideCurrent();
    if (j) guideSubs(j, true);
    guideShow(k, true);
  }

  // Follows one mention through the boxes that ran for it. With reduced motion there
  // is no travel: the finished figure is shown.
  function guidePlay(i) {
    guideStop();
    if (!guideEntities()[i]) return;
    guideSetOpen(true);
    guideFollow(i);
    const j = guideCurrent();
    guideHint(j);
    if (guideCalm()) {
      guideFinish(j);
      el.pipe.setAttribute("data-play", "done");
      guidePlayLabel(true);
      return;
    }
    guidePlayLabel(false);
    guideSubs(j, false);
    el.pipe.setAttribute("data-play", "playing");
    const chip = guideChip(i);
    const node = guide.nodes;
    let t = 100;
    let v;
    guideLater(t, () => {
      if (chip) guideFly(chip, j.mention);
    });
    guideLater((t += 350), () => {
      v = guideShow("reader");
      guideFly(node.reader, j.mention);
      guideMark("reader");
      v.snip.classList.add("is-scan");
    });
    guideLater((t += 1100), () => {
      v.out.classList.add("is-on");
      if (v.end) guideLater(450, () => v.end.classList.add("is-on"));
    });
    t += 900;
    if (j.stops.includes("retriever")) {
      const lanes = j.all.length ? (j.lanes.length > 1 ? j.lanes.length + 1 : 1) : 0;
      guideLater(t, () => {
        v = guideShow("retriever");
        guideFly(node.retriever);
        guideMark("retriever");
        v.lanes.forEach((chips, lane) => chips.forEach((c, n) => guideLater(250 + lane * 650 + n * 130, () => c.classList.add("is-on"))));
      });
      t += 1250 + lanes * 650;
    }
    if (j.stops.includes("selector")) {
      // The ones above the chosen term are considered first; a term at the top is
      // settled at once. With no term chosen, every one listed is considered.
      const looks = j.pickAt > 0 ? j.pickAt + 1 : j.pickAt === 0 || j.outcome === "failed" ? 0 : j.short.length;
      const words = j.why ? j.why.split(/\s+/).filter(Boolean).length : 0;
      const pace = words ? Math.min(55, 3500 / words) : 0;
      const settle = 500 + looks * 650;
      guideLater(t, () => {
        v = guideShow("selector");
        guideFly(node.selector);
        guideMark("selector");
        for (let n = 0; n < looks; n += 1) {
          guideLater(500 + n * 650, () =>
            v.short.forEach((c, at) => {
              c.classList.toggle("is-consider", at === n);
              c.classList.toggle("is-passed", at < n);
            }),
          );
        }
        guideLater(settle, () => {
          v.done();
          if (v.verdict) v.verdict.classList.add("is-on");
        });
        (v.words || []).forEach((w, n) => guideLater(settle + 250 + n * pace, () => w.classList.add("is-on")));
      });
      t += settle + 250 + words * pace + 650;
    }
    if (j.stops.includes("result")) {
      guideLater(t, () => {
        guide.nodes.result.querySelector(".guide-nn").textContent = j.resultName;
        if (!j.term) guide.nodes.result.setAttribute("data-outcome", "none");
        guideFly(node.result, j.resultName);
        guideMark("result");
      });
      t += 900;
    }
    guideLater(t, () => {
      guide.timers = [];
      guide.pos = null;
      guideFinish(j);
      el.pipe.setAttribute("data-play", "done");
      guidePlayLabel(true);
    });
  }

  // ---------- run ----------

  function setRunning(on) {
    if (state.running === on) return;
    state.running = on;
    syncRun();
    el.run.setAttribute("aria-busy", on ? "true" : "false");
    el.results.setAttribute("aria-busy", on ? "true" : "false");
    el.run.querySelector(".run-label").textContent = on ? "Mapping…" : "Map this text";
    syncStale();
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
    // On a phone the steps can be below the screen's edge. With the figure open, the
    // bar and the clock are what has to be seen: the figure is taller than a screen.
    reveal(guideOpen() ? el.progress : el.pipe, "nearest");
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
    if (state.running) return;
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
      if (text.trim()) showTextError("long", tooLongText());
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

    const ctx = {
      provider,
      model: payload.model,
      selector: el.selectorField.hidden ? effectiveSelector() : payload.selector || "",
      started: performance.now(),
      startedAt: Date.now(),
      hidden: document.visibilityState === "hidden",
      edited: false,
      abort: null,
    };
    state.runCtx = ctx;
    state.ran = true;
    // The text is the visitor's submission from here on, also when it is the example
    // this page put in the box: a later health answer may take the recorded example
    // away, but not the text that is being mapped.
    state.prefilled = false;
    renderSteps(ctx.selector);
    setRunning(true);
    // Said once for screen readers; sighted visitors have the steps and the button.
    announce("Mapping your text. This usually takes 10 to 30 seconds.");
    try {
      // The key is read here, for this call, and held nowhere else.
      const data = await api("POST", "/api/ner", payload, provider ? el.ownKey.value.trim() : undefined, ctx);
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
    // The run is over and what it has to say is on the page (an error, against what it
    // was submitted with). Now the form may change for the next run: who pays, as session
    // storage has it if the page was restored meanwhile, and then a health answer that
    // arrived during the run.
    if (state.payerCheckDue) {
      state.payerCheckDue = false;
      reconcilePayer();
    }
    const held = state.heldHealth;
    state.heldHealth = null;
    if (held) settleHealth(held);
    refreshHealth();
  }

  // ---------- wiring ----------

  function loadExample() {
    state.touched = true;
    // As typing does: a result that arrives now does not take the focus or the scroll.
    if (state.runCtx) state.runCtx.edited = true;
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
        // Timers do not run while a phone sleeps: a request past its time is cancelled now.
        const ctx = state.runCtx;
        if (ctx && ctx.abort && Date.now() - ctx.startedAt > REQUEST_TIMEOUT_MS) ctx.abort();
      }
      return;
    }
    // A tab that slept through a wait learns that the wait is over.
    if (visible && (allowanceTimer !== null || state.limitNotice)) refreshHealth({ expire: true });
  });
  for (const select of [el.model, el.selector]) {
    select.addEventListener("change", () => {
      rememberModels();
      renderAllowance();
    });
  }
  el.example.addEventListener("click", loadExample);
  el.noEntitiesExample.addEventListener("click", loadExample);
  el.clear.addEventListener("click", () => {
    state.touched = true;
    if (state.runCtx) state.runCtx.edited = true;
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
  // A page restored from the back-forward cache (`persisted`) is the document as it was
  // left, key field included: who pays is checked against session storage before it can
  // submit (reconcilePayer), and the free-runs line is read again.
  window.addEventListener("pageshow", (ev) => {
    updateCounter();
    syncStale();
    if (!ev.persisted) return;
    if (state.running) {
      // The request did not survive the cache; if it fails, the message says so.
      if (state.runCtx) state.runCtx.hidden = true;
      state.payerCheckDue = true;
      return;
    }
    reconcilePayer();
    refreshHealth({ expire: true });
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
    if (state.running) {
      // Not reachable through the disabled control; kept so that nothing else can
      // change who pays for a run that is under way.
      el.provider.value = state.runCtx && state.runCtx.provider ? state.runCtx.provider.id : "site";
      return;
    }
    forgetKey();
    rememberPayer(el.provider.value);
    applyProvider();
    // A message about the other payer (a used-up free limit, a busy model) is not left
    // standing over the new choice.
    hideError();
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
    // A key entered again settles who pays on a page that could not read its storage.
    if (state.payer && state.payer.kind === "unread" && currentProvider() && el.ownKey.value.trim()) releaseRuns();
  });
  el.forgetKey.addEventListener("click", () => {
    if (state.running) return;
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

  for (const node of el.pipe.querySelectorAll(".guide-node")) {
    guide.nodes[node.getAttribute("data-k")] = node;
    node.addEventListener("click", () => guideTap(node.getAttribute("data-k")));
  }
  el.guideToggle.addEventListener("click", () => guideSetOpen(!guideOpen()));
  el.guidePlay.addEventListener("click", () => guidePlay(guide.at));
  guideReset();

  fillTopK(DEFAULT_LIMITS.top_k_max);
  el.topK.value = "5";
  updateCounter();
  // Every handler is installed: Run, disabled in index.html, can be enabled.
  syncRun();
  loadHealth();
  loadExampleResult();
})();
