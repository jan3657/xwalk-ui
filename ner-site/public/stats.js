// The private stats page (/stats). ES2020, no dependencies, no build step.
// It calls one endpoint, GET /api/stats, and nothing else. Everything the server sends
// is inserted with textContent; charts are SVG built with DOM calls and sized through
// attributes and CSS classes (the Content-Security-Policy allows no inline style).
"use strict";

(() => {
  // The stats passcode: one session storage entry of its own. It leaves the page only
  // as the X-Stats-Passcode header of GET /api/stats: never in a URL, a log or the DOM.
  const PASSCODE_ENTRY = "xwalk-stats-passcode";
  const STATS_PATH = "/api/stats";
  const REQUEST_TIMEOUT_MS = 30000;
  const SVG_NS = "http://www.w3.org/2000/svg";
  const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
  const WEEKDAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
  // Tiers in stacking order, bottom first. "none" is not a tier the server sends: it is
  // the day's runs that were refused before a tier was chosen (runs minus the four).
  const TIERS = [
    { key: "opus", name: "Opus" },
    { key: "sonnet", name: "Sonnet" },
    { key: "other", name: "Other" },
    { key: "own", name: "Own key", hatched: true },
    { key: "none", name: "Refused, no tier" },
  ];
  const MESSAGES = {
    required: "Enter the stats passcode.",
    invalid: "That passcode is not right. Check it and try again.",
    shape: "A passcode has only plain letters, digits and symbols. Check it and try again.",
    off: "Stats are not enabled on this site.",
    unavailable: "The stats database is unavailable. Try again shortly.",
    network: "Network error: the server could not be reached.",
    timeout: "The request took too long and was cancelled. Try again.",
  };

  const $ = (id) => document.getElementById(id);
  const el = {
    form: $("stats-form"),
    passcode: $("stats-passcode"),
    days: $("days"),
    show: $("show"),
    forget: $("forget"),
    message: $("message"),
    messageText: $("message-text"),
    status: $("status"),
    empty: $("empty"),
    emptyText: $("empty-text"),
    dashboard: $("dashboard"),
    unpriced: $("unpriced"),
    runsChart: $("runs-chart"),
    runsReadout: $("runs-readout"),
    runsLegend: $("runs-legend"),
    usdChart: $("usd-chart"),
    usdReadout: $("usd-readout"),
  };

  const state = { data: null, picked: null, sticky: false, request: 0, controller: null, charts: [] };

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

  function svg(tag, attrs, ...children) {
    const node = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "text") node.textContent = v;
      else node.setAttribute(k, String(v));
    }
    for (const c of children) if (c) node.append(c);
    return node;
  }

  function clear(node) {
    while (node.firstChild) node.removeChild(node.firstChild);
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
      /* storage blocked: the passcode simply is not remembered */
    }
  }

  // ---------- reading the answer: anything missing or odd becomes 0, "" or [] ----------

  const num = (v) => (typeof v === "number" && Number.isFinite(v) ? v : 0);
  const str = (v) => (typeof v === "string" ? v : "");
  const rows = (v) => (Array.isArray(v) ? v.filter((r) => r && typeof r === "object") : []);
  const obj = (v) => (v && typeof v === "object" && !Array.isArray(v) ? v : {});

  // ---------- formatting ----------

  const int = (n) => Math.round(num(n)).toLocaleString("en-US");
  const usd = (n) =>
    "$" + num(n).toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  const plural = (n, one, many) => `${int(n)} ${Math.round(num(n)) === 1 ? one : many}`;

  function percent(part, whole) {
    if (!(whole > 0)) return "n/a";
    const p = (100 * part) / whole;
    return `${p >= 10 || p === 0 ? Math.round(p) : p.toFixed(1)}%`;
  }

  // Run times: always seconds with one decimal.
  const seconds = (ms) => `${(num(ms) / 1000).toFixed(1)} s`;

  // "2026-11-18" -> {y, m, d}, or null. No Date parsing: these are UTC days as text.
  function dayParts(day) {
    const m = /^(\d{4})-(\d{2})-(\d{2})$/.exec(str(day));
    if (!m) return null;
    const p = { y: Number(m[1]), m: Number(m[2]), d: Number(m[3]) };
    return p.m >= 1 && p.m <= 12 ? p : null;
  }

  function shortDay(day) {
    const p = dayParts(day);
    return p ? `${p.d} ${MONTHS[p.m - 1]}` : str(day) || "unknown day";
  }

  function longDay(day) {
    const p = dayParts(day);
    if (!p) return shortDay(day);
    const weekday = WEEKDAYS[new Date(Date.UTC(p.y, p.m - 1, p.d)).getUTCDay()];
    return `${weekday} ${p.d} ${MONTHS[p.m - 1]}`;
  }

  let regionNames = null;
  try {
    regionNames = new Intl.DisplayNames(["en"], { type: "region" });
  } catch (_) {
    regionNames = null;
  }

  function countryName(code) {
    if (typeof code !== "string" || !code) return "Unknown";
    if (!/^[A-Z]{2}$/.test(code) || !regionNames) return code;
    try {
      const name = regionNames.of(code);
      return name && name !== code ? `${name} (${code})` : code;
    } catch (_) {
      return code;
    }
  }

  // The day's runs that reached no tier, so that a bar is as tall as the day's runs.
  function tierCounts(day) {
    const counts = {};
    let sum = 0;
    for (const t of TIERS) {
      if (t.key === "none") continue;
      counts[t.key] = Math.max(0, num(day[t.key]));
      sum += counts[t.key];
    }
    counts.none = Math.max(0, num(day.runs) - sum);
    return counts;
  }

  // ---------- messages ----------

  function showMessage(text) {
    el.messageText.textContent = text;
    el.message.hidden = false;
  }

  function hideMessage() {
    el.message.hidden = true;
    el.messageText.textContent = "";
  }

  function setBusy(busy) {
    el.show.setAttribute("aria-busy", busy ? "true" : "false");
    el.show.disabled = busy;
  }

  function hideData() {
    state.data = null;
    state.picked = null;
    el.dashboard.hidden = true;
    el.empty.hidden = true;
    el.status.textContent = "";
  }

  // ---------- the one request ----------

  class StatsError extends Error {
    constructor(kind, message) {
      super(message);
      this.kind = kind;
    }
  }

  async function fetchStats(passcode, days, signal) {
    let response;
    try {
      response = await fetch(`${STATS_PATH}?days=${days}`, {
        method: "GET",
        // X-Requested-With as on the site's other API calls; it carries no secret.
        headers: { "X-Stats-Passcode": passcode, "X-Requested-With": "xwalk-ner" },
        signal,
        credentials: "same-origin",
        cache: "no-store",
        // A request that carries the passcode is never followed to another address.
        redirect: "error",
      });
    } catch (err) {
      const aborted = err && err.name === "AbortError";
      throw new StatsError(aborted ? "timeout" : "network", aborted ? MESSAGES.timeout : MESSAGES.network);
    }
    let data = null;
    try {
      data = JSON.parse(await response.text());
    } catch (_) {
      data = null;
    }
    const code = str(obj(obj(data).error).code);
    if (response.status === 401) {
      const kind = code === "stats_passcode_required" ? "required" : "invalid";
      throw new StatsError(kind, MESSAGES[kind]);
    }
    if (response.status === 404) throw new StatsError("off", MESSAGES.off);
    if (response.status === 503) throw new StatsError("unavailable", MESSAGES.unavailable);
    if (!response.ok || obj(data).ok !== true) {
      throw new StatsError("other", `The stats could not be loaded (HTTP ${response.status}).`);
    }
    return data;
  }

  async function load() {
    const passcode = el.passcode.value;
    hideMessage();
    if (!passcode) {
      hideData();
      showMessage(MESSAGES.required);
      el.passcode.focus();
      return;
    }
    // A header value must be plain characters; anything else would make fetch throw.
    if (!/^[\x20-\x7e]+$/.test(passcode)) {
      hideData();
      showMessage(MESSAGES.shape);
      el.passcode.focus();
      return;
    }
    const days = ["7", "30", "90"].includes(el.days.value) ? el.days.value : "30";

    if (state.controller) state.controller.abort();
    const controller = new AbortController();
    const request = (state.request += 1);
    state.controller = controller;
    let timedOut = false;
    const timer = setTimeout(() => {
      timedOut = true;
      controller.abort();
    }, REQUEST_TIMEOUT_MS);
    setBusy(true);
    el.status.textContent = "Loading.";
    try {
      const data = await fetchStats(passcode, days, controller.signal);
      if (request !== state.request) return;
      storageSet(PASSCODE_ENTRY, passcode);
      el.forget.hidden = false;
      render(data, Number(days));
    } catch (err) {
      // Replaced by a newer request (not a timeout): that one reports.
      if (request !== state.request || (controller.signal.aborted && !timedOut)) return;
      hideData();
      const kind = err instanceof StatsError ? err.kind : "other";
      showMessage(err instanceof StatsError ? err.message : "The stats could not be shown.");
      if (kind === "required" || kind === "invalid") {
        // A refused passcode is not kept; it stays in the field to be corrected.
        storageSet(PASSCODE_ENTRY, "");
        el.forget.hidden = true;
        el.passcode.focus();
        el.passcode.select();
      }
    } finally {
      clearTimeout(timer);
      if (request === state.request) {
        setBusy(false);
        state.controller = null;
      }
    }
  }

  // ---------- rendering ----------

  function render(data, askedDays) {
    const totals = obj(data.totals);
    const byDay = rows(data.by_day);
    const days = byDay.length || num(data.days) || askedDays;
    state.data = data;
    state.picked = null;

    setStatusLine(data, byDay, days);
    const runs = num(totals.runs);
    if (runs <= 0) {
      el.dashboard.hidden = true;
      el.emptyText.textContent =
        `No runs were recorded in the last ${plural(days, "day", "days")}. ` +
        "Runs appear here once someone uses the demo.";
      el.empty.hidden = false;
      return;
    }
    el.empty.hidden = true;
    el.dashboard.hidden = false;

    renderTiles(totals, byDay, days);
    renderModels(rows(data.by_model));
    renderCounts($("rows-code"), rows(data.by_code), "code", runs, "No errors in this period.", (v) => ({
      text: str(v) || "unknown",
      cls: "code",
    }));
    renderCounts($("rows-fallback"), rows(data.by_fallback), "reason", runs, "No run stepped down in this period.", (v) => ({
      text: str(v) || "unknown",
      cls: "code",
    }));
    renderCounts($("rows-country"), rows(data.by_country), "country", runs, "No countries recorded.", (v) => ({
      text: countryName(v),
      cls: null,
    }));
    renderDays(byDay);
    renderLegend();
    drawCharts();
  }

  function setStatusLine(data, byDay, days) {
    const parts = [`Last ${plural(days, "day", "days")}`];
    if (byDay.length) {
      const first = shortDay(byDay[0].day);
      const last = shortDay(byDay[byDay.length - 1].day);
      parts[0] += first === last ? `: ${first}` : `: ${first} to ${last}`;
    }
    parts[0] += " (UTC days).";
    const at = new Date(str(data.generated_at));
    if (str(data.generated_at) && !Number.isNaN(at.getTime())) {
      const hh = String(at.getUTCHours()).padStart(2, "0");
      const mm = String(at.getUTCMinutes()).padStart(2, "0");
      parts.push(`Updated ${at.getUTCDate()} ${MONTHS[at.getUTCMonth()]} at ${hh}:${mm} UTC.`);
    }
    el.status.textContent = parts.join(" ");
  }

  function renderTiles(totals, byDay, days) {
    const runs = num(totals.runs);
    const ok = num(totals.ok);
    const errors = "errors" in totals ? num(totals.errors) : Math.max(0, runs - ok);
    const stepped = byDay.reduce((sum, d) => sum + num(d.stepped_down), 0);
    const tiers = { opus: 0, sonnet: 0, other: 0, own: 0, none: 0 };
    for (const d of byDay) {
      const c = tierCounts(d);
      for (const k of Object.keys(tiers)) tiers[k] += c[k];
    }

    $("t-usd").textContent = usd(totals.estimated_usd);
    $("t-usd-note").textContent =
      `About ${usd(num(totals.estimated_usd) / Math.max(1, days))} a day, on the site's keys. An estimate.`;
    $("t-step").textContent = percent(stepped, runs);
    $("t-step-note").textContent = `${int(stepped)} of ${plural(runs, "run", "runs")} ended on a lower model.`;
    $("t-runs").textContent = int(runs);
    $("t-runs-note").textContent = TIERS.filter((t) => tiers[t.key] > 0)
      .map((t) => `${t.key === "none" ? "refused" : t.name} ${int(tiers[t.key])}`)
      .join(", ");
    $("t-ok").textContent = int(ok);
    $("t-ok-note").textContent = `${percent(ok, runs)} of runs. ${plural(errors, "error", "errors")}.`;
    $("t-visitors").textContent = int(totals.visitors);

    const unpriced = num(totals.unpriced_runs);
    el.unpriced.hidden = unpriced <= 0;
    el.unpriced.textContent =
      unpriced > 0
        ? `${plural(unpriced, "run", "runs")} on the site's keys used a model with no list price. ` +
          "They add nothing to the estimate, so the real spend is higher."
        : "";
  }

  // `label` names the column where a narrow screen shows the row without its header.
  function cell(text, cls, label) {
    return h("td", { class: cls || null, "data-label": label || null, text });
  }

  function noRows(tbody, columns, text) {
    tbody.append(h("tr", null, h("td", { class: "no-rows", colspan: columns, text })));
  }

  function renderModels(models) {
    const tbody = $("rows-model");
    clear(tbody);
    if (!models.length) return noRows(tbody, 6, "No models recorded.");
    for (const m of models) {
      const site = m.credential !== "visitor";
      const tokens = num(m.prompt_tokens) + num(m.completion_tokens);
      // A visitor's key costs the site nothing; a site model with tokens and no dollars
      // has no list price.
      let dollars = usd(m.estimated_usd);
      if (!site) dollars = "visitor pays";
      else if (num(m.estimated_usd) === 0 && tokens > 0) dollars = "no price";
      tbody.append(
        h(
          "tr",
          null,
          str(m.model)
            ? h("th", { scope: "row", class: "code", text: m.model })
            : h("th", { scope: "row", class: "quiet", text: "No model (refused first)" }),
          cell(str(m.credential) || "unknown", null, "Key"),
          cell(int(m.runs), "num", "Runs"),
          cell(int(m.prompt_tokens), "num", "Tokens in"),
          cell(int(m.completion_tokens), "num", "Tokens out"),
          cell(dollars, dollars.startsWith("$") ? "num" : "num quiet", "Est. $"),
        ),
      );
    }
  }

  // `runs` is every run of the period: the share column is a share of all runs.
  function renderCounts(tbody, items, field, runs, emptyText, label) {
    clear(tbody);
    if (!items.length) return noRows(tbody, 3, emptyText);
    for (const item of items) {
      const { text, cls } = label(item[field]);
      // A code may wrap after an underscore (text nodes and <wbr>, never markup).
      const name = h("th", { scope: "row", class: cls });
      text.split(/(?<=_)/).forEach((piece, i) => {
        if (i) name.append(h("wbr"));
        name.append(document.createTextNode(piece));
      });
      tbody.append(h("tr", null, name, cell(int(item.runs), "num"), cell(percent(num(item.runs), runs), "num")));
    }
  }

  function renderDays(byDay) {
    const tbody = $("rows-day");
    clear(tbody);
    if (!byDay.length) return noRows(tbody, 12, "No days in this answer.");
    for (let i = byDay.length - 1; i >= 0; i -= 1) {
      const d = byDay[i];
      const quiet = num(d.runs) <= 0;
      tbody.append(
        h(
          "tr",
          { class: quiet ? "none" : null },
          h("th", { scope: "row", text: str(d.day) || "unknown" }),
          cell(int(d.runs), "num"),
          cell(int(d.ok), "num"),
          cell(usd(d.estimated_usd), "num"),
          cell(int(d.stepped_down), "num"),
          cell(int(d.visitors), "num"),
          cell(int(d.opus), "num"),
          cell(int(d.sonnet), "num"),
          cell(int(d.other), "num"),
          cell(int(d.own), "num"),
          // A day without runs has no run time; 0.0 s would read as "instant".
          cell(quiet ? "none" : seconds(d.p50_ms), "num"),
          cell(quiet ? "none" : seconds(d.p95_ms), "num"),
        ),
      );
    }
  }

  // ---------- charts ----------

  function swatch(tier) {
    const box = svg("svg", { width: 12, height: 12, viewBox: "0 0 12 12", "aria-hidden": "true", focusable: "false" });
    box.append(svg("rect", { width: 12, height: 12, class: `tier-${tier.key}` }));
    if (tier.hatched) {
      for (const x of [-6, 0, 6]) {
        box.append(svg("line", { x1: x, y1: 12, x2: x + 12, y2: 0, class: "hatch-line" }));
      }
    }
    return box;
  }

  function renderLegend() {
    clear(el.runsLegend);
    for (const tier of TIERS) el.runsLegend.append(h("li", null, swatch(tier), tier.name));
  }

  // A round step that gives about `target` ticks up to `max`.
  function niceStep(max, target, least) {
    const raw = Math.max(max, least) / target;
    const pow = 10 ** Math.floor(Math.log10(raw));
    const step = [1, 2, 5, 10].map((m) => m * pow).find((s) => s >= raw) || 10 * pow;
    return Math.max(step, least);
  }

  function runsSentence(d) {
    const c = tierCounts(d);
    const parts = TIERS.filter((t) => c[t.key] > 0).map((t) => `${t.name} ${int(c[t.key])}`);
    const stepped = num(d.stepped_down);
    return (
      `${longDay(d.day)}: ${plural(d.runs, "run", "runs")}` +
      (parts.length ? `. ${parts.join(", ")}` : "") +
      (stepped > 0 ? `. ${int(stepped)} stepped down` : "") +
      "."
    );
  }

  function usdSentence(d) {
    return `${longDay(d.day)}: ${usd(d.estimated_usd)} estimated, ${plural(d.runs, "run", "runs")}.`;
  }

  function peak(byDay, value) {
    let best = null;
    for (const d of byDay) if (best === null || value(d) > value(best)) best = d;
    return best;
  }

  // One bar chart. `stack(day)` returns [{cls, value, fill}] bottom first.
  function drawChart(host, spec) {
    clear(host);
    const byDay = spec.byDay;
    const n = byDay.length;
    const width = Math.max(240, Math.floor(host.clientWidth));
    const height = 190;
    const m = { top: 8, right: 2, bottom: 22, left: spec.left };
    const plotW = width - m.left - m.right;
    const plotH = height - m.top - m.bottom;
    const max = byDay.reduce((top, d) => Math.max(top, spec.total(d)), 0);
    const step = niceStep(max, 4, spec.least);
    const top = Math.max(step, Math.ceil(max / step - 1e-9) * step);
    const y = (v) => m.top + plotH - (v / top) * plotH;
    const slot = plotW / n;
    const gap = slot >= 8 ? 2 : slot >= 3 ? 1 : 0;
    const barW = Math.max(1, Math.min(32, slot - gap));
    const x = (i) => m.left + i * slot + (slot - barW) / 2;

    const root = svg("svg", {
      width,
      height,
      viewBox: `0 0 ${width} ${height}`,
      role: "group",
      tabindex: 0,
      "aria-label": `${spec.label}. Use the left and right arrow keys to read each day. The same numbers are in the By day table.`,
    });

    if (spec.defs) root.append(spec.defs());

    for (let v = 0; v <= top + step / 2; v += step) {
      const yy = Math.round(y(v)) + 0.5;
      root.append(svg("line", { x1: m.left, x2: width - m.right, y1: yy, y2: yy, class: v === 0 ? "base-line" : "grid-line" }));
      root.append(svg("text", { x: m.left - 6, y: yy + 4, "text-anchor": "end", text: spec.tick(v) }));
    }

    const band = svg("rect", { class: "band", x: 0, y: m.top, width: Math.max(slot, barW), height: plotH });
    root.append(band);

    byDay.forEach((d, i) => {
      let base = 0;
      for (const part of spec.stack(d)) {
        if (!(part.value > 0)) continue;
        const y1 = y(base + part.value);
        const y0 = y(base);
        // A hairline between segments when the bar is wide enough to spare it.
        const inset = barW >= 6 && base > 0 && y0 - y1 > 2 ? 1 : 0;
        root.append(
          svg("rect", {
            class: `bar ${part.cls}`,
            fill: part.fill || null,
            x: x(i),
            y: y1,
            width: barW,
            height: Math.max(0.5, y0 - y1 - inset),
          }),
        );
        base += part.value;
      }
    });

    // Day labels: as many as fit without touching, counted back from the newest day.
    const every = Math.max(1, Math.ceil(n / Math.max(1, Math.floor(plotW / 58))));
    for (let i = n - 1; i >= 0; i -= every) {
      const centre = m.left + i * slot + slot / 2;
      const last = i === n - 1 && centre > width - 24;
      const first = i < every && centre < m.left + 20;
      root.append(
        svg("text", {
          x: last ? width - m.right : first ? m.left : centre,
          y: height - 6,
          "text-anchor": last ? "end" : first ? "start" : "middle",
          text: shortDay(byDay[i].day),
        }),
      );
    }

    // One target per day, the full height of the plot: easier to hit than a thin bar.
    byDay.forEach((day, i) => {
      const hit = svg("rect", { class: "hit", x: m.left + i * slot, y: m.top, width: slot, height: plotH });
      hit.append(svg("title", { text: spec.sentence(day) }));
      hit.addEventListener("pointerenter", (e) => {
        if (e.pointerType === "mouse") pick(i, false);
      });
      hit.addEventListener("click", () => pick(i, true));
      root.append(hit);
    });
    root.addEventListener("pointerleave", (e) => {
      if (e.pointerType === "mouse" && !state.sticky) pick(null, false);
    });
    root.addEventListener("keydown", (e) => {
      const at = state.picked === null ? n : state.picked;
      let next = null;
      if (e.key === "ArrowLeft") next = Math.max(0, at - 1);
      else if (e.key === "ArrowRight") next = Math.min(n - 1, at === n ? n - 1 : at + 1);
      else if (e.key === "Home") next = 0;
      else if (e.key === "End") next = n - 1;
      else if (e.key === "Escape") next = -1;
      if (next === null) return;
      e.preventDefault();
      pick(next < 0 ? null : next, next >= 0);
    });

    host.append(root);
    return {
      show(i) {
        const on = i !== null && i < n;
        band.classList.toggle("on", on);
        if (on) band.setAttribute("x", String(m.left + i * slot + (slot - Math.max(slot, barW)) / 2));
        spec.readout.classList.toggle("picked", on);
        spec.readout.textContent = on ? spec.sentence(byDay[i]) : spec.summary;
      },
    };
  }

  // Both charts show the same day. `sticky`: chosen by a tap, a click or a key, so it
  // stays when the pointer leaves; picking the same day again lets go of it.
  function pick(i, sticky) {
    if (sticky && state.sticky && state.picked === i) {
      i = null;
      sticky = false;
    }
    state.picked = i;
    state.sticky = i !== null && sticky;
    for (const chart of state.charts) chart.show(i);
  }

  function hatchDefs() {
    const pattern = svg("pattern", {
      id: "hatch-own",
      width: 4,
      height: 4,
      patternUnits: "userSpaceOnUse",
      patternTransform: "rotate(45)",
    });
    pattern.append(svg("rect", { width: 4, height: 4, class: "tier-own" }));
    pattern.append(svg("line", { x1: 0, y1: 0, x2: 0, y2: 4, class: "hatch-line" }));
    return svg("defs", null, pattern);
  }

  function drawCharts() {
    if (!state.data || el.dashboard.hidden) return;
    const byDay = rows(state.data.by_day);
    state.charts = [];
    if (!byDay.length) {
      clear(el.runsChart);
      clear(el.usdChart);
      el.runsReadout.textContent = "No days in this answer.";
      el.usdReadout.textContent = "No days in this answer.";
      return;
    }
    const days = plural(byDay.length, "day", "days");
    const busiest = peak(byDay, (d) => num(d.runs));
    const dearest = peak(byDay, (d) => num(d.estimated_usd));
    const totals = obj(state.data.totals);

    state.charts.push(
      drawChart(el.runsChart, {
        byDay,
        label: "Runs per day, stacked by tier",
        left: 30,
        least: 1,
        total: (d) => Math.max(num(d.runs), 0),
        tick: (v) => int(v),
        stack: (d) => {
          const c = tierCounts(d);
          return TIERS.map((t) => ({
            cls: t.hatched ? "" : `tier-${t.key}`,
            fill: t.hatched ? "url(#hatch-own)" : null,
            value: c[t.key],
          }));
        },
        defs: hatchDefs,
        sentence: runsSentence,
        readout: el.runsReadout,
        summary: `${days}, ${plural(totals.runs, "run", "runs")}. Busiest: ${shortDay(busiest.day)} with ${int(busiest.runs)}.`,
      }),
    );
    state.charts.push(
      drawChart(el.usdChart, {
        byDay,
        label: "Estimated dollars per day",
        left: 42,
        least: 0.01,
        total: (d) => Math.max(num(d.estimated_usd), 0),
        tick: (v) => usd(v),
        stack: (d) => [{ cls: "bar-usd", value: Math.max(num(d.estimated_usd), 0) }],
        sentence: usdSentence,
        readout: el.usdReadout,
        summary: `${days}, ${usd(totals.estimated_usd)} estimated. Highest: ${shortDay(dearest.day)} with ${usd(dearest.estimated_usd)}.`,
      }),
    );
    const keep = state.picked !== null && state.picked < byDay.length ? state.picked : null;
    state.picked = keep;
    for (const chart of state.charts) chart.show(keep);
  }

  // ---------- wiring ----------

  el.form.addEventListener("submit", (e) => {
    e.preventDefault();
    load();
  });

  el.days.addEventListener("change", () => {
    if (el.passcode.value) load();
  });

  el.forget.addEventListener("click", () => {
    if (state.controller) state.controller.abort();
    state.request += 1;
    setBusy(false);
    storageSet(PASSCODE_ENTRY, "");
    el.passcode.value = "";
    el.forget.hidden = true;
    hideMessage();
    hideData();
    el.passcode.focus();
  });

  // Charts are drawn at the width they have, so that text stays the size it is set at.
  let lastWidth = 0;
  let redraw = 0;
  const onResize = () => {
    const width = el.runsChart.clientWidth;
    if (!width || width === lastWidth) return;
    lastWidth = width;
    window.clearTimeout(redraw);
    redraw = window.setTimeout(drawCharts, 60);
  };
  if (typeof ResizeObserver === "function") new ResizeObserver(onResize).observe(el.runsChart);
  else window.addEventListener("resize", onResize);

  // A passcode kept from earlier in this tab goes back in the field and is used.
  const kept = storageGet(PASSCODE_ENTRY);
  if (kept) {
    el.passcode.value = kept;
    el.forget.hidden = false;
    load();
  } else {
    el.passcode.focus();
  }
})();
