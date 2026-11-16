// Page logic. The model itself runs in worker.js; this file only handles the interface.
(function () {
  const config = window.HW_CONFIG;
  const $ = (id) => document.getElementById(id);
  let worker = null;
  let nextId = 1;
  const pending = new Map();
  let busy = false;
  let workerNote = "";   // latest note from the worker (e.g. an extra download in progress)

  function call(action, args, onProgress) {
    return new Promise((resolve, reject) => {
      const id = nextId++;
      pending.set(id, { resolve, reject, onProgress });
      worker.postMessage({ id, action, args });
    });
  }

  function setStatus(element, text, isError) {
    element.textContent = text;
    element.classList.toggle("error", Boolean(isError));
  }

  // Run one model job at a time, with a progress bar and elapsed time.
  async function run(name, action, args, workingText) {
    if (busy) return null;
    busy = true;
    const status = $(name + "-status"), bar = $(name + "-progress");
    const buttons = document.querySelectorAll("#app button, #photo-ui button");
    buttons.forEach((b) => (b.disabled = true));
    bar.hidden = false;
    bar.firstElementChild.style.width = "0%";
    const started = performance.now();
    const seconds = () => Math.round((performance.now() - started) / 1000);
    setStatus(status, workingText);
    workerNote = "";
    const timer = setInterval(() => setStatus(status, `${workerNote || workingText} ${seconds()} s`), 1000);
    try {
      const result = await call(action, args, (f) => (bar.firstElementChild.style.width = Math.round(f * 100) + "%"));
      setStatus(status, `Done in ${seconds()} s.`);
      return result;
    } catch (error) {
      setStatus(status, error.message, true);
      return null;
    } finally {
      clearInterval(timer);
      bar.hidden = true;
      buttons.forEach((b) => (b.disabled = false));
      busy = false;
    }
  }

  // ---------------------------------------------------------------- start-up
  $("start").addEventListener("click", async () => {
    $("start").disabled = true;
    const status = $("load-status");
    worker = new Worker("worker.js");
    worker.onmessage = (event) => {
      const message = event.data;
      if (message.type === "status") { workerNote = message.text; if (!busy) setStatus(status, message.text); return; }
      const job = pending.get(message.id);
      if (!job) return;
      if (message.type === "progress") { if (job.onProgress) job.onProgress(message.fraction); return; }
      pending.delete(message.id);
      if (message.type === "done") job.resolve(message.result); else job.reject(new Error(message.message));
    };
    worker.onerror = (event) => setStatus(status, "The model could not start: " + event.message, true);
    try {
      const styles = await call("init", config);
      const select = $("style");
      for (const name of styles) select.add(new Option("built-in " + name, name));
      select.add(new Option("none (the model's own hand)", "none"));
      select.value = styles.includes("7") ? "7" : styles[0];
      $("loader").hidden = true;
      $("app").hidden = false;
      if (config.ENABLE_IMAGE_CLONING) $("photo-ui").hidden = false;
      resizePad();
    } catch (error) {
      setStatus(status, "The model could not start: " + error.message, true);
      $("start").disabled = false;
    }
  });

  // ---------------------------------------------------------------- writing
  $("bias").addEventListener("input", () => ($("bias-value").textContent = Number($("bias").value).toFixed(2)));

  $("generate").addEventListener("click", async () => {
    const svg = await run("generate", "generate", {
      text: $("text").value, style: $("style").value, bias: $("bias").value,
      color: $("color").value, justify: $("justify").value,
    }, "Writing...");
    if (!svg) return;
    $("output").innerHTML = svg;
    $("output").hidden = false;
    const link = $("download");
    if (link.href) URL.revokeObjectURL(link.href);
    link.href = URL.createObjectURL(new Blob([svg], { type: "image/svg+xml" }));
    link.hidden = false;
  });

  // ---------------------------------------------------------------- drawing pad
  const pad = $("pad"), ctx = pad.getContext("2d");
  let strokes = [], current = null;

  function resizePad() {
    ctx.lineWidth = 2.5; ctx.lineCap = "round"; ctx.lineJoin = "round"; ctx.strokeStyle = "#111";
  }
  function point(event) {
    const box = pad.getBoundingClientRect();
    return [(event.clientX - box.left) * pad.width / box.width, (event.clientY - box.top) * pad.height / box.height];
  }
  function extend(event) {
    const [x, y] = point(event);
    const n = current.x.length;
    if (n && Math.hypot(x - current.x[n - 1], y - current.y[n - 1]) < 0.5) return;
    if (n) { ctx.beginPath(); ctx.moveTo(current.x[n - 1], current.y[n - 1]); ctx.lineTo(x, y); ctx.stroke(); }
    current.x.push(x); current.y.push(y);
  }
  pad.addEventListener("pointerdown", (event) => {
    pad.setPointerCapture(event.pointerId);
    current = { x: [], y: [] };
    extend(event);
  });
  pad.addEventListener("pointermove", (event) => {
    if (!current) return;
    const events = event.getCoalescedEvents ? event.getCoalescedEvents() : [event];
    (events.length ? events : [event]).forEach(extend);
  });
  function endStroke() {
    if (current && current.x.length > 1) strokes.push(current);
    current = null;
  }
  pad.addEventListener("pointerup", endStroke);
  pad.addEventListener("pointercancel", endStroke);
  $("clear").addEventListener("click", () => {
    strokes = []; ctx.clearRect(0, 0, pad.width, pad.height);
    setStatus($("capture-status"), "");
  });

  function showStyle(name, result) {
    const status = $(name + "-status");
    setStatus(status, `${status.textContent} Quality check: ${result.rating} (${result.score}). ${result.message}`,
              result.rating === "poor");
    const sample = $(name + "-sample");
    sample.innerHTML = result.sample;
    sample.hidden = false;
    const select = $("style");
    if (![...select.options].some((o) => o.value === "mine")) select.add(new Option("mine (this tab only)", "mine"), 0);
    select.value = "mine";
  }

  $("capture").addEventListener("click", async () => {
    if (!strokes.length) { setStatus($("capture-status"), "Write a few words in the box first.", true); return; }
    // the pad is 220 units tall; the model's stroke code expects about 200
    const scaled = strokes.map((s) => ({ x: s.x.map((v) => v * 200 / 220), y: s.y.map((v) => v * 200 / 220) }));
    const result = await run("capture", "capture", { strokes: scaled, text: $("sample-text").value },
                             "Reading your sample and testing it...");
    if (result) showStyle("capture", result);
  });

  // ---------------------------------------------------------------- photo (off unless enabled in config.js)
  $("photo-go").addEventListener("click", async () => {
    const file = $("photo-file").files[0];
    if (!file) { setStatus($("photo-status"), "Choose an image first.", true); return; }
    const bytes = new Uint8Array(await file.arrayBuffer());
    const result = await run("photo", "trace", { bytes, text: $("photo-text").value }, "Tracing the image...");
    if (result) showStyle("photo", result);
  });
})();
