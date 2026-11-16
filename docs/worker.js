// Runs the handwriting model off the main thread, in Python (Pyodide) inside this browser tab.
// Nothing here talks to a server other than to download the page's own static files.
let pyodide = null;
let api = null;
let config = null;
let tracingReady = false;

const PY_FILES = ["hwlite/__init__.py", "hwlite/model.py", "hwlite/strokes.py", "hwlite/writer.py",
                  "hwlite/quality.py", "hwlite/trace.py", "site_api.py"];

function status(text) { postMessage({ type: "status", text }); }

async function fetchBytes(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Could not download ${url} (${response.status})`);
  return new Uint8Array(await response.arrayBuffer());
}

async function init(cfg) {
  config = cfg;
  status("Downloading Python (about 10 MB)...");
  importScripts(config.PYODIDE_URL + "pyodide.js");
  pyodide = await loadPyodide({ indexURL: config.PYODIDE_URL });
  status("Downloading NumPy...");
  await pyodide.loadPackage("numpy");
  status("Downloading the handwriting model (7 MB)...");
  pyodide.FS.mkdirTree("/app/hwlite");
  pyodide.FS.mkdirTree("/data");
  for (const file of PY_FILES) pyodide.FS.writeFile("/app/" + file, await fetchBytes("py/" + file));
  pyodide.FS.writeFile("/data/weights.npz", await fetchBytes("model/weights.npz"));
  pyodide.FS.writeFile("/data/styles.npz", await fetchBytes("model/styles.npz"));
  status("Starting the model...");
  pyodide.runPython("import sys; sys.path.insert(0, '/app')");
  api = pyodide.pyimport("site_api");
  return JSON.parse(api.init());
}

async function ensureTracing() {
  if (tracingReady) return;
  status("Downloading the image tools (about 45 MB, first time only)...");
  await pyodide.loadPackage(["scipy", "scikit-image", "opencv-python", "pillow"]);
  tracingReady = true;
}

onmessage = async (event) => {
  const { id, action, args } = event.data;
  const progress = (fraction) => postMessage({ type: "progress", id, fraction });
  try {
    let result;
    if (action === "init") {
      result = await init(args);
    } else if (action === "generate") {
      result = api.generate(args.text, args.style, args.bias, args.color, args.justify, progress);
    } else if (action === "capture") {
      result = JSON.parse(api.capture(JSON.stringify(args.strokes), args.text, progress));
    } else if (action === "trace") {
      if (!config.ENABLE_IMAGE_CLONING) throw new Error("Image cloning is switched off in config.js.");
      await ensureTracing();
      status("Reconstructing pen strokes and testing them...");
      result = JSON.parse(api.trace(pyodide.toPy(args.bytes), args.text, progress));
    } else {
      throw new Error("Unknown action " + action);
    }
    postMessage({ type: "done", id, result });
  } catch (error) {
    // Python errors arrive as a traceback; the last line is the readable part
    const lines = String(error.message || error).trim().split("\n");
    const last = lines[lines.length - 1].replace(/^\w*(Error|Exception):\s*/, "");
    postMessage({ type: "error", id, message: last });
  }
};
