// Settings for the demo page.
//
// ENABLE_IMAGE_CLONING: the code that turns a *photo* of handwriting into a style
// ships with this page (py/hwlite/trace.py) but its interface is switched off
// here, as a small deliberate barrier to casual misuse (forged homework,
// notes, prescriptions). Researchers who want to study it can set this to true
// in their own copy; everything still runs only in their browser.
window.HW_CONFIG = {
  ENABLE_IMAGE_CLONING: false,
  PYODIDE_URL: "https://cdn.jsdelivr.net/pyodide/v0.26.4/full/",
};
