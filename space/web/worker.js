importScripts("https://cdn.jsdelivr.net/pyodide/v0.27.3/full/pyodide.js");
let py = null;
async function init() {
  py = await loadPyodide(); await py.loadPackage("numpy");
  const files = ["demo_core.py", "genesis/__init__.py", "genesis/engine.py", "genesis/genome.py", "genesis/surrogate.py", "genesis/town.py"];
  py.FS.mkdirTree("/app/genesis");
  for (const f of files) { const r = await fetch(f); if (r.ok) py.FS.writeFile("/app/" + f, await r.text()); }
  // wasm32 shim: numpy.bincount refuses int64 input on 32-bit platforms; cast to the native index type (same result).
  py.runPython(`
import numpy as np
_bc = np.bincount
def _bincount(x, weights=None, minlength=0):
    return _bc(np.asarray(x).astype(np.intp), weights=weights, minlength=minlength)
np.bincount = _bincount
import sys; sys.path.insert(0, '/app'); import demo_core`);
  postMessage({ type: "ready" });
}
const ready = init().catch(e => postMessage({ type: "error", msg: String(e) }));
onmessage = async (ev) => {
  await ready;
  try {
    self.progress = (k, n) => postMessage({ type: "progress", k, n });
    py.globals.set("args", py.toPy(ev.data));
    const out = py.runPython(`
import json, js
a = args.to_py() if hasattr(args, "to_py") else args
a = demo_core.clamp_args(a.get("generations"), a.get("population"), a.get("seed"))
res = demo_core.run(a["generations"], a["population"], a["seed"], time_cap=120, on_gen=lambda k, n: js.progress(k, n))
json.dumps(dict(res=res, md=demo_core.summary_md(res)))`);
    postMessage({ type: "done", data: JSON.parse(out) });
  } catch (e) { postMessage({ type: "error", msg: String(e) }); }
};
