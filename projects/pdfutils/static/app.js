const KEY_STORAGE = "pdfutils.geminiApiKey";

// ---- tabs ----
document.querySelectorAll(".tab").forEach(btn => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach(b => b.classList.toggle("active", b === btn));
    document.querySelectorAll(".panel").forEach(p => {
      p.classList.toggle("active", p.id === "tab-" + btn.dataset.tab);
    });
  });
});

// ---- file list ----
const fileInput = document.getElementById("files");
const fileList  = document.getElementById("file-list");
fileInput.addEventListener("change", () => {
  fileList.innerHTML = "";
  [...fileInput.files].forEach(f => {
    const li = document.createElement("li");
    li.textContent = `${f.name} — ${(f.size / 1024).toFixed(0)} KB`;
    fileList.appendChild(li);
  });
});

// ---- output helper ----
const output = document.getElementById("output");
function showOutput(msg, isError = false) {
  output.textContent = msg;
  output.classList.add("show");
  output.classList.toggle("error", isError);
}

// ---- API key handling ----
const keyInput  = document.getElementById("api-key");
const keyStatus = document.getElementById("key-status");
const savedKey = localStorage.getItem(KEY_STORAGE);
if (savedKey) {
  keyInput.value = savedKey;
  keyStatus.textContent = "key loaded from localStorage";
}
document.getElementById("save-key").addEventListener("click", () => {
  const k = keyInput.value.trim();
  if (!k) { keyStatus.textContent = "nothing to save"; return; }
  localStorage.setItem(KEY_STORAGE, k);
  keyStatus.textContent = "key saved";
});
document.getElementById("remove-key").addEventListener("click", () => {
  localStorage.removeItem(KEY_STORAGE);
  keyInput.value = "";
  keyStatus.textContent = "stored key removed";
});

// ---- download helper ----
async function postAndDownload(url, formData, button, label) {
  if (!fileInput.files.length) {
    showOutput("Load at least one PDF first (tab: Load document(s)).", true);
    return;
  }
  [...fileInput.files].forEach(f => formData.append("files", f, f.name));

  const prev = button.textContent;
  button.disabled = true;
  button.textContent = "Working…";
  try {
    const resp = await fetch(url, { method: "POST", body: formData });
    if (!resp.ok) {
      let msg = `HTTP ${resp.status}`;
      try { msg = (await resp.json()).error || msg; } catch (_) {}
      showOutput("Error: " + msg, true);
      return;
    }
    const blob = await resp.blob();
    const cd = resp.headers.get("Content-Disposition") || "";
    const m = /filename="?([^"]+)"?/.exec(cd);
    const name = m ? m[1] : label;
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = name;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(a.href);
    showOutput(`Done. Downloaded ${name}.`);
  } catch (e) {
    showOutput("Network error: " + e.message, true);
  } finally {
    button.disabled = false;
    button.textContent = prev;
  }
}

// ---- split by pages ----
function parseRanges(text) {
  const out = [];
  for (const raw of text.split(/\r?\n/)) {
    const line = raw.trim();
    if (!line) continue;
    // "start-end" or "start-end:name"
    const m = /^(\d+)\s*-\s*(\d+)\s*(?::\s*(.+))?$/.exec(line);
    if (!m) throw new Error(`Bad range line: "${line}"`);
    out.push({
      start: parseInt(m[1], 10),
      end:   parseInt(m[2], 10),
      name:  (m[3] || "").trim() || `pages_${m[1]}-${m[2]}`,
    });
  }
  return out;
}

document.getElementById("do-pages").addEventListener("click", async (ev) => {
  let ranges;
  try { ranges = parseRanges(document.getElementById("ranges").value); }
  catch (e) { showOutput(e.message, true); return; }
  if (!ranges.length) { showOutput("Enter at least one page range.", true); return; }

  const fd = new FormData();
  fd.append("ranges", JSON.stringify(ranges));
  await postAndDownload("/api/split/by-pages", fd, ev.currentTarget, "split_by_pages.zip");
});

// ---- split by chapters ----
document.getElementById("do-chapters").addEventListener("click", async (ev) => {
  const key = (keyInput.value || localStorage.getItem(KEY_STORAGE) || "").trim();
  if (!key) { showOutput("Enter a Gemini API key first.", true); return; }
  const fd = new FormData();
  fd.append("api_key", key);
  await postAndDownload("/api/split/by-chapters", fd, ev.currentTarget, "split_by_chapters.zip");
});
