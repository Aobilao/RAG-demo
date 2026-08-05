"use strict";

const $ = (id) => document.getElementById(id);
const esc = (s) =>
  String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

let noticeTimer;
function notice(message) {
  const el = $("notice");
  el.textContent = message;
  el.hidden = false;
  clearTimeout(noticeTimer);
  noticeTimer = setTimeout(() => (el.hidden = true), 5200);
}

/* Textareas grow with their content rather than scrolling. */
function autoGrow(el) {
  el.style.height = "auto";
  el.style.height = `${el.scrollHeight}px`;
}
document.querySelectorAll(".query textarea").forEach((el) => {
  el.addEventListener("input", () => autoGrow(el));
});

/* ── Modes ─────────────────────────────────────────────────────────── */
document.querySelectorAll(".mode").forEach((mode) => {
  mode.addEventListener("click", () => {
    document.querySelectorAll(".mode").forEach((m) => {
      const on = m === mode;
      m.classList.toggle("is-on", on);
      if (on) m.setAttribute("aria-current", "page");
      else m.removeAttribute("aria-current");
    });
    document.querySelectorAll(".view").forEach((v) =>
      v.classList.toggle("is-on", v.id === `view-${mode.dataset.view}`)
    );
    if (mode.dataset.view === "gallery") loadGalleryState();
  });
});

/* ── Index state ───────────────────────────────────────────────────── */
function setState(kind, text) {
  $("tick").className = `tick ${kind}`;
  $("state-text").textContent = text;
}

const describe = ({ chunks, sources }) =>
  `${chunks.toLocaleString("vi-VN")} đoạn · ${sources.length} tài liệu`;

async function loadState() {
  setState("work", "đang mở chỉ mục");
  try {
    const res = await fetch("/api/status");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    if (data.chunks) setState("on", describe(data));
    else setState("off", "chỉ mục trống — thêm PDF vào docs/");
  } catch (err) {
    setState("off", "không kết nối được máy chủ");
    notice(`Không đọc được trạng thái chỉ mục: ${err.message}`);
  }
}

$("rescan").addEventListener("click", async () => {
  const button = $("rescan");
  button.disabled = true;
  setState("work", "đang quét docs/");
  try {
    const res = await fetch("/api/refresh", { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    setState("on", describe(data));
    notice("Đã quét lại thư mục docs/.");
  } catch (err) {
    setState("off", "quét lại thất bại");
    notice(`Quét lại thất bại: ${err.message}`);
  } finally {
    button.disabled = false;
  }
});

/* ── Marginal notes ────────────────────────────────────────────────── */
function noteMarkup(e) {
  return `
    <div class="note" id="note-${e.rank}" data-rank="${e.rank}">
      <p class="note-head">
        <span class="note-n">${e.rank}</span>
        <span class="note-src" title="${esc(e.source)}">${esc(e.source)}</span>
        <span class="note-pg">tr. ${esc(e.pages)}</span>
        <span class="note-sc">${e.score.toFixed(3)}</span>
      </p>
      <p class="note-text">${esc(e.text)}</p>
    </div>`;
}

function renderNotes(host, evidence, heading) {
  host.innerHTML =
    `<p class="apparatus-head">${heading}</p>` + evidence.map(noteMarkup).join("");
  host.querySelectorAll(".note").forEach((note) => {
    note.addEventListener("click", () => {
      note.classList.toggle("open");
      alignNotes(host);
    });
  });
}

/* Place each note beside the line that cites it.

   Cited notes go first, in the order their citations appear, so that an uncited
   one cannot sit between two of them and push the later one past its line. Notes
   are pushed down when they would collide, so they stay in reading order and
   never overlap. Whatever was never cited settles underneath, still visible — a
   retrieval demo that hides its misses is not demonstrating retrieval.

   `settled` marks the answer as complete. Until it is there are no citations to
   anchor to, and no passage can yet be called uncited, since the sentence citing
   it may still be arriving; notes queue up in rank order and glide into place
   once the answer lands. */
function alignNotes(host, prose) {
  if (window.matchMedia("(max-width: 860px)").matches) return;

  const column = prose || host.parentElement.querySelector(".prose");
  if (!column) return;

  const settled = host.dataset.settled === "1";

  const origin = host.getBoundingClientRect().top;
  const anchors = new Map();
  column.querySelectorAll(".cite").forEach((cite) => {
    const rank = cite.dataset.rank;
    if (!anchors.has(rank)) {
      anchors.set(rank, cite.getBoundingClientRect().top - origin);
    }
  });

  const cited = [];
  const uncited = [];
  for (const note of host.querySelectorAll(".note")) {
    (anchors.has(note.dataset.rank) ? cited : uncited).push(note);
  }
  cited.sort((a, b) => anchors.get(a.dataset.rank) - anchors.get(b.dataset.rank));

  let floor = 26;
  for (const note of cited) {
    const top = Math.max(floor, anchors.get(note.dataset.rank) - 2);
    note.style.top = `${top}px`;
    note.classList.remove("unused");
    note.classList.add("placed");
    floor = top + note.offsetHeight + 20;
  }
  for (const note of uncited) {
    note.style.top = `${floor}px`;
    note.classList.toggle("unused", settled);
    note.classList.add("placed");
    floor += note.offsetHeight + 20;
  }

  host.style.minHeight = `${floor}px`;
}

function lightUp(rank) {
  const note = $(`note-${rank}`);
  if (!note) return;

  document.querySelectorAll(".note.lit, .cite.lit").forEach((el) => el.classList.remove("lit"));
  note.classList.add("lit", "open");
  document.querySelectorAll(`.cite[data-rank="${rank}"]`).forEach((c) => c.classList.add("lit"));

  alignNotes(note.parentElement);
  const box = note.getBoundingClientRect();
  if (box.top < 70 || box.bottom > innerHeight - 20) {
    note.scrollIntoView({ behavior: "smooth", block: "center" });
  }
}

/* Rewrite [1] / [2][3] into buttons pointing at their passage. */
function markCitations(container) {
  container.innerHTML = esc(container.textContent).replace(
    /\[(\d+)\]/g,
    (_, n) => `<button class="cite" type="button" data-rank="${n}"
                 aria-label="Xem đoạn ${n}">${n}</button>`
  );
  container.querySelectorAll(".cite").forEach((cite) =>
    cite.addEventListener("click", () => lightUp(cite.dataset.rank))
  );
}

const WAIT = '<div class="wait"><i></i><i></i><i></i></div>';

/* ── Ask ───────────────────────────────────────────────────────────── */
let asking = false;

/* Ask a question and render the answer as it streams.

   The question is cleared on submit: it has been asked and is about to be answered
   below, and leaving it in the box invites re-reading it as though still pending.
   It goes back on failure, so a long question is never lost to a dropped
   connection.

   Notes are placed the moment the sources arrive rather than at the end. They are
   positioned absolutely, so notes with no `top` yet would otherwise pile up at the
   origin for the whole of the stream. */
async function ask(submitEvent) {
  submitEvent.preventDefault();
  const question = $("question").value.trim();
  if (!question || asking) return;

  asking = true;
  $("ask-go").disabled = true;
  $("ask-invite").hidden = true;
  $("ask-out").hidden = false;
  $("answer").textContent = "";
  $("ask-foot").textContent = "";
  $("apparatus").innerHTML = WAIT;
  $("apparatus").dataset.settled = "";

  $("question").value = "";
  autoGrow($("question"));

  const started = performance.now();
  let answer = "";
  let evidence = [];

  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        top_n: Number($("ask-top-n").value),
        mode: $("ask-mode").value,
        rerank: $("ask-rerank").checked,
        temperature: Number($("ask-temperature").value),
      }),
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);

    const reader = res.body.pipeThrough(new TextDecoderStream()).getReader();
    let buffer = "";

    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += value;

      const frames = buffer.split("\n\n");
      buffer = frames.pop();

      for (const frame of frames) {
        const name = frame.match(/^event: (.+)$/m)?.[1];
        const raw = frame.match(/^data: ([\s\S]*)$/m)?.[1];
        if (!name || raw === undefined) continue;
        const data = JSON.parse(raw);

        if (name === "sources") {
          evidence = data;
          renderNotes($("apparatus"), data, `${data.length} đoạn truy xuất`);
          alignNotes($("apparatus"), $("answer"));
        } else if (name === "token") {
          answer += data;
          $("answer").textContent = answer;
        } else if (name === "empty") {
          $("apparatus").innerHTML = "";
          $("answer").textContent =
            "Không có đoạn nào trong docs/ khớp với câu hỏi này. Thử diễn đạt lại, hoặc tăng số đoạn truy xuất.";
        } else if (name === "error") {
          throw new Error(data);
        }
      }
    }

    if (answer) {
      markCitations($("answer"));
      $("apparatus").dataset.settled = "1";
      alignNotes($("apparatus"), $("answer"));

      const cited = new Set([...document.querySelectorAll("#answer .cite")].map((c) => c.dataset.rank));
      const seconds = ((performance.now() - started) / 1000).toFixed(1);
      $("ask-foot").textContent =
        `${cited.size}/${evidence.length} đoạn được trích dẫn · ${$("ask-mode").value}` +
        `${$("ask-rerank").checked ? " + rerank" : ""} · ${seconds}s`;
    }
  } catch (err) {
    $("answer").textContent = "";
    $("apparatus").innerHTML = "";
    $("ask-out").hidden = true;
    $("ask-invite").hidden = false;
    $("question").value = question;
    autoGrow($("question"));
    notice(`Không hoàn thành câu trả lời: ${err.message}`);
  } finally {
    asking = false;
    $("ask-go").disabled = false;
  }
}

$("ask-form").addEventListener("submit", ask);
$("question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("ask-form").requestSubmit(); }
});

/* Notes are positioned from measured geometry, so they must be re-placed
   whenever the column reflows. */
let resizeTimer;
addEventListener("resize", () => {
  clearTimeout(resizeTimer);
  resizeTimer = setTimeout(() => {
    if (!$("ask-out").hidden) alignNotes($("apparatus"), $("answer"));
  }, 140);
});

/* ── Gallery ───────────────────────────────────────────────────────── */

/* The gallery is only indexed on the server the first time it is asked for, since
   captioning every image costs real time. Loading its status is deferred to the
   first visit to the tab for the same reason, rather than paid at page load. */
async function loadGalleryState() {
  $("gallery-count").textContent = "đang mở thư viện ảnh…";
  try {
    const res = await fetch("/api/gallery/status");
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const { count } = await res.json();
    $("gallery-count").textContent =
      count ? `${count.toLocaleString("vi-VN")} ảnh` : "chưa có ảnh nào trong images/";
  } catch (err) {
    $("gallery-count").textContent = "không mở được thư viện ảnh";
    notice(`Không đọc được thư viện ảnh: ${err.message}`);
  }
}

$("gallery-rescan").addEventListener("click", async () => {
  const button = $("gallery-rescan");
  button.disabled = true;
  $("gallery-count").textContent = "đang quét images/…";
  try {
    const res = await fetch("/api/gallery/refresh", { method: "POST" });
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail || `HTTP ${res.status}`);
    $("gallery-count").textContent = `${data.count.toLocaleString("vi-VN")} ảnh`;
    notice("Đã quét lại thư mục images/.");
  } catch (err) {
    $("gallery-count").textContent = "quét lại thất bại";
    notice(`Quét lại thất bại: ${err.message}`);
  } finally {
    button.disabled = false;
  }
});

function tileMarkup(match) {
  return `
    <figure class="tile">
      <a href="${match.url}" target="_blank" rel="noopener">
        <img src="${match.url}" alt="${esc(match.caption)}" loading="lazy">
      </a>
      <figcaption>
        <span class="tile-cap">${esc(match.caption)}</span>
        <span class="tile-sc">${match.score.toFixed(3)}</span>
      </figcaption>
    </figure>`;
}

let searchingGallery = false;

async function searchGallery(submitEvent) {
  submitEvent.preventDefault();
  const query = $("gallery-query").value.trim();
  if (!query || searchingGallery) return;

  searchingGallery = true;
  $("gallery-go").disabled = true;
  $("gallery-invite").hidden = true;
  $("gallery-out").hidden = false;
  $("gallery-out").innerHTML = WAIT;

  try {
    const res = await fetch("/api/gallery/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, top_n: Number($("gallery-top-n").value) }),
    });
    const matches = await res.json();
    if (!res.ok) throw new Error(matches.detail || `HTTP ${res.status}`);

    $("gallery-out").innerHTML = matches.length
      ? matches.map(tileMarkup).join("")
      : '<p class="invite-note">Không có ảnh nào khớp. Hãy thêm ảnh vào images/ rồi quét lại.</p>';
  } catch (err) {
    $("gallery-out").innerHTML = "";
    notice(`Không tìm được ảnh: ${err.message}`);
  } finally {
    searchingGallery = false;
    $("gallery-go").disabled = false;
  }
}

$("gallery-form").addEventListener("submit", searchGallery);
$("gallery-query").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("gallery-form").requestSubmit(); }
});

loadState();
