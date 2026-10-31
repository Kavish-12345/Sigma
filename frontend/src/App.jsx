import { useEffect, useRef, useState } from "react";
import { FIELD_ORDER, emptyForm, getMaterialsSummary, getOptions, getRecommendation, missingFields, toPayload } from "./api.js";
import { addToHistory, clearHistory, clearSession, loadHistory, loadSession, removeFromHistory, saveSession } from "./storage.js";
import ChatWindow from "./components/ChatWindow.jsx";
import NoAIForm from "./components/NoAIForm.jsx";
import SummaryCard from "./components/SummaryCard.jsx";
import Results from "./components/Results.jsx";
import { GearLoader, Logo } from "./components/mech.jsx";

const saved = loadSession() || {};

function timeAgo(iso) {
  const s = (Date.now() - new Date(iso).getTime()) / 1000;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)} min ago`;
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`;
  return new Date(iso).toLocaleDateString();
}

export default function App({ start, onStartConsumed, onHome }) {
  const [options, setOptions] = useState(null);
  const [mode, setMode] = useState(saved.mode || "chat");
  const [form, setForm] = useState(saved.form || emptyForm());
  const [messages, setMessages] = useState(saved.messages || []);
  const [question, setQuestion] = useState(saved.question || null);
  const [result, setResult] = useState(saved.result || null);
  const [priceOverrides, setPriceOverrides] = useState(saved.priceOverrides || {});
  const [history, setHistory] = useState(loadHistory());
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [menu, setMenu] = useState(null); // "req" | "history" | null
  const [pendingPrompt, setPendingPrompt] = useState(null);
  const [summary, setSummary] = useState(null);
  const menuRef = useRef(null);

  useEffect(() => {
    getOptions().then(setOptions).catch((e) => setError("Backend not reachable: " + e.message));
    getMaterialsSummary().then(setSummary).catch(() => {});
  }, []);

  // arriving from the homepage: start fresh in the chosen mode (optionally with an example prompt)
  useEffect(() => {
    if (!start) return;
    newSearch();
    setMode(start.mode || "chat");
    if (start.prompt) setPendingPrompt(start.prompt);
    onStartConsumed?.();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [start]);

  // keep the current session in browser storage (survives refresh)
  useEffect(() => {
    saveSession({ mode, form, messages, question, result, priceOverrides });
  }, [mode, form, messages, question, result, priceOverrides]);

  useEffect(() => {
    const close = (e) => menuRef.current && !menuRef.current.contains(e.target) && setMenu(null);
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, []);

  async function runRecommend(f = form, overrides = priceOverrides, save = true) {
    setBusy(true);
    setError("");
    try {
      const res = await getRecommendation(toPayload(f, overrides));
      if (!res.results.length && res.problem_fields?.length) {
        recover(res, f);
        return;
      }
      setResult(res);
      if (save && res.results.length) {
        setHistory(
          addToHistory({
            id: Date.now(),
            at: new Date().toISOString(),
            part_name: f.part_name || "Untitled part",
            top: res.results[0].display_name,
            form: f,
            priceOverrides: overrides,
            messages,
            result: res,
          })
        );
      }
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  // Empty result: clear the answers that caused it and ask those questions again.
  function recover(res, f) {
    const cleared = { ...f, user_words: { ...(f.user_words || {}) } };
    res.problem_fields.forEach((k) => {
      cleared[k] = null;
      delete cleared.user_words[k];
    });
    setForm(cleared);
    setResult(null);
    if (mode === "chat") {
      setMessages((m) => [
        ...m,
        { role: "assistant", content: "", warn: res.message, recorded: [] },
        { role: "assistant", content: "Let's fix that. Please answer this question again." },
      ]);
      setQuestion(res.question);
    } else {
      setError(res.message + " Please change the highlighted answers and review again.");
    }
  }

  function setOverride(group, value) {
    const next = { ...priceOverrides };
    if (value === null || value === "" || Number(value) <= 0) delete next[group];
    else next[group] = Number(value);
    setPriceOverrides(next);
    runRecommend(form, next, false);
  }

  function newSearch() {
    setForm(emptyForm());
    setMessages([]);
    setQuestion(null);
    setResult(null);
    setPriceOverrides({});
    setError("");
    setMode("chat");
    clearSession();
  }

  function openHistory(h) {
    setForm(h.form);
    setPriceOverrides(h.priceOverrides || {});
    setMessages(h.messages || []);
    setQuestion(null);
    setResult(h.result);
    setMenu(null);
    setError("");
  }

  const complete = missingFields(form).length === 0;
  const done = FIELD_ORDER.length - missingFields(form).length;

  return (
    <div className="app">
      <header className="topbar">
        <div className="brand" onClick={onHome} title="Home">
          <Logo size={32} />
          <div>
            <h1 className="wordmark">SIGMA</h1>
            <p>Material selection engine</p>
          </div>
        </div>

        {options && (
          <div className="top-actions" ref={menuRef}>
            {!result && mode === "chat" && (
              <div className="pill-wrap">
                <button className={`pill ${complete ? "ok" : ""}`} onClick={() => setMenu(menu === "req" ? null : "req")}>
                  Requirements · {done}/{FIELD_ORDER.length} {menu === "req" ? "▴" : "▾"}
                </button>
                {menu === "req" && (
                  <div className="pill-menu">
                    <SummaryCard variant="list" options={options} form={form} />
                  </div>
                )}
              </div>
            )}

            <div className="pill-wrap">
              <button className="pill" onClick={() => setMenu(menu === "history" ? null : "history")}>
                Recent {history.length ? `· ${history.length}` : ""} {menu === "history" ? "▴" : "▾"}
              </button>
              {menu === "history" && (
                <div className="pill-menu history">
                  <div className="history-head">
                    <strong>Recent searches</strong>
                    {history.length > 0 && (
                      <button className="link small" onClick={() => setHistory(clearHistory())}>
                        Clear all
                      </button>
                    )}
                  </div>
                  {history.length === 0 && <p className="muted small">No searches yet. Results you confirm are saved in this browser.</p>}
                  {history.map((h) => (
                    <div key={h.id} className="history-item" onClick={() => openHistory(h)}>
                      <div>
                        <div className="h-title">{h.part_name}</div>
                        <div className="muted small">
                          #1 {h.top} · {timeAgo(h.at)}
                        </div>
                      </div>
                      <button
                        className="x"
                        title="Remove"
                        onClick={(e) => {
                          e.stopPropagation();
                          setHistory(removeFromHistory(h.id));
                        }}
                      >
                        ×
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <button className="pill" onClick={newSearch} title="Start a new search">+ New</button>
            {!result && (
              <div className="mode-toggle" role="tablist">
                <button className={mode === "chat" ? "active" : ""} onClick={() => setMode("chat")}>
                  Chat
                </button>
                <button className={mode === "form" ? "active" : ""} onClick={() => setMode("form")}>
                  Form (No AI)
                </button>
              </div>
            )}
          </div>
        )}
      </header>

      {error && <div className="error">{error}</div>}

      {busy && (
        <div className="overlay">
          <GearLoader label={result ? "Re-ranking with your price" : "Ranking materials"} />
        </div>
      )}

      {!options && !error && <div className="muted center">Loading…</div>}

      {options && !result && mode === "chat" && (
        <ChatWindow
          options={options}
          messages={messages}
          setMessages={setMessages}
          form={form}
          setForm={setForm}
          question={question}
          setQuestion={setQuestion}
          pendingPrompt={pendingPrompt}
          onPromptConsumed={() => setPendingPrompt(null)}
          busy={busy}
          onConfirm={() => runRecommend()}
          onEdit={() => setMode("form")}
        />
      )}

      {options && !result && mode === "form" && (
        <main className="main page-scroll">
          <div className="input-layout">
            <section className="panel">
              <NoAIForm options={options} form={form} setForm={setForm} onReview={() => setError("")} />
            </section>
            <aside className="panel side">
              <SummaryCard
                options={options}
                form={form}
                complete={complete}
                busy={busy}
                onConfirm={() => runRecommend()}
                onEdit={() => setMode("form")}
              />
            </aside>
          </div>
        </main>
      )}

      {result && (
        <main className="main page-scroll">
          <Results
            summary={summary}
            result={result}
            options={options}
            form={form}
            busy={busy}
            priceOverrides={priceOverrides}
            onOverride={setOverride}
            onEdit={() => setResult(null)}
            onNew={newSearch}
          />
          <footer className="footer">
            Suitability from an ML model trained on the department dataset · Ranking by TOPSIS · Prices are cited market
            ranges (indicative, not quotes). The chat only collects requirements; it never selects materials.
          </footer>
        </main>
      )}
    </div>
  );
}
