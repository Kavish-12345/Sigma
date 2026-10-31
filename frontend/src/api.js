// All backend calls. The backend is stateless; React keeps the conversation and form.

async function request(path, options = {}) {
  const res = await fetch(path, {
    headers: { "Content-Type": "application/json" },
    ...options,
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      detail = (await res.json()).detail || detail;
    } catch (_) {}
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return res.json();
}

export const getOptions = () => request("/api/options");
export const getPrices = () => request("/api/prices");
export const getMaterialsSummary = () => request("/api/materials/summary");
export const sendChat = (messages, form) =>
  request("/api/chat", { method: "POST", body: JSON.stringify({ messages, form }) });
// Streaming chat: calls onEvent for each server event (token / recorded / done / error).
export async function streamChat(messages, form, onEvent, directAnswer = null) {
  const res = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      messages: messages.map(({ role, content }) => ({ role, content })),
      form,
      direct_answer: directAnswer,
    }),
  });
  if (!res.ok || !res.body) throw new Error(`Chat failed (${res.status})`);
  const reader = res.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });
    let i;
    while ((i = buffer.indexOf("\n\n")) >= 0) {
      const line = buffer.slice(0, i).trim();
      buffer = buffer.slice(i + 2);
      if (line.startsWith("data:")) onEvent(JSON.parse(line.slice(5)));
    }
  }
}

export const getRecommendation = (payload) =>
  request("/api/recommend", { method: "POST", body: JSON.stringify(payload) });

export const REQUIRED = ["part_name", "function", "design_inputs", "properties", "primary_property", "environment", "weight_priority", "cost_priority"];
export const OPTIONAL = ["volume_cm3", "min_limits", "include", "avoid"];
export const FIELD_ORDER = [...REQUIRED, ...OPTIONAL];

export const FIELD_LABELS = {
  part_name: "Part name",
  part_detail: "Part detail",
  function: "Function",
  design_inputs: "Loads & sizes",
  properties: "Properties",
  primary_property: "Most important",
  environment: "Environment",
  weight_priority: "Low weight",
  cost_priority: "Low cost",
  volume_cm3: "Part volume",
  min_limits: "Minimum limits",
  include: "Must include",
  avoid: "Avoid",
};

export function emptyForm() {
  const f = {};
  FIELD_ORDER.forEach((k) => (f[k] = null));
  f.user_words = {};
  f.suggested = [];
  f.design_draft = {};
  return f;
}

// design input fields of one function (from /api/options)
export const designInputs = (options, fn) => options?.fields?.function?.find((o) => o.value === fn)?.inputs || [];

// show_if: the input applies only for these values of another (select) input
export const inputShown = (i, vals) => !i.show_if || i.show_if.in.includes(vals[i.show_if.key]);

// required = a number with no default that is not optional; need_one = at least one of a group
export function designComplete(inputs, vals) {
  const shown = inputs.filter((i) => inputShown(i, vals));
  const group = shown.filter((i) => i.need_one);
  return (
    shown.every((i) => i.type === "select" || i.optional || "default" in i || (vals[i.key] !== "" && vals[i.key] != null && Number(vals[i.key]) > 0)) &&
    (!group.length || group.some((i) => Number(vals[i.key]) > 0))
  );
}

// Same conversion for chat and No-AI form, so both give identical results.
export function toPayload(form, priceOverrides = {}, topN = 5) {
  const vol = form.volume_cm3;
  return {
    part_name: form.part_name || null,
    function: form.function || "not_sure",
    design_inputs: form.design_inputs && typeof form.design_inputs === "object" ? form.design_inputs : null,
    properties: form.properties && form.properties.length ? form.properties : ["not_sure"],
    primary_property: form.primary_property && form.primary_property !== "not_sure" ? form.primary_property : null,
    environment: form.environment || "not_sure",
    weight_priority: form.weight_priority || "not_sure",
    cost_priority: form.cost_priority || "not_sure",
    volume_cm3: typeof vol === "number" && vol > 0 ? vol : null,
    min_limits: form.min_limits && typeof form.min_limits === "object" ? form.min_limits : {},
    include: Array.isArray(form.include) ? form.include : [],
    avoid: Array.isArray(form.avoid) ? form.avoid : [],
    part_notes: Array.isArray(form.notes) ? form.notes : [],
    part_type: form.part_type || null,
    price_overrides: priceOverrides,
    top_n: topN,
  };
}

export const missingFields = (form) => FIELD_ORDER.filter((k) => form[k] === null || form[k] === undefined);
