// Turn stored form values into readable text using /api/options.

export function optionLabel(options, field, value) {
  const list = options?.fields?.[field];
  const hit = list && list.find((o) => o.value === value);
  return hit ? hit.label : value;
}

export function groupLabel(options, key) {
  const fam = { ferrous: "Steels & cast irons", aluminium: "Aluminium alloys", copper: "Copper alloys", magnesium: "Magnesium alloys" };
  return fam[key] || options?.material_groups?.[key] || key;
}

export function formatValue(options, field, value) {
  if (value === null || value === undefined) return null;
  if (value === "not_sure") return "Not sure";
  switch (field) {
    case "part_name":
      return value;
    case "function":
      return optionLabel(options, field, value).split(" (")[0];
    case "design_inputs": {
      const fnOpt =
        options?.fields?.function?.find((o) => Object.keys(value).every((k) => o.inputs.some((i) => i.key === k))) || null;
      const specs = fnOpt?.inputs || [];
      return Object.entries(value)
        .filter(([, v]) => v !== null && v !== "" && v !== 0)
        .map(([k, v]) => {
          const s = specs.find((i) => i.key === k);
          if (s?.type === "select") return s.options.find((o) => o.value === v)?.label.split(" (")[0] || v;
          return `${s?.label || k} ${v}${s?.unit ? " " + s.unit : ""}`;
        })
        .join(", ");
    }
    case "properties":
      return value.includes("not_sure") ? "Not sure" : value.join(", ");
    case "primary_property":
      return value;
    case "volume_cm3":
      return typeof value === "number" ? `${Number((value / 1e6).toPrecision(6))} m³` : "Not given"; // stored in cm³
    case "min_limits": {
      const keys = Object.keys(value || {});
      if (!keys.length) return "None";
      return keys.map((k) => `${options?.min_limits?.[k] || k}: ${value[k]}`).join(", ");
    }
    case "include":
    case "avoid":
      return value.length ? value.map((v) => groupLabel(options, v)).join(", ") : "None";
    default:
      return optionLabel(options, field, value);
  }
}
