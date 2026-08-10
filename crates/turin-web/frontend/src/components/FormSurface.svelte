<script lang="ts">
  import type { JsonValue, UiFormField, UiNode } from "../lib/types";
  import { displayValue } from "../lib/format";

  export let node: UiNode;
  export let running = false;
  export let onSubmit: (action: string, params: Record<string, JsonValue>) => void;

  $: fields = (node.fields ?? []).filter((field): field is UiFormField => typeof field !== "string");
  let values: Record<string, JsonValue> = {};
  let initializedFor = "";
  $: if (node.id !== initializedFor) initialize();

  function initialize() {
    initializedFor = node.id ?? node.title ?? "form";
    const staticParams = node.params && typeof node.params === "object" && !Array.isArray(node.params) ? node.params : {};
    values = { ...staticParams };
    for (const field of fields) {
      if (!(field.name in values) && field.default !== undefined) values[field.name] = field.default;
    }
  }

  function inputType(field: UiFormField): string {
    const kind = field.kind?.toLowerCase() ?? "text";
    if (["password", "secret", "passphrase"].includes(kind)) return "password";
    if (["number", "integer", "float", "decimal"].includes(kind)) return "number";
    return "text";
  }

  function update(field: UiFormField, raw: string | boolean) {
    const kind = field.kind?.toLowerCase() ?? "text";
    if (typeof raw === "boolean") values = { ...values, [field.name]: raw };
    else if (["number", "integer", "float", "decimal"].includes(kind)) values = { ...values, [field.name]: raw === "" ? null : Number(raw) };
    else values = { ...values, [field.name]: raw };
  }

  function encodeOption(value: JsonValue): string {
    return JSON.stringify(value);
  }

  function updateOption(field: UiFormField, raw: string) {
    try {
      values = { ...values, [field.name]: JSON.parse(raw) as JsonValue };
    } catch {
      values = { ...values, [field.name]: raw };
    }
  }

  function selectedOption(field: UiFormField): string {
    const value = values[field.name];
    return value === undefined ? "" : encodeOption(value);
  }

  function submit(event: SubmitEvent) {
    event.preventDefault();
    if (node.action) onSubmit(node.action, values);
  }
</script>

<section class="form-surface surface-card">
  <header class="surface-header"><div><span class="surface-kicker">Input</span><h2>{node.title}</h2></div></header>
  <form onsubmit={submit}>
    <div class="form-grid">
      {#each fields as field (field.name)}
        <label class:full-field={["multiline", "markdown", "textarea"].includes(field.kind?.toLowerCase() ?? "")}>
          <span>{field.label}{field.required ? " *" : ""}</span>
          {#if ["multiline", "markdown", "textarea"].includes(field.kind?.toLowerCase() ?? "")}
            <textarea rows="4" required={field.required} value={displayValue(values[field.name]) === "-" ? "" : displayValue(values[field.name])} oninput={event => update(field, event.currentTarget.value)}></textarea>
          {:else if field.options?.length}
            <select required={field.required} value={selectedOption(field)} onchange={event => updateOption(field, event.currentTarget.value)}>
              <option value="" disabled>Select an option</option>
              {#each field.options as option}<option value={encodeOption(option)}>{displayValue(option)}</option>{/each}
            </select>
          {:else if ["bool", "boolean", "checkbox"].includes(field.kind?.toLowerCase() ?? "")}
            <span class="switch-field"><input type="checkbox" checked={values[field.name] === true} onchange={event => update(field, event.currentTarget.checked)} /><i></i><em>{values[field.name] === true ? "On" : "Off"}</em></span>
          {:else}
            <input type={inputType(field)} required={field.required} value={displayValue(values[field.name]) === "-" ? "" : displayValue(values[field.name])} oninput={event => update(field, event.currentTarget.value)} />
          {/if}
        </label>
      {/each}
    </div>
    <div class="form-actions"><button class="primary-button" type="submit" disabled={running}>{running ? "Running..." : "Submit"}</button></div>
  </form>
</section>
