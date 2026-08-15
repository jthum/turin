<script lang="ts">
  import { onMount } from "svelte";
  import { indentWithTab } from "@codemirror/commands";
  import {
    HighlightStyle,
    StreamLanguage,
    indentUnit,
    syntaxHighlighting,
  } from "@codemirror/language";
  import { lua } from "@codemirror/legacy-modes/mode/lua";
  import { EditorState, Prec } from "@codemirror/state";
  import { EditorView, keymap } from "@codemirror/view";
  import { tags } from "@lezer/highlight";
  import { basicSetup } from "codemirror";

  export let value: string;
  export let ariaLabel = "Lua source";
  export let onChange: (value: string) => void;
  export let onSave: () => void;
  export let onCursorChange: (line: number, column: number) => void;

  let host: HTMLDivElement;
  let view: EditorView | null = null;
  let applyingExternalValue = false;

  const turinTheme = EditorView.theme({
    "&": {
      height: "100%",
      minHeight: "410px",
      backgroundColor: "var(--surface-raised)",
      color: "var(--ink)",
      fontSize: "12px",
    },
    "&.cm-focused": { outline: "none" },
    ".cm-scroller": {
      fontFamily: '"JetBrains Mono", "SFMono-Regular", monospace',
      lineHeight: "1.7",
      overflow: "auto",
    },
    ".cm-content": {
      minHeight: "100%",
      padding: "14px 0",
      caretColor: "var(--ink)",
    },
    ".cm-line": { padding: "0 18px 0 9px" },
    ".cm-cursor, .cm-dropCursor": { borderLeftColor: "var(--accent)" },
    ".cm-selectionBackground, &.cm-focused .cm-selectionBackground, ::selection": {
      backgroundColor: "color-mix(in srgb, var(--blue) 20%, transparent)",
    },
    ".cm-activeLine": {
      backgroundColor: "color-mix(in srgb, var(--accent) 5%, transparent)",
    },
    ".cm-gutters": {
      border: "0",
      borderRight: "1px solid var(--line)",
      backgroundColor: "var(--surface-muted)",
      color: "var(--faint)",
    },
    ".cm-activeLineGutter": {
      backgroundColor: "color-mix(in srgb, var(--accent) 8%, var(--surface-muted))",
      color: "var(--accent-strong)",
    },
    ".cm-lineNumbers .cm-gutterElement": {
      minWidth: "34px",
      padding: "0 9px 0 5px",
      fontSize: "10px",
    },
    ".cm-foldGutter .cm-gutterElement": {
      width: "16px",
      color: "var(--faint)",
      fontSize: "10px",
    },
    ".cm-matchingBracket": {
      borderBottom: "1px solid var(--accent)",
      backgroundColor: "var(--accent-soft)",
      color: "var(--accent-strong)",
    },
    ".cm-searchMatch": {
      outline: "1px solid color-mix(in srgb, var(--warning) 65%, transparent)",
      backgroundColor: "color-mix(in srgb, var(--warning) 16%, transparent)",
    },
    ".cm-searchMatch.cm-searchMatch-selected": {
      backgroundColor: "color-mix(in srgb, var(--accent) 22%, transparent)",
    },
    ".cm-panels": {
      borderBottom: "1px solid var(--line)",
      backgroundColor: "var(--surface-muted)",
      color: "var(--ink)",
    },
    ".cm-search": { gap: "6px", padding: "7px 9px" },
    ".cm-search input": {
      border: "1px solid var(--line-strong)",
      borderRadius: "6px",
      outline: "none",
      backgroundColor: "var(--surface-raised)",
      color: "var(--ink)",
      padding: "4px 7px",
    },
    ".cm-search input:focus": { borderColor: "var(--accent)" },
    ".cm-search button": {
      border: "1px solid var(--line)",
      borderRadius: "6px",
      backgroundColor: "var(--surface-raised)",
      color: "var(--muted)",
      padding: "4px 7px",
      fontSize: "10px",
    },
    ".cm-tooltip": {
      border: "1px solid var(--line)",
      borderRadius: "8px",
      backgroundColor: "var(--surface-raised)",
      color: "var(--ink)",
      boxShadow: "var(--shadow-lg)",
      overflow: "hidden",
    },
    ".cm-tooltip-autocomplete > ul > li[aria-selected]": {
      backgroundColor: "var(--accent-soft)",
      color: "var(--accent-strong)",
    },
  });

  const turinHighlight = HighlightStyle.define([
    { tag: [tags.keyword, tags.controlKeyword, tags.operatorKeyword], color: "var(--accent-strong)", fontWeight: "650" },
    { tag: [tags.function(tags.variableName), tags.definition(tags.variableName)], color: "var(--blue)" },
    { tag: [tags.propertyName, tags.attributeName], color: "var(--blue)" },
    { tag: [tags.string, tags.special(tags.string)], color: "var(--success)" },
    { tag: [tags.number, tags.bool, tags.null], color: "var(--warning)" },
    { tag: [tags.comment, tags.docComment], color: "var(--faint)", fontStyle: "italic" },
    { tag: [tags.operator, tags.punctuation], color: "var(--muted)" },
    { tag: tags.invalid, color: "var(--danger)", textDecoration: "underline wavy" },
  ]);

  function reportCursor(state: EditorState) {
    const head = state.selection.main.head;
    const line = state.doc.lineAt(head);
    onCursorChange(line.number, head - line.from + 1);
  }

  function syncValue(nextValue: string) {
    if (!view || view.state.doc.toString() === nextValue) return;
    applyingExternalValue = true;
    view.dispatch({ changes: { from: 0, to: view.state.doc.length, insert: nextValue } });
    applyingExternalValue = false;
  }

  $: syncValue(value);

  onMount(() => {
    const saveBinding = {
      key: "Mod-s",
      preventDefault: true,
      run: () => {
        onSave();
        return true;
      },
    };
    view = new EditorView({
      parent: host,
      state: EditorState.create({
        doc: value,
        extensions: [
          basicSetup,
          StreamLanguage.define(lua),
          EditorState.tabSize.of(2),
          indentUnit.of("  "),
          EditorView.contentAttributes.of({ "aria-label": ariaLabel, spellcheck: "false" }),
          EditorView.updateListener.of(update => {
            if (update.docChanged && !applyingExternalValue) onChange(update.state.doc.toString());
            if (update.docChanged || update.selectionSet) reportCursor(update.state);
          }),
          Prec.high(keymap.of([saveBinding, indentWithTab])),
          turinTheme,
          syntaxHighlighting(turinHighlight),
        ],
      }),
    });
    reportCursor(view.state);

    return () => {
      view?.destroy();
      view = null;
    };
  });
</script>

<div class="source-code-editor" bind:this={host}></div>
