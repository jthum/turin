import type { JsonValue, SessionSummary } from "./types";

export function titleForSession(session: SessionSummary): string {
  const title = session.metadata?.title;
  return typeof title === "string" && title.trim() ? title : humanize(session.agent_id);
}

export function humanize(value: string): string {
  return value
    .replace(/[._-]+/g, " ")
    .replace(/\b\w/g, letter => letter.toUpperCase());
}

export function sameSession(left: string | null | undefined, right: string | null | undefined): boolean {
  if (!left || !right) return left === right;
  return left.split("@", 1)[0] === right.split("@", 1)[0];
}

export function messageText(content: JsonValue): string {
  if (typeof content === "string") return content;
  if (Array.isArray(content)) {
    return content.map(messageText).filter(Boolean).join("\n");
  }
  if (content && typeof content === "object") {
    const candidate = content.text ?? content.content ?? content.value;
    return candidate === undefined ? "" : messageText(candidate);
  }
  return "";
}

export function displayValue(value: unknown): string {
  if (value === null || value === undefined || value === "") return "-";
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return JSON.stringify(value);
}

export function shortDate(value: string): string {
  const date = parseTurinDate(value);
  if (Number.isNaN(date.valueOf())) return value;
  return new Intl.DateTimeFormat(undefined, {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  }).format(date);
}

export function messageTimestamp(value: string): string {
  const date = parseTurinDate(value);
  if (Number.isNaN(date.valueOf())) return value;
  const now = new Date();
  const sameDay = date.getFullYear() === now.getFullYear()
    && date.getMonth() === now.getMonth()
    && date.getDate() === now.getDate();
  return new Intl.DateTimeFormat(undefined, sameDay
    ? { hour: "numeric", minute: "2-digit" }
    : { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }
  ).format(date);
}

export function fullDate(value: string): string {
  const date = parseTurinDate(value);
  if (Number.isNaN(date.valueOf())) return value;
  return new Intl.DateTimeFormat(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
  }).format(date);
}

function parseTurinDate(value: string): Date {
  const withoutZone = /^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}(?:\.\d+)?$/;
  const normalized = withoutZone.test(value) ? `${value.replace(" ", "T")}Z` : value;
  return new Date(normalized);
}
