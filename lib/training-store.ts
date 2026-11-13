// Хранилище обучения. Живёт в браузере (localStorage), отдельно от встроенных примеров модели.
//
// unknown — фразы, на которые модель не нашла ответа;
// learned — то, чему модель научили вручную (вопросы и ответы на них).

import { useSyncExternalStore } from "react";

/** Почему фраза попала в хранилище */
export type UnknownReason = "none" | "weak" | "flagged";

export const REASONS: Record<UnknownReason, string> = {
  none: "ответ не найден",
  weak: "модель ответила неуверенно",
  flagged: "ответ не подошёл",
};

export type UnknownPhrase = {
  text: string;
  reason?: UnknownReason;
  /** Что модель ответила на эту фразу */
  answer?: string;
  /** Сколько раз фразу писали */
  count: number;
  /** Когда писали в последний раз (мс) */
  lastAt: number;
};

export type LearnedGroup = {
  id: string;
  /** Варианты вопроса */
  q: string[];
  /** Варианты ответа */
  a: string[];
  createdAt: number;
};

export type Training = { unknown: UnknownPhrase[]; learned: LearnedGroup[] };

const KEY = "ai-tokenizer:training";
const EMPTY: Training = { unknown: [], learned: [] };
const MAX_UNKNOWN = 200;

let raw: string | null = null;
let snapshot: Training = EMPTY;
/** localStorage недоступен (приватный режим и т. п.) — держим данные только в памяти вкладки */
let memoryOnly = false;
const listeners = new Set<() => void>();

function read(): Training {
  if (memoryOnly) return snapshot;
  let current: string | null = null;
  try {
    current = localStorage.getItem(KEY);
  } catch {
    memoryOnly = true;
    return snapshot;
  }
  if (current === raw) return snapshot;
  raw = current;
  try {
    const data = current ? JSON.parse(current) : EMPTY;
    snapshot = {
      unknown: Array.isArray(data.unknown) ? data.unknown : [],
      learned: Array.isArray(data.learned) ? data.learned : [],
    };
  } catch {
    snapshot = EMPTY;
  }
  return snapshot;
}

function write(next: Training) {
  snapshot = next;
  raw = JSON.stringify(next);
  try {
    localStorage.setItem(KEY, raw);
  } catch {
    memoryOnly = true;
  }
  listeners.forEach((l) => l());
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  window.addEventListener("storage", listener);
  return () => {
    listeners.delete(listener);
    window.removeEventListener("storage", listener);
  };
}

/** Данные хранилища; компонент перерисуется, когда они изменятся (даже в другой вкладке) */
export function useTraining(): Training {
  return useSyncExternalStore(subscribe, read, () => EMPTY);
}

/** Одинаковыми считаем фразы, которые отличаются только регистром, пробелами и знаками на конце */
export function samePhrase(a: string, b: string) {
  const clean = (s: string) => s.trim().toLowerCase().replace(/\s+/g, " ").replace(/[.!?…]+$/, "");
  return clean(a) === clean(b);
}

/** Кладём фразу в хранилище: ответа нет, ответ неуверенный или пользователь отметил его как неподходящий */
export function addUnknown(text: string, reason: UnknownReason = "none", answer?: string) {
  text = text.trim();
  if (!text) return;
  const data = read();
  const existing = data.unknown.find((u) => samePhrase(u.text, text));
  const unknown = existing
    ? data.unknown.map((u) =>
        u === existing
          ? {
              ...u,
              count: u.count + 1,
              lastAt: Date.now(),
              // «не подошёл» важнее автоматических пометок
              reason: u.reason === "flagged" ? u.reason : reason,
              answer: answer ?? u.answer,
            }
          : u,
      )
    : [{ text, reason, answer, count: 1, lastAt: Date.now() }, ...data.unknown].slice(0, MAX_UNKNOWN);
  write({ ...data, unknown });
}

export function removeUnknown(text: string) {
  const data = read();
  write({ ...data, unknown: data.unknown.filter((u) => !samePhrase(u.text, text)) });
}

export function clearUnknown() {
  write({ ...read(), unknown: [] });
}

/** Ручное обучение: на такие вопросы отвечать такими ответами */
export function teach(questions: string[], answers: string[]) {
  const q = questions.map((x) => x.trim()).filter(Boolean);
  const a = answers.map((x) => x.trim()).filter(Boolean);
  if (q.length === 0 || a.length === 0) return false;
  const data = read();
  const group: LearnedGroup = { id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6), q, a, createdAt: Date.now() };
  write({
    // Выученные фразы из списка «без ответа» убираем
    unknown: data.unknown.filter((u) => !q.some((x) => samePhrase(x, u.text))),
    learned: [group, ...data.learned],
  });
  return true;
}

export function removeLearned(id: string) {
  const data = read();
  write({ ...data, learned: data.learned.filter((g) => g.id !== id) });
}

/** Исправить выученное: новые вопросы и ответы вместо старых */
export function updateLearned(id: string, questions: string[], answers: string[]) {
  const q = questions.map((x) => x.trim()).filter(Boolean);
  const a = answers.map((x) => x.trim()).filter(Boolean);
  if (q.length === 0 || a.length === 0) return false;
  const data = read();
  write({ ...data, learned: data.learned.map((g) => (g.id === id ? { ...g, q, a } : g)) });
  return true;
}

const isStrings = (x: unknown): x is string[] => Array.isArray(x) && x.every((v) => typeof v === "string");

/**
 * Загрузка хранилища из JSON (того же вида, что даёт «Скачать JSON»).
 * Данные добавляются к имеющимся; повторы пропускаются. Возвращает, сколько добавлено.
 */
export function importTraining(json: string): { learned: number; unknown: number } {
  const parsed: unknown = JSON.parse(json);
  if (!parsed || typeof parsed !== "object") throw new Error("Это не файл хранилища.");
  const input = parsed as { learned?: unknown; unknown?: unknown };
  const data = read();

  const learned = [...data.learned];
  let addedLearned = 0;
  for (const item of Array.isArray(input.learned) ? input.learned : []) {
    const g = item as Partial<LearnedGroup>;
    if (!isStrings(g.q) || !isStrings(g.a) || g.q.length === 0 || g.a.length === 0) continue;
    const key = JSON.stringify([g.q, g.a]);
    if (learned.some((x) => JSON.stringify([x.q, x.a]) === key)) continue;
    learned.push({
      id: Date.now().toString(36) + Math.random().toString(36).slice(2, 6),
      q: g.q,
      a: g.a,
      createdAt: typeof g.createdAt === "number" ? g.createdAt : Date.now(),
    });
    addedLearned++;
  }

  const unknown = [...data.unknown];
  let addedUnknown = 0;
  for (const item of Array.isArray(input.unknown) ? input.unknown : []) {
    const u = item as Partial<UnknownPhrase>;
    if (typeof u.text !== "string" || !u.text.trim()) continue;
    const text = u.text.trim();
    // Уже выученное и уже записанное не добавляем
    if (unknown.some((x) => samePhrase(x.text, text))) continue;
    if (learned.some((g) => g.q.some((q) => samePhrase(q, text)))) continue;
    unknown.push({
      text,
      reason: u.reason && u.reason in REASONS ? u.reason : "none",
      answer: typeof u.answer === "string" ? u.answer : undefined,
      count: typeof u.count === "number" ? u.count : 1,
      lastAt: typeof u.lastAt === "number" ? u.lastAt : Date.now(),
    });
    addedUnknown++;
  }

  if (addedLearned === 0 && addedUnknown === 0 && !Array.isArray(input.learned) && !Array.isArray(input.unknown)) {
    throw new Error("В файле нет разделов learned и unknown.");
  }
  write({ learned, unknown: unknown.slice(0, MAX_UNKNOWN) });
  return { learned: addedLearned, unknown: addedUnknown };
}
