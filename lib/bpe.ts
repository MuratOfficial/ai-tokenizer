// Простейший BPE (Byte Pair Encoding) — тот же принцип, что в токенизаторах GPT.
// 1. Режем текст на символы.
// 2. Находим самую частую пару соседних токенов.
// 3. Склеиваем её в новый токен. Повторяем.

export type VocabEntry = {
  id: number;
  text: string;
  /** Из каких двух токенов склеен (нет у базовых символов) */
  left?: number;
  right?: number;
  /** Номер шага, на котором появился (0 — базовый символ) */
  step: number;
};

export type Candidate = { a: number; b: number; count: number };

export type Step = {
  a: number;
  b: number;
  count: number;
  newId: number;
  /** Самые частые пары на этом шаге (первая — победитель) */
  candidates: Candidate[];
  /** Последовательность токенов после склейки */
  tokens: number[];
};

export type BpeResult = {
  vocab: VocabEntry[];
  /** Последовательность до склеек: один символ — один токен */
  initial: number[];
  steps: Step[];
};

const MAX_MERGES = 200;
/** Пара токенов (a, b) кодируется одним числом a * KEY_BASE + b */
const KEY_BASE = 1_000_000;

/** canMerge — необязательное правило: какие два токена разрешено склеивать */
export function trainBpe(
  text: string,
  maxMerges = MAX_MERGES,
  canMerge?: (left: string, right: string) => boolean,
): BpeResult {
  const vocab: VocabEntry[] = [];
  const charIds = new Map<string, number>();

  const initial = Array.from(text).map((ch) => {
    let id = charIds.get(ch);
    if (id === undefined) {
      id = vocab.length;
      charIds.set(ch, id);
      vocab.push({ id, text: ch, step: 0 });
    }
    return id;
  });

  const steps: Step[] = [];
  let tokens = initial;

  while (steps.length < maxMerges) {
    // Считаем, сколько раз встречается каждая пара соседей
    const counts = new Map<number, Candidate>();
    for (let i = 0; i < tokens.length - 1; i++) {
      const key = tokens[i] * KEY_BASE + tokens[i + 1];
      const c = counts.get(key);
      if (c) c.count++;
      else counts.set(key, { a: tokens[i], b: tokens[i + 1], count: 1 });
    }

    let pairs = [...counts.values()];
    // На больших текстах пары, встреченные один раз, не нужны: склеивать их всё равно нельзя
    if (pairs.length > 500) pairs = pairs.filter((c) => c.count >= 2);
    if (canMerge) pairs = pairs.filter((c) => canMerge(vocab[c.a].text, vocab[c.b].text));

    // Map хранит порядок вставки, а sort стабилен — при равенстве побеждает пара, встреченная раньше
    const candidates = pairs.sort((x, y) => y.count - x.count).slice(0, 5);
    const best = candidates[0];
    if (!best || best.count < 2) break;

    const newId = vocab.length;
    vocab.push({
      id: newId,
      text: vocab[best.a].text + vocab[best.b].text,
      left: best.a,
      right: best.b,
      step: steps.length + 1,
    });

    tokens = merge(tokens, best.a, best.b, newId);
    steps.push({ a: best.a, b: best.b, count: best.count, newId, candidates, tokens });
  }

  return { vocab, initial, steps };
}

function merge(tokens: number[], a: number, b: number, newId: number): number[] {
  const out: number[] = [];
  for (let i = 0; i < tokens.length; i++) {
    if (tokens[i] === a && tokens[i + 1] === b) {
      out.push(newId);
      i++;
    } else {
      out.push(tokens[i]);
    }
  }
  return out;
}

/** Индексы токенов, которые будут склеены парой (a, b) */
export function pairPositions(tokens: number[], a: number, b: number): Set<number> {
  const set = new Set<number>();
  for (let i = 0; i < tokens.length - 1; i++) {
    if (tokens[i] === a && tokens[i + 1] === b) {
      set.add(i);
      set.add(i + 1);
      i++;
    }
  }
  return set;
}
