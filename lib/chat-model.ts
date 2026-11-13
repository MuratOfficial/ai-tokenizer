// Крошечная, но настоящая языковая модель для чата.
//
// Как она отвечает:
// 1. Режет сообщение на токены (BPE, обученный на примерах диалогов).
// 2. Находит в примерах вопросы, похожие на сообщение.
// 3. Собирает ответ по одному токену: смотрит на несколько последних токенов ответа
//    и считает, какие токены шли после них в ответах на похожие вопросы.
//
// У больших моделей шаги 2–3 делает нейросеть, но принцип тот же:
// «оценить вероятность каждого следующего токена и выбрать один».

import { trainBpe, type VocabEntry } from "./bpe";
import { fallbackAnswers, groups } from "./chat-corpus";

export const START = -1;
export const END = -2;
const UNKNOWN = -3;
export const END_TEXT = "<конец>";

/** Сколько последних токенов ответа модель учитывает */
const MAX_CONTEXT = 4;
const BPE_MERGES = 600;
const MIN_SIMILARITY = 0.3;

export type Tok = { id: number; text: string };
type Answer = { text: string; tokens: number[] };
type Question = { text: string; stems: string[]; norm: number };
type Group = { questions: Question[]; answers: Answer[] };

export type Model = {
  vocab: VocabEntry[];
  charIds: Map<string, number>;
  merges: Map<string, number>;
  groups: Group[];
  fallback: Answer[];
  idf: Map<string, number>;
  maxIdf: number;
  answerCount: number;
};

export type Match = { question: string; similarity: number };
export type ActiveAnswer = Answer & { weight: number };

export type Retrieval = {
  userTokens: Tok[];
  matches: Match[];
  /** Похожих вопросов не нашлось — используются общие ответы */
  fallback: boolean;
  active: ActiveAnswer[];
};

export type Candidate = {
  id: number;
  text: string;
  /** Вероятность, 0..1 (уже с учётом температуры) */
  prob: number;
  /** Сколько раз токен шёл после такого контекста в похожих ответах */
  count: number;
  /** Пример ответа, где это встретилось */
  example: string;
};

export type Step = {
  /** Номер токена в ответе, с 1 */
  index: number;
  /** Последние токены ответа, на которые смотрела модель (пусто — начало ответа) */
  context: Tok[];
  /** Кандидаты, от самого вероятного */
  cands: Candidate[];
  /** Индекс выбранного кандидата */
  picked?: number;
};

let cached: Model | null = null;

export function getModel(): Model {
  if (cached) return cached;

  const text = [...groups.flatMap((g) => [...g.q, ...g.a]), ...fallbackAnswers].join("\n");
  // Как в настоящих токенизаторах: пробел бывает только в начале токена, знаки препинания — отдельно.
  // Поэтому токены получаются словами и кусочками слов, а не обрывками фраз.
  const { vocab } = trainBpe(
    text,
    BPE_MERGES,
    (left, right) => !/\s/.test(right) && !/[\n.,!?:=+]/.test(left + right),
  );

  const charIds = new Map<string, number>();
  const merges = new Map<string, number>();
  for (const v of vocab) {
    if (v.left === undefined) charIds.set(v.text, v.id);
    else merges.set(v.left + "," + v.right, v.id);
  }

  const model: Model = {
    vocab,
    charIds,
    merges,
    groups: [],
    fallback: [],
    idf: new Map(),
    maxIdf: 1,
    answerCount: 0,
  };

  // Чем реже слово встречается в вопросах, тем оно важнее для поиска похожих
  const allQuestions = groups.flatMap((g) => g.q.map(stems));
  const df = new Map<string, number>();
  for (const q of allQuestions) for (const s of q) df.set(s, (df.get(s) ?? 0) + 1);
  for (const [s, n] of df) model.idf.set(s, Math.log(1 + allQuestions.length / n));
  model.maxIdf = Math.log(1 + allQuestions.length);

  const toAnswer = (a: string): Answer => ({ text: a, tokens: encode(model, a).map((t) => t.id) });
  model.groups = groups.map((g) => ({
    questions: g.q.map((q) => {
      const st = stems(q);
      return { text: q, stems: st, norm: norm(model, st) };
    }),
    answers: g.a.map(toAnswer),
  }));
  model.fallback = fallbackAnswers.map(toAnswer);
  model.answerCount = groups.reduce((s, g) => s + g.a.length, 0) + fallbackAnswers.length;

  cached = model;
  return model;
}

export function tokenText(model: Model, id: number) {
  return id === END ? END_TEXT : (model.vocab[id]?.text ?? "?");
}

/** Режем текст на токены: начинаем с символов и применяем выученные склейки по порядку */
export function encode(model: Model, text: string): Tok[] {
  let toks: Tok[] = Array.from(text).map((ch) => ({ id: model.charIds.get(ch) ?? UNKNOWN, text: ch }));
  for (;;) {
    // Ищем пару, которую выучили раньше всех (у неё самый маленький номер)
    let best = Infinity;
    for (let i = 0; i < toks.length - 1; i++) {
      const id = model.merges.get(toks[i].id + "," + toks[i + 1].id);
      if (id !== undefined && id < best) best = id;
    }
    if (best === Infinity) return toks;

    const { left, right, text: merged } = model.vocab[best];
    const out: Tok[] = [];
    for (let i = 0; i < toks.length; i++) {
      if (toks[i].id === left && toks[i + 1]?.id === right) {
        out.push({ id: best, text: merged });
        i++;
      } else {
        out.push(toks[i]);
      }
    }
    toks = out;
  }
}

/** Признаки текста для поиска похожих вопросов */
function stems(text: string): string[] {
  const words = text.toLowerCase().replace(/ё/g, "е").match(/[a-zа-я0-9]+/g) ?? [];
  // Основа слова ловит разные окончания («токен» и «токены»), а слово целиком («=токен») даёт точному совпадению больший вес
  return [...new Set(words.flatMap((w) => [w.length > 5 ? w.slice(0, 5) : w, "=" + w]))];
}

function weightOf(model: Model, stem: string) {
  return model.idf.get(stem) ?? model.maxIdf;
}

function norm(model: Model, st: string[]) {
  return Math.sqrt(st.reduce((s, x) => s + weightOf(model, x) ** 2, 0));
}

/** Шаги 1–2: токены сообщения и похожие вопросы из примеров */
export function retrieve(model: Model, text: string): Retrieval {
  const userTokens = encode(model, text);
  const us = stems(text);
  const uNorm = norm(model, us);

  const scored = model.groups
    .map((group) => {
      let best = { question: "", similarity: 0 };
      for (const q of group.questions) {
        let dot = 0;
        for (const s of q.stems) if (us.includes(s)) dot += weightOf(model, s) ** 2;
        const similarity = uNorm && q.norm ? dot / (uNorm * q.norm) : 0;
        if (similarity > best.similarity) best = { question: q.text, similarity };
      }
      return { group, ...best };
    })
    .sort((a, b) => b.similarity - a.similarity);

  const top = scored[0]?.similarity ?? 0;
  if (top < MIN_SIMILARITY) {
    return {
      userTokens,
      matches: [],
      fallback: true,
      active: model.fallback.map((a) => ({ ...a, weight: 1 })),
    };
  }

  const chosen = scored.filter((x) => x.similarity >= Math.max(MIN_SIMILARITY, top * 0.75)).slice(0, 4);
  return {
    userTokens,
    matches: chosen.map(({ question, similarity }) => ({ question, similarity })),
    fallback: false,
    // Чем больше вопрос похож на сообщение, тем весомее его ответы
    active: chosen.flatMap((x) => x.group.answers.map((a) => ({ ...a, weight: x.similarity ** 2 }))),
  };
}

/** Шаг 3: кандидаты на следующий токен и их вероятности */
export function nextStep(model: Model, active: ActiveAnswer[], out: number[], temp: number): Step {
  const soFar = [START, ...out];
  const found = new Map<number, { weight: number; count: number; example: string; exampleWeight: number }>();
  let context: number[] = [];

  // Берём самый длинный контекст, для которого в примерах есть продолжение
  for (let n = Math.min(MAX_CONTEXT, soFar.length); n >= 1 && found.size === 0; n--) {
    context = soFar.slice(-n);
    for (const a of active) {
      const seq = [START, ...a.tokens, END];
      for (let i = n; i < seq.length; i++) {
        let same = true;
        for (let j = 0; j < n && same; j++) same = seq[i - n + j] === context[j];
        if (!same) continue;
        const c = found.get(seq[i]) ?? { weight: 0, count: 0, example: a.text, exampleWeight: 0 };
        c.weight += a.weight;
        c.count++;
        if (a.weight > c.exampleWeight) {
          c.example = a.text;
          c.exampleWeight = a.weight;
        }
        found.set(seq[i], c);
      }
    }
  }
  if (found.size === 0) found.set(END, { weight: 1, count: 0, example: "", exampleWeight: 0 });

  const entries = [...found.entries()].sort((a, b) => b[1].weight - a[1].weight);
  const total = entries.reduce((s, [, c]) => s + c.weight, 0);
  // Температура: 0 — только лучший; выше 1 — шансы выравниваются
  const scaled = entries.map(([, c], i) => (temp === 0 ? (i === 0 ? 1 : 0) : Math.pow(c.weight / total, 1 / temp)));
  const scaledTotal = scaled.reduce((s, x) => s + x, 0);

  return {
    index: out.length + 1,
    context: context.filter((id) => id !== START).map((id) => ({ id, text: tokenText(model, id) })),
    cands: entries.map(([id, c], i) => ({
      id,
      text: tokenText(model, id),
      prob: scaled[i] / scaledTotal,
      count: c.count,
      example: c.example,
    })),
  };
}

/** Случайный выбор с учётом вероятностей */
export function sample(step: Step): number {
  let r = Math.random();
  for (let i = 0; i < step.cands.length; i++) {
    r -= step.cands[i].prob;
    if (r < 0) return i;
  }
  return 0;
}
