// Крошечная, но настоящая языковая модель для чата.
//
// Как она отвечает:
// 1. Вспоминает разговор: имя собеседника, прошлое сообщение, свой последний вопрос.
// 2. Режет сообщение на токены (BPE, обученный на примерах диалогов).
// 3. Находит в примерах вопросы, похожие на сообщение (с учётом памяти).
// 4. Собирает ответ по одному токену: смотрит на несколько последних токенов ответа
//    и считает, какие токены шли после них в ответах на похожие вопросы.
//
// У больших моделей шаги 3–4 делает нейросеть, а «память» — это просто весь разговор,
// который каждый раз целиком подаётся на вход. Но принцип тот же:
// «оценить вероятность каждого следующего токена и выбрать один».

import { trainBpe, type VocabEntry } from "./bpe";
import { fallbackAnswers, follows, groups } from "./chat-corpus";

export const START = -1;
export const END = -2;
const UNKNOWN = -3;
export const END_TEXT = "<конец>";

/** Сколько последних токенов ответа модель учитывает */
const MAX_CONTEXT = 4;
const BPE_MERGES = 600;
const MIN_SIMILARITY = 0.3;
/** Насколько прошлое сообщение помогает выбрать тему */
const TOPIC_BONUS = 0.2;
/** Бонус группе, которая понимает ответ на вопрос модели */
const AFTER_BONUS = 0.35;
/** Оценка группы «подходит любой ответ на вопрос модели» */
const ANY_REPLY_SCORE = 0.85;
/** Во сколько раз реже модель повторяет уже сказанное */
const REPEAT_PENALTY = 0.15;

export type Tok = { id: number; text: string };
type Question = { text: string; stems: string[]; norm: number };
type Follow = { tag: string; q: string[] };
type Group = {
  questions: Question[];
  answers: string[];
  follow?: Follow;
  after?: string;
  lacks?: "name" | "last";
  when?: "newName" | "oldName";
};

export type Model = {
  vocab: VocabEntry[];
  charIds: Map<string, number>;
  merges: Map<string, number>;
  groups: Group[];
  idf: Map<string, number>;
  maxIdf: number;
  topicCount: number;
  answerCount: number;
};

/** Что модель помнит о разговоре */
export type Memory = {
  /** Имя собеседника */
  name?: string;
  /** Предыдущее сообщение собеседника */
  last?: string;
  /** Вопрос, который модель задала в прошлом ответе и на который ждёт ответ */
  pending?: { tag: string; question: string };
  /** Что модель уже говорила — чтобы не повторяться */
  said: string[];
};

export const emptyMemory = (): Memory => ({ said: [] });

export type Match = {
  question: string;
  similarity: number;
  /** Сообщение понято как ответ на вопрос модели */
  reply?: boolean;
};
export type ActiveAnswer = { text: string; tokens: number[]; weight: number };

export type Retrieval = {
  userTokens: Tok[];
  matches: Match[];
  /** Похожих вопросов не нашлось — используются общие ответы */
  fallback: boolean;
  active: ActiveAnswer[];
  /** Память на момент ответа (уже с тем, что узнали из этого сообщения) */
  memory: Memory;
  /** Как память повлияла на ответ — для показа в панели */
  used: string[];
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

  const followTexts = Object.values(follows).flatMap((f) => f.q);
  const text = [...groups.flatMap((g) => [...g.q, ...g.a]), ...followTexts, ...fallbackAnswers]
    .join("\n")
    .replace(/\{\w+\}/g, "");

  // Как в настоящих токенизаторах: пробел бывает только в начале токена, знаки препинания — отдельно.
  // Поэтому токены получаются словами и кусочками слов, а не обрывками фраз.
  const { vocab } = trainBpe(
    text,
    BPE_MERGES,
    (left, right) => !/\s/.test(right) && !/[\n.,!?:=+«»]/.test(left + right),
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
    idf: new Map(),
    maxIdf: 1,
    topicCount: groups.filter((g) => !g.after && !g.lacks).length,
    answerCount: groups.reduce((s, g) => s + g.a.length, 0) + fallbackAnswers.length,
  };

  // Чем реже слово встречается в вопросах, тем оно важнее для поиска похожих
  const allQuestions = groups.flatMap((g) => g.q.map(stems));
  const df = new Map<string, number>();
  for (const q of allQuestions) for (const s of q) df.set(s, (df.get(s) ?? 0) + 1);
  for (const [s, n] of df) model.idf.set(s, Math.log(1 + allQuestions.length / n));
  model.maxIdf = Math.log(1 + allQuestions.length);

  model.groups = groups.map((g) => ({
    questions: g.q.map((q) => {
      const st = stems(q);
      return { text: q, stems: st, norm: norm(model, st) };
    }),
    answers: g.a,
    follow: g.follow ? follows[g.follow] : undefined,
    after: g.after,
    lacks: g.lacks,
    when: g.when,
  }));

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

/** Насколько текст (его признаки) похож на вопрос из примеров, 0..1 */
function similarity(model: Model, st: string[], stNorm: number, q: Question) {
  if (!stNorm || !q.norm) return 0;
  let dot = 0;
  for (const s of q.stems) if (st.includes(s)) dot += weightOf(model, s) ** 2;
  return dot / (stNorm * q.norm);
}

/** Все ли символы знакомы модели (иначе текст нельзя будет собрать из токенов) */
function known(model: Model, text: string) {
  return Array.from(text).every((ch) => model.charIds.has(ch));
}

const NOT_A_NAME = ["нет", "да", "не", "зачем", "почему", "никак", "секрет", "привет", "хорошо", "ладно", "незнаю"];

/** Пытаемся узнать имя из сообщения */
function findName(model: Model, text: string, askedName: boolean): string | undefined {
  const word = "([A-Za-zА-Яа-яЁё]{2,20})";
  let m = text.match(new RegExp("(?:меня зовут|мо[её] имя|зови меня|называй меня)\\s+" + word, "i"));
  // Если модель сама спросила имя, ответом может быть просто слово: «Мурат» или «я Мурат»
  if (!m && askedName) m = text.trim().match(new RegExp("^(?:я\\s+)?" + word + "[.!]?$", "i"));
  if (!m || NOT_A_NAME.includes(m[1].toLowerCase())) return undefined;
  const name = m[1][0].toUpperCase() + m[1].slice(1);
  return known(model, name) ? name : undefined;
}

/** Подставляем в ответ то, что модель помнит. null — если нужного в памяти нет */
function fill(model: Model, template: string, memory: Memory, userText: string): string | null {
  let ok = true;
  const text = template.replace(/\{(\w+)\}/g, (_, key: string) => {
    let value: string | undefined;
    if (key === "name") value = memory.name;
    if (key === "last") value = memory.last && shorten(memory.last, 40);
    if (key === "reply") {
      const reply = userText.trim().replace(/[.!?]+$/, "");
      if (reply && reply.length <= 25) value = reply[0].toUpperCase() + reply.slice(1);
    }
    if (!value || !known(model, value)) ok = false;
    return value ?? "";
  });
  return ok ? text : null;
}

function shorten(text: string, max: number) {
  return text.length > max ? text.slice(0, max).trim() + "..." : text;
}

/** Шаги 1–3: память, токены сообщения и похожие вопросы из примеров */
export function retrieve(model: Model, text: string, before: Memory): Retrieval {
  const used: string[] = [];
  const pending = before.pending;

  const newName = findName(model, text, pending?.tag === "yourName");
  const memory: Memory = newName ? { ...before, name: newName } : before;
  if (newName) used.push(`Из сообщения взято имя «${newName}» и записано в память.`);

  const userTokens = encode(model, text);
  const us = stems(text);
  const uNorm = norm(model, us);
  const topic = before.last ? stems(before.last) : [];
  const topicNorm = norm(model, topic);

  const scored = model.groups
    .map((group) => {
      const result = { group, question: "", score: 0, reply: false, topicHelped: false };
      if (group.lacks && memory[group.lacks]) return result;
      if (group.after && group.after !== pending?.tag) return result;
      if (group.when === "newName" && !newName) return result;
      if (group.when === "oldName" && (newName || !memory.name)) return result;

      let sim = 0;
      let topicSim = 0;
      for (const q of group.questions) {
        const s = similarity(model, us, uNorm, q);
        if (s > sim) {
          sim = s;
          result.question = q.text;
        }
        topicSim = Math.max(topicSim, similarity(model, topic, topicNorm, q));
      }

      if (group.after) {
        result.reply = true;
        if (group.questions.length === 0) {
          // Сообщение с вопросительным знаком — скорее новый вопрос, чем ответ
          result.score = text.includes("?") ? ANY_REPLY_SCORE / 2 : ANY_REPLY_SCORE;
          result.question = "любой ответ";
        } else if (sim > 0) {
          result.score = sim + AFTER_BONUS;
        }
      } else if (sim >= 0.2) {
        // Прошлое сообщение помогает понять, о чём речь: «а зачем они нужны?»
        result.score = sim + TOPIC_BONUS * topicSim;
        result.topicHelped = topicSim > 0.3;
      }
      return result;
    })
    .filter((x) => x.score > 0)
    .sort((a, b) => b.score - a.score);

  // Отбираем лучшие группы, у которых есть чем ответить
  const chosen: { item: (typeof scored)[number]; answers: string[] }[] = [];
  let top = 0;
  for (const item of scored) {
    if (chosen.length === 4) break;
    if (top) {
      // Если есть почти точное совпадение, остальные вопросы только мешают
      const cutoff = top > 0.95 ? Math.max(0.95, top - 0.1) : Math.max(MIN_SIMILARITY, top * 0.75);
      if (item.score < cutoff) break;
    } else if (item.score < MIN_SIMILARITY) {
      break;
    }
    const answers = item.group.answers
      .map((a) => fill(model, a, memory, text))
      .filter((a): a is string => a !== null);
    if (answers.length === 0) continue;
    // Если сообщение — ответ на вопрос модели, посторонние темы не подмешиваем
    if (chosen[0]?.item.reply && !item.reply) continue;
    if (!top) top = item.score;
    chosen.push({ item, answers });
  }

  const toActive = (texts: string[], weight: number): ActiveAnswer[] =>
    texts.map((t) => ({
      text: t,
      tokens: encode(model, t).map((x) => x.id),
      weight: weight * (before.said.includes(t) ? REPEAT_PENALTY : 1),
    }));

  let active: ActiveAnswer[];
  if (chosen.length === 0) {
    active = toActive(fallbackAnswers, 1);
  } else {
    active = chosen.flatMap(({ item, answers }) =>
      answers.flatMap((a) => {
        // После ответа модель задаёт встречный вопрос — если сам ответ не вопрос
        const tails = item.group.follow && !a.endsWith("?") ? item.group.follow.q : [""];
        return toActive(
          tails.map((tail) => (tail ? a + " " + tail : a)),
          item.score ** 2 / tails.length,
        );
      }),
    );

    if (pending && chosen.some((c) => c.item.reply)) {
      used.push(`Сообщение понято как ответ на вопрос модели «${pending.question}».`);
    }
    if (chosen.some((c) => c.item.topicHelped)) {
      used.push(`Прошлое сообщение «${shorten(before.last!, 30)}» добавило вес близким темам.`);
    }
    const templates = chosen.flatMap((c) => c.item.group.answers).join(" ");
    if (memory.name && templates.includes("{name}")) used.push(`В ответ подставлено имя «${memory.name}».`);
    if (templates.includes("{last}")) used.push("В ответ подставлено прошлое сообщение.");
  }
  if (active.some((a) => before.said.includes(a.text))) {
    used.push("Уже звучавшие ответы получили меньший вес, чтобы не повторяться.");
  }

  return {
    userTokens,
    matches: chosen.map(({ item }) => ({
      question: item.question,
      similarity: Math.min(1, item.score),
      reply: item.reply,
    })),
    fallback: chosen.length === 0,
    active,
    memory,
    used,
  };
}

/** После ответа: что запомнить на будущее */
export function remember(memory: Memory, userText: string, botText: string): Memory {
  // Если ответ закончился одним из встречных вопросов — ждём на него ответ
  let pending: Memory["pending"];
  for (const f of Object.values(follows)) {
    for (const q of f.q) {
      if (botText.endsWith(q) && q.length > (pending?.question.length ?? 0)) pending = { tag: f.tag, question: q };
    }
  }
  return {
    name: memory.name,
    last: userText,
    pending,
    said: [...memory.said, botText].slice(-30),
  };
}

/** Шаг 4: кандидаты на следующий токен и их вероятности */
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
