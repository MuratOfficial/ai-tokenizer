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
import { fallbackAnswers, follows, groups as builtIn, unknownWordAnswers, type CorpusGroup } from "./chat-corpus";

export const START = -1;
export const END = -2;
const UNKNOWN = -3;
export const END_TEXT = "<конец>";

/** Сколько последних токенов ответа модель учитывает */
const MAX_CONTEXT = 4;
const BPE_MERGES = 600;
/** Ниже этого сходства модель честно говорит «не знаю», а не угадывает */
export const MIN_SIMILARITY = 0.45;
/** Если в сообщении есть незнакомое слово, совпадение должно быть почти точным */
const MIN_SIMILARITY_WITH_UNKNOWN = 0.75;
/** Ниже этого сходства ответ считается неуверенным и попадает в хранилище на проверку */
const WEAK_SIMILARITY = 0.6;
/** Насколько прошлое сообщение помогает выбрать тему */
const TOPIC_BONUS = 0.2;
/** Во сколько раз падает сходство, если главное слово сообщения в вопросе не встретилось */
const KEY_MISS_PENALTY = 0.5;
/** Бонус группе, которая понимает ответ на вопрос модели */
const AFTER_BONUS = 0.35;
/** Оценка группы «подходит любой ответ на вопрос модели» */
const ANY_REPLY_SCORE = 0.72;
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
  /** Основы всех слов, которые модель видела в вопросах и ответах */
  knownStems: Set<string>;
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

/** Признак текста (основа слова или слово целиком) и его важность */
export type Feature = { name: string; weight: number };

/** Как посчитана оценка одного вопроса из примеров */
export type SearchRow = {
  question: string;
  /** Совпавшие признаки */
  common: Feature[];
  /** Сумма квадратов весов совпавших признаков */
  dot: number;
  /** «Длина» вопроса из примеров */
  qNorm: number;
  /** Сходство = dot / (длина сообщения × длина вопроса), с учётом штрафа за главное слово */
  sim: number;
  /** Сходство до штрафа */
  rawSim: number;
  /** Главное слово сообщения в этом вопросе не встретилось */
  keyMissed: boolean;
  bonus: number;
  bonusWhy: string;
  /** Итоговая оценка = сходство + бонус */
  score: number;
  /** Вес ответов этой группы = оценка² */
  weight: number;
  chosen: boolean;
  reply: boolean;
};

/** Расчёт поиска похожих вопросов — для показа в панели */
export type Search = {
  features: Feature[];
  /** Главное слово сообщения — самое редкое из знакомых */
  key?: string;
  /** «Длина» сообщения: корень из суммы квадратов весов признаков */
  uNorm: number;
  rows: SearchRow[];
};

export type Retrieval = {
  userTokens: Tok[];
  matches: Match[];
  /** Похожих вопросов не нашлось — используются общие ответы */
  fallback: boolean;
  /** Совпадение есть, но слабое: модель отвечает неуверенно */
  weak: boolean;
  /** Незнакомые слова в сообщении */
  strange: string[];
  active: ActiveAnswer[];
  /** Память на момент ответа (уже с тем, что узнали из этого сообщения) */
  memory: Memory;
  /** Как память повлияла на ответ — для показа в панели */
  used: string[];
  search: Search;
};

export type Candidate = {
  id: number;
  text: string;
  /** Сумма весов ответов, где этот токен шёл после контекста */
  weight: number;
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
  /** Сумма весов всех кандидатов: вероятность до температуры = вес / total */
  total: number;
  /** Температура, с которой считались вероятности */
  temp: number;
  /** Индекс выбранного кандидата */
  picked?: number;
};

let cached: Model | null = null;
let cachedKey = "";

/**
 * Собирает («обучает») модель. extra — то, чему модель научили вручную:
 * эти вопросы и ответы добавляются к встроенным примерам, и словарь токенов учится заново.
 */
export function getModel(extra: { q: string[]; a: string[] }[] = []): Model {
  const key = JSON.stringify(extra.map((g) => [g.q, g.a]));
  if (cached && cachedKey === key) return cached;
  const groups: CorpusGroup[] = [...builtIn, ...extra.map((g) => ({ q: g.q, a: g.a }))];

  const followTexts = Object.values(follows).flatMap((f) => f.q);
  const text = [...groups.flatMap((g) => [...g.q, ...g.a]), ...followTexts, ...fallbackAnswers, ...unknownWordAnswers]
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
    knownStems: new Set(),
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
  for (const w of text.toLowerCase().replace(/ё/g, "е").match(/[a-zа-я0-9]+/g) ?? []) model.knownStems.add(stemOf(w));

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
  cachedKey = key;
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

/** Слова сообщения, которых нет ни в одном вопросе из примеров */
export function unknownWords(model: Model, text: string): string[] {
  const words = text.toLowerCase().replace(/ё/g, "е").match(/[a-zа-я0-9]+/g) ?? [];
  return [...new Set(words)].filter((w) => !model.knownStems.has(stemOf(w)));
}

// Частые русские окончания, от длинных к коротким
const ENDINGS = (
  "иями ями ами ого его ому ему ыми ими аешь яешь ешь ишь ете ите ают яют уют ует ать ять ить еть " +
  "ался алась ился илась аю яю ую ая яя ое ее ые ие ый ий ой юю ом ем ам ям ах ях ов ев ей ть ет ит ют ут ат ят " +
  "а я о е ы и у ю ь й"
).split(" ");

/** Основа слова: отрезаем окончание, чтобы «токены» и «токен», «ночи» и «ночам» считались одним словом */
function stemOf(word: string) {
  if (!/[а-я]/.test(word)) return word;
  for (const ending of ENDINGS) {
    if (word.length - ending.length >= 3 && word.endsWith(ending)) return word.slice(0, -ending.length);
  }
  return word;
}

/** Слова, с которых обычно начинается новый вопрос или просьба, а не ответ */
const NEW_TOPIC_STARTS =
  "расскажи скажи объясни покажи напиши придумай посоветуй дай что как почему зачем кто где когда сколько какой какая какое какие знаешь умеешь можешь".split(
    " ",
  );

/** Похоже ли сообщение на ответ на вопрос модели (а не на новый вопрос) */
function looksLikeReply(text: string) {
  const words: string[] = text.toLowerCase().match(/[a-zа-яё0-9]+/g) ?? [];
  if (text.includes("?") || words.length === 0 || words.length > 8) return false;
  // Прощание, благодарность и приветствие — тоже не ответ на вопрос
  if (words.some((w) => ["пока", "спасибо", "привет", "свидания"].includes(w))) return false;
  return !NEW_TOPIC_STARTS.includes(words[0]);
}

/** Незнакомые содержательные слова и числа — в том виде, как их написали */
function strangeWords(model: Model, text: string, skip?: string): string[] {
  const words = text.match(/[A-Za-zА-Яа-яЁё0-9]+/g) ?? [];
  return [...new Set(words)].filter((w) => {
    if (skip && w.toLowerCase() === skip.toLowerCase()) return false;
    if (w.length < 4 && !/^[0-9]+$/.test(w)) return false;
    return !model.knownStems.has(stemOf(w.toLowerCase().replace(/ё/g, "е")));
  });
}

/** Признаки текста для поиска похожих вопросов */
function stems(text: string): string[] {
  const words = text.toLowerCase().replace(/ё/g, "е").match(/[a-zа-я0-9]+/g) ?? [];
  // Сравниваем по основам слов, чтобы «фильм», «фильмы» и «фильмов» считались одним и тем же
  return [...new Set(words.map(stemOf))];
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

/** То же сходство, но с подробностями: какие признаки совпали */
function compare(model: Model, st: string[], stNorm: number, q: Question) {
  const common: Feature[] = [];
  let dot = 0;
  for (const s of q.stems) {
    if (!st.includes(s)) continue;
    const weight = weightOf(model, s);
    common.push({ name: s, weight });
    dot += weight ** 2;
  }
  return { common, dot, sim: stNorm && q.norm ? dot / (stNorm * q.norm) : 0 };
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
function fill(model: Model, template: string, memory: Memory, userText: string, unknown?: string): string | null {
  let ok = true;
  const text = template.replace(/\{(\w+)\}/g, (_, key: string) => {
    let value: string | undefined;
    if (key === "name") value = memory.name;
    if (key === "unknown") value = unknown;
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

  // Главное слово сообщения: самое редкое из тех, что встречаются в вопросах примеров
  let key: string | undefined;
  for (const st of us) {
    if (model.idf.has(st) && (key === undefined || weightOf(model, st) > weightOf(model, key))) key = st;
  }
  const replyLike = looksLikeReply(text);
  const strange = strangeWords(model, text, newName);

  const scored = model.groups
    .map((group) => {
      const result = {
        group,
        question: "",
        score: 0,
        reply: false,
        topicHelped: false,
        common: [] as Feature[],
        dot: 0,
        qNorm: 0,
        sim: 0,
        bonus: 0,
        bonusWhy: "",
        rawSim: 0,
        keyMissed: false,
      };
      if (group.lacks && memory[group.lacks]) return result;
      if (group.after && group.after !== pending?.tag) return result;
      if (group.when === "newName" && !newName) return result;
      if (group.when === "oldName" && (newName || !memory.name)) return result;

      let sim = 0;
      let topicSim = 0;
      for (const q of group.questions) {
        const c = compare(model, us, uNorm, q);
        // Совпали только общие слова («расскажи про…»), а главное — нет: скорее всего, тема другая
        const strongest = Math.max(0, ...c.common.map((f) => f.weight));
        const keyMissed = key !== undefined && strongest < 0.7 * weightOf(model, key);
        const value = c.sim * (keyMissed ? KEY_MISS_PENALTY : 1);
        if (value > sim) {
          sim = value;
          result.rawSim = c.sim;
          result.keyMissed = keyMissed;
          result.question = q.text;
          result.common = c.common;
          result.dot = c.dot;
          result.qNorm = q.norm;
          result.sim = value;
        }
        topicSim = Math.max(topicSim, similarity(model, topic, topicNorm, q));
      }

      if (group.after) {
        result.reply = true;
        if (group.questions.length === 0) {
          // Новый вопрос или просьбу («расскажи…», «что…?») ответом не считаем
          if (!replyLike) return result;
          result.score = ANY_REPLY_SCORE;
          result.question = "любой ответ";
          result.bonus = result.score;
          result.bonusWhy = "подходит любой ответ на вопрос модели";
        } else if (sim > 0) {
          result.score = sim + AFTER_BONUS;
          result.bonus = AFTER_BONUS;
          result.bonusWhy = "это ответ на вопрос модели";
        }
      } else if (sim >= 0.2) {
        // Прошлое сообщение помогает понять, о чём речь: «а зачем они нужны?»
        result.bonus = TOPIC_BONUS * topicSim;
        result.bonusWhy = "похоже на прошлое сообщение";
        result.score = sim + result.bonus;
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
    } else if (!item.reply && strange.length > 0 && item.sim < MIN_SIMILARITY_WITH_UNKNOWN) {
      // Есть незнакомое слово, а совпадение неточное — лучше признаться, чем угадывать
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
    const aboutWord = strange.length
      ? unknownWordAnswers
          .map((a) => fill(model, a, memory, text, strange[0]))
          .filter((a): a is string => a !== null)
      : [];
    active = toActive(aboutWord.length ? aboutWord : fallbackAnswers, 1);
    if (aboutWord.length) used.push(`Слово «${strange[0]}» модели незнакомо, поэтому она не стала угадывать.`);
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

  const search: Search = {
    features: us.map((name) => ({ name, weight: weightOf(model, name) })),
    key,
    uNorm,
    rows: scored.slice(0, 6).map((x) => ({
      question: x.question,
      common: x.common,
      dot: x.dot,
      qNorm: x.qNorm,
      sim: x.sim,
      rawSim: x.rawSim,
      keyMissed: x.keyMissed,
      bonus: x.bonus,
      bonusWhy: x.bonusWhy,
      score: x.score,
      weight: x.score ** 2,
      chosen: chosen.some((c) => c.item === x),
      reply: x.reply,
    })),
  };

  return {
    userTokens,
    matches: chosen.map(({ item }) => ({
      question: item.question,
      similarity: Math.min(1, item.score),
      reply: item.reply,
    })),
    fallback: chosen.length === 0,
    weak: chosen.length > 0 && !chosen[0].item.reply && chosen[0].item.sim < WEAK_SIMILARITY,
    strange,
    active,
    memory,
    used,
    search,
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
    total,
    temp,
    context: context.filter((id) => id !== START).map((id) => ({ id, text: tokenText(model, id) })),
    cands: entries.map(([id, c], i) => ({
      id,
      text: tokenText(model, id),
      weight: c.weight,
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
