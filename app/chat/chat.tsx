"use client";

import Link from "next/link";
import { useEffect, useMemo, useRef, useState, type CSSProperties, type FormEvent } from "react";
import { suggestions } from "@/lib/chat-corpus";
import {
  END,
  emptyMemory,
  getModel,
  nextStep,
  remember,
  retrieve,
  sample,
  type ActiveAnswer,
  type Match,
  type Memory,
  type Search,
  type Step,
  type Tok,
} from "@/lib/chat-model";
import { Chip, show } from "../chip";

const MAX_TOKENS = 60;
const SHOWN_CANDIDATES = 5;

/** Сколько длится показ кандидатов и показ выбранного токена, мс */
const SPEEDS = {
  slow: { label: "Медленно", think: 3400, pick: 1900 },
  normal: { label: "Средне", think: 1900, pick: 1100 },
  fast: { label: "Быстро", think: 650, pick: 400 },
} as const;
type Speed = keyof typeof SPEEDS;

type UserMsg = { role: "user"; text: string };
type BotMsg = {
  role: "bot";
  userTokens: Tok[];
  matches: Match[];
  fallback: boolean;
  /** Что модель помнила, когда отвечала */
  memory: Memory;
  /** Как память повлияла на ответ */
  used: string[];
  /** Расчёт поиска похожих вопросов */
  search: Search;
  /** Ответы-примеры, из которых собирается ответ, и их веса */
  material: { text: string; weight: number }[];
  /** Шаги, которые уже закончились: в каждом выбран токен */
  steps: Step[];
};
type Msg = UserMsg | BotMsg;

/** Шаг, который идёт прямо сейчас: think — показаны кандидаты, pick — токен выбран */
type Live = { phase: "think" | "pick"; step: Step };

function pct(p: number) {
  if (p === 0) return "0%";
  if (p < 0.01) return "<1%";
  return Math.round(p * 100) + "%";
}

function times(n: number) {
  const d = n % 10;
  const h = n % 100;
  return `${n} ${d >= 2 && d <= 4 && (h < 12 || h > 14) ? "раза" : "раз"}`;
}

function hue(id: number) {
  return { "--h": (((id * 47) % 360) + 360) % 360 } as CSSProperties;
}

export default function Chat() {
  const model = useMemo(() => getModel(), []);
  const [messages, setMessages] = useState<Msg[]>([]);
  const [input, setInput] = useState("");
  const [live, setLive] = useState<Live | null>(null);
  const [paused, setPaused] = useState(false);
  const [speed, setSpeed] = useState<Speed>("normal");
  const [temp, setTemp] = useState(1);
  /** Какой токен готового ответа рассматриваем: [номер сообщения, номер шага] */
  const [inspect, setInspect] = useState<[number, number] | null>(null);

  const active = useRef<ActiveAnswer[]>([]);
  const memory = useRef<Memory>(emptyMemory());
  const listRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const panelRef = useRef<HTMLElement>(null);
  const stepRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [messages, live]);

  // Держим текущий шаг на виду в панели
  const liveIndex = live?.step.index;
  useEffect(() => {
    if (liveIndex && panelRef.current && stepRef.current) {
      panelRef.current.scrollTo({ top: stepRef.current.offsetTop - 64, behavior: "smooth" });
    }
  }, [liveIndex]);

  function send(text: string) {
    text = text.trim();
    if (!text || live) return;
    const r = retrieve(model, text, memory.current);
    active.current = r.active;
    setMessages([
      ...messages,
      { role: "user", text },
      { role: "bot", userTokens: r.userTokens, matches: r.matches, fallback: r.fallback, memory: r.memory,
        used: r.used,
        search: r.search,
        material: r.active.map((a) => ({ text: a.text, weight: a.weight })).sort((a, b) => b.weight - a.weight),
        steps: [],
      },
    ]);
    setInput("");
    setInspect(null);
    setPaused(false);
    setLive({ phase: "think", step: nextStep(model, r.active, [], temp) });
  }

  /** Ответ закончен: запоминаем разговор для следующих сообщений */
  function finish(steps: Step[]) {
    const bot = messages[messages.length - 1] as BotMsg;
    const user = messages[messages.length - 2] as UserMsg;
    const botText = steps
      .map((s) => s.cands[s.picked!])
      .filter((c) => c.id !== END)
      .map((c) => c.text)
      .join("");
    memory.current = remember(bot.memory, user.text, botText);
    setLive(null);
    inputRef.current?.focus();
  }

  function reset() {
    memory.current = emptyMemory();
    setMessages([]);
    setLive(null);
    setInspect(null);
  }

  /** Один шаг вперёд: показать выбор или перейти к следующему токену */
  function advance() {
    if (!live) return;
    if (live.phase === "think") {
      setLive({ phase: "pick", step: { ...live.step, picked: sample(live.step) } });
      return;
    }
    const last = messages[messages.length - 1] as BotMsg;
    const steps = [...last.steps, live.step];
    setMessages([...messages.slice(0, -1), { ...last, steps }]);

    const done = live.step.cands[live.step.picked!].id === END || steps.length >= MAX_TOKENS;
    if (done) {
      finish(steps);
    } else {
      const out = steps.map((s) => s.cands[s.picked!].id);
      setLive({ phase: "think", step: nextStep(model, active.current, out, temp) });
    }
  }

  useEffect(() => {
    if (!live || paused) return;
    // Если кандидат всего один, разглядывать нечего — идём быстрее
    const factor = live.step.cands.length === 1 ? 0.3 : 1;
    const t = setTimeout(advance, SPEEDS[speed][live.phase] * factor);
    return () => clearTimeout(t);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [live, paused, speed]);

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    send(input);
  }

  // Что показывать в панели: идущий шаг, выбранный токен или последний токен последнего ответа
  const lastBotIndex = messages.length - 1;
  const lastBot = messages[lastBotIndex]?.role === "bot" ? (messages[lastBotIndex] as BotMsg) : null;
  let viewMsg: BotMsg | null = lastBot;
  let viewStep: Step | null = live?.step ?? null;
  let viewPhase: Live["phase"] = live?.phase ?? "pick";
  let viewAt: [number, number] | null = null;
  let viewMsgIndex = lastBotIndex;
  if (!live) {
    viewPhase = "pick";
    if (inspect && messages[inspect[0]]?.role === "bot") {
      viewMsg = messages[inspect[0]] as BotMsg;
      viewStep = viewMsg.steps[inspect[1]] ?? null;
      viewAt = inspect;
      viewMsgIndex = inspect[0];
    } else if (lastBot && lastBot.steps.length) {
      viewStep = lastBot.steps[lastBot.steps.length - 1];
      viewAt = [lastBotIndex, lastBot.steps.length - 1];
    }
  }

  // Расход токенов: большие модели при каждом ответе перечитывают всю историю разговора
  const usage: UsageRow[] = [];
  let history = 0;
  messages.forEach((m, i) => {
    if (m.role !== "bot") return;
    let output = m.steps.filter((st) => st.cands[st.picked!].id !== END).length;
    if (i === lastBotIndex && live?.phase === "pick" && live.step.cands[live.step.picked!].id !== END) output++;
    usage.push({ index: i, history, message: m.userTokens.length, output });
    history += m.userTokens.length + output;
  });
  const totalIn = usage.reduce((sum, u) => sum + u.history + u.message, 0);
  const totalOut = usage.reduce((sum, u) => sum + u.output, 0);

  return (
    <div className="chatPage">
      <header className="chatTop">
        <div>
          <h1>Чат с мини-моделью</h1>
          <p>Напишите сообщение и смотрите, как ответ собирается токен за токеном.</p>
        </div>
        <div className="topBtns">
          {messages.length > 0 && (
            <button type="button" className="navLink" onClick={reset} title="Очистить чат и память модели">
              Новый разговор
            </button>
          )}
          <Link href="/" className="navLink">
            ← Токенизация
          </Link>
        </div>
      </header>

      <div className="chatBody">
        <section className="chatCol">
          <div className="messages" ref={listRef}>
            {messages.length === 0 && (
              <div className="welcome">
                <div className="welcomeIcon">💬</div>
                <h2>Спросите что-нибудь</h2>
                <p>
                  Это крошечная учебная модель: она знает около {model.topicCount} тем, помнит разговор и сама задаёт вопросы. Зато
                  справа видно каждый её шаг.
                </p>
                <div className="suggest">
                  {suggestions.map((s) => (
                    <button key={s} type="button" onClick={() => send(s)}>
                      {s}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {messages.map((m, mi) =>
              m.role === "user" ? (
                <div key={mi} className="msg user">
                  <div className="bubble">{m.text}</div>
                </div>
              ) : (
                <div key={mi} className="msg bot">
                  <div className="avatar">ИИ</div>
                  <div className="bubble">
                    {m.steps.map((s, si) => {
                      const c = s.cands[s.picked!];
                      if (c.id === END) return null;
                      const on = viewAt?.[0] === mi && viewAt[1] === si;
                      return (
                        <button
                          key={si}
                          type="button"
                          className={`tok ${on ? "on" : ""}`}
                          style={hue(c.id)}
                          disabled={!!live}
                          onClick={() => setInspect([mi, si])}
                          title="Показать, как выбран этот токен"
                        >
                          {c.text}
                        </button>
                      );
                    })}
                    {live && mi === lastBotIndex && live.phase === "pick" && live.step.cands[live.step.picked!].id !== END && (
                      <span className="tok fresh" style={hue(live.step.cands[live.step.picked!].id)}>
                        {live.step.cands[live.step.picked!].text}
                      </span>
                    )}
                    {live && mi === lastBotIndex && <span className="caret" />}
                  </div>
                </div>
              ),
            )}
          </div>

          <form className="composer" onSubmit={onSubmit}>
            <input
              ref={inputRef}
              value={input}
              maxLength={200}
              placeholder={live ? "Модель подбирает ответ…" : "Напишите сообщение…"}
              onChange={(e) => setInput(e.target.value)}
              aria-label="Сообщение"
            />
            {live ? (
              <button type="button" className="send stop" onClick={() => finish(lastBot?.steps ?? [])} title="Остановить">
                ■
              </button>
            ) : (
              <button type="submit" className="send" disabled={!input.trim()} title="Отправить">
                ↑
              </button>
            )}
          </form>
          {usage.length > 0 && (
            <p className="usageLine">
              Токены за разговор: вход <b>{totalIn}</b> · выход <b>{totalOut}</b> · всего{" "}
              <b>{totalIn + totalOut}</b>
            </p>
          )}
        </section>

        <aside className="panel" ref={panelRef}>
          <div className="panelHead">
            <h2>Под капотом</h2>
            {live && (
              <div className="panelBtns">
                <button type="button" onClick={() => setPaused(!paused)}>
                  {paused ? "▶ Дальше" : "⏸ Пауза"}
                </button>
                {paused && (
                  <button type="button" onClick={advance}>
                    Шаг ▶
                  </button>
                )}
              </div>
            )}
          </div>

          <div className="settings">
            <div className="seg" role="group" aria-label="Скорость">
              {(Object.keys(SPEEDS) as Speed[]).map((k) => (
                <button key={k} type="button" className={k === speed ? "active" : ""} onClick={() => setSpeed(k)}>
                  {SPEEDS[k].label}
                </button>
              ))}
            </div>
            <label className="tempRow">
              <span>
                Температура <b>{temp.toFixed(1)}</b>
              </span>
              <input
                type="range"
                min={0}
                max={1.5}
                step={0.1}
                value={temp}
                onChange={(e) => setTemp(Number(e.target.value))}
              />
            </label>
          </div>

          {!viewMsg ? (
            <div className="panelEmpty">
              <p>Здесь появится разбор ответа в четыре шага:</p>
              <ol>
                <li>Модель вспоминает разговор: ваше имя, прошлое сообщение, свой вопрос.</li>
                <li>Ваше сообщение режется на токены.</li>
                <li>Модель находит похожие вопросы в своих обучающих примерах.</li>
                <li>
                  Ответ собирается по одному токену: для каждого видны кандидаты, их вероятности и
                  выбор.
                </li>
              </ol>
            </div>
          ) : (
            <>
              <div className="block">
                <h3>1. Память о разговоре</h3>
                <MemoryView memory={viewMsg.memory} used={viewMsg.used} />
              </div>

              <div className="block">
                <h3>2. Сообщение → токены</h3>
                <div className="tokens">
                  {viewMsg.userTokens.map((t, i) => (
                    <Chip key={i} entry={t} showId={t.id >= 0} />
                  ))}
                </div>
              </div>

              <div className="block">
                <h3>3. Похожие вопросы из обучения</h3>
                {viewMsg.fallback ? (
                  <p className="muted">
                    Ничего похожего не нашлось, поэтому модель собирает ответ из общих фраз вроде «я
                    не понял».
                  </p>
                ) : (
                  <>
                    <div className="matches">
                      {viewMsg.matches.map((m) => (
                        <div key={m.question} className="bar">
                          <span className="barPair" title={m.reply ? "Понято как ответ на вопрос модели" : undefined}>
                            {m.reply && "↩ "}«{m.question}»
                          </span>
                          <span className="barTrack">
                            <span className="barFill grow" style={{ width: `${m.similarity * 100}%` }} />
                          </span>
                          <span className="barCount">{pct(m.similarity)}</span>
                        </div>
                      ))}
                    </div>
                    <p className="muted small">
                      Ответы на эти вопросы — материал для ответа. Чем больше сходство, тем больше
                      их вес.
                    </p>
                  </>
                )}
                <SearchCalc search={viewMsg.search} fallback={viewMsg.fallback} />
              </div>

              {viewStep && (
                <div className="block" ref={stepRef} key={`${viewMsgIndex}-${viewStep.index}`}>
                  <h3>4. Выбор токена №{viewStep.index}</h3>
                  {viewStep.context.length === 0 ? (
                    <p className="muted">
                      Ответ только начинается. Модель смотрит, с каких токенов начинались ответы на
                      похожие вопросы.
                    </p>
                  ) : (
                    <>
                      <p className="muted">
                        Модель смотрит на конец ответа и ищет, что шло после этих токенов в похожих
                        примерах:
                      </p>
                      <div className="tokens context">
                        {viewStep.context.map((t, i) => (
                          <Chip key={i} entry={t} />
                        ))}
                        <span className="arrow">→ ?</span>
                      </div>
                    </>
                  )}

                  <Candidates step={viewStep} phase={viewPhase} />

                  <details className="calc">
                    <summary>Откуда веса: {viewMsg.material.length} ответов-примеров</summary>
                    <p className="calcStep">
                      Вес кандидата — это сумма весов ответов, в которых этот токен стоит сразу
                      после показанного контекста. Вероятность = вес кандидата ÷ сумма весов всех
                      кандидатов.
                    </p>
                    <ul className="material">
                      {viewMsg.material.map((a, i) => (
                        <li key={i}>
                          <b>{a.weight.toFixed(2)}</b> {a.text}
                        </li>
                      ))}
                    </ul>
                  </details>

                  <p className={`status ${viewPhase}`}>
                    {viewPhase === "think" ? (
                      <>⏳ Модель оценивает кандидатов…</>
                    ) : (
                      <PickedNote step={viewStep} />
                    )}
                  </p>
                </div>
              )}

              <div className="block">
                <h3>5. Расход токенов</h3>
                <UsageView rows={usage} current={viewMsgIndex} totalIn={totalIn} totalOut={totalOut} />
              </div>
            </>
          )}

          <p className="about">
            Это настоящая, но крошечная модель: {model.answerCount} примеров ответов и словарь из{" "}
            {model.vocab.length} токенов. Вероятности не выдуманы — они считаются по тому, что шло
            после таких же токенов в похожих примерах. У больших моделей эту работу делает
            нейросеть с миллиардами параметров.
          </p>
        </aside>
      </div>
    </div>
  );
}

type UsageRow = { index: number; history: number; message: number; output: number };

function featureLabel(name: string) {
  return name.startsWith("=") ? `«${name.slice(1)}» целиком` : `основа «${name}»`;
}

/** Как модель искала похожие вопросы: все числа расчёта */
function SearchCalc({ search, fallback }: { search: Search; fallback: boolean }) {
  return (
    <details className="calc" open>
      <summary>Как найдены совпадения — расчёт</summary>

      <p className="calcStep">
        <b>а)</b> Сообщение превращается в признаки: основы слов и слова целиком. Чем реже признак
        встречается в примерах, тем больше его вес.
      </p>
      <div className="feats">
        {search.features.map((f) => (
          <span key={f.name} className="feat">
            {featureLabel(f.name)} <i>{f.weight.toFixed(1)}</i>
          </span>
        ))}
        {search.features.length === 0 && <span className="feat">слов не найдено</span>}
      </div>
      <code className="calcLine">
        длина сообщения = √(сумма квадратов весов) = {search.uNorm.toFixed(2)}
      </code>

      <p className="calcStep">
        <b>б)</b> Сообщение сравнивается с каждым вопросом из примеров: сходство = сумма квадратов
        весов совпавших признаков ÷ (длина сообщения × длина вопроса). Лучшие результаты:
      </p>
      {search.rows.length === 0 && <p className="calcStep">Ни один вопрос не совпал ни одним словом.</p>}
      {search.rows.map((r, i) => (
        <div key={i} className={`calcRow ${r.chosen ? "chosen" : ""}`}>
          <div className="calcHead">
            <span>
              {r.reply && "↩ "}«{r.question}»
            </span>
            <b>{r.score.toFixed(2)}</b>
            <em>{r.chosen ? "взят" : "отсечён"}</em>
          </div>
          {r.common.length > 0 && (
            <>
              <div className="feats">
                {r.common.map((f) => (
                  <span key={f.name} className="feat hit">
                    {featureLabel(f.name)} <i>{f.weight.toFixed(1)}²</i>
                  </span>
                ))}
              </div>
              <code className="calcLine">
                сходство = {r.dot.toFixed(1)} ÷ ({search.uNorm.toFixed(2)} × {r.qNorm.toFixed(2)}) ={" "}
                {r.sim.toFixed(2)}
              </code>
            </>
          )}
          {r.bonus > 0.005 && (
            <code className="calcLine">
              {r.common.length > 0 ? `${r.sim.toFixed(2)} + ${r.bonus.toFixed(2)}` : r.bonus.toFixed(2)} (
              {r.bonusWhy}) = {r.score.toFixed(2)}
            </code>
          )}
          {r.chosen && (
            <code className="calcLine">
              вес ответов = {r.score.toFixed(2)}² = {r.weight.toFixed(2)}
            </code>
          )}
        </div>
      ))}

      <p className="calcStep">
        <b>в)</b> В работу идут вопросы с оценкой от 0.30, близкие к лучшему (не ниже 75% его
        оценки). При почти точном совпадении остальные отбрасываются, а если сообщение понято как
        ответ на вопрос модели (↩), посторонние темы не подмешиваются.{fallback && " Здесь таких нет, поэтому модель берёт общие ответы «я не понял»."}
      </p>
    </details>
  );
}

/** Расход токенов по ответам: вход (история + сообщение) и выход */
function UsageView({
  rows,
  current,
  totalIn,
  totalOut,
}: {
  rows: UsageRow[];
  current: number;
  totalIn: number;
  totalOut: number;
}) {
  const row = rows.find((r) => r.index === current);
  const max = Math.max(1, ...rows.map((r) => r.history + r.message + r.output));
  if (!row) return null;
  const input = row.history + row.message;
  return (
    <>
      <div className="usageNums">
        <div>
          <b>{input}</b>
          <span>вход</span>
        </div>
        <div>
          <b>{row.output}</b>
          <span>выход</span>
        </div>
        <div>
          <b>{input + row.output}</b>
          <span>этот ответ</span>
        </div>
        <div>
          <b>{totalIn + totalOut}</b>
          <span>весь разговор</span>
        </div>
      </div>
      <code className="calcLine">
        вход = история {row.history} + сообщение {row.message} = {input}
      </code>
      <code className="calcLine">
        этот ответ = вход {input} + выход {row.output} = {input + row.output}
      </code>
      <code className="calcLine">
        весь разговор = вход {totalIn} + выход {totalOut} = {totalIn + totalOut}
      </code>

      <div className="usageRows">
        {rows.map((r, i) => (
          <div key={r.index} className={`usageRow ${r.index === current ? "current" : ""}`}>
            <span>№{i + 1}</span>
            <span className="usageBar">
              <span className="seg history" style={{ width: `${(r.history / max) * 100}%` }} />
              <span className="seg message" style={{ width: `${(r.message / max) * 100}%` }} />
              <span className="seg output" style={{ width: `${(r.output / max) * 100}%` }} />
            </span>
            <span className="usageSum">{r.history + r.message + r.output}</span>
          </div>
        ))}
      </div>
      <div className="legend">
        <span>
          <i className="seg history" /> история
        </span>
        <span>
          <i className="seg message" /> сообщение
        </span>
        <span>
          <i className="seg output" /> ответ
        </span>
      </div>
      <p className="muted small">
        Большая модель перед каждым ответом перечитывает весь разговор, поэтому вход растёт с
        каждым сообщением — и именно за эти токены платят. Наша мини-модель историю не
        перечитывает (у неё короткая память), но расход посчитан так, как его считают у настоящих.
      </p>
    </>
  );
}

function MemoryView({ memory, used }: { memory: Memory; used: string[] }) {
  const empty = !memory.name && !memory.last && !memory.pending;
  return (
    <>
      {empty ? (
        <p className="muted">Пока пусто: это первое сообщение разговора.</p>
      ) : (
        <dl className="memory">
          {memory.name && (
            <>
              <dt>Имя собеседника</dt>
              <dd>{memory.name}</dd>
            </>
          )}
          {memory.last && (
            <>
              <dt>Прошлое сообщение</dt>
              <dd>«{memory.last}»</dd>
            </>
          )}
          {memory.pending && (
            <>
              <dt>Мой вопрос, на который жду ответ</dt>
              <dd>«{memory.pending.question}»</dd>
            </>
          )}
          {memory.said.length > 0 && (
            <>
              <dt>Уже сказано ответов</dt>
              <dd>{memory.said.length}</dd>
            </>
          )}
        </dl>
      )}
      {used.length > 0 && (
        <ul className="used">
          {used.map((u) => (
            <li key={u}>{u}</li>
          ))}
        </ul>
      )}
      <p className="muted small">
        У больших моделей отдельной памяти нет: им каждый раз заново подают на вход весь разговор
        в виде токенов.
      </p>
    </>
  );
}

function Candidates({ step, phase }: { step: Step; phase: Live["phase"] }) {
  const picked = phase === "pick" ? step.picked : undefined;
  // Показываем лучших, а если выпал редкий токен — добавляем и его
  const shown = step.cands.map((_, i) => i).filter((i) => i < SHOWN_CANDIDATES || i === picked);
  const hidden = step.cands.length - shown.length;
  const hiddenProb = step.cands.reduce((s, c, i) => (shown.includes(i) ? s : s + c.prob), 0);

  return (
    <div className="cands">
      {shown.map((i) => {
        const c = step.cands[i];
        const state = picked === undefined ? "" : picked === i ? "picked" : "faded";
        return (
          <div key={i} className={`cand ${state}`}>
            <span className="candTop">
              <span className="barPair">{show(c.text)}</span>
              <span className="barTrack">
                <span className="barFill grow" style={{ width: `${c.prob * 100}%` }} />
              </span>
              <span className="barCount">{pct(c.prob)}</span>
            </span>
            {c.count > 0 && (
              <span className="candWhy">
                Встретился {times(c.count)}, например: «{c.example}»
              </span>
            )}
            <span className="candCalc">
              вес {c.weight.toFixed(2)} ÷ {step.total.toFixed(2)} = {pct(c.weight / step.total)}
              {step.temp !== 1 && ` → при температуре ${step.temp.toFixed(1)}: ${pct(c.prob)}`}
            </span>
          </div>
        );
      })}
      {hidden > 0 && (
        <div className="cand rest">
          <span className="candTop">
            <span className="barPair">…ещё {hidden}</span>
            <span className="barTrack">
              <span className="barFill" style={{ width: `${hiddenProb * 100}%` }} />
            </span>
            <span className="barCount">{pct(hiddenProb)}</span>
          </span>
        </div>
      )}
    </div>
  );
}

function PickedNote({ step }: { step: Step }) {
  const c = step.cands[step.picked!];
  if (step.cands.length === 1) {
    return (
      <>
        ✅ Вариант всего один — <b>«{show(c.text)}»</b>. После такого начала в примерах шло только
        это.
      </>
    );
  }
  return (
    <>
      🎲 Выбран <b>«{show(c.text)}»</b> ({pct(c.prob)})
      {step.picked! > 0
        ? " — не самый вероятный: сработала случайность."
        : " — самый вероятный вариант."}
      {c.id === END && " Это сигнал остановиться: ответ готов."}
    </>
  );
}
