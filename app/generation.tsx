"use client";

import { useEffect, useState } from "react";
import { END, fakeId, scenarios, withTemperature, type Option } from "@/lib/scenarios";
import { Chip, show } from "./chip";

const SPEED_MS = 1600;

type Pick = {
  option: Option;
  /** Вероятность в момент выбора, % */
  prob: number;
  /** Место среди кандидатов (1 — самый вероятный) */
  rank: number;
  of: number;
  by: "model" | "user";
};

function entry(text: string) {
  return { id: fakeId(text), text };
}

function pct(p: number) {
  if (p === 0) return "0%";
  if (p < 1) return "<1%";
  return Math.round(p) + "%";
}

export default function Generation() {
  const [index, setIndex] = useState(0);
  const [path, setPath] = useState<Pick[]>([]);
  const [temp, setTemp] = useState(1);
  const [playing, setPlaying] = useState(false);

  const sc = scenarios[index];
  const nodeId = path.length ? path[path.length - 1].option.to : sc.start;
  const node = nodeId ? sc.nodes[nodeId] : null;
  const dist = node ? withTemperature(node.options, temp) : null;
  const isPlaying = playing && node !== null;

  function choose(i: number, by: Pick["by"]) {
    if (!node || !dist) return;
    const option = node.options[i];
    const rank = node.options.filter((x) => x.p > option.p).length + 1;
    setPath([...path, { option, prob: dist.probs[i], rank, of: node.options.length, by }]);
  }

  /** Случайный выбор с учётом вероятностей — так и работает настоящая модель */
  function sample() {
    if (!dist) return;
    const total = dist.probs.reduce((s, x) => s + x, 0);
    let r = Math.random() * total;
    let i = 0;
    while (i < dist.probs.length - 1 && r >= dist.probs[i]) r -= dist.probs[i++];
    choose(i, "model");
  }

  useEffect(() => {
    if (!isPlaying) return;
    const t = setTimeout(sample, SPEED_MS);
    return () => clearTimeout(t);
    // Перезапускаем таймер после каждого выбранного токена
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isPlaying, path]);

  function reset(newIndex = index) {
    setIndex(newIndex);
    setPath([]);
    setPlaying(false);
  }

  const answer = path.map((x) => x.option.t).filter((t) => t !== END);
  const focus = new Set(node?.focus ?? []);
  const ids = [...sc.prompt, ...answer].map(fakeId);

  return (
    <section className="card">
      <h2>5. Как ИИ подбирает ответ</h2>
      <p className="muted">
        Ответ не достаётся из памяти целиком. На каждом шаге модель оценивает вероятность{" "}
        <b>каждого</b> токена словаря, выбирает один, дописывает его — и повторяет. Выбирайте
        токены сами или дайте выбрать модели: ответ каждый раз может получиться другим.
      </p>

      <div className="examples">
        {scenarios.map((s, i) => (
          <button
            key={s.label}
            type="button"
            className={`example ${i === index ? "active" : ""}`}
            onClick={() => reset(i)}
          >
            {s.label}
          </button>
        ))}
      </div>
      <p className="hint">💡 {sc.about}</p>

      <div className="rowLabel">
        Запрос, разрезанный на токены{" "}
        {focus.size > 0 && <small>(обведены те, на которые модель сейчас опирается сильнее всего)</small>}
      </div>
      <div className="tokens">
        {sc.prompt.map((t, i) => (
          <Chip key={i} entry={entry(t)} showId mark={focus.has(i) ? "pair" : undefined} />
        ))}
      </div>

      <div className="rowLabel">Ответ — по одному токену:</div>
      <div className="tokens answer">
        {answer.map((t, i) => (
          <Chip
            key={`${index}-${i}`}
            entry={entry(t)}
            showId
            mark={i === path.length - 1 ? "new" : undefined}
          />
        ))}
        {node ? <span className="caret" /> : <span className="endMark">■ конец</span>}
      </div>

      <div className="controls genControls">
        <button type="button" onClick={() => reset()} title="Сначала">⏮</button>
        <button
          type="button"
          onClick={() => {
            setPlaying(false);
            setPath(path.slice(0, -1));
          }}
          disabled={path.length === 0}
          title="Отменить последний токен"
        >
          ◀
        </button>
        <button
          type="button"
          onClick={() => {
            setPlaying(false);
            sample();
          }}
          disabled={!node}
        >
          🎲 Выбрать как модель
        </button>
        <button
          type="button"
          className="play"
          onClick={() => {
            if (isPlaying) return setPlaying(false);
            if (!node) setPath([]);
            setPlaying(true);
          }}
        >
          {isPlaying ? "⏸ Пауза" : "▶ Играть"}
        </button>
      </div>

      <label className="temp">
        <span>
          Температура: <b>{temp.toFixed(1)}</b>
        </span>
        <input
          type="range"
          min={0}
          max={1.5}
          step={0.1}
          value={temp}
          onChange={(e) => setTemp(Number(e.target.value))}
        />
        <small>
          {temp === 0
            ? "Всегда берётся самый вероятный токен — ответ каждый раз одинаковый."
            : temp < 0.8
              ? "Низкая: модель осторожна, редкие варианты почти не выпадают."
              : temp <= 1.2
                ? "Обычная: вероятности как есть."
                : "Высокая: шансы выравниваются, ответы разнообразнее — и ошибок больше."}
        </small>
      </label>

      {node && dist ? (
        <div className="explain">
          <p>
            <b>Шаг {path.length + 1}.</b> {node.note}
          </p>
          <p className="muted candHead">Кандидаты на следующий токен — нажмите, чтобы выбрать самому:</p>
          <div className="cands">
            {node.options.map((x, i) => (
              <button key={i} type="button" className="cand" onClick={() => choose(i, "user")}>
                <span className="candTop">
                  <span className="barPair">{show(x.t)}</span>
                  <span className="barTrack">
                    <span className="barFill" style={{ width: `${dist.probs[i]}%` }} />
                  </span>
                  <span className="barCount">{pct(dist.probs[i])}</span>
                </span>
                <span className="candWhy">{x.why}</span>
              </button>
            ))}
            <div className="cand rest">
              <span className="candTop">
                <span className="barPair">…остальные</span>
                <span className="barTrack">
                  <span className="barFill" style={{ width: `${dist.rest}%` }} />
                </span>
                <span className="barCount">{pct(dist.rest)}</span>
              </span>
              <span className="candWhy">
                Все прочие токены словаря (около 100 000). У каждого шанс ничтожный, но не нулевой — чем выше температура, тем больше их общая доля. В этой симуляции выбрать их нельзя.
              </span>
            </div>
          </div>
        </div>
      ) : (
        <div className="explain">
          <p>
            <b>Ответ готов.</b> Нажмите «Играть» ещё раз — при температуре выше 0 ответ, скорее
            всего, получится другим.
          </p>
        </div>
      )}

      {path.length > 0 && (
        <>
          <div className="rowLabel">Почему ответ получился именно таким:</div>
          <ol className="history">
            {path.map((x, i) => (
              <li key={i}>
                <Chip entry={entry(x.option.t)} />
                <span className="historyText">
                  <b>{pct(x.prob)}</b> · {x.rank}-й из {x.of} по вероятности ·{" "}
                  {x.by === "user" ? "выбрали вы" : "выбрала модель"}
                  {x.by === "model" && x.rank > 1 && " (случайность: выпал не самый вероятный)"}
                  <br />
                  <span className="candWhy">{x.option.why}</span>
                </span>
              </li>
            ))}
          </ol>
        </>
      )}

      <div className="rowLabel">Что модель видит на этом шаге:</div>
      <code className="ids">[{ids.join(", ")}]</code>

      <p className="hint">
        Это симуляция: вероятности, пояснения и номера токенов написаны вручную для наглядности.
        Настоящая модель вычисляет вероятности нейросетью и не объясняет их словами — пояснения
        здесь показывают, какие закономерности из текстов она обычно улавливает.
      </p>
    </section>
  );
}
