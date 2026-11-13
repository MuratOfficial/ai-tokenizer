"use client";

import { useEffect, useState } from "react";
import { END, fakeId, scenarios } from "@/lib/scenarios";
import { Chip, show } from "./chip";

const SPEED_MS = 1400;

function entry(text: string) {
  return { id: fakeId(text), text };
}

export default function Generation() {
  const [index, setIndex] = useState(0);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(false);

  const sc = scenarios[index];
  const total = sc.steps.length;
  const isPlaying = playing && step < total;

  useEffect(() => {
    if (!isPlaying) return;
    const t = setInterval(() => setStep((s) => s + 1), SPEED_MS);
    return () => clearInterval(t);
  }, [isPlaying]);

  function goTo(n: number) {
    setPlaying(false);
    setStep(Math.max(0, Math.min(total, n)));
  }

  const picked = sc.steps.slice(0, step).map((s) => s.options[s.pick ?? 0][0]);
  const answer = picked.filter((t) => t !== END);
  const last = step > 0 ? sc.steps[step - 1] : null;
  const lastPick = last ? (last.pick ?? 0) : 0;
  const done = step === total;
  const ids = [...sc.prompt, ...answer].map(fakeId);

  return (
    <section className="card">
      <h2>5. Как ИИ подбирает ответ</h2>
      <p className="muted">
        Ответ не достаётся из памяти целиком. Модель смотрит на все токены запроса, оценивает
        вероятность каждого следующего токена, выбирает один, дописывает его — и повторяет, пока
        не выберет специальный токен <code>{END}</code>.
      </p>

      <div className="examples">
        {scenarios.map((s, i) => (
          <button
            key={s.label}
            type="button"
            className={`example ${i === index ? "active" : ""}`}
            onClick={() => {
              setIndex(i);
              setStep(0);
              setPlaying(true);
            }}
          >
            {s.label}
          </button>
        ))}
      </div>

      <div className="rowLabel">Запрос, разрезанный на токены:</div>
      <div className="tokens">
        {sc.prompt.map((t, i) => (
          <Chip key={i} entry={entry(t)} showId />
        ))}
      </div>

      <div className="rowLabel">Ответ — по одному токену:</div>
      <div className="tokens answer">
        {answer.map((t, i) => (
          <Chip
            key={`${index}-${i}`}
            entry={entry(t)}
            showId
            mark={i === step - 1 ? "new" : undefined}
          />
        ))}
        {done ? <span className="endMark">■ конец</span> : <span className="caret" />}
      </div>

      <div className="controls genControls">
        <button type="button" onClick={() => goTo(0)} title="В начало">⏮</button>
        <button type="button" onClick={() => goTo(step - 1)} title="Шаг назад">◀</button>
        <button
          type="button"
          className="play"
          onClick={() => {
            if (isPlaying) return setPlaying(false);
            if (done) setStep(0);
            setPlaying(true);
          }}
        >
          {isPlaying ? "⏸ Пауза" : "▶ Играть"}
        </button>
        <button type="button" onClick={() => goTo(step + 1)} title="Следующий токен">▶</button>
        <span className="stepLabel">
          Токен {step} из {total}
        </span>
      </div>

      <div className="explain">
        {last ? (
          <>
            <p>
              Кандидаты на токен №{step} и их вероятности. Выбран{" "}
              <b>«{show(last.options[lastPick][0])}»</b>
              {lastPick > 0 &&
                " — не самый вероятный: немного случайности делает ответы разнообразнее"}
              .{done && " Это сигнал остановиться — ответ готов."}
            </p>
            <div className="bars">
              {last.options.map(([text, p], i) => (
                <div key={i} className={`bar ${i === lastPick ? "win" : ""}`}>
                  <span className="barPair">{show(text)}</span>
                  <span className="barTrack">
                    <span className="barFill" style={{ width: `${p}%` }} />
                  </span>
                  <span className="barCount">{p}%</span>
                </div>
              ))}
            </div>
          </>
        ) : (
          <p>
            <b>Начало.</b> Запрос превращён в токены. Нажмите «Играть» или ▶, чтобы модель выбрала
            первый токен ответа.
          </p>
        )}
      </div>

      <div className="rowLabel">Что модель видит на этом шаге:</div>
      <code className="ids">[{ids.join(", ")}]</code>

      <p className="hint">
        Это симуляция: вероятности и номера токенов заданы вручную для наглядности. Настоящая
        модель вычисляет вероятность для каждого из ~100 000 токенов словаря на каждом шаге.
      </p>
    </section>
  );
}
