"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";
import { pairPositions, trainBpe, type VocabEntry } from "@/lib/bpe";
import { examples } from "@/lib/examples";
import { Chip, show } from "./chip";
import Generation from "./generation";

const MAX_TEXT = 500;
const SPEED_MS = 800;

/** Дерево: из каких кусочков склеен токен */
function Tree({ id, vocab }: { id: number; vocab: VocabEntry[] }) {
  const entry = vocab[id];
  if (entry.left === undefined || entry.right === undefined) {
    return (
      <div className="tree">
        <Chip entry={entry} />
        <span className="treeNote">символ</span>
      </div>
    );
  }
  return (
    <div className="tree">
      <Chip entry={entry} />
      <span className="treeNote">шаг {entry.step}</span>
      <div className="treeKids">
        <Tree id={entry.left} vocab={vocab} />
        <Tree id={entry.right} vocab={vocab} />
      </div>
    </div>
  );
}

export default function Tokenizer() {
  const [text, setText] = useState(examples[0].text);
  const [step, setStep] = useState(0);
  const [playing, setPlaying] = useState(true);
  const [selected, setSelected] = useState<number | null>(null);

  const bpe = useMemo(() => trainBpe(text), [text]);
  const total = bpe.steps.length;
  const cur = Math.min(step, total);
  const isPlaying = playing && cur < total;

  useEffect(() => {
    if (!isPlaying) return;
    const t = setInterval(() => setStep((s) => s + 1), SPEED_MS);
    return () => clearInterval(t);
  }, [isPlaying]);

  function load(newText: string, autoplay: boolean) {
    setText(newText);
    setSelected(null);
    setStep(autoplay ? 0 : Number.MAX_SAFE_INTEGER);
    setPlaying(autoplay);
  }

  function goTo(n: number) {
    setPlaying(false);
    setStep(Math.max(0, Math.min(total, n)));
  }

  const { vocab } = bpe;
  const merge = cur > 0 ? bpe.steps[cur - 1] : null;
  const before = cur > 1 ? bpe.steps[cur - 2].tokens : bpe.initial;
  const tokens = merge ? merge.tokens : bpe.initial;
  const pairIdx = merge ? pairPositions(before, merge.a, merge.b) : new Set<number>();
  const knownVocab = vocab.filter((v) => v.step <= cur);
  const baseCount = vocab.filter((v) => v.step === 0).length;

  // Что показывать в дереве: выбранный токен или самый длинный из текущих
  const treeId =
    selected !== null && selected < vocab.length && vocab[selected].step <= cur
      ? selected
      : (merge?.newId ?? null);

  const example = examples.find((e) => e.text === text);
  const chars = bpe.initial.length;
  const saved = chars ? Math.round((1 - tokens.length / chars) * 100) : 0;

  return (
    <main className="page">
      <header>
        <h1>Симулятор токенизации ИИ</h1>
        <p className="lead">
          ИИ не читает буквы и слова — он читает числа. Сначала текст режется на кусочки
          — <b>токены</b>, и каждому даётся номер. Здесь видно, как это происходит под капотом.
        </p>
        <Link href="/chat" className="cta">
          💬 Попробовать в чате: напишите сообщение и смотрите, как подбирается ответ →
        </Link>
      </header>

      <section className="card">
        <h2>1. Введите текст или выберите пример</h2>
        <textarea
          value={text}
          maxLength={MAX_TEXT}
          rows={3}
          spellCheck={false}
          placeholder="Напишите что-нибудь…"
          onChange={(e) => load(e.target.value, false)}
        />
        <div className="examples">
          {examples.map((e) => (
            <button
              key={e.label}
              type="button"
              className={`example ${e.text === text ? "active" : ""}`}
              onClick={() => load(e.text, true)}
            >
              {e.label}
            </button>
          ))}
        </div>
        {example && <p className="hint">💡 {example.hint}</p>}
      </section>

      {chars === 0 ? (
        <p className="empty">Введите текст, чтобы увидеть токены.</p>
      ) : (
        <>
          <section className="card">
            <h2>2. Под капотом: склейка шаг за шагом</h2>
            <ol className="how">
              <li>Режем текст на отдельные символы.</li>
              <li>Ищем пару соседей, которая встречается чаще всего.</li>
              <li>Склеиваем её в новый токен — и повторяем.</li>
            </ol>

            <div className="controls">
              <button type="button" onClick={() => goTo(0)} title="В начало">⏮</button>
              <button type="button" onClick={() => goTo(cur - 1)} title="Шаг назад">◀</button>
              <button
                type="button"
                className="play"
                onClick={() => {
                  if (isPlaying) return setPlaying(false);
                  if (cur >= total) setStep(0);
                  else setStep(cur);
                  setPlaying(true);
                }}
              >
                {isPlaying ? "⏸ Пауза" : "▶ Играть"}
              </button>
              <button type="button" onClick={() => goTo(cur + 1)} title="Шаг вперёд">▶</button>
              <button type="button" onClick={() => goTo(total)} title="В конец">⏭</button>
              <input
                type="range"
                min={0}
                max={total}
                value={cur}
                onChange={(e) => goTo(Number(e.target.value))}
                aria-label="Шаг"
              />
              <span className="stepLabel">
                Шаг {cur} из {total}
              </span>
            </div>

            <div className="explain">
              {merge ? (
                <>
                  <div className="formula">
                    <Chip entry={vocab[merge.a]} mark="pair" />
                    <span>+</span>
                    <Chip entry={vocab[merge.b]} mark="pair" />
                    <span>→</span>
                    <Chip entry={vocab[merge.newId]} mark="new" showId />
                  </div>
                  <p>
                    Самая частая пара встречается <b>{merge.count}</b>{" "}
                    {plural(merge.count, "раз", "раза", "раз")} — склеиваем её в новый токен{" "}
                    <b>№{merge.newId}</b>.
                    {cur === total && " Больше повторяющихся пар нет — готово!"}
                  </p>
                  <div className="bars">
                    {merge.candidates.map((c, i) => (
                      <div key={i} className={`bar ${i === 0 ? "win" : ""}`}>
                        <span className="barPair">
                          {show(vocab[c.a].text)}
                          <i>+</i>
                          {show(vocab[c.b].text)}
                        </span>
                        <span className="barTrack">
                          <span
                            className="barFill"
                            style={{ width: `${(c.count / merge.count) * 100}%` }}
                          />
                        </span>
                        <span className="barCount">×{c.count}</span>
                      </div>
                    ))}
                  </div>
                </>
              ) : (
                <p>
                  <b>Начало.</b> Текст разрезан на символы: каждый символ — отдельный токен.
                  {total === 0
                    ? " В этом тексте нет повторяющихся пар — склеивать нечего."
                    : " Нажмите «Играть» или листайте шаги."}
                </p>
              )}
            </div>

            {merge && (
              <>
                <div className="rowLabel">Было — пара обведена:</div>
                <div className="tokens dim">
                  {before.map((id, i) => (
                    <Chip key={i} entry={vocab[id]} mark={pairIdx.has(i) ? "pair" : undefined} />
                  ))}
                </div>
              </>
            )}
            <div className="rowLabel">
              {merge ? "Стало — новый токен подсвечен:" : "Токены:"}{" "}
              <small>(нажмите на токен, чтобы увидеть, из чего он собран)</small>
            </div>
            <div className="tokens">
              {tokens.map((id, i) => (
                <Chip
                  key={`${cur}-${i}`}
                  entry={vocab[id]}
                  mark={id === treeId ? "selected" : id === merge?.newId ? "new" : undefined}
                  onClick={() => setSelected(id)}
                />
              ))}
            </div>

            <div className="stats">
              <div>
                <b>{chars}</b>
                <span>символов</span>
              </div>
              <div>
                <b>{tokens.length}</b>
                <span>токенов сейчас</span>
              </div>
              <div>
                <b>{saved}%</b>
                <span>сжатие</span>
              </div>
              <div>
                <b>{knownVocab.length}</b>
                <span>размер словаря</span>
              </div>
            </div>
          </section>

          <section className="card">
            <h2>3. Из чего собран токен</h2>
            {treeId === null ? (
              <p className="muted">
                Пока все токены — одиночные символы. Сделайте шаг вперёд или нажмите на любой
                склеенный токен выше.
              </p>
            ) : (
              <>
                <p className="muted">
                  Токен <b>№{treeId}</b> «{show(vocab[treeId].text)}» — сверху результат, снизу
                  кусочки, из которых он склеен.
                </p>
                <div className="treeWrap">
                  <Tree id={treeId} vocab={vocab} />
                </div>
              </>
            )}
          </section>

          <section className="card">
            <h2>4. Что в итоге видит модель</h2>
            <p className="muted">Не текст, а список номеров токенов:</p>
            <code className="ids">[{tokens.join(", ")}]</code>

            <h3>Словарь</h3>
            <p className="muted">
              {baseCount} базовых {plural(baseCount, "символ", "символа", "символов")} +{" "}
              {knownVocab.length - baseCount}{" "}
              {plural(knownVocab.length - baseCount, "склеенный", "склеенных", "склеенных")}.
              У настоящих моделей в словаре 50–200 тысяч токенов.
            </p>
            <div className="tokens">
              {knownVocab.map((v) => (
                <Chip
                  key={v.id}
                  entry={v}
                  showId
                  mark={v.id === treeId ? "selected" : undefined}
                  onClick={() => setSelected(v.id)}
                />
              ))}
            </div>
          </section>
        </>
      )}

      <Generation />

      <footer>
        Упрощённый алгоритм BPE (Byte Pair Encoding). Настоящие токенизаторы работают так же, но
        обучаются заранее на огромном количестве текстов.
      </footer>
    </main>
  );
}

function plural(n: number, one: string, few: string, many: string) {
  const d = n % 10;
  const h = n % 100;
  if (d === 1 && h !== 11) return one;
  if (d >= 2 && d <= 4 && (h < 12 || h > 14)) return few;
  return many;
}
