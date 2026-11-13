"use client";

import Link from "next/link";
import { useMemo, useRef, useState, type ChangeEvent, type FormEvent } from "react";
import { emptyMemory, getModel, MIN_SIMILARITY, retrieve, unknownWords, type Model } from "@/lib/chat-model";
import {
  clearUnknown,
  importTraining,
  REASONS,
  removeLearned,
  removeUnknown,
  teach,
  updateLearned,
  useTraining,
  type LearnedGroup,
  type Training,
  type UnknownPhrase,
} from "@/lib/training-store";
import { Chip } from "../chip";

const lines = (text: string) => text.split("\n").map((x) => x.trim()).filter(Boolean);

function download(data: Training) {
  const url = URL.createObjectURL(new Blob([JSON.stringify(data, null, 2)], { type: "application/json" }));
  const a = document.createElement("a");
  a.href = url;
  a.download = "training.json";
  a.click();
  URL.revokeObjectURL(url);
}

export default function Train() {
  const training = useTraining();
  const model = useMemo(() => getModel(training.learned), [training.learned]);
  const fileRef = useRef<HTMLInputElement>(null);
  const [importNote, setImportNote] = useState("");

  async function onImport(e: ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0];
    e.target.value = "";
    if (!file) return;
    try {
      const added = importTraining(await file.text());
      setImportNote(`Загружено: выученных групп — ${added.learned}, фраз без ответа — ${added.unknown}.`);
    } catch (err) {
      setImportNote("Не удалось загрузить: " + (err instanceof SyntaxError ? "это не JSON." : (err as Error).message));
    }
  }

  return (
    <main className="page">
      <header className="trainTop">
        <div>
          <h1>Обучение мини-модели</h1>
          <p className="lead">
            Сюда попадают фразы, на которые модель не нашла ответа, ответила неуверенно или ответ
            вам не подошёл. Напишите, как нужно отвечать, — и модель выучит это.
          </p>
        </div>
        <div className="topBtns">
          <Link href="/chat" className="navLink">
            ← Чат
          </Link>
          <Link href="/" className="navLink">
            Токенизация
          </Link>
        </div>
      </header>

      <div className="stats trainStats">
        <div>
          <b>{training.unknown.length}</b>
          <span>фраз без ответа</span>
        </div>
        <div>
          <b>{training.learned.length}</b>
          <span>выучено вручную</span>
        </div>
        <div>
          <b>{model.topicCount}</b>
          <span>тем знает модель</span>
        </div>
        <div>
          <b>{model.vocab.length}</b>
          <span>токенов в словаре</span>
        </div>
      </div>

      <section className="card">
        <div className="cardHead">
          <h2>1. Хранилище: фразы без ответа</h2>
          {training.unknown.length > 0 && (
            <button type="button" className="ghost" onClick={clearUnknown}>
              Очистить
            </button>
          )}
        </div>
        {training.unknown.length === 0 ? (
          <p className="muted">
            Пока пусто. Напишите в <Link href="/chat">чате</Link> что-нибудь, чего модель не знает, —
            например «купи слона», — и фраза появится здесь.
          </p>
        ) : (
          <div className="items">
            {training.unknown.map((u) => (
              <UnknownItem key={u.text} item={u} model={model} />
            ))}
          </div>
        )}
      </section>

      <section className="card">
        <h2>2. Обучить своей фразе</h2>
        <p className="muted">
          Можно научить модель и заранее. Несколько вариантов вопроса и ответа — каждый с новой
          строки: чем их больше, тем разнообразнее ответы.
        </p>
        <TeachForm />
      </section>

      <section className="card">
        <div className="cardHead">
          <h2>3. Выучено вручную</h2>
          <div className="headBtns">
            <input ref={fileRef} type="file" accept="application/json,.json" hidden onChange={onImport} />
            <button type="button" className="ghost" onClick={() => fileRef.current?.click()}>
              Загрузить JSON
            </button>
            {(training.learned.length > 0 || training.unknown.length > 0) && (
              <button type="button" className="ghost" onClick={() => download(training)}>
                Скачать JSON
              </button>
            )}
          </div>
        </div>
        {importNote && <p className="hint importNote">{importNote}</p>}
        {training.learned.length === 0 ? (
          <p className="muted">Здесь появится всё, чему вы научите модель.</p>
        ) : (
          <div className="items">
            {training.learned.map((g) => (
              <LearnedItem key={g.id} group={g} />
            ))}
          </div>
        )}
      </section>

      <footer>
        Хранилище лежит в этом браузере (localStorage) отдельно от встроенных примеров. После
        обучения модель пересобирается: новые вопросы участвуют в поиске совпадений, а словарь
        токенов учится заново с учётом новых ответов.
      </footer>
    </main>
  );
}

/** Одна фраза без ответа: почему не нашлось совпадение и форма обучения */
function UnknownItem({ item, model }: { item: UnknownPhrase; model: Model }) {
  const [answers, setAnswers] = useState("");
  const r = useMemo(() => retrieve(model, item.text, emptyMemory()), [model, item.text]);
  const strange = useMemo(() => unknownWords(model, item.text), [model, item.text]);
  const best = r.search.rows.find((row) => !row.reply);

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    teach([item.text], lines(answers));
  }

  return (
    <div className="item">
      <div className="itemHead">
        <div>
          <b className="phraseTitle">«{item.text}»</b>
          <span className="meta">
            <span className={`reason ${item.reason ?? "none"}`}>{REASONS[item.reason ?? "none"]}</span> ·
            спрашивали: {item.count} · {new Date(item.lastAt).toLocaleString("ru-RU")}
          </span>
        </div>
        <button type="button" className="ghost" onClick={() => removeUnknown(item.text)}>
          Удалить
        </button>
      </div>

      <div className="why">
        {item.answer && (
          <div>
            <span className="tag">модель ответила</span>
            <span className="phrase answer">{item.answer}</span>
          </div>
        )}
        <div>
          <span className="tag">токены</span>
          <span className="tokens">
            {r.userTokens.map((t, i) => (
              <Chip key={i} entry={t} showId={t.id >= 0} />
            ))}
          </span>
        </div>
        <div>
          <span className="tag">незнакомые слова</span>
          {strange.length ? (
            strange.map((w) => (
              <span key={w} className="phrase">
                {w}
              </span>
            ))
          ) : (
            <span className="meta">нет — слова знакомы, но вместе не похожи ни на один вопрос</span>
          )}
        </div>
        <div>
          <span className="tag">ближайший вопрос</span>
          {best ? (
            <span className="meta">
              «{best.question}» — сходство {best.sim.toFixed(2)}, а нужно не меньше{" "}
              {MIN_SIMILARITY.toFixed(2)}
            </span>
          ) : (
            <span className="meta">ни один вопрос из примеров не совпал ни одним словом</span>
          )}
        </div>
      </div>

      <form className="teachForm" onSubmit={onSubmit}>
        <textarea
          rows={2}
          value={answers}
          maxLength={600}
          placeholder={"Как нужно отвечать? Каждый вариант ответа — с новой строки"}
          onChange={(e) => setAnswers(e.target.value)}
          aria-label={`Ответы на «${item.text}»`}
        />
        <button type="submit" className="primary" disabled={lines(answers).length === 0}>
          Обучить
        </button>
      </form>
    </div>
  );
}

/** Выученная группа: просмотр и правка */
function LearnedItem({ group }: { group: LearnedGroup }) {
  const [editing, setEditing] = useState(false);
  const [questions, setQuestions] = useState("");
  const [answers, setAnswers] = useState("");

  if (editing) {
    return (
      <form
        className="item teachForm two"
        onSubmit={(e) => {
          e.preventDefault();
          if (updateLearned(group.id, lines(questions), lines(answers))) setEditing(false);
        }}
      >
        <label>
          <span>Вопросы</span>
          <textarea rows={3} value={questions} maxLength={600} onChange={(e) => setQuestions(e.target.value)} />
        </label>
        <label>
          <span>Ответы</span>
          <textarea rows={3} value={answers} maxLength={600} onChange={(e) => setAnswers(e.target.value)} />
        </label>
        <div className="formFoot">
          <button
            type="submit"
            className="primary"
            disabled={lines(questions).length === 0 || lines(answers).length === 0}
          >
            Сохранить
          </button>
          <button type="button" className="ghost" onClick={() => setEditing(false)}>
            Отмена
          </button>
        </div>
      </form>
    );
  }

  return (
    <div className="item">
      <div className="itemHead">
        <div className="pairs">
          <div>
            <span className="tag">вопросы</span>
            {group.q.map((q) => (
              <span key={q} className="phrase">
                «{q}»
              </span>
            ))}
          </div>
          <div>
            <span className="tag">ответы</span>
            {group.a.map((a) => (
              <span key={a} className="phrase answer">
                {a}
              </span>
            ))}
          </div>
        </div>
        <div className="headBtns">
          <button
            type="button"
            className="ghost"
            onClick={() => {
              setQuestions(group.q.join("\n"));
              setAnswers(group.a.join("\n"));
              setEditing(true);
            }}
          >
            Изменить
          </button>
          <button type="button" className="ghost" onClick={() => removeLearned(group.id)}>
            Забыть
          </button>
        </div>
      </div>
    </div>
  );
}

function TeachForm() {
  const [questions, setQuestions] = useState("");
  const [answers, setAnswers] = useState("");
  const [saved, setSaved] = useState(false);

  function onSubmit(e: FormEvent) {
    e.preventDefault();
    if (teach(lines(questions), lines(answers))) {
      setQuestions("");
      setAnswers("");
      setSaved(true);
    }
  }

  return (
    <form className="teachForm two" onSubmit={onSubmit}>
      <label>
        <span>Вопросы</span>
        <textarea
          rows={3}
          value={questions}
          maxLength={600}
          placeholder={"купи слона\nты купишь слона"}
          onChange={(e) => {
            setQuestions(e.target.value);
            setSaved(false);
          }}
        />
      </label>
      <label>
        <span>Ответы</span>
        <textarea
          rows={3}
          value={answers}
          maxLength={600}
          placeholder={"Все так говорят, а ты купи слона.\nСлона? У меня нет денег."}
          onChange={(e) => {
            setAnswers(e.target.value);
            setSaved(false);
          }}
        />
      </label>
      <div className="formFoot">
        <button
          type="submit"
          className="primary"
          disabled={lines(questions).length === 0 || lines(answers).length === 0}
        >
          Обучить
        </button>
        {saved && <span className="savedNote">✓ Выучено. Проверьте в чате.</span>}
      </div>
    </form>
  );
}
