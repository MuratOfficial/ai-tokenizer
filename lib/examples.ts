export type Example = { label: string; text: string; hint: string };

export const examples: Example[] = [
  {
    label: "Мама мыла раму",
    text: "мама мыла раму, мама мыла маму, мама мыла раму",
    hint: "Классика: мало разных букв, много повторов — слова быстро становятся целыми токенами.",
  },
  {
    label: "Скороговорка",
    text: "Карл у Клары украл кораллы, а Клара у Карла украла кларнет",
    hint: "Похожие слова делят общие кусочки: «Кла», «ра», «укра».",
  },
  {
    label: "Шла Саша",
    text: "Шла Саша по шоссе и сосала сушку. Шла Саша по шоссе и сосала сушку.",
    hint: "Предложение повторено дважды — в конце оно сожмётся в считаные токены.",
  },
  {
    label: "Однокоренные",
    text: "учить учитель учительница ученик учёба обучение учебник научить",
    hint: "Токенизатор сам «находит» корень «уч» — хотя ничего не знает о грамматике.",
  },
  {
    label: "banana",
    text: "banana bandana banana bandana banana",
    hint: "Знаменитый пример: «an» и «ana» повторяются внутри одного слова.",
  },
  {
    label: "English",
    text: "the cat sat on the mat. the cat ate the rat. the rat sat on the cat.",
    hint: "Частое слово «the» вместе с пробелом становится одним токеном — как в настоящем GPT.",
  },
  {
    label: "token-слова",
    text: "token tokens tokenize tokenizer tokenization tokenized",
    hint: "Общая основа «token» станет токеном, а окончания останутся отдельными кусочками.",
  },
  {
    label: "abracadabra",
    text: "abracadabra abracadabra abracadabra",
    hint: "Смотрите, как «ab» → «abra» → «abracadabra» растёт шаг за шагом.",
  },
  {
    label: "аааааааа",
    text: "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
    hint: "32 одинаковые буквы: каждая склейка удваивает токен — 1, 2, 4, 8, 16…",
  },
  {
    label: "Ха-ха-ха",
    text: "ха-ха-ха-ха-ха-ха-ха-ха",
    hint: "Чистый повтор сжимается лучше всего.",
  },
  {
    label: "Панграмма",
    text: "Съешь же ещё этих мягких французских булок, да выпей чаю",
    hint: "Почти нет повторов — склеивать нечего, текст почти не сжимается.",
  },
  {
    label: "Код",
    text: "function add(a, b) { return a + b; }\nfunction sub(a, b) { return a - b; }\nfunction mul(a, b) { return a * b; }",
    hint: "В коде много шаблонов: «function », «(a, b) { return a » станут длинными токенами.",
  },
  {
    label: "JSON",
    text: '{"name": "Alice", "age": 30}\n{"name": "Bob", "age": 25}\n{"name": "Carol", "age": 41}',
    hint: "Ключи повторяются — структура сжимается, а данные остаются по буквам.",
  },
  {
    label: "Ссылки",
    text: "https://example.com/home https://example.com/about https://example.com/blog",
    hint: "Общий префикс адреса превращается в один токен.",
  },
  {
    label: "Числа",
    text: "2024 2025 2026 1024 2048 4096 2024 2025",
    hint: "Числа режутся на куски цифр — поэтому ИИ бывает трудно считать.",
  },
  {
    label: "Эмодзи",
    text: "Привет 👋 мир 🌍! Привет 👋 ИИ 🤖! Привет 👋 всем 🎉!",
    hint: "Эмодзи — такие же символы, и они тоже склеиваются с соседями.",
  },
  {
    label: "Два языка",
    text: "hello мир, привет world, hello world, привет мир",
    hint: "Токенизатору всё равно, какой язык: он видит только частоту пар.",
  },
  {
    label: "ДНК",
    text: "ATGCATGCATTAGCATGCATGCATTAGC",
    hint: "Всего 4 буквы — тот же алгоритм применяют и в биоинформатике.",
  },
];
