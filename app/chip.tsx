import type { CSSProperties } from "react";

/** Делаем невидимые символы видимыми */
export function show(text: string) {
  return text.replace(/ /g, "·").replace(/\n/g, "↵").replace(/\t/g, "→");
}

export function Chip({
  entry,
  mark,
  showId,
  onClick,
}: {
  entry: { id: number; text: string };
  mark?: "pair" | "new" | "selected";
  showId?: boolean;
  onClick?: () => void;
}) {
  return (
    <button
      type="button"
      className={`chip ${mark ?? ""}`}
      style={{ "--h": (entry.id * 47) % 360 } as CSSProperties}
      onClick={onClick}
      title={`Токен №${entry.id}`}
    >
      <span className="chipText">{show(entry.text)}</span>
      {showId && <span className="chipId">{entry.id}</span>}
    </button>
  );
}
