import type { Metadata } from "next";
import Train from "./train";
import "./train.css";

export const metadata: Metadata = {
  title: "Обучение мини-модели",
  description: "Фразы, на которые модель не нашла ответа, и ручное обучение.",
};

export default function TrainPage() {
  return <Train />;
}
