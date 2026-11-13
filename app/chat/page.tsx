import type { Metadata } from "next";
import Chat from "./chat";
import "./chat.css";

export const metadata: Metadata = {
  title: "Чат с мини-моделью — как ИИ подбирает ответ",
  description: "Напишите сообщение и смотрите, как модель токен за токеном собирает ответ.",
};

export default function ChatPage() {
  return <Chat />;
}
