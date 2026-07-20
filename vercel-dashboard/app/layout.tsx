import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "실험 진행 대시보드",
  description: "시계열 이상 탐지 실험 운영 현황",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="ko">
      <body>{children}</body>
    </html>
  );
}
