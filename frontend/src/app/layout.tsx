import type { Metadata } from "next";
import { Plus_Jakarta_Sans } from "next/font/google";
import "./globals.css";

const jakartaSans = Plus_Jakarta_Sans({
  subsets: ["latin"],
  weight: ["300", "400", "500", "600", "700", "800"],
  variable: "--font-jakarta",
  display: "swap",
});

export const metadata: Metadata = {
  title: "Research Agent — AI-powered document analysis",
  description:
    "Upload PDF, DOCX, or TXT files and get instant AI-powered insights, comparisons, and diagrams.",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${jakartaSans.className} bg-[#0f0f13] text-white antialiased`}>
        {children}
      </body>
    </html>
  );
}
