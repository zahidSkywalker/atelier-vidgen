import type { Metadata, Viewport } from "next";
import { Manrope } from "next/font/google";
import "./globals.css";
import { Toaster } from "@/components/ui/sonner";

const manrope = Manrope({
  variable: "--font-manrope",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Atelier VidGen — Cinematic Video Generation",
  description:
    "Text-to-video and image-to-video studio powered by Kandinsky 6.0 Lite on the Kaggle B lane. Dark, responsive, built for any device.",
  icons: {
    icon: "https://z-cdn.chatglm.cn/z-ai/static/logo.svg",
  },
  applicationName: "Atelier VidGen",
  openGraph: {
    title: "Atelier VidGen",
    description: "Cinematic 5-second clips with synced audio — T2V & I2V.",
    siteName: "Atelier VidGen",
    type: "website",
  },
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  themeColor: "#09090b",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" suppressHydrationWarning className="dark">
      <body
        className={`${manrope.variable} font-sans antialiased bg-zinc-950 text-zinc-100 min-h-screen flex flex-col`}
      >
        {children}
        <Toaster position="top-center" richColors closeButton />
      </body>
    </html>
  );
}
