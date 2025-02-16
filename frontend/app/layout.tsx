import { JetBrains_Mono } from "next/font/google";
import "./globals.css";

const jetBrainsMono = JetBrains_Mono({
    variable: "--font-jet-brains-mono",
    subsets: ["latin"],
});

export default function RootLayout({
    children,
}: Readonly<{
    children: React.ReactNode;
}>) {
    return (
        <html lang="en">
            <body className={`${jetBrainsMono.variable} font-mono`}>
                {children}
            </body>
        </html>
    );
}
