import { NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

export async function GET() {
    const uploadDir = path.join(process.cwd(), "public", "uploads");
    try {
        const files = await fs.readdir(uploadDir);
        const imageFiles = files.filter((file) =>
            file.match(/\.(jpg|jpeg|png|gif)$/)
        );
        return NextResponse.json({
            images: imageFiles.map((file) => `/uploads/${file}`),
        });
    } catch (error) {
        console.error("Error reading upload directory:", error);
        return NextResponse.json({ images: [] });
    }
}
