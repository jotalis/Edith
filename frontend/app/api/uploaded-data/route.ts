// app/api/uploaded-data/route.ts
import { NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

export async function GET() {
    const uploadDir = path.join(process.cwd(), "public", "uploads");
    try {
        const files = await fs.readdir(uploadDir);
        const images = files
            .filter((file) => file.match(/\.(jpg|jpeg|png|gif)$/i))
            .map((file) => `/uploads/${file}`);
        const glbs = files
            .filter((file) => file.match(/\.glb$/i))
            .map((file) => `/uploads/${file}`);

        return NextResponse.json({ images, glbs });
    } catch (error) {
        console.error("Error reading upload directory:", error);
        return NextResponse.json({ images: [], glbs: [] });
    }
}
