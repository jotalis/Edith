import { NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

export async function DELETE() {
    const uploadDir = path.join(process.cwd(), "public", "uploads");
    try {
        const files = await fs.readdir(uploadDir);
        // Delete every file in the uploads directory.
        await Promise.all(
            files.map((file) => fs.unlink(path.join(uploadDir, file)))
        );
        return NextResponse.json({ message: "Uploads cleared." });
    } catch (error) {
        console.error("Error clearing uploads:", error);
        return NextResponse.json(
            { message: "Failed to clear uploads." },
            { status: 500 }
        );
    }
}
