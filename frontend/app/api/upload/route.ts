import { NextResponse } from "next/server";
import { promises as fs } from "fs";
import path from "path";

export const config = {
    runtime: "nodejs", 
};

export async function POST(request: Request) {
    // Parse the incoming form data
    const formData = await request.formData();
    const images = formData.getAll("images");
    const glbFiles = formData.getAll("glbFiles");

    // Define the upload directory (ensure the directory exists)
    const uploadDir = path.join(process.cwd(), "public", "uploads");
    await fs.mkdir(uploadDir, { recursive: true });

    // Save image files
    const savedImages: string[] = [];
    for (const file of images) {
        if (file instanceof File) {
            const buffer = Buffer.from(await file.arrayBuffer());
            const filePath = path.join(uploadDir, file.name);
            await fs.writeFile(filePath, buffer);
            // Return a public URL (assuming /public is served as the root)
            savedImages.push(`/uploads/${file.name}`);
        }
    }

    // Save GLB files
    const savedGlbFiles: string[] = [];
    for (const file of glbFiles) {
        if (file instanceof File) {
            const buffer = Buffer.from(await file.arrayBuffer());
            const filePath = path.join(uploadDir, file.name);
            await fs.writeFile(filePath, buffer);
            savedGlbFiles.push(`/uploads/${file.name}`);
        }
    }

    // Return a JSON response with the file URLs
    return NextResponse.json({
        message: "Files uploaded successfully",
        images: savedImages,
        glbFiles: savedGlbFiles,
    });
}
