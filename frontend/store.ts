import { create } from "zustand";

interface MediaStore {
    images: string[];
    glbFiles: string[];
    addImages: (newImages: string[]) => void;
    addGlbFiles: (newGlbFiles: string[]) => void;
}

export const useMediaStore = create<MediaStore>((set) => ({
    images: [],
    glbFiles: [],
    addImages: (newImages) =>
        set((state) => {
            console.log("Adding images to store:", newImages);
            return { images: [...state.images, ...newImages] };
        }),
    addGlbFiles: (newGlbFiles) =>
        set((state) => {
            console.log("Adding GLB files to store:", newGlbFiles);
            return { glbFiles: [...state.glbFiles, ...newGlbFiles] };
        }),
}));
