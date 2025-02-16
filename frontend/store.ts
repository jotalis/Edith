import { create } from "zustand";

interface MediaStore {
    images: string[];
    glbFiles: string[];
    depth: number;
    heading: number;
    addImages: (newImages: string[]) => void;
    addGlbFiles: (newGlbFiles: string[]) => void;
    clearImages: () => void;
    clearGlbs: () => void;
    setImages: (images: string[]) => void;
    setGlbs: (glbFiles: string[]) => void;
    setDepth: (depth: number) => void;
    setHeading: (heading: number) => void;
}

export const useMediaStore = create<MediaStore>((set) => ({
    images: [],
    glbFiles: [],
    depth: 0,
    heading: 0,
    addImages: (newImages) =>
        set((state) => ({ images: [...state.images, ...newImages] })),
    addGlbFiles: (newGlbFiles) =>
        set((state) => ({ glbFiles: [...state.glbFiles, ...newGlbFiles] })),
    clearImages: () => set({ images: [] }),
    clearGlbs: () => set({ glbFiles: [] }),
    setImages: (images) => set({ images }),
    setGlbs: (glbFiles) => set({ glbFiles }),
    setDepth: (depth) => set({ depth }),
    setHeading: (heading) => set({ heading }),
}));
