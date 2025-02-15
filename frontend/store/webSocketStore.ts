import { create } from "zustand";

interface WebSocketData {
    images: string[]; // base64 image strings
    glbModels: string[]; // Blob URLs for GLB files
    addImage: (img: string) => void;
    addGlb: (glbBlob: Blob) => void;
}

export const useWebSocketStore = create<WebSocketData>((set) => ({
    images: [],
    glbModels: [],
    addImage: (img: string) =>
        set((state) => ({ images: [...state.images, img] })),
    addGlb: (glbBlob: Blob) => {
        // create a URL from the binary blob so we can reference it later.
        const blobUrl = URL.createObjectURL(glbBlob);
        return set((state) => ({ glbModels: [...state.glbModels, blobUrl] }));
    },
}));
