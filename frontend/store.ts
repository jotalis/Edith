import { create } from "zustand";

export type Stage =
    | "Original"
    | "Bounding Box"
    | "Segmentation"
    | "Metric Depth"
    | "MoGe"
    | "Final";

export enum StageNumber {
    Original = 0,
    "Bounding Box" = 1,
    Segmentation = 2,
    "Metric Depth" = 3,
    MoGe = 4,
    Final = 5,
}

// Define the media files for each stage
const STAGE_MEDIA: Record<
    StageNumber,
    { path: string; type: "image" | "glb" }
> = {
    [StageNumber.Original]: { path: "/uploads/original.jpg", type: "image" },
    [StageNumber["Bounding Box"]]: {
        path: "/uploads/bounding_box.jpg",
        type: "image",
    },
    [StageNumber.Segmentation]: {
        path: "/uploads/segmentation.jpg",
        type: "image",
    },
    [StageNumber["Metric Depth"]]: {
        path: "/uploads/depth.jpg",
        type: "image",
    },
    [StageNumber.MoGe]: { path: "/uploads/mesh.glb", type: "glb" },
    [StageNumber.Final]: { path: "/uploads/segmented_mesh.glb", type: "glb" },
};

// Helper functions for stage conversion
const stageToNumber = (stage: Stage): number => StageNumber[stage];
const numberToStage = (num: number): Stage => {
    const stageKey = StageNumber[num] as keyof typeof StageNumber;
    return stageKey as Stage;
};

interface MediaStore {
    currentStage: StageNumber;
    getCurrentMedia: () => { path: string; type: "image" | "glb" };
    nextStage: () => void;
    previousStage: () => void;
    setStageByNumber: (stageNum: number) => void;
    getStageNumber: () => number;
}

export const useMediaStore = create<MediaStore>((set, get) => ({
    currentStage: StageNumber.Original,

    getCurrentMedia: () => {
        return STAGE_MEDIA[get().currentStage];
    },

    nextStage: () =>
        set((state) => ({
            currentStage:
                (state.currentStage + 1) % Object.keys(StageNumber).length,
        })),

    previousStage: () =>
        set((state) => ({
            currentStage:
                (state.currentStage - 1 + Object.keys(StageNumber).length) %
                Object.keys(StageNumber).length,
        })),

    setStageByNumber: (stageNum: number) =>
        set({
            currentStage: stageNum,
        }),

    getStageNumber: () => get().currentStage,
}));

export { stageToNumber, numberToStage, STAGE_MEDIA };
