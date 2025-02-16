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

// Add this near the top of the file
const REQUIRED_FILES = [
    "/uploads/original.jpg",
    "/uploads/bounding_box.jpg",
    "/uploads/segmentation.jpg",
    "/uploads/depth.jpg",
    "/uploads/mesh.glb",
    "/uploads/segmented_mesh.glb",
];

interface MediaStore {
    currentStage: StageNumber;
    getCurrentMedia: () => { path: string; type: "image" | "glb" };
    nextStage: () => void;
    previousStage: () => void;
    setStageByNumber: (stageNum: number) => void;
    getStageNumber: () => number;
    checkRequiredFiles: () => Promise<boolean>;
    areFilesChecked: boolean;
    areFilesMissing: boolean;
}

export const useMediaStore = create<MediaStore>((set, get) => ({
    currentStage: StageNumber.Original,

    getCurrentMedia: () => {
        return STAGE_MEDIA[get().currentStage];
    },

    nextStage: () =>
        set((state) => {
            // Only increment if not at the last stage
            if (state.currentStage < Object.keys(STAGE_MEDIA).length - 1) {
                return { currentStage: state.currentStage + 1 };
            }
            return state; // Return unchanged state if at last stage
        }),

    previousStage: () =>
        set((state) => {
            // Only decrement if not at the first stage
            if (state.currentStage > 0) {
                return { currentStage: state.currentStage - 1 };
            }
            return state; // Return unchanged state if at first stage
        }),

    setStageByNumber: (stageNum: number) =>
        set({
            currentStage: stageNum,
        }),

    getStageNumber: () => get().currentStage,

    areFilesChecked: false,
    areFilesMissing: false,

    checkRequiredFiles: async () => {
        try {
            const results = await Promise.all(
                REQUIRED_FILES.map(async (file) => {
                    const response = await fetch(file, { method: "HEAD" });
                    return response.ok;
                })
            );

            const allFilesExist = results.every((exists) => exists);
            set({ areFilesChecked: true, areFilesMissing: !allFilesExist });
            return allFilesExist;
        } catch (error) {
            set({ areFilesChecked: true, areFilesMissing: true });
            return false;
        }
    },
}));

export { stageToNumber, numberToStage, STAGE_MEDIA };
