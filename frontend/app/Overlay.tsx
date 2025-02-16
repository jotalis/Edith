"use client";

import React from "react";
import {
    useMediaStore,
    StageNumber,
    type Stage,
    numberToStage,
} from "../store";
import { motion } from "framer-motion";

const Overlay = () => {
    const stage = numberToStage(useMediaStore((state) => state.currentStage));
    const stageNum = useMediaStore((state) => state.getStageNumber());
    const areFilesMissing = useMediaStore((state) => state.areFilesMissing);

    const handleClear = async () => {
        try {
            const res = await fetch("/api/clear-uploads", { method: "DELETE" });
            if (!res.ok) {
                console.error("Failed to clear uploads from the server");
            }
        } catch (error) {
            console.error("Error clearing uploads:", error);
        }
    };

    const GlassLabel = ({
        children,
        className,
        ...props
    }: {
        children: React.ReactNode;
        className?: string;
    } & React.HTMLAttributes<HTMLDivElement>) => {
        return (
            <div
                className={`${className} px-2 py-1 bg-gray-100 bg-opacity-20 backdrop-blur-md border-[0.5px] border-white/40 font-mono text-sm text-white max-h-fit max-w-fit`}
                {...props}
            >
                {children}
            </div>
        );
    };

    const ProgressBar = ({
        completed,
        stageName,
    }: {
        completed: boolean;
        stageName: string;
    }) => {
        return (
            <div className="flex flex-row gap-6 items-center">
                <div className="relative flex w-52 h-1.5 bg-gray-100 bg-opacity-20 backdrop-blur-md border-[0.5px] border-white/40 font-mono text-sm text-white overflow-hidden">
                    {completed ? (
                        <div className="h-full bg-white w-full" />
                    ) : (
                        <motion.div
                            className="absolute h-full bg-white/50"
                            initial={{ width: 32, left: -32 }}
                            animate={{
                                left: ["-32px", "208px"],
                            }}
                            transition={{
                                duration: 1.5,
                                repeat: Infinity,
                                ease: "linear",
                                repeatType: "loop",
                            }}
                        />
                    )}
                </div>
                <div className="text-white text-xs">{stageName}</div>
            </div>
        );
    };

    const StageDot = ({ filled }: { filled: boolean }) => {
        return (
            <div className="flex size-3.5 rounded-full border border-white items-center justify-center">
                {filled && <div className="size-1.5 rounded-full bg-white" />}
            </div>
        );
    };

    // takes stage number between 1-5 depending on stage
    const StageBar = ({ stageNum }: { stageNum: number }) => {
        return (
            <div className="flex flex-row items-center gap-4">
                <StageDot filled={stageNum >= 1} />
                <StageDot filled={stageNum >= 2} />
                <StageDot filled={stageNum >= 3} />
                <StageDot filled={stageNum >= 4} />
                <StageDot filled={stageNum >= 5} />
                <div className="text-xs text-white">{stageNum}/5 steps</div>
            </div>
        );
    };

    // Update this function to include current stage and all previous stages
    const getCurrentAndPreviousStages = () => {
        const allStages: Stage[] = Object.keys(StageNumber)
            .filter((key) => isNaN(Number(key)))
            .filter(
                (stage) => StageNumber[stage as keyof typeof StageNumber] > 0
            ) as Stage[];
        // Include one more stage than the current stageNum to show the "in progress" stage
        return allStages.slice(0, stageNum + 1);
    };

    // Update this helper to determine if stage is completed
    const isStageCompleted = (stageName: string) => {
        const stageIndex = StageNumber[stageName as keyof typeof StageNumber];
        // If it's a previous stage, it's completed
        // If it's the current stage, it's not completed
        return stageIndex <= stageNum;
    };

    return (
        <div className="absolute inset-0 z-10 p-5 text-sm font-mono">
            <div className="flex flex-col h-full w-full justify-between">
                <div className="flex flex-row w-full h-fit justify-between">
                    <div className="flex gap-3">
                        <div className="px-2 py-1 bg-gray-100 uppercase text-black">
                            Edith
                        </div>
                        <GlassLabel className="flex gap-3 items-center">
                            <div className="relative flex items-center justify-center">
                                <div className="z-10 h-2.5 w-2.5 rounded-full bg-emerald-300" />
                            </div>
                            <div>Hardware connected</div>
                        </GlassLabel>
                    </div>
                    {/* TODO: replace hardcoded values */}
                    <div className="flex gap-3">
                        <GlassLabel>CPU 51%</GlassLabel>
                        <GlassLabel>GPU 83%</GlassLabel>
                        <GlassLabel>RAM 75%</GlassLabel>
                    </div>
                </div>
                <div className="flex flex-row w-full h-fit justify-between items-end">
                    {!areFilesMissing ? (
                        <div className="flex flex-col gap-5">
                            <div className="flex flex-col gap-2">
                                {stageNum >= 0 &&
                                    getCurrentAndPreviousStages().map(
                                        (stageName) => (
                                            <ProgressBar
                                                key={stageName}
                                                completed={isStageCompleted(
                                                    stageName
                                                )}
                                                stageName={stageName}
                                            />
                                        )
                                    )}
                            </div>
                            <StageBar stageNum={stageNum} />
                        </div>
                    ) : null}
                    <GlassLabel
                        onClick={handleClear}
                        className="cursor-pointer"
                    >
                        Clear
                    </GlassLabel>
                </div>
            </div>
        </div>
    );
};

export default Overlay;
