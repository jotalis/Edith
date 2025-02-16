"use client";

import React, { useState } from "react";
import { useMediaStore } from "../store";

const Overlay = () => {
    const [stageName, setStageName] = useState("Bounding box");
    const [stageNum, setStageNum] = useState(1);

    // Retrieve clear functions from the store
    const clearImages = useMediaStore((state) => state.clearImages);
    const clearGlbs = useMediaStore((state) => state.clearGlbs);

    // New clear handler moved from SlideShow.tsx
    const handleClear = async () => {
        try {
            const res = await fetch("/api/clear-uploads", { method: "DELETE" });
            if (res.ok) {
                clearImages();
                clearGlbs();
            } else {
                console.error("Failed to clear uploads from the server");
            }
        } catch (error) {
            console.error("Error clearing uploads:", error);
        }
    };

    // Updated GlassLabel to forward extra props such as onClick
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

    // takes progress between 0-1
    const ProgressBar = ({
        progress,
        stageName,
    }: {
        progress: number;
        stageName: string;
    }) => {
        return (
            <div className="flex flex-row gap-6 items-center">
                <div className="flex w-52 h-1.5 bg-gray-100 bg-opacity-20 backdrop-blur-md border-[0.5px] border-white/40 font-mono text-sm text-white">
                    <div
                        className="h-full bg-white"
                        style={{ width: `${progress * 100}%` }}
                    />
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
            <div className="flex flex-row items-center gap-5">
                <StageDot filled={stageNum >= 1} />
                <StageDot filled={stageNum >= 2} />
                <StageDot filled={stageNum >= 3} />
                <StageDot filled={stageNum >= 4} />
                <StageDot filled={stageNum >= 5} />
                <div className="text-xs text-white">{stageNum}/5 steps</div>
            </div>
        );
    };

    return (
        <div className="absolute inset-0 z-10 p-5 text-sm font-mono">
            <div className="flex flex-col h-full w-full justify-between">
                <div className="flex flex-row w-full h-fit justify-between">
                    <div className="flex gap-3">
                        <div className="px-2 py-1 bg-gray-100 uppercase">
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
                {/* TODO: implement dynamic progress bars based on stage */}
                <div className="flex flex-row w-full h-fit justify-between items-end">
                    <div className="flex flex-col gap-5">
                        <ProgressBar progress={0.5} stageName={stageName} />
                        <StageBar stageNum={stageNum} />
                    </div>
                    {/* Updated the "Clear" button to attach the handleClear functionality */}
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
