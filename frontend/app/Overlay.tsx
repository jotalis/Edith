import React from "react";

const Overlay = () => {
    const GlassLabel = ({
        children,
        className,
    }: {
        children: React.ReactNode;
        className?: string;
    }) => {
        return (
            <div
                className={`${className} px-2 py-1 bg-gray-100 bg-opacity-20 backdrop-blur-md border-[0.5px] border-white/40 font-mono text-sm text-white`}
            >
                {children}
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
                <div className="flex flex-row w-full h-fit justify-between">
                    <GlassLabel>Test</GlassLabel>
                </div>
            </div>
        </div>
    );
};

export default Overlay;
