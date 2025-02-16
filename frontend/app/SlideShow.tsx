"use client";

import React, { useEffect, useState, useCallback, useRef, FC } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { useMediaStore, STAGE_MEDIA } from "../store";
import { StageNumber } from "../store";
import { motion } from "framer-motion";

// -----------------------------------------------------------------------------
// ThreeDModel Component
// -----------------------------------------------------------------------------
interface ThreeDModelProps {
    glbUrl: string;
}

const ThreeDModel: FC<ThreeDModelProps> = ({ glbUrl }) => {
    const mountRef = useRef<HTMLDivElement>(null);
    const controlsRef = useRef<OrbitControls | null>(null);

    const initScene = useCallback(
        (mountNode: HTMLDivElement) => {
            const width = mountNode.clientWidth;
            const height = mountNode.clientHeight;

            // Log dimensions and GLB URL for debugging
            console.log("Initializing scene:", { width, height, glbUrl });

            // Scene setup
            const scene = new THREE.Scene();
            const camera = new THREE.PerspectiveCamera(
                75,
                width / height,
                0.1,
                1000
            );
            // Move camera back so we can see the model
            camera.position.set(0, 0, 5);

            const renderer = new THREE.WebGLRenderer({
                antialias: true,
                alpha: true,
            });
            renderer.setSize(width, height);
            renderer.setClearColor(0x000000, 0);

            // Clear any existing canvas
            while (mountNode.firstChild) {
                mountNode.removeChild(mountNode.firstChild);
            }
            mountNode.appendChild(renderer.domElement);

            // Set the canvas style to ensure it receives events
            renderer.domElement.style.position = "absolute";
            renderer.domElement.style.zIndex = "20";
            renderer.domElement.style.touchAction = "none";

            // Improve OrbitControls setup
            const orbitControls = new OrbitControls(
                camera,
                renderer.domElement
            );
            orbitControls.enableDamping = true;
            orbitControls.dampingFactor = 0.05;
            orbitControls.enablePan = true;
            orbitControls.enableZoom = true;
            orbitControls.enableRotate = true;
            orbitControls.minDistance = 2;
            orbitControls.maxDistance = 10;
            controlsRef.current = orbitControls;

            // Improve lighting
            const ambientLight = new THREE.AmbientLight(0xffffff, 0.5);
            scene.add(ambientLight);

            const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
            directionalLight.position.set(5, 5, 5);
            scene.add(directionalLight);

            // Add type safety to GLTF loader
            const loader = new GLTFLoader();
            loader.load(
                glbUrl,
                (gltf: { scene: THREE.Object3D }) => {
                    console.log("GLTF loaded successfully:", gltf);
                    scene.add(gltf.scene);

                    // Center the model
                    const box = new THREE.Box3().setFromObject(gltf.scene);
                    const center = box.getCenter(new THREE.Vector3());
                    gltf.scene.position.x = -center.x;
                    gltf.scene.position.y = -center.y;
                    gltf.scene.position.z = -center.z;

                    // Scale up the model more
                    gltf.scene.scale.setScalar(3.5);

                    // Auto-adjust camera to fit model, but even closer
                    const size = box.getSize(new THREE.Vector3());
                    const maxDim = Math.max(size.x, size.y, size.z);
                    camera.position.z = maxDim * 1.0;
                },
                undefined,
                (error: unknown) =>
                    console.error(
                        "An error occurred while loading the GLB:",
                        error
                    )
            );

            // Animation loop
            const animate = () => {
                requestAnimationFrame(animate);
                orbitControls.update();
                renderer.render(scene, camera);
            };
            animate();

            // Handle resizing
            const handleResize = () => {
                const newWidth = mountNode.clientWidth;
                const newHeight = mountNode.clientHeight;
                camera.aspect = newWidth / newHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(newWidth, newHeight);
            };

            window.addEventListener("resize", handleResize);

            return () => {
                orbitControls.dispose();
                renderer.dispose();
                if (mountNode.contains(renderer.domElement)) {
                    mountNode.removeChild(renderer.domElement);
                }
                window.removeEventListener("resize", handleResize);
            };
        },
        [glbUrl]
    );

    useEffect(() => {
        if (mountRef.current) {
            return initScene(mountRef.current);
        }
    }, [initScene]);

    return (
        <div className="absolute inset-0" style={{ zIndex: 30 }}>
            <div
                ref={mountRef}
                className="w-full h-full"
                style={{
                    position: "relative",
                    touchAction: "none",
                    pointerEvents: "auto",
                }}
            />
        </div>
    );
};

// -----------------------------------------------------------------------------
// SlideShow Component
// -----------------------------------------------------------------------------
export default function SlideShow() {
    const {
        getCurrentMedia,
        nextStage,
        previousStage,
        currentStage,
        checkRequiredFiles,
        areFilesChecked,
        areFilesMissing,
    } = useMediaStore();
    const currentMedia = getCurrentMedia();
    const showOverlays = currentStage !== StageNumber.Final;
    const [isTransitioning, setIsTransitioning] = useState(false);
    const [previousMedia, setPreviousMedia] = useState(currentMedia);

    // Add opacity control for 3D models
    const [modelOpacity, setModelOpacity] = useState(1);

    // Add blur control for 3D models
    const [modelBlur, setModelBlur] = useState(0);

    // Add new state for autoplay
    const [isAutoPlaying, setIsAutoPlaying] = useState(false);
    const autoPlayTimeoutRef = useRef<NodeJS.Timeout | null>(null);

    // Check required files on mount
    useEffect(() => {
        checkRequiredFiles();
    }, [checkRequiredFiles]);

    // Handle auto-play functionality
    useEffect(() => {
        if (
            !isAutoPlaying ||
            isTransitioning ||
            !areFilesChecked ||
            areFilesMissing
        )
            return;

        const randomDuration = Math.floor(
            Math.random() * (3000 - 2000 + 1) + 2000
        ); // Random duration between 2-3 seconds

        autoPlayTimeoutRef.current = setTimeout(() => {
            if (currentStage < StageNumber.Final) {
                nextStage();
            } else {
                setIsAutoPlaying(false); // Stop when reaching the final stage
            }
        }, randomDuration);

        return () => {
            if (autoPlayTimeoutRef.current) {
                clearTimeout(autoPlayTimeoutRef.current);
            }
        };
    }, [
        currentStage,
        isTransitioning,
        nextStage,
        isAutoPlaying,
        areFilesChecked,
        areFilesMissing,
    ]);

    // Update transition effect when media changes
    useEffect(() => {
        if (currentMedia !== previousMedia) {
            setIsTransitioning(true);
            setModelOpacity(0);
            setModelBlur(10); // Start with blur

            // Sequence the transitions
            const sequence = async () => {
                // First phase - blur and fade out current
                await new Promise((resolve) => setTimeout(resolve, 250));

                // Second phase - switch content and start fade in
                setModelOpacity(1);

                // Final phase - remove blur and complete transition
                await new Promise((resolve) => setTimeout(resolve, 250));
                setModelBlur(0);
                setIsTransitioning(false);
                setPreviousMedia(currentMedia);
            };

            sequence();

            return () => {
                setModelBlur(0);
                setModelOpacity(1);
            };
        }
    }, [currentMedia, previousMedia]);

    // Handle keyboard navigation
    const handleKeyDown = useCallback(
        (e: KeyboardEvent) => {
            if (e.key === "ArrowRight") {
                nextStage();
            } else if (e.key === "ArrowLeft") {
                previousStage();
            }
        },
        [nextStage, previousStage]
    );

    useEffect(() => {
        window.addEventListener("keydown", handleKeyDown);
        return () => window.removeEventListener("keydown", handleKeyDown);
    }, [handleKeyDown]);

    // Update the return statement to show loading or missing files message
    if (!areFilesChecked) {
        return (
            <div className="relative h-screen w-screen bg-black flex items-center justify-center">
                <p className="text-white text-xl">Checking required files...</p>
            </div>
        );
    }

    if (areFilesMissing) {
        return (
            <div className="relative h-screen w-screen bg-black flex items-center justify-center">
                <div className="flex flex-col gap-5 justify-center items-center">
                    <p className="text-white font-mono text-center">
                        Waiting for files
                    </p>
                    <div className="flex flex-row gap-6 items-center">
                        <div className="relative flex w-52 h-1.5 bg-gray-100 bg-opacity-20 backdrop-blur-md border-[0.5px] border-white/40 font-mono text-sm text-white overflow-hidden">
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
                        </div>
                    </div>
                </div>
            </div>
        );
    }

    return (
        <div className="relative h-screen w-screen bg-black flex items-center justify-center">
            {currentMedia && (
                <>
                    {/* Show previous media during transition */}
                    {isTransitioning && (
                        <div
                            className="absolute inset-0 transition-all duration-500"
                            style={{
                                opacity: 0,
                                filter: `blur(${modelBlur}px)`,
                            }}
                        >
                            {previousMedia?.type === "image" ? (
                                <img
                                    src={previousMedia.path}
                                    alt=""
                                    className="w-full h-full object-cover"
                                />
                            ) : (
                                <div className="relative w-full h-full">
                                    <ThreeDModel
                                        glbUrl={previousMedia?.path || ""}
                                    />
                                </div>
                            )}
                        </div>
                    )}

                    {/* Current media */}
                    {currentMedia.type === "image" ? (
                        <div className="relative w-full h-full overflow-hidden">
                            <img
                                key={currentMedia.path + "-blur"}
                                src={currentMedia.path}
                                alt=""
                                className="absolute inset-0 w-full h-full object-cover"
                                style={{
                                    filter: "blur(15px)",
                                    transform: "scale(1.1)",
                                    transition: "filter 400ms ease-in-out",
                                }}
                            />

                            <img
                                key={currentMedia.path}
                                src={currentMedia.path}
                                alt=""
                                className="relative w-full h-full object-cover transition-all duration-[400ms]"
                                style={{
                                    maskImage:
                                        "radial-gradient(circle, white 60%, transparent 100%)",
                                    WebkitMaskImage:
                                        "radial-gradient(circle, white 60%, transparent 100%)",
                                    filter: isTransitioning
                                        ? "blur(7px)"
                                        : "blur(0px)",
                                    opacity: isTransitioning ? 0 : 1,
                                }}
                            />

                            {showOverlays && (
                                <>
                                    <div className="absolute inset-0 bg-grid-small-white/[0.2]" />
                                    <div className="absolute inset-0 bg-black/30" />
                                </>
                            )}
                        </div>
                    ) : (
                        <div className="relative w-full h-full">
                            {/* Main 3D model container - no blur here to allow interactions */}
                            <div
                                className="relative w-full h-full transition-opacity duration-500"
                                style={{ opacity: modelOpacity }}
                            >
                                <ThreeDModel
                                    key={currentMedia.path}
                                    glbUrl={currentMedia.path}
                                />
                            </div>

                            {/* Separate blur overlay that doesn't block interactions */}
                            {(isTransitioning || modelBlur > 0) && (
                                <div
                                    className="absolute inset-0 transition-all duration-500 pointer-events-none"
                                    style={{
                                        filter: `blur(${modelBlur}px)`,
                                        backgroundColor: "rgba(0,0,0,0.1)",
                                    }}
                                />
                            )}

                            {showOverlays && (
                                <div className="absolute inset-0 z-10 pointer-events-none">
                                    <div className="absolute inset-0 bg-grid-small-white/[0.2]" />
                                    <div className="absolute inset-0 bg-black/30" />
                                </div>
                            )}
                        </div>
                    )}
                </>
            )}
        </div>
    );
}
