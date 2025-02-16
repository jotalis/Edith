"use client";

import React, { useEffect, useState, useCallback, useRef, FC } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { useMediaStore, STAGE_MEDIA } from "../store";
import { StageNumber } from "../store";

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
    const { getCurrentMedia, nextStage, previousStage, currentStage } =
        useMediaStore();
    const currentMedia = getCurrentMedia();
    const showOverlays = currentStage !== StageNumber.Final;
    // Add state for transition
    const [isTransitioning, setIsTransitioning] = useState(false);
    const [previousMedia, setPreviousMedia] = useState(currentMedia);

    // Update transition effect when media changes
    useEffect(() => {
        if (currentMedia !== previousMedia) {
            setIsTransitioning(true);
            const timer = setTimeout(() => {
                setIsTransitioning(false);
                setPreviousMedia(currentMedia);
            }, 500); // Duration matches CSS transition
            return () => clearTimeout(timer);
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

    return (
        <div className="relative h-screen w-screen bg-black flex items-center justify-center">
            {currentMedia && (
                <>
                    {currentMedia.type === "image" ? (
                        <div className="relative w-full h-full overflow-hidden">
                            {/* Previous image */}
                            {isTransitioning &&
                                previousMedia?.type === "image" && (
                                    <img
                                        key={previousMedia.path + "-previous"}
                                        src={previousMedia.path}
                                        alt=""
                                        className="absolute inset-0 w-full h-full object-cover transition-all duration-500"
                                        style={{
                                            filter: "blur(0px)",
                                            opacity: 0,
                                        }}
                                    />
                                )}

                            {/* Current image with transition */}
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
                            <ThreeDModel
                                key={currentMedia.path}
                                glbUrl={currentMedia.path}
                            />
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
