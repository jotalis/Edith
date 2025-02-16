"use client";

import React, { useEffect, useState, useCallback, useRef, FC } from "react";
import * as THREE from "three";
import { GLTFLoader } from "three/examples/jsm/loaders/GLTFLoader";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls";
import { useMediaStore } from "../store";

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
            camera.position.z = 0;

            const renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(width, height);
            mountNode.appendChild(renderer.domElement);

            // Add orbit controls for interaction
            const orbitControls = new OrbitControls(
                camera,
                renderer.domElement
            );
            orbitControls.enableDamping = true;
            orbitControls.dampingFactor = 0.05;
            orbitControls.enablePan = true;
            orbitControls.enableZoom = true;
            orbitControls.enableRotate = true;
            // orbitControls.autoRotateSpeed = 2.0;
            // Set to false if you do not want user control
            orbitControls.enabled = true;
            controlsRef.current = orbitControls;

            // Lighting setup
            const ambientLight = new THREE.AmbientLight(0xffffff, 1);
            scene.add(ambientLight);

            const directionalLight = new THREE.DirectionalLight(0xffffff, 1);
            directionalLight.position.set(5, 5, 5).normalize();
            scene.add(directionalLight);

            // Load GLB Model using the provided URL
            const loader = new GLTFLoader();
            loader.load(
                glbUrl,
                (gltf) => {
                    console.log("GLTF loaded successfully:", gltf);
                    scene.add(gltf.scene);
                    gltf.scene.position.set(0, 0, 0);
                    gltf.scene.scale.set(1, 1, 1);
                },
                undefined,
                (error) =>
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
        <div className="h-full w-full overflow-hidden rounded-2xl bg-black">
            <div style={{ width: "100%", height: "100%" }}>
                <div
                    ref={mountRef}
                    style={{ width: "100%", height: "100%" }}
                ></div>
            </div>
        </div>
    );
};

// -----------------------------------------------------------------------------
// SlideShow Component
// -----------------------------------------------------------------------------
export default function SlideShow() {
    // Get images and GLB files from the zustand store
    const images = useMediaStore((state) => state.images);
    const glbs = useMediaStore((state) => state.glbFiles);
    const setImages = useMediaStore((state) => state.setImages);
    const setGlbs = useMediaStore((state) => state.setGlbs);
    const clearImages = useMediaStore((state) => state.clearImages);
    const clearGlbs = useMediaStore((state) => state.clearGlbs);

    // currentIndex is used for both images and GLBs combined
    const [currentIndex, setCurrentIndex] = useState(0);

    // Fetch uploaded data from a single endpoint every 5 seconds
    useEffect(() => {
        async function fetchData() {
            try {
                const res = await fetch("/api/uploaded-data");
                if (res.ok) {
                    const data = await res.json();
                    console.log("Fetched data:", data);
                    setImages(data.images);
                    setGlbs(data.glbs);
                }
            } catch (error) {
                console.error("Error fetching uploaded data:", error);
            }
        }
        fetchData();
        const interval = setInterval(fetchData, 5000);
        return () => clearInterval(interval);
    }, [setImages, setGlbs]);

    // Total slides is the sum of images and GLB files
    const totalSlides = images.length + glbs.length;

    // Ensure currentIndex stays within bounds
    useEffect(() => {
        if (totalSlides === 0) {
            setCurrentIndex(0);
        } else if (currentIndex >= totalSlides) {
            setCurrentIndex(totalSlides - 1);
        }
    }, [totalSlides, currentIndex]);

    // Handle arrow key navigation for combined slides
    const handleKeyDown = useCallback(
        (e: KeyboardEvent) => {
            if (totalSlides === 0) return;
            if (e.key === "ArrowRight") {
                setCurrentIndex((prev) => (prev + 1) % totalSlides);
            } else if (e.key === "ArrowLeft") {
                setCurrentIndex(
                    (prev) => (prev - 1 + totalSlides) % totalSlides
                );
            }
        },
        [totalSlides]
    );

    useEffect(() => {
        window.addEventListener("keydown", handleKeyDown);
        return () => window.removeEventListener("keydown", handleKeyDown);
    }, [handleKeyDown]);

    // Function to clear both images and GLBs from the server
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

    return (
        <div className="relative h-screen w-screen bg-black flex items-center justify-center">
            {totalSlides > 0 && (
                <>
                    {currentIndex < images.length ? (
                        <div className="relative w-full h-full overflow-hidden">
                            {/* blurred background image */}
                            <img
                                key={images[currentIndex] + "-blur"}
                                src={images[currentIndex]}
                                alt=""
                                className="absolute inset-0 w-full h-full object-cover"
                                style={{
                                    filter: "blur(50px)",
                                    transform: "scale(1.1)",
                                }}
                            />
                            {/* sharp image with radial mask for faded edges */}
                            <img
                                key={images[currentIndex]}
                                src={images[currentIndex]}
                                alt=""
                                className="relative w-full h-full object-cover"
                                style={{
                                    maskImage:
                                        "radial-gradient(circle, white 60%, transparent 100%)",
                                    WebkitMaskImage:
                                        "radial-gradient(circle, white 60%, transparent 100%)",
                                }}
                            />
                            {/* bg grid overlay */}
                            <div className="absolute inset-0 bg-grid-small-white/[0.2]" />
                        </div>
                    ) : (
                        <ThreeDModel
                            key={glbs[currentIndex - images.length]} // force remount when URL changes
                            glbUrl={glbs[currentIndex - images.length]}
                        />
                    )}
                </>
            )}

            {/* Move the black overlay here */}
            <div className="absolute inset-0 bg-black/30" />

            {/* Small Circular Clear Button (Bottom Right) */}
            <button
                onClick={handleClear}
                className="absolute bottom-4 right-4 w-10 h-10 bg-black bg-opacity-50 rounded-full flex items-center justify-center hover:bg-opacity-70 focus:outline-none"
            >
                <svg
                    xmlns="http://www.w3.org/2000/svg"
                    className="h-6 w-6 text-white"
                    fill="none"
                    viewBox="0 0 24 24"
                    stroke="currentColor"
                >
                    <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth="2"
                        d="M6 18L18 6M6 6l12 12"
                    />
                </svg>
            </button>
        </div>
    );
}
