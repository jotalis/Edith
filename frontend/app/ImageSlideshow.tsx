"use client";

import { useEffect } from "react";
import { useMediaStore } from "../store";

export default function ImageSlideshow() {
    const images = useMediaStore((state) => state.images);
    const addImages = useMediaStore((state) => state.addImages);

    // fetch uploaded images periodically
    useEffect(() => {
        async function fetchImages() {
            const res = await fetch("/api/uploaded-images");
            if (res.ok) {
                const data = await res.json();
                addImages(data.images);
            }
        }

        fetchImages(); 
        const interval = setInterval(fetchImages, 5000);

        return () => clearInterval(interval); 
    }, [addImages]);

    if (images.length === 0) {
        return <div>No images uploaded yet.</div>;
    }

    return (
        <div>
            {images.map((src, index) => (
                <img
                    key={index}
                    src={src}
                    alt={`Slide ${index + 1}`}
                    style={{ width: "400px" }}
                />
            ))}
        </div>
    );
}
