"use client";

import { useWebSocketStore } from "../../store/webSocketStore";

export default function MediaGallery() {
    const images = useWebSocketStore((state) => state.images);
    const glbModels = useWebSocketStore((state) => state.glbModels);

    return (
        <div>
            <h2>Received Images</h2>
            <div style={{ display: "flex", gap: "10px", flexWrap: "wrap" }}>
                {images.map((img, index) => (
                    <img
                        key={index}
                        src={img}
                        alt={`Image ${index}`}
                        style={{ maxWidth: "200px", border: "1px solid #ccc" }}
                    />
                ))}
            </div>
            <h2>Received 3D Models (GLB)</h2>
            <div>
                {glbModels.map((modelUrl, index) => (
                    <div key={index} style={{ marginBottom: "1rem" }}>
                        {/* for now, provide a link to view/download the GLB file */}
                        <a
                            href={modelUrl}
                            target="_blank"
                            rel="noopener noreferrer"
                        >
                            View GLB Model {index + 1}
                        </a>
                        {/* integrate a 3D viewer here later */}
                    </div>
                ))}
            </div>
        </div>
    );
}
