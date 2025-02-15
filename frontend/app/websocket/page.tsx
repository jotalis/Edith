"use client";

import { useEffect } from "react";
import { useWebSocketStore } from "../../store/webSocketStore";

export default function WebSocketPage() {
    const addImage = useWebSocketStore((state) => state.addImage);
    const addGlb = useWebSocketStore((state) => state.addGlb);

    useEffect(() => {
        const socket = new WebSocket("ws://172.24.75.90:8000");
        socket.binaryType = "arraybuffer";

        socket.onopen = () => {
            console.log("WebSocket connected");
        };

        socket.onmessage = (event: MessageEvent<any>) => {
            // Check if the message is text or binary
            if (typeof event.data === "string") {
                // You might receive JSON with a type field, or a raw base64 string.
                try {
                    let data;
                    try {
                        data = JSON.parse(event.data);
                    } catch {}
                    if (data && data.type === "image" && data.image) {
                        // If the message is JSON with a type field, use the provided image data.
                        addImage(data.image);
                    } else {
                        // Otherwise, treat the whole string as the base64 image.
                        addImage(event.data);
                    }
                } catch (error) {
                    console.error("Error processing image message:", error);
                }
            } else if (event.data instanceof ArrayBuffer) {
                // Binary message: assume it's a GLB file.
                try {
                    const blob = new Blob([event.data], {
                        type: "model/gltf-binary",
                    });
                    addGlb(blob);
                } catch (error) {
                    console.error("Error processing binary message:", error);
                }
            } else {
                console.warn("Received an unknown message type:", event.data);
            }
        };

        socket.onerror = (error) => {
            console.error("WebSocket error:", error);
        };

        socket.onclose = () => {
            console.log("WebSocket closed");
        };

        return () => {
            socket.close();
        };
    }, [addImage, addGlb]);

    return (
        // <div>
        //     <h1>WebSocket Connection</h1>
        //     <p>Listening for new images and 3D models...</p>
        // </div>
        <></>
    );
}
