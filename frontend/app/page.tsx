"use client";

import dynamic from "next/dynamic";

const WebSocketPage = dynamic(() => import("./websocket/page"), { ssr: false });
const MediaGallery = dynamic(() => import("./components/MediaGallery"), {
    ssr: false,
});

export default function HomePage() {
    return (
        <div>
            <WebSocketPage />
            <MediaGallery />
        </div>
    );
}
