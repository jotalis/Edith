import SlideShow from "./SlideShow";
import Overlay from "./Overlay";

export default function Home() {
    return (
        <main className="relative min-h-screen bg-gray-200">
            <SlideShow />
            <Overlay />
        </main>
    );
}
