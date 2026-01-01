export const Background = () => {
    return (
        <div className="fixed inset-0 -z-30 bg-[#02040a] overflow-hidden">
            {/* Mesh Gradient 1 */}
            <div className="absolute top-[-20%] left-[-10%] w-[50%] h-[50%] bg-teal-900/10 rounded-full blur-[120px] mix-blend-screen animate-pulse-slow" />
            {/* Mesh Gradient 2 */}
            <div className="absolute bottom-[-10%] right-[-10%] w-[40%] h-[40%] bg-indigo-900/10 rounded-full blur-[100px] mix-blend-screen" />

            {/* Grid Pattern Overlay */}
            <div className="absolute inset-0 bg-[linear-gradient(rgba(255,255,255,0.01)_1px,transparent_1px),linear-gradient(90deg,rgba(255,255,255,0.01)_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_60%_50%_at_50%_0%,#000_70%,transparent_100%)]" />

            {/* Noise Texture */}
            <div className="absolute inset-0 opacity-[0.03] bg-repeat [background-image:url('data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSI0IiBoZWlnaHQ9IjQiPjxyZWN0IHdpZHRoPSI0IiBoZWlnaHQ9IjQiIGZpbGw9IiNmZmYiLz48cmVjdCB3aWR0aD0iMSIgaGVpZ2h0PSIxIiBmaWxsPSIjMDAwIi8+PC9zdmc+')] mix-blend-overlay pointer-events-none" />
        </div>
    );
};
