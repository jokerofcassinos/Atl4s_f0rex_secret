"use client";

import { motion } from "framer-motion";
import { useState } from "react";

export const HomeView = () => {
    const [hoverTooltip, setHoverTooltip] = useState(false);

    return (
        <div className="relative w-full h-full flex flex-col justify-center items-center font-sans">
            {/* Main Typography */}
            <motion.div
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                transition={{ delay: 0.2, duration: 0.8 }}
                className="text-center z-10"
            >
                <span className="block text-white/50 text-sm tracking-[0.2em] mb-2 lowercase">seja</span>
                <h1 className="text-6xl md:text-8xl font-black text-white mb-2 tracking-tighter uppercase drop-shadow-[0_0_15px_rgba(255,255,255,0.3)]">
                    BEM-VINDO
                </h1>
                <span className="block text-white/50 text-sm tracking-[0.3em] mb-6 lowercase">de volta</span>
                <motion.div
                    className="relative inline-block"
                    animate={{ skewX: [0, -5, 0, 5, 0], x: [0, -2, 2, 0] }} // Subtle glitch trigger
                    transition={{ repeat: Infinity, repeatDelay: 5, duration: 0.2 }}
                >
                    <h2 className="text-7xl md:text-9xl font-black text-white tracking-widest uppercase font-mono bg-clip-text text-transparent bg-gradient-to-b from-white to-white/70">
                        UNKNOWN
                    </h2>
                </motion.div>
            </motion.div>

            {/* Interactive Tooltip Point */}
            <div className="absolute top-2/3 left-2/3 transform -translate-x-1/2 -translate-y-1/2 group">
                <div
                    className="w-4 h-4 bg-white rounded-full relative cursor-pointer"
                    onMouseEnter={() => setHoverTooltip(true)}
                    onMouseLeave={() => setHoverTooltip(false)}
                >
                    <div className="absolute inset-0 bg-white rounded-full animate-ping opacity-75"></div>
                    {/* Connecting Line */}
                    <motion.div
                        className="absolute top-1/2 left-full h-[1px] bg-white/40 origin-left"
                        initial={{ width: 0 }}
                        animate={{ width: hoverTooltip ? 100 : 0 }}
                        transition={{ duration: 0.3 }}
                    />

                    {/* Tooltip Box */}
                    <motion.div
                        className="absolute top-1/2 mt-8 left-[120px] -translate-y-1/2 w-64 p-4 border border-white/20 bg-black/60 backdrop-blur-md rounded-xl text-xs text-white/80 font-mono pointer-events-none"
                        initial={{ opacity: 0, x: -10 }}
                        animate={{ opacity: hoverTooltip ? 1 : 0, x: hoverTooltip ? 0 : -10 }}
                        transition={{ duration: 0.3, delay: 0.1 }}
                    >
                        <div className="font-bold text-cyan-400 mb-1">complexo 4tlas</div>
                        Alta tecnologia de trade desenvolvida para ações automáticas e inteligentes...
                    </motion.div>
                </div>
            </div>

            {/* Footer */}
            <div className="absolute bottom-8 w-full text-center">
                <p className="text-xs text-white/40 tracking-[0.5em] uppercase">
                    nós somos o futuro, nós somos a inovação
                </p>
            </div>
        </div>
    );
};
