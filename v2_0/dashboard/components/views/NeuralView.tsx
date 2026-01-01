"use client";

import { Button } from "@/components/ui/Button";
import { motion } from "framer-motion";
import { useState } from "react";

const SUBSYSTEMS = [
    "4tlas_trend",
    "4tlas_micro",
    "4tlas_neuralweight",
    "4tlas_kinematics",
    "4tlas_volatility",
    "4tlas_cycle",
    "4tlas_patterns"
];

export const NeuralView = () => {
    const [manualOverride, setManualOverride] = useState<string | null>("sell"); // Default active as per image

    return (
        <div className="relative w-full h-full flex items-center justify-between px-16">

            {/* Left: Subsystems List */}
            <div className="flex flex-col space-y-2 font-mono text-xs z-10 w-1/4">
                <div className="text-white/40 mb-4 tracking-widest text-[10px] uppercase">sistemas inicializados:</div>
                {SUBSYSTEMS.map((sys, i) => (
                    <motion.div
                        key={sys}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.1 }}
                        className="flex items-center gap-2"
                    >
                        <span className={i === 0 ? "text-white font-bold" : "text-white/60"}>
                            {sys}
                        </span>
                        {i === 0 && <span className="w-1 h-1 bg-cyan-400 rounded-full animate-pulse" />}
                    </motion.div>
                ))}

                <div className="mt-8 p-4 border border-white/10 rounded-xl bg-black/40 backdrop-blur-sm relative overflow-hidden group">
                    <span className="text-[10px] text-white/30 uppercase block mb-1">4tlas_trend</span>
                    <div className="h-1 w-full bg-white/10 rounded-full overflow-hidden">
                        <motion.div
                            className="h-full bg-cyan-500"
                            animate={{ width: ["0%", "100%"] }}
                            transition={{ duration: 2, repeat: Infinity }}
                        />
                    </div>
                    <p className="text-[9px] text-white/50 mt-2 leading-tight">
                        o sistema é responsável por identificar as tendências...
                    </p>
                </div>
            </div>

            {/* Center: Neural Core Animation */}
            <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 z-0 pointer-events-none">
                <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
                    className="relative w-96 h-96 flex items-center justify-center"
                >
                    {/* Abstract Neural Shape (Simulated with rotating rings/dots) */}
                    {/* Outer Ring */}
                    <div className="absolute inset-0 border border-cyan-500/10 rounded-full border-dashed" />

                    {/* Inner Structure (Simulating a Dodecahedron or 3D mesh via CSS) */}
                    <div className="absolute inset-10 border border-white/5 rotate-45" />
                    <div className="absolute inset-10 border border-white/5 -rotate-45" />

                    {/* Nodes */}
                    {[...Array(8)].map((_, i) => (
                        <motion.div
                            key={i}
                            className="absolute w-2 h-2 bg-cyan-400 rounded-full shadow-[0_0_15px_#0ff]"
                            style={{
                                top: '50%',
                                left: '50%',
                                transform: `rotate(${i * 45}deg) translate(140px) rotate(-${i * 45}deg)`
                            }}
                        />
                    ))}

                    {/* Core Text */}
                    <div className="absolute inset-0 flex items-center justify-center">
                        <h1 className="text-6xl font-black text-transparent bg-clip-text bg-gradient-to-b from-white to-white/10 tracking-tighter mix-blend-overlay">
                            NEURAL<br />CORTEX
                        </h1>
                    </div>
                </motion.div>

                {/* Market Regime */}
                <div className="absolute top-full left-1/2 -translate-x-1/2 mt-8 text-center">
                    <div className="text-xs text-white/40 mb-1 tracking-widest uppercase">market regime</div>
                    <div className="text-4xl font-black text-white tracking-widest uppercase animate-pulse">TRENDING</div>
                </div>
            </div>


            {/* Right: Manual Controls */}
            <div className="flex flex-col space-y-6 w-1/4 items-end z-10">
                <div className="text-right mb-4">
                    <span className="text-xs text-white/40 tracking-[0.2em] uppercase">manual override</span>
                </div>

                <div className="flex flex-col gap-4 w-full max-w-[200px]">
                    <Button
                        active={manualOverride === 'buy'}
                        onClick={() => setManualOverride(manualOverride === 'buy' ? null : 'buy')}
                        className="w-full uppercase tracking-widest border-white/30"
                    >
                        force buy
                    </Button>
                    <Button
                        active={manualOverride === 'sell'}
                        onClick={() => setManualOverride(manualOverride === 'sell' ? null : 'sell')}
                        className="w-full uppercase tracking-widest border-white/30"
                    >
                        force sell
                    </Button>
                    <Button
                        active={manualOverride === 'panic'}
                        onClick={() => setManualOverride(manualOverride === 'panic' ? null : 'panic')}
                        className="w-full uppercase tracking-widest border-red-500/30 text-red-400 hover:bg-red-950/30 hover:border-red-500"
                    >
                        panic close
                    </Button>
                </div>

                <div className="text-[10px] text-white/30 font-mono mt-8 text-right">
                    <div>drawdown previsto: 00</div>
                    <div>stop-loss previsto: 00</div>
                </div>
            </div>
        </div>
    );
};
