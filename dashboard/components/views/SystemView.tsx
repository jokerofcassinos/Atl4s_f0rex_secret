"use client";

import { Card } from "@/components/ui/Card";
import { motion } from "framer-motion";

const METRICS = [
    { label: "equity", value: "$100.00", progress: 75 },
    { label: "balance", value: "$100.00", progress: 80 },
    { label: "profit", value: "+$0.00", progress: 10 },
    { label: "win rate", value: "0%", progress: 50 },
];

export const SystemView = () => {
    return (
        <div className="w-full h-full flex flex-col p-8 md:p-12">
            {/* Header */}
            <div className="mb-12">
                <span className="text-xs text-white/40 tracking-[0.3em] uppercase block mb-2">4tlas systems</span>
                <h1 className="text-5xl font-black text-white tracking-wide uppercase">
                    Relatorio de Sistema
                </h1>
            </div>

            <div className="flex flex-row gap-8 h-full">
                {/* Metrics Grid */}
                <div className="flex-1 grid grid-cols-4 gap-6 h-64">
                    {METRICS.map((metric, i) => (
                        <Card key={metric.label} active className="flex flex-col items-center justify-center p-4 bg-white/5 border-white/10 group">
                            <div className="text-[10px] font-bold text-white/70 uppercase tracking-widest mb-4 absolute top-4 left-4">{metric.label}</div>

                            {/* Donut Chart */}
                            <div className="relative w-24 h-24 flex items-center justify-center">
                                <svg className="w-full h-full -rotate-90">
                                    <circle
                                        cx="48" cy="48" r="40"
                                        stroke="currentColor" strokeWidth="8"
                                        fill="transparent"
                                        className="text-white/10"
                                    />
                                    <motion.circle
                                        cx="48" cy="48" r="40"
                                        stroke="currentColor" strokeWidth="8"
                                        fill="transparent"
                                        className="text-white shape-rendering-geometricPrecision"
                                        strokeDasharray={251.2}
                                        strokeDashoffset={251.2 * (1 - metric.progress / 100)}
                                        initial={{ strokeDashoffset: 251.2 }}
                                        animate={{ strokeDashoffset: 251.2 * (1 - metric.progress / 100) }}
                                        transition={{ duration: 1.5, delay: i * 0.2 }}
                                    />
                                </svg>
                                {/* Hover Tooltip/Value */}
                                <div className="absolute inset-0 flex items-center justify-center opacity-0 group-hover:opacity-100 transition-opacity duration-300">
                                    <div className="bg-black/80 backdrop-blur text-white text-xs px-2 py-1 rounded border border-white/20">
                                        {metric.value}
                                    </div>
                                </div>
                            </div>
                        </Card>
                    ))}
                </div>

                {/* Offline Status */}
                <div className="w-1/3 flex flex-col justify-start">
                    <div className="text-xl text-white/70 mb-2">STATUS</div>
                    <h2 className="text-6xl font-black text-white uppercase mb-4 tracking-tighter">OFFLINE</h2>
                    <div className="text-xs text-white/50 space-y-2 font-mono">
                        <p>ligado pela ultima vez em <span className="text-white font-bold">quarta-feira, 22:00</span></p>
                        <p>operou por <span className="text-white font-bold">1 hora e 10 minutos</span></p>
                    </div>
                </div>
            </div>

            {/* Bottom Graph Area */}
            <div className="mt-8 w-full h-32 border border-white/20 rounded-2xl bg-black/20 relative overflow-hidden flex items-center justify-center">
                <div className="text-white/20 text-4xl font-thin tracking-widest lowercase">grafico</div>
                {/* Scanline */}
                <div className="absolute top-0 left-0 w-full h-[2px] bg-cyan-500/50 shadow-[0_0_20px_#0ff] animate-[scanline_3s_linear_infinite]" />
                <div className="absolute inset-0 bg-gradient-to-t from-cyan-900/10 to-transparent" />
            </div>
        </div>
    );
};
