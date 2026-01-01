"use client";

import { Card } from "@/components/ui/Card";
import { useEffect, useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";

const INITIAL_LOGS = [
    "[S Y S T E M] interface loaded. waiting for neural link...",
    "[15:31:18] command sent: force_buy",
    "[15:31:18] command sent: force_buy",
    "[15:31:18] command sent: force_buy",
    "[15:31:19] command sent: force_buy",
    "[15:31:20] market status: volatile",
    "[15:31:21] neural network: optimizing weights...",
    "[15:31:22] success: transaction executed"
];

export const LogsView = () => {
    const [logs, setLogs] = useState<string[]>([]);

    useEffect(() => {
        let currentIndex = 0;
        const interval = setInterval(() => {
            if (currentIndex < INITIAL_LOGS.length) {
                setLogs(prev => [...prev, INITIAL_LOGS[currentIndex]]);
                currentIndex++;
            } else {
                clearInterval(interval);
            }
        }, 150); // Speed of typewriter

        return () => clearInterval(interval);
    }, []);

    return (
        <div className="w-full h-full flex items-center justify-center p-8 md:p-20">
            <Card className="w-full max-w-4xl h-[60vh] flex flex-col font-mono text-sm border-white/10 bg-black/40">
                <div className="mb-4">
                    <h2 className="text-2xl font-black uppercase tracking-wider text-white">Logs do Sistema</h2>
                    <div className="h-[1px] w-full bg-gradient-to-r from-white/20 to-transparent mt-2" />
                </div>

                <div className="flex-1 overflow-y-auto space-y-1 scrollbar-none masked-overflow">
                    <AnimatePresence>
                        {logs.map((log, i) => (
                            <motion.div
                                key={i}
                                initial={{ opacity: 0, x: -10 }}
                                animate={{ opacity: 1, x: 0 }}
                                className="text-white/70 hover:text-white transition-colors cursor-default"
                            >
                                <span className="text-cyan-500/50 mr-2">➜</span>
                                {log}
                            </motion.div>
                        ))}
                    </AnimatePresence>
                    {/* Gradient fade at bottom is handled by parent Card but can be enhanced here */}
                    <div className="h-12 w-full absolute bottom-0 left-0 bg-gradient-to-t from-black/80 to-transparent pointer-events-none rounded-b-3xl" />
                </div>
            </Card>
        </div>
    );
};
