"use client";

import { useState } from "react";
import { Background } from "@/components/layout/Background";
import { Navbar } from "@/components/layout/Navbar";
import { CustomCursor } from "@/components/ui/CustomCursor";
import { HomeView } from "@/components/views/HomeView";
import { LogsView } from "@/components/views/LogsView";
import { NeuralView } from "@/components/views/NeuralView";
import { SystemView } from "@/components/views/SystemView";
import { AnimatePresence, motion } from "framer-motion";
import { Hexagon } from "lucide-react";

export const Dashboard = () => {
    const [activeTab, setActiveTab] = useState("home");

    return (
        <div className="relative w-screen h-screen overflow-hidden text-white selection:bg-cyan-500/30">
            <CustomCursor />
            <Background />

            <Navbar activeTab={activeTab} onTabChange={setActiveTab} />

            {/* Decorative Icon */}
            <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 60, repeat: Infinity, ease: "linear" }}
                className="fixed top-8 right-8 z-40 opacity-50"
            >
                <Hexagon className="w-8 h-8 text-white/40" strokeWidth={1} />
                {/* Inner static hexagon for wireframe feel */}
                <div className="absolute inset-0 flex items-center justify-center">
                    <div className="w-4 h-4 border border-white/20 rounded-full" />
                </div>
            </motion.div>

            {/* Side Lists Decoration */}
            <div className="fixed right-8 top-1/2 -translate-y-1/2 text-[10px] text-right space-y-2 text-white/30 hidden md:block z-20 font-mono tracking-widest">
                <div>• inovação</div>
                <div>• performance</div>
                <div>• tecnologia</div>
            </div>


            <main className="w-full h-full relative z-10 pt-24 pb-8 px-8">
                <AnimatePresence mode="wait">
                    <motion.div
                        key={activeTab}
                        initial={{ opacity: 0, y: -20, filter: "blur(10px)" }}
                        animate={{ opacity: 1, y: 0, filter: "blur(0px)" }}
                        exit={{ opacity: 0, y: 20, filter: "blur(10px)" }}
                        transition={{ duration: 0.5, ease: "easeInOut" }}
                        className="w-full h-full"
                    >
                        {activeTab === "home" && <HomeView />}
                        {activeTab === "logs" && <LogsView />}
                        {activeTab === "neural" && <NeuralView />}
                        {activeTab === "system" && <SystemView />}
                    </motion.div>
                </AnimatePresence>
            </main>
        </div>
    );
};
