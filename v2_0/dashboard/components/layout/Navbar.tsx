"use client";

import { Button } from "@/components/ui/Button";
import { motion } from "framer-motion";

interface NavbarProps {
    activeTab: string;
    onTabChange: (tab: string) => void;
}

const tabs = [
    { id: "home", label: "home" },
    { id: "system", label: "system" },
    { id: "neural", label: "neural" },
    { id: "logs", label: "logs" },
];

export const Navbar = ({ activeTab, onTabChange }: NavbarProps) => {
    return (
        <motion.nav
            initial={{ y: -100, opacity: 0 }}
            animate={{ y: 0, opacity: 1 }}
            transition={{ duration: 0.8, ease: "circOut" }}
            className="fixed top-8 left-1/2 -translate-x-1/2 z-50 p-2 rounded-full border border-white/10 bg-black/20 backdrop-blur-md flex gap-4"
        >
            {tabs.map((tab) => (
                <Button
                    key={tab.id}
                    active={activeTab === tab.id}
                    onClick={() => onTabChange(tab.id)}
                >
                    {tab.label}
                </Button>
            ))}
        </motion.nav>
    );
};
