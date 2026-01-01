import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
    active?: boolean;
    children: React.ReactNode;
}

export const Button = ({ active, className, children, ...props }: ButtonProps) => {
    return (
        <button
            className={cn(
                "relative px-6 py-2 rounded-full text-sm font-medium transition-all duration-300 border border-transparent",
                active
                    ? "bg-white text-black shadow-[0_0_20px_rgba(255,255,255,0.5)] border-white"
                    : "bg-transparent text-white/70 border-white/30 hover:bg-white/5 hover:text-white hover:border-white/50",
                className
            )}
            {...props}
        >
            {active && (
                <motion.div
                    layoutId="active-glow"
                    className="absolute inset-0 rounded-full bg-white blur-md -z-10 opacity-50"
                    initial={false}
                    transition={{ type: "spring", stiffness: 500, damping: 30 }}
                />
            )}
            {children}
        </button>
    );
};
