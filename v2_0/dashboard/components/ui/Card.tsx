import { cn } from "@/lib/utils";
import { HTMLMotionProps, motion } from "framer-motion";

interface CardProps extends HTMLMotionProps<"div"> {
    children: React.ReactNode;
    active?: boolean; // For System View interactivity
}

export const Card = ({ className, children, active, ...props }: CardProps) => {
    return (
        <motion.div
            className={cn(
                "relative rounded-3xl border border-white/10 bg-white/5 backdrop-blur-md overflow-hidden transition-all duration-300",
                active && "hover:border-cyan-500/50 hover:shadow-[0_0_30px_rgba(8,145,178,0.2)]",
                className
            )}
            {...props}
        >
            {/* Optional Noise Texture or Inner Glow could be added here */}
            <div className="absolute inset-0 bg-gradient-to-br from-white/5 to-transparent pointer-events-none" />
            <div className="relative z-10 p-6 h-full w-full">{children}</div>
        </motion.div>
    );
};
