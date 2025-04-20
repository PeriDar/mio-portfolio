"use client";
import React, { useState, useRef } from "react";
import {
  motion,
  AnimatePresence,
  useScroll,
  useMotionValueEvent,
} from "motion/react";
import { cn } from "@/lib/utils";

export const FloatingNav: React.FC<{
  navItems: { name: string; link: string; icon?: React.ReactNode }[];
  className?: string;
}> = ({ navItems, className }) => {
  const { scrollY } = useScroll();
  const [visible, setVisible] = useState(true);
  const lastScrollY = useRef(0);

  // Ogni volta che scrollY cambia, confronto con il precedente
  useMotionValueEvent(scrollY, "change", (current) => {
    const prev = lastScrollY.current;

    // Se siamo in alto nella pagina (<50px), mostro sempre la navbar
    if (current < 50) {
      setVisible(true);
    } else {
      // scroll verso il basso → nascondi
      if (current > prev) {
        setVisible(false);
      }
      // scroll verso l’alto → mostra
      else if (current < prev) {
        setVisible(true);
      }
    }

    lastScrollY.current = current;
  });

  return (
    <AnimatePresence>
      {visible && (
        <motion.div
          // stato iniziale all’apparizione
          initial={{ opacity: 0, y: -100 }}
          // animazione di entrata
          animate={{ opacity: 1, y: 0 }}
          // animazione di uscita
          exit={{ opacity: 0, y: -100 }}
          transition={{ duration: 0.2 }}
          className={cn(
            "flex max-w-fit fixed top-10 inset-x-0 mx-auto",
            "border border-transparent dark:border-white/[0.2] rounded-full",
            "dark:bg-black bg-white",
            "shadow-[0px_2px_3px_-1px_rgba(0,0,0,0.1),0px_1px_0px_0px_rgba(25,28,33,0.02),0px_0px_0px_1px_rgba(25,28,33,0.08)]",
            "z-[5000] pr-2 pl-8 py-2 items-center justify-center space-x-4",
            className
          )}
        >
          {navItems.map((navItem, idx) => (
            <a
              key={idx}
              href={navItem.link}
              className={cn(
                "relative items-center flex space-x-1",
                "text-neutral-600 dark:text-neutral-50",
                "hover:text-neutral-500 dark:hover:text-neutral-300"
              )}
            >
              <span className="block sm:hidden">{navItem.icon}</span>
              <span className="hidden sm:block text-sm">{navItem.name}</span>
            </a>
          ))}

          <button className="relative px-4 py-2 text-sm font-medium border border-neutral-200 dark:border-white/[0.2] rounded-full text-black dark:text-white">
            <span>Login</span>
            <span className="absolute inset-x-0 mx-auto -bottom-px h-px w-1/2 bg-gradient-to-r from-transparent via-blue-500 to-transparent" />
          </button>
        </motion.div>
      )}
    </AnimatePresence>
  );
};
