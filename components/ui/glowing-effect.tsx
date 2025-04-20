"use client";

import { memo, useCallback, useEffect, useRef, ReactNode } from "react";
import { cn } from "@/lib/utils";
import { animate } from "motion/react";

interface GlowingEffectProps {
  blur?: number;
  inactiveZone?: number;
  proximity?: number;
  spread?: number;
  variant?: "default" | "white";
  glow?: boolean;
  disabled?: boolean;
  movementDuration?: number;
  borderWidth?: number;
  className?: string;
  children?: ReactNode;
}

const GlowingEffect = memo(
  ({
    blur = 0,
    inactiveZone = 0.7,
    proximity = 0,
    spread = 20,
    variant = "default",
    glow = false,
    disabled = true,
    movementDuration = 2,
    borderWidth = 1,
    className,
    children,
  }: GlowingEffectProps) => {
    const containerRef = useRef<HTMLDivElement>(null);
    const lastPosition = useRef({ x: 0, y: 0 });
    const raf = useRef(0);

    const handleMove = useCallback(
      (e?: MouseEvent | { x: number; y: number }) => {
        if (!containerRef.current) return;
        if (raf.current) cancelAnimationFrame(raf.current);

        raf.current = requestAnimationFrame(() => {
          const el = containerRef.current!;
          const { left, top, width, height } = el.getBoundingClientRect();
          const mouseX = e?.x ?? lastPosition.current.x;
          const mouseY = e?.y ?? lastPosition.current.y;
          if (e) lastPosition.current = { x: mouseX, y: mouseY };

          // distanza dal centro
          const cx = left + width / 2;
          const cy = top + height / 2;
          const dist = Math.hypot(mouseX - cx, mouseY - cy);
          const inactiveR = 0.5 * Math.min(width, height) * inactiveZone;
          if (dist < inactiveR) {
            el.style.setProperty("--active", "0");
            return;
          }

          // sei dentro al box + proximity?
          const isActive =
            mouseX > left - proximity &&
            mouseX < left + width + proximity &&
            mouseY > top - proximity &&
            mouseY < top + height + proximity;
          el.style.setProperty("--active", isActive ? "1" : "0");
          if (!isActive) return;

          // calcolo angolo e animazione
          const current = parseFloat(el.style.getPropertyValue("--start")) || 0;
          const target =
            (180 * Math.atan2(mouseY - cy, mouseX - cx)) / Math.PI + 90;
          const diff = ((target - current + 180) % 360) - 180;
          animate(current, current + diff, {
            duration: movementDuration,
            ease: [0.16, 1, 0.3, 1],
            onUpdate: (v) => el.style.setProperty("--start", String(v)),
          });
        });
      },
      [inactiveZone, proximity, movementDuration]
    );

    useEffect(() => {
      if (disabled) return;
      const onScroll = () => handleMove();
      const onMove = (e: PointerEvent) => handleMove(e);
      window.addEventListener("scroll", onScroll, { passive: true });
      document.body.addEventListener("pointermove", onMove, { passive: true });
      return () => {
        cancelAnimationFrame(raf.current);
        window.removeEventListener("scroll", onScroll);
        document.body.removeEventListener("pointermove", onMove);
      };
    }, [handleMove, disabled]);

    return (
      <div
        ref={containerRef}
        className={cn("relative overflow-visible", className)}
        style={
          {
            "--blur": `${blur}px`,
            "--spread": spread,
            "--start": 0,
            "--active": 0,
            "--glowingeffect-border-width": `${borderWidth}px`,
            "--repeating-conic-gradient-times": 5,
            "--gradient":
              variant === "white"
                ? `repeating-conic-gradient(
                from 236.84deg at 50% 50%,
                var(--black),
                var(--black) calc(25% / var(--repeating-conic-gradient-times))
              )`
                : `radial-gradient(circle, #dd7bbb 10%, #dd7bbb00 20%),
               radial-gradient(circle at 40% 40%, #d79f1e 5%, #d79f1e00 15%),
               radial-gradient(circle at 60% 60%, #5a922c 10%, #5a922c00 20%),
               radial-gradient(circle at 40% 60%, #4c7894 10%, #4c789400 20%),
               repeating-conic-gradient(
                 from 236.84deg at 50% 50%,
                 #dd7bbb 0%,
                 #d79f1e calc(25% / var(--repeating-conic-gradient-times)),
                 #5a922c calc(50% / var(--repeating-conic-gradient-times)),
                 #4c7894 calc(75% / var(--repeating-conic-gradient-times)),
                 #dd7bbb calc(100% / var(--repeating-conic-gradient-times))
               )`,
          } as React.CSSProperties
        }
      >
        {/* bordo invisibile che diventa visibile al glow */}
        <div
          className={cn(
            "pointer-events-none absolute -inset-px rounded-[inherit]",
            // definiamo un border-width uguale a --glowingeffect-border-width,
            // ma trasparente di default (così prende lo spazio, senza disegnarsi).
            "border-[var(--glowingeffect-border-width)] border-transparent",
            // quando glow=true facciamo apparire il bordo colore normale
            !disabled && glow && "border-current opacity-[var(--active)]",
            // variante bianca: il glow diventa bianco
            variant === "white" && !disabled && glow && "border-white"
          )}
        />

        {/* il vero gradiente con mask */}
        <div
          className={cn(
            "pointer-events-none absolute inset-0 rounded-[inherit] transition-opacity after:content-['']",
            blur > 0 && "blur-[var(--blur)]",
            "after:absolute after:inset-[calc(-1*var(--glowingeffect-border-width))]",
            "after:rounded-[inherit]",
            "after:[border:var(--glowingeffect-border-width)_solid_transparent]",
            "after:[background:var(--gradient)]",
            "after:background-attachment-fixed",
            "after:opacity-[var(--active)]",
            "after:transition-opacity after:duration-300",
            "after:[mask-clip:padding-box,border-box]",
            "after:[mask-composite:intersect]",
            "after:[mask-image:linear-gradient(#0000,#0000),conic-gradient(from_calc((var(--start)-var(--spread))*1deg),#00000000_0deg,#fff,#00000000_calc(var(--spread)*2deg))]"
          )}
        />

        {/* finalmente i tuoi figli */}
        {children}
      </div>
    );
  }
);

GlowingEffect.displayName = "GlowingEffect";
export { GlowingEffect };
