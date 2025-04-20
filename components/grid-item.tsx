import React from "react";
import { GlowingEffect } from "./ui/glowing-effect";

interface GridItemProps {
  area: string;
  title: React.ReactNode;
  TitleImg?: string;
  description: React.ReactNode;
  img?: string; // es. "/images/stock_terminal.jpg"
}

export const GridItem = ({
  area,
  title,
  description,
  img,
  TitleImg,
}: GridItemProps) => (
  <li className={`min-h-[14rem] list-none ${area}`}>
    <GlowingEffect
      className="h-full rounded-3xl"
      disabled={false}
      glow={true}
      proximity={64}
      inactiveZone={0.01}
      spread={40}
      borderWidth={2.3}
    >
      {/* questa ha overflow-hidden solo per l’immagine */}
      <div className="relative h-full rounded-3xl border p-2 md:rounded-3xl md:p-3 overflow-hidden">
        {img && (
          <img
            src={img}
            alt={TitleImg}
            className="absolute inset-0 h-full w-full object-cover pointer-events-none"
          />
        )}
        <div className="relative flex h-full flex-col justify-between gap-6 rounded-3xl border-0.75 p-6 md:p-6 bg-white/80 dark:bg-black/30 backdrop-blur-xs">
          <h3 className="font-sans text-xl font-semibold text-black dark:text-white md:text-2xl">
            {title}
          </h3>
          <p className="font-sans text-sm text-black dark:text-white md:text-base">
            {description}
          </p>
        </div>
      </div>
    </GlowingEffect>
  </li>
);

export default GridItem;
