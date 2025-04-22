"use client";
import { animate, motion } from "motion/react";
import React, { useEffect, useState } from "react";
import { cn } from "@/lib/utils";
import { useMemo } from "react";

function prng(seed: number) {
  const x = Math.sin(seed) * 10_000;
  return x - Math.floor(x);
}

export function SparklesIcons() {
  return (
    <CardSkeletonContainer>
      <Skeleton />
    </CardSkeletonContainer>
  );
}

const Skeleton = () => {
  const scale = [1, 1.1, 1];
  const transform = ["translateY(0px)", "translateY(-4px)", "translateY(0px)"];
  const sequence = [
    [
      ".circle-1",
      {
        scale,
        transform,
      },
      { duration: 0.8 },
    ],
    [
      ".circle-2",
      {
        scale,
        transform,
      },
      { duration: 0.8 },
    ],
    [
      ".circle-3",
      {
        scale,
        transform,
      },
      { duration: 0.8 },
    ],
    [
      ".circle-4",
      {
        scale,
        transform,
      },
      { duration: 0.8 },
    ],
    [
      ".circle-5",
      {
        scale,
        transform,
      },
      { duration: 0.8 },
    ],
  ];

  useEffect(() => {
    animate(sequence, {
      // @ts-ignore
      repeat: Infinity,
      repeatDelay: 1,
    });
  }, []);
  return (
    <div className="p-1 overflow-hidden h-full relative flex items-center justify-center">
      <div className="flex flex-row shrink-0 justify-center items-center gap-2">
        <Container className="h-8 w-8 circle-1">
          <img src="/images/git.svg" alt="" className="h-5 w-5" />
        </Container>
        <Container className="h-12 w-12 circle-2">
          <img src="/images/sql.svg" alt="" className="h-7 w-7" />
        </Container>
        <Container className="circle-3">
          <img src="/images/python.svg" alt="" />
        </Container>
        <Container className="h-12 w-12 circle-4">
          <img src="/images/R.svg" alt="" className="h-7 w-7" />
        </Container>
        <Container className="h-8 w-8 circle-5">
          <img src="/images/tableau.svg" alt="" className="h-5 w-5" />
        </Container>
      </div>

      <div className="h-40 w-px absolute top-20 m-auto z-40 bg-gradient-to-b from-transparent via-cyan-500 to-transparent animate-move">
        <div className="w-10 h-32 top-1/2 -translate-y-1/2 absolute -left-10">
          <Sparkles />
        </div>
      </div>
    </div>
  );
};

export const Sparkles = ({ count = 12 }: { count?: number }) => {
  const stars = useMemo(
    () =>
      Array.from({ length: count }, (_, i) => {
        const top = prng(i + 1) * 100; // %
        const left = prng(i + 101) * 100; // %
        const driftY = prng(i + 201) * 2 - 1; // px
        const driftX = prng(i + 301) * 2 - 1; // px
        const opacity = prng(i + 401); // 0‑1
        const duration = prng(i + 501) * 2 + 4; // 4‑6 s

        return { top, left, driftX, driftY, opacity, duration };
      }),
    [count]
  );

  return (
    <div className="absolute inset-0">
      {stars.map((s, i) => (
        <motion.span
          key={i}
          className="absolute inline-block bg-black dark:bg-white"
          style={{
            top: `${s.top}%`,
            left: `${s.left}%`,
            width: 2,
            height: 2,
            borderRadius: "50%",
            zIndex: 1,
          }}
          animate={{
            top: `calc(${s.top}% + ${s.driftY}px)`,
            left: `calc(${s.left}% + ${s.driftX}px)`,
            opacity: s.opacity,
            scale: [1, 1.2, 0],
          }}
          transition={{
            duration: s.duration,
            repeat: Infinity,
            ease: "linear",
          }}
        />
      ))}
    </div>
  );
};

export const CardSkeletonContainer = ({
  className,
  children,
  showGradient = true,
}: {
  className?: string;
  children: React.ReactNode;
  showGradient?: boolean;
}) => {
  return (
    <div
      className={cn(
        "h-[15rem] md:h-[20rem] rounded-xl z-40",
        className,
        showGradient &&
          "bg-neutral-300 dark:bg-[rgba(40,40,40,0.70)] [mask-image:radial-gradient(50%_50%_at_50%_50%,white_0%,transparent_100%)]"
      )}
    >
      {children}
    </div>
  );
};

const Container = ({
  className,
  children,
}: {
  className?: string;
  children: React.ReactNode;
}) => {
  return (
    <div
      className={cn(
        `h-16 w-16 rounded-full flex items-center justify-center bg-[rgba(248,248,248,0.01)]
    shadow-[0px_0px_8px_0px_rgba(248,248,248,0.25)_inset,0px_32px_24px_-16px_rgba(0,0,0,0.40)]
    `,
        className
      )}
    >
      {children}
    </div>
  );
};
