import { section } from "motion/react-client";
import React from "react";
import { SparklesIcons } from "@/components/sparkle-icons";
import { GlareCard } from "./ui/glare-card";

const TechStack = () => {
  return (
    <section className="min-h-screen p-15">
      <div>
        <h2 className="text-center font-bold md:tracking-wider mb-10 text-5xl lg:text-6xl xl:text-6xl">
          My tech Stack
        </h2>
        <SparklesIcons />
        <div className="w-full flex flex-col items-center gap-5 md:grid md:grid-cols-2 md:grid-rows-2 md:place-items-center lg:gap-10 pt-3">
          <GlareCard className="flex flex-col items-center justify-center">
            <svg
              width="66"
              height="65"
              viewBox="0 0 66 65"
              fill="none"
              xmlns="http://www.w3.org/2000/svg"
              className="h-14 w-14 text-white"
            >
              <path
                d="M8 8.05571C8 8.05571 54.9009 18.1782 57.8687 30.062C60.8365 41.9458 9.05432 57.4696 9.05432 57.4696"
                stroke="currentColor"
                strokeWidth="15"
                strokeMiterlimit="3.86874"
                strokeLinecap="round"
              />
            </svg>
          </GlareCard>
          <GlareCard className="flex flex-col items-center justify-center">
            <img
              className="h-full w-full absolute inset-0 object-cover"
              src="https://images.unsplash.com/photo-1512618831669-521d4b375f5d?q=80&w=3388&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D"
            />
          </GlareCard>
          <GlareCard className="flex flex-col items-start justify-end py-8 px-6">
            <p className="font-bold text-white text-lg">The greatest trick</p>
            <p className="font-normal text-base text-neutral-200 mt-4">
              The greatest trick the devil ever pulled was to convince the world
              that he didn&apos;t exist.
            </p>
          </GlareCard>
          <GlareCard className="flex flex-col items-start justify-end py-8 px-6">
            <p className="font-bold text-white text-lg">The greatest trick</p>
            <p className="font-normal text-base text-neutral-200 mt-4">
              The greatest trick the devil ever pulled was to convince the world
              that he didn&apos;t exist.
            </p>
          </GlareCard>
        </div>
      </div>
    </section>
  );
};

export default TechStack;
