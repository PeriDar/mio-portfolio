import { Spotlight } from "./ui/spotlight";
import { cn } from "@/lib/utils";
import { TextGenerateEffect } from "./ui/text-generate-effect";
import MagicButton from "./ui/magic-button";
import { section } from "motion/react-client";

const Hero = () => {
  return (
    <section id="hero" className="relative min-h-screen pt-15">
      <div className="pb-15 pt-10">
        <div>
          <Spotlight
            className="-top-40 -left-10 md:-left-32 md:-top-20 h-screen"
            fill="white"
          />
          <Spotlight
            className="top-10 left-full h-[80vh] w-[50vw]"
            fill="purple"
          />
          <Spotlight className="top-28 left-80 h-[80vh] w-[50vw]" fill="blue" />
        </div>

        <div className="absolute top-0 left-0 flex h-screen w-full items-center justify-center bg-white dark:bg-black-100">
          <div
            className={cn(
              "absolute inset-0",
              "[background-size:80px_80px]",
              "[background-image:linear-gradient(to_right,#e4e4e7_1px,transparent_1px),linear-gradient(to_bottom,#e4e4e7_1px,transparent_1px)]",
              "dark:[background-image:linear-gradient(to_right,rgba(38,38,38,0.5)_1px,transparent_1px),linear-gradient(to_bottom,rgba(38,38,38,0.5)_1px,transparent_1px)]"
            )}
          />
          {/* Radial gradient for the container to give a faded look */}
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-white [mask-image:radial-gradient(ellipse_at_center,transparent_20%,black)] dark:bg-black-100" />
        </div>
        <div className="flex justify-center relative my-20 z-10">
          <div
            className="max-w[89vw] md:max-w-2xl lg: max-w-[60vw] flex flex-col items-center
          justify-center"
          >
            <h2
              className="uppercase tracking-widest text-xs
            text-center text-blue-100 max-w-80"
            >
              Hi, i&apos;m Dario, A Data Science student based in Italy
            </h2>
            <TextGenerateEffect
              className="text-center text-[40px] md:text-5xl lg:text-6xl"
              words="Data Science Meets Economics & Deep Learning"
            />
            <p className="text-center md:tracking-wider mb-4 text-sm md:text-lg lg:text-2xl">
              My passion is to solve real-world problems using Python, R, and
              statistical modeling.
            </p>

            <MagicButton
              href="/CV_Dario_Perico.pdf" // percorso relativo a /public
              filename="CV_Dario_Perico.pdf"
            />
          </div>
        </div>
      </div>
    </section>
  );
};

export default Hero;
