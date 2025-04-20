import { div, p, section, span } from "motion/react-client";
import React from "react";
import GridItem from "./grid-item";
import { FaGraduationCap } from "react-icons/fa";
import { FaLocationDot } from "react-icons/fa6";
import { MdWavingHand } from "react-icons/md";
import { FaGear } from "react-icons/fa6";

const Grid = () => {
  return (
    <section id="about">
      <h2 className="text-center font-bold md:tracking-wider mb-10 text-5xl lg:text-6xl xl:text-6xl">
        About Me
      </h2>
      <ul className="grid grid-cols-1 grid-rows-none gap-4 md:grid-cols-12 md:grid-rows-3 lg:gap-4 xl:max-h-[34rem] xl:grid-rows-2">
        <GridItem
          area="md:[grid-area:1/1/2/13] xl:[grid-area:1/1/2/9]"
          img="/images/stock_terminal.jpg"
          title={
            <>
              <span>
                Hi, I'm Dario Perico{" "}
                <MdWavingHand className="float-start mr-2 mt-1" />
              </span>
            </>
          }
          description={
            <>
              <span>
                I’m currently pursuing a Master’s in Data Science, focusing on
                the intersection of economics, statistics, and modern machine
                learning.
              </span>
              <br />
              <br />
              <span>
                I’m especially drawn to projects where I can combine analytical
                thinking with real-world impact — from understanding financial
                systems to building predictive models with Python and R.
              </span>
            </>
          }
        />

        <GridItem
          area="md:[grid-area:2/1/3/7] xl:[grid-area:2/1/3/6]"
          img="/images/bg-gradient-purple-azure.jpg"
          title={
            <>
              <span>
                Education <FaGraduationCap className="float-start mr-2 mt-1" />
              </span>
            </>
          }
          description={
            <>
              <span className="font-bold">
                M.Sc. in Economics and Data Analysis (Data Science track)
              </span>
              University of Bergamo (2023 – expected 2025)
              <br />
              <br />
              <span className="font-bold">
                B.Sc. Economics, Banking and Finance
              </span>{" "}
              University of Milan-Bicocca (2020 – 2023)
            </>
          }
        />
        <GridItem
          img="/images/bg-globe-gradient.png"
          area="md:[grid-area:2/7/3/13] xl:[grid-area:1/9/2/13]"
          title={
            <>
              <span>
                Location <FaLocationDot className="float-start mr-2 mt-1" />
              </span>
            </>
          }
          description={
            <>
              <span className="font-bold">Based near Milan, Italy</span>
              <br />
              <span>
                Open to remote and on-site opportunities, with an international
                outlook.
              </span>
            </>
          }
        />
        <GridItem
          area="md:[grid-area:3/1/4/13] xl:[grid-area:2/6/3/13]"
          img="/images/bg-tech.jpg"
          title={
            <>
              <span>
                Tech Stack Highlights{" "}
                <FaGear className="float-start mr-2 mt-1" />
              </span>
            </>
          }
          description="I work mainly with Python, R and SQL, using tools like Jupyter, Git, and Tableau; powered by libraries such as Pandas, Scikit-learn, and the tidyverse."
        />
      </ul>
    </section>
  );
};

export default Grid;
