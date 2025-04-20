import { div, p, section, span } from "motion/react-client";
import React from "react";
import GridItem from "./grid-item";
import { FaGraduationCap } from "react-icons/fa";

const Grid = () => {
  return (
    <section id="about">
      <h2 className="text-center font-bold md:tracking-wider mb-10 text-sm md:text-lg lg:text-5xl">
        About Me
      </h2>
      <ul className="grid grid-cols-1 grid-rows-none gap-4 md:grid-cols-12 md:grid-rows-3 lg:gap-4 xl:max-h-[34rem] xl:grid-rows-2">
        <GridItem
          area="md:[grid-area:1/1/2/7] xl:[grid-area:1/1/2/9]"
          title="Hi, I'm Dario Perico"
          description="I'm currently pursuing my Master's Degree in Data Science, combining my passions for econometrics, macroeconomics, and Bayesian statistics to uncover insights hidden in data. I love transforming complex problems into clear, actionable solutions."
        />

        <GridItem
          area="md:[grid-area:1/7/2/13] xl:[grid-area:2/1/3/5]"
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
          area="md:[grid-area:2/7/3/13] xl:[grid-area:1/9/2/13]"
          title="This card is also built by Cursor"
          description="I'm not even kidding. Ask my mom if you don't believe me."
        />
        <GridItem
          img="/images/stock_terminal.jpg"
          area="md:[grid-area:3/1/4/13] xl:[grid-area:2/5/3/13]"
          title="Coming soon on Aceternity UI"
          description="I'm writing the code as I record this, no shit."
        />
      </ul>
    </section>
  );
};

export default Grid;
