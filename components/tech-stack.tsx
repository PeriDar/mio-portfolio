import { section } from "motion/react-client";
import React from "react";
import { SparklesIcons } from "@/components/sparkle-icons";
import { HoverEffect } from "./ui/card-hover-effect";
const TechStack = () => {
  return (
    <section id="techStack" className="min-h-screen p-15">
      <div>
        <h2 className="text-center font-bold md:tracking-wider mb-10 text-5xl lg:text-6xl xl:text-6xl">
          My tech Stack
        </h2>
        <SparklesIcons />
        <div className="max-w-7xl mx-auto">
          <HoverEffect items={projects} />
        </div>
      </div>
    </section>
  );
};

export default TechStack;

export const projects = [
  {
    id: "langs",
    title: "Languages",
    description:
      "I mainly work with Python, which is my go-to language for data analysis, modeling, and scripting. I also use R, especially in combination with tidyverse tools for statistical analysis and visualization. I'm comfortable writing SQL for data querying and manipulation, and I often document projects using Markdown or RMarkdown. Additionally, I have basic experience with JavaScript, primarily for web development purposes.",
  },
  {
    id: "lib",
    title: "Data Analysis Libraries",
    description:
      "My core Python stack includes NumPy and Pandas for numerical computation and data wrangling, along with Statsmodels and SciPy for statistical modeling and optimization tasks. For data visualization, I use Matplotlib and Seaborn to create both standard and statistical plots.In R, I work with the tidyverse collection — especially dplyr and ggplot2 — to perform structured analysis and produce clear, elegant graphics.",
  },
  {
    id: "ml-dl",
    title: "Machine Learning & Deep Learning",
    description:
      "I use Scikit-learn and XGBoost for classical machine learning tasks such as regression, classification, and clustering.For deep learning, I primarily work with PyTorch, and I'm currently starting to explore TensorFlow and Keras. In R, I occasionally use tidymodels, though my machine learning workflows are mostly Python-based",
  },
  {
    id: "BI",
    title: "BI & Data Visualization Tools",
    description:
      "For dashboarding and business reporting, I work with Tableau and Power BI. When interactivity is needed in Python, I use Plotly to create dynamic, shareable visualizations. I'm also interested in exploring D3.js in the future, especially for custom visualizations in web-based data storytelling.",
  },
];
