import { section } from "motion/react-client";
import React from "react";

const Footer = () => {
  return (
    <section id="footer" className="min-h-screen p-15">
      <h2 className="bg-clip-text text-transparent text-center bg-gradient-to-b from-neutral-900 to-neutral-700 dark:from-blue-500 dark:to-neutral-100 text-5xl lg:text-7xl font-sans py-2 md:py-10 relative z-20 font-bold tracking-tight">
        Always open to new opportunities & collaborations. <br /> Let's connect!
      </h2>
      <p className="max-w-xl mx-auto text-sm md:text-lg text-neutral-700 dark:text-neutral-400 text-center">
        I'm currently available for internships, research projects, roles in
        Data Science, ML or Data Analysis.
      </p>
      <div className="flex justify-center gap-6 mt-6">
        {/* GitHub Button */}
        <a
          href="https://github.com/PeriDar"
          target="_blank"
          rel="noopener noreferrer"
          className="w-12 h-12 rounded-full flex items-center justify-center bg-neutral-100 hover:bg-purple-600 transition"
        >
          <img
            src="/images/GitHub-Logo.svg"
            alt="GitHub"
            className="w-10 h-10"
          />
        </a>

        {/* LinkedIn Button */}
        <a
          href="https://www.linkedin.com/in/darioperico/"
          target="_blank"
          rel="noopener noreferrer"
          className="w-12 h-12 rounded-full flex items-center justify-center bg-neutral-100 hover:bg-purple-600 transition"
        >
          <img
            src="/images/LinkedIn-Logo.svg"
            alt="LinkedIn"
            className="w-10 h-10"
          />
        </a>
      </div>

      <footer className="absolute bottom-6 w-full float-end">
        © 2025 Dario Perico — Built with Next.js & Tailwind CSS
      </footer>
    </section>
  );
};

export default Footer;
