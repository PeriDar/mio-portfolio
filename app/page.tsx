import Image from "next/image";
import Link from "next/link";
import Hero from "@/components/hero";
import { FloatingNav } from "@/components/ui/floating-navbar";
import { FaHome } from "react-icons/fa";
import Grid from "@/components/about-me-grid";
import TechStack from "@/components/tech-stack";
import Footer from "@/components/footer";

export default function Home() {
  return (
    <main
      className="relative bg-black-100 flex justify-center
    items-center flex-col overflow-hidden mx-auto sm:px-10 px-5"
    >
      <div className="max-w-7xl w-full">
        <FloatingNav
          navItems={[
            { name: "Home", link: "#hero", icon: <FaHome /> },
            { name: "About", link: "#about" },
            { name: "Tech", link: "#techStack" },
            { name: "Contacts", link: "#footer" },
          ]}
        />
        <Hero />
        <Grid />
        <TechStack />
        <Footer />
      </div>

      {/* Codice per aggiungere peogetti */}
      <div className="relative flex flex-col items-center">
        <h1 className="text-3xl font-bold">Benvenuto nel mio portfolio</h1>
        <p className="mt-4">Ecco un progetto disponibile:</p>
        <Link
          href="/projects/testing-strategies"
          className="text-blue-600 underline"
        >
          Vai al progetto "Testing Strategies"
        </Link>
      </div>
    </main>
  );
}
