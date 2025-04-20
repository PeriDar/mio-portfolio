import React from "react";

interface MagicButtonProps {
  href: string; // path del file
  filename?: string; // nome suggerito al download
  children?: React.ReactNode;
}

const MagicButton: React.FC<MagicButtonProps> = ({
  href,
  filename,
  children = "Download my CV",
}) => (
  <a
    href={href}
    download={filename ?? true}
    className="relative inline-flex h-12 overflow-hidden rounded-2xl p-[1px] focus:outline-none focus:ring-2 focus:ring-slate-400 focus:ring-offset-2 focus:ring-offset-slate-50"
  >
    <span className="absolute inset-[-1000%] animate-[spin_2s_linear_infinite] bg-[conic-gradient(from_90deg_at_50%_50%,#E2CBFF_0%,#393BB2_50%,#E2CBFF_100%)]" />
    <span className="hover:bg-slate-900 inline-flex h-full w-full items-center justify-center rounded-2xl bg-slate-950 px-7 text-base font-medium text-white backdrop-blur-3xl">
      {children}
    </span>
  </a>
);

export default MagicButton;
