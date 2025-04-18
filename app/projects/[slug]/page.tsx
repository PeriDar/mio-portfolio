import fs from "fs";
import path from "path";
import matter from "gray-matter";
import ReactMarkdown from "react-markdown";
import rehypeHighlight from "rehype-highlight";
import "highlight.js/styles/github-dark.css"; // stile evidenziazione

export default async function ProjectPage(props: { params: { slug: string } }) {
  // ✅ estrai lo slug in una variabile subito all’inizio
  const { slug } = props.params;

  const filePath = path.join(
    process.cwd(),
    "content",
    "projects",
    slug,
    `${slug}.md`
  );

  if (!fs.existsSync(filePath)) {
    return (
      <div className="p-8 text-center text-red-600">
        Progetto non trovato 😢
        <br />({filePath})
      </div>
    );
  }

  const fileContent = fs.readFileSync(filePath, "utf8");
  const { content, data } = matter(fileContent);

  return (
    <main
      className="prose-dark w-full max-w-[95%] sm:max-w-2xl md:max-w-3xl lg:max-w-4xl xl:max-w-5xl
     px-4 sm:px-6 lg:px-8 py-8"
    >
      <ReactMarkdown rehypePlugins={[rehypeHighlight]}>{content}</ReactMarkdown>
    </main>
  );
}
