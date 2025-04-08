import { config } from "dotenv";
import { OpenAIEmbeddings } from "@langchain/openai";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import path from "path";

config();

const STORE_PATH = path.join(process.cwd(), "vector_store");
const embeddings = new OpenAIEmbeddings({
  openAIApiKey: process.env.OPENAI_API_KEY,
});

// Preload vector store when the function starts
let vectorStore;
(async () => {
  try {
    console.log("Loading vector store...");
    vectorStore = await HNSWLib.load(STORE_PATH, embeddings);
    console.log("Vector store loaded successfully");
  } catch (error) {
    console.error("Failed to load vector store:", error.message);
  }
})();

// Timeout wrapper for embedding query
async function embedQueryWithTimeout(query, timeoutMs = 5000) {
  const timeout = new Promise((_, reject) => {
    setTimeout(() => reject(new Error("Embedding query timed out")), timeoutMs);
  });
  return Promise.race([embeddings.embedQuery(query), timeout]);
}

export default async function handler(req, res) {
  console.log("Handler invoked");
  if (req.method !== "GET") return res.status(405).json({ error: "Method not allowed" });
  const query = req.query.q;
  if (!query) return res.status(400).json({ error: "Query parameter 'q' is required" });

  if (!vectorStore) {
    console.log("Vector store not loaded");
    return res.status(500).json({ error: "Vector store not loaded" });
  }

  try {
    console.time("embedQuery");
    const queryEmbedding = await embedQueryWithTimeout(query, 5000);
    console.timeEnd("embedQuery");

    console.time("similaritySearch");
    const results = await vectorStore.similaritySearchVector(queryEmbedding, 3);
    console.timeEnd("similaritySearch");

    res.status(200).json(results.map((doc) => ({ content: doc.pageContent, metadata: doc.metadata })));
  } catch (error) {
    console.log("Error:", error.message);
    res.status(500).json({ error: error.message });
  }
}