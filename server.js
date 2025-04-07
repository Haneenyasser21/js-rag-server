import { config } from "dotenv";
import { OpenAIEmbeddings } from "@langchain/openai";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import path from "path";

config();

const STORE_PATH = path.join(process.cwd(), "vector_store");
const embeddings = new OpenAIEmbeddings({
  openAIApiKey: process.env.OPENAI_API_KEY,
});

let vectorStore;
async function loadVectorStore() {
  if (!vectorStore) {
    vectorStore = await HNSWLib.load(STORE_PATH, embeddings);
  }
  return vectorStore;
}

export default async function handler(req, res) {
  if (req.method !== "GET") return res.status(405).json({ error: "Method not allowed" });
  const query = req.query.q;
  if (!query) return res.status(400).json({ error: "Query parameter 'q' is required" });

  try {
    const store = await loadVectorStore();
    const queryEmbedding = await embeddings.embedQuery(query);
    const results = await store.similaritySearchVector(queryEmbedding, 5);
    res.status(200).json(results.map((doc) => ({ content: doc.pageContent, metadata: doc.metadata })));
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
}