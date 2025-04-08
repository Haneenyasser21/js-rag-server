import express from "express";
import { config } from "dotenv";
import { OpenAIEmbeddings } from "@langchain/openai";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import path from "path";

config();

const app = express();
const PORT = process.env.PORT || 3000;
const STORE_PATH = path.join(process.cwd(), "vector_store");

const embeddings = new OpenAIEmbeddings({
  openAIApiKey: process.env.OPENAI_API_KEY,
});

let vectorStore;
(async () => {
  try {
    vectorStore = await HNSWLib.load(STORE_PATH, embeddings);
    console.log("Vector store loaded successfully");
  } catch (error) {
    console.error("Failed to load vector store:", error.message);
  }
})();

app.get("/search", async (req, res) => {
  try {
    const query = req.query.q;
    if (!query) return res.status(400).json({ error: "Query parameter 'q' is required" });

    const queryEmbedding = await embeddings.embedQuery(query);
    const results = await vectorStore.similaritySearchVector(queryEmbedding, 5);
    res.json(results.map((doc) => ({ content: doc.pageContent, metadata: doc.metadata })));
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

app.listen(PORT, () => {
  console.log(`Server running on port ${PORT}`);
});