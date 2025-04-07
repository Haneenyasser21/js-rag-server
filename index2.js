import { config } from "dotenv";
import path from "path";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { TextLoader } from "langchain/document_loaders/fs/text";
import fs from "fs";

config();

const DATA_PATH = path.join(process.cwd(), "data", "books");
const STORE_PATH = path.join(process.cwd(), "vector_store");

// 1. Load documents
async function loadDocuments() {
  const loader = new DirectoryLoader(DATA_PATH, {
    ".md": (path) => new TextLoader(path),
  });
  const docs = await loader.load();
  console.log(`📄 Loaded ${docs.length} documents`);
  return docs;
}

// 2. Split text into chunks
async function splitText(docs) {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 300,
    chunkOverlap: 100,
  });
  const chunks = await splitter.splitDocuments(docs);
  console.log(`✂️ Split into ${chunks.length} chunks`);
  return chunks;
}

// 3. Save to HNSWLib
async function saveToVectorStore(chunks) {
  if (!fs.existsSync(STORE_PATH)) {
    fs.mkdirSync(STORE_PATH, { recursive: true });
  }

  const embeddings = new OpenAIEmbeddings({
    openAIApiKey: process.env.OPENAI_API_KEY,
  });

  console.log("Storing vectors in HNSWLib...");
  const vectorStore = await HNSWLib.fromDocuments(chunks, embeddings);

  console.log("Saving to disk...");
  await vectorStore.save(STORE_PATH);

  console.log(`✅ Saved to ${STORE_PATH}`);
}

// 4. Main function (called last)
async function main() {
  try {
    const documents = await loadDocuments();
    const chunks = await splitText(documents);
    await saveToVectorStore(chunks);
  } catch (error) {
    console.error("❌ Error:", error.message);
  }
}

main();