import { config } from "dotenv";
import fs from "fs/promises";
import path from "path";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { Chroma } from "@langchain/community/vectorstores/chroma";
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";
import { TextLoader } from "langchain/document_loaders/fs/text";
import fsExtra from "fs-extra";

// Load env variables
config();

const DATA_PATH = "C:/Users/Dell/js_project/data/books";
const CHROMA_PATH = "C:/Users/Dell/js_project/chroma";

async function main() {
  const documents = await loadDocuments();
  const chunks = await splitText(documents);
  await saveToChroma(chunks);
}

async function loadDocuments() {
  const loader = new DirectoryLoader(DATA_PATH, {
    ".md": (path) => new TextLoader(path),
  });

  const docs = await loader.load();
  console.log(`📄 Loaded ${docs.length} documents`);
  return docs;
}

async function splitText(docs) {
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 300,
    chunkOverlap: 100,
  });

  const chunks = await splitter.splitDocuments(docs);
  console.log(`✂️ Split into ${chunks.length} chunks`);

  if (chunks[0]) {
    console.log("📑 Example chunk:", chunks[0].pageContent);
    console.log("📎 Metadata:", chunks[0].metadata);
  }

  return chunks;
}

async function saveToChroma(chunks) {
  if (await fsExtra.pathExists(CHROMA_PATH)) {
    await fsExtra.remove(CHROMA_PATH);
  }

  const embeddings = new OpenAIEmbeddings({
    openAIApiKey: process.env.OPENAI_API_KEY,
  });

  const db = await Chroma.fromDocuments(chunks, embeddings, {
    collectionName: "books",
    persistDirectory: CHROMA_PATH,
  });

  await db.persist();
  console.log(`✅ Saved ${chunks.length} chunks to ${CHROMA_PATH}`);
}

main();
