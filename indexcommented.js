// Import environment variables from .env file
import { config } from "dotenv";

// Import path module for working with file/directory paths
import path from "path";

// Import text splitter for dividing documents into chunks
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";

// Import OpenAI embeddings for converting text to vectors
import { OpenAIEmbeddings } from "@langchain/openai";

// Import local vector store implementation
import { HNSWLib } from "@langchain/community/vectorstores/hnswlib";

// Import directory loader for loading multiple files
import { DirectoryLoader } from "langchain/document_loaders/fs/directory";

// Import text loader for loading individual text files
import { TextLoader } from "langchain/document_loaders/fs/text";

// Import file system module for directory operations
import fs from "fs";

// Load environment variables from .env file
config();

// Define path to directory containing documents to process
const DATA_PATH = path.join(process.cwd(), "data", "books");

// Define path where vector store will be saved
const STORE_PATH = path.join(process.cwd(), "vector_store");

/**
 * Loads documents from specified directory
 * @returns {Promise<Array>} Array of loaded documents
 */
async function loadDocuments() {
  // Create directory loader that processes .md files with TextLoader
  const loader = new DirectoryLoader(DATA_PATH, {
    ".md": (path) => new TextLoader(path), // Load markdown files as text
  });
  
  // Load all documents from directory
  const docs = await loader.load();
  
  // Log number of loaded documents
  console.log(`📄 Loaded ${docs.length} documents`);
  
  return docs;
}

/**
 * Splits documents into smaller chunks
 * @param {Array} docs - Documents to split
 * @returns {Promise<Array>} Array of document chunks
 */
async function splitText(docs) {
  // Initialize text splitter with chunk size and overlap
  const splitter = new RecursiveCharacterTextSplitter({
    chunkSize: 300,      // Maximum characters per chunk
    chunkOverlap: 100,    // Characters overlapping between chunks
  });
  
  // Split documents into chunks
  const chunks = await splitter.splitDocuments(docs);
  
  // Log number of created chunks
  console.log(`✂️ Split into ${chunks.length} chunks`);
  
  return chunks;
}

/**
 * Creates and saves vector store from document chunks
 * @param {Array} chunks - Document chunks to vectorize and store
 */
async function saveToVectorStore(chunks) {
  // Create storage directory if it doesn't exist
  if (!fs.existsSync(STORE_PATH)) {
    fs.mkdirSync(STORE_PATH, { recursive: true });
  }

  // Initialize OpenAI embeddings with API key from environment
  const embeddings = new OpenAIEmbeddings({
    openAIApiKey: process.env.OPENAI_API_KEY,
  });

  console.log("Storing vectors in HNSWLib...");
  
  // Create vector store from documents and embeddings
  const vectorStore = await HNSWLib.fromDocuments(chunks, embeddings);

  console.log("Saving to disk...");
  
  // Persist vector store to disk
  await vectorStore.save(STORE_PATH);

  // Confirm successful save
  console.log(`✅ Saved to ${STORE_PATH}`);
}

/**
 * Main execution function
 */
async function main() {
  try {
    // 1. Load documents
    const documents = await loadDocuments();
    
    // 2. Split documents into chunks
    const chunks = await splitText(documents);
    
    // 3. Create and save vector store
    await saveToVectorStore(chunks);
  } catch (error) {
    // Handle any errors in the process
    console.error("❌ Error:", error.message);
  }
}

// Execute main function
main();