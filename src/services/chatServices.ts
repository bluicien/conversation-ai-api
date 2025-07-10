import dotenv from "dotenv";
import { GoogleGenAI, HarmCategory, HarmBlockThreshold, Content } from "@google/genai";
import { ChatHistory } from '../models/chat'; // Corrected: Changed => to from
import * as fs from 'fs/promises'; // For file system operations
import * as path from 'path'; // For path manipulation
import pdf from 'pdf-parse'; // For parsing PDF files

dotenv.config();

const genAI = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY as string });

if (!genAI) {
    console.error("Failed to connect to Gemini API");
}

// --- Embedding Model Configuration ---
const EMBEDDING_MODEL_NAME = "gemini-embedding-exp-03-07";

// --- Data Structure for Embedded Documents ---
interface EmbeddedDocument {
    id: string; // Unique identifier for the document
    content: string; // The original text content of the document
    embedding: number[]; // The embedding vector
}

// In-memory store for embedded documents. In a real application, this would be a vector database.
const knowledgeBase: EmbeddedDocument[] = [];

// --- Function to Load and Embed Documents ---
/**
 * Loads documents from a specified directory, processes their content,
 * and generates embeddings for each document. Now includes PDF handling.
 * @param directoryPath The path to the directory containing markdown/json/pdf files.
 */
export const loadAndEmbedDocuments = async (directoryPath: string) => {
    console.log(`Loading and embedding documents from: ${directoryPath}`);
    try {
        const files = await fs.readdir(directoryPath);

        for (const file of files) {
            const filePath = path.join(directoryPath, file);
            const fileExtension = path.extname(file).toLowerCase();
            let content = '';

            // Read file content based on its type
            if (fileExtension === '.md' || fileExtension === '.json') {
                content = await fs.readFile(filePath, 'utf-8');
                if (fileExtension === '.json') {
                    try {
                        const jsonData = JSON.parse(content);
                        content = JSON.stringify(jsonData, null, 2); // Pretty print for embedding context
                    } catch (jsonError) {
                        console.warn(`Could not parse JSON file ${file}:`, jsonError);
                        // Fallback to raw content if parsing fails
                    }
                }
            } else if (fileExtension === '.pdf') {
                try {
                    const dataBuffer = await fs.readFile(filePath);
                    const pdfData = await pdf(dataBuffer);
                    content = pdfData.text; // Extract text content from PDF
                    console.log(`Successfully extracted text from PDF: ${file}`);
                } catch (pdfError) {
                    console.error(`Error parsing PDF file ${file}:`, pdfError);
                    continue; // Skip this file if PDF parsing fails
                }
            } else {
                console.log(`Skipping unsupported file type: ${file}`);
                continue;
            }

            if (content.trim().length === 0) {
                console.warn(`Skipping empty file: ${file}`);
                continue;
            }

            // Generate embedding for the document content
            const result = await genAI.models.embedContent({
                model: EMBEDDING_MODEL_NAME,
                contents: content, // Pass content as a string directly
                config: {
                    taskType: "SEMANTIC_SIMILARITY",
                }
            });
            
            // Corrected: Access embedding values from result.embeddings[0].values
            const embedding = result.embeddings?.[0]?.values as number[] | undefined;

            if (embedding && embedding.length > 0) {
                knowledgeBase.push({
                    id: file, // Using filename as ID for simplicity
                    content: content,
                    embedding: embedding,
                });
                console.log(`Embedded document: ${file}`);
            } else {
                console.warn(`Failed to generate embedding for ${file}. Embedding result was:`, result); // Log the full result for debugging
            }
        }
        console.log(`Finished embedding ${knowledgeBase.length} documents.`);
    } catch (error) {
        console.error("Error loading and embedding documents:", error);
    }
};

// --- Cosine Similarity Calculation ---
/**
 * Calculates the cosine similarity between two vectors.
 * @param vecA First vector.
 * @param vecB Second vector.
 * @returns Cosine similarity score.
 */
function cosineSimilarity(vecA: number[], vecB: number[]): number {
    if (vecA.length !== vecB.length) {
        throw new Error("Vectors must be of the same length to calculate cosine similarity.");
    }

    let dotProduct = 0;
    let magnitudeA = 0;
    let magnitudeB = 0;

    for (let i = 0; i < vecA.length; i++) {
        dotProduct += vecA[i] * vecB[i];
        magnitudeA += vecA[i] * vecA[i];
        magnitudeB += vecB[i] * vecB[i];
    }

    magnitudeA = Math.sqrt(magnitudeA);
    magnitudeB = Math.sqrt(magnitudeB);

    if (magnitudeA === 0 || magnitudeB === 0) {
        return 0; // Avoid division by zero
    }

    return dotProduct / (magnitudeA * magnitudeB);
}

// --- Function to Find Relevant Documents ---
/**
 * Finds the most relevant documents from the knowledge base based on a query.
 * @param query The user's query string.
 * @param topN The number of top relevant documents to return.
 * @returns An array of relevant EmbeddedDocument objects.
 */
async function findRelevantDocuments(query: string, topN: number = 3): Promise<EmbeddedDocument[]> {
    if (knowledgeBase.length === 0) {
        console.warn("Knowledge base is empty. No documents to retrieve.");
        return [];
    }

    // Corrected: Call embedContent directly on genAI.models and pass model/contents/config
    const queryResult = await genAI.models.embedContent({
        model: EMBEDDING_MODEL_NAME,
        contents: query, // Pass query as a string directly
        config: {
            taskType: "SEMANTIC_SIMILARITY",
        }
    });
    // Corrected: Access embedding values from queryResult.embeddings[0].values
    const queryEmbedding = queryResult.embeddings?.[0]?.values;

    if (!queryEmbedding || queryEmbedding.length === 0) {
        console.error("Failed to generate embedding for the query.");
        return [];
    }

    // Calculate similarity for all documents
    const similarities = knowledgeBase.map(doc => ({
        doc,
        similarity: cosineSimilarity(queryEmbedding, doc.embedding),
    }));

    // Sort by similarity in descending order and take the top N
    similarities.sort((a, b) => b.similarity - a.similarity);

    // Filter out documents with very low similarity if desired
    const relevantDocs = similarities
        .filter(item => item.similarity > 0.5) // Example threshold, adjust as needed
        .slice(0, topN)
        .map(item => item.doc);

    console.log(`Found ${relevantDocs.length} relevant documents for query.`);
    return relevantDocs;
}

// --- Main Function to Send Message to Gemini with Context ---
export const sendGeminiMessage = async (history: ChatHistory) => {
    const MODEL_NAME = "gemini-2.0-flash"; // Gemini Model to use.

    const instructionHistory = generateSystemPrompt();
    const chatHistory = createGeminiHistoryParts(history);

    // Corrected: Check if chatHistory is empty before pop()
    if (chatHistory.length === 0) {
        throw new Error("Chat history is empty. Cannot extract user message.");
    }

    const userMessagePart = chatHistory[chatHistory.length - 1].parts![0]; // Access last element without pop
    if (!userMessagePart || typeof userMessagePart.text !== "string") {
        throw new Error("No valid user message found in chat history.");
    }
    const userMessage = userMessagePart.text;

    // Remove the last message (user's current message) from history for context retrieval
    // This ensures the model doesn't see the current question twice (once in history, once with context)
    const historyWithoutCurrentMessage = chatHistory.slice(0, chatHistory.length -1);


    // --- Retrieve relevant documents based on the user's message ---
    const relevantDocuments = await findRelevantDocuments(userMessage);

    // --- Construct context from relevant documents ---
    let contextString = "";
    if (relevantDocuments.length > 0) {
        contextString = "Here is some relevant information from the knowledge base:\n\n";
        relevantDocuments.forEach((doc, index) => {
            contextString += `--- Document ${index + 1} (${doc.id}) ---\n${doc.content}\n\n`;
        });
        contextString += "Please use the above information to answer the user's question if relevant.\n\n";
    }

    // Prepend the context to the full history
    const fullHistory: Content[] = [
        ...instructionHistory,
        ...(contextString ? [{ role: "user", parts: [{ text: contextString }] }] : []), // Add context as a user message
        ...historyWithoutCurrentMessage, // Original chat history without the current message
    ];

    try {
        // Corrected: Use genAI.chats.create directly for chat models, as per documentation
        const chatSession = genAI.chats.create({
            model: MODEL_NAME,
            history: fullHistory, // Pass the full history including context
            config: {
                maxOutputTokens: 500,
                temperature: 0.7,
                safetySettings: [
                    {
                        category: HarmCategory.HARM_CATEGORY_HARASSMENT,
                        threshold: HarmBlockThreshold.BLOCK_NONE,
                    },
                    {
                        category: HarmCategory.HARM_CATEGORY_HATE_SPEECH,
                        threshold: HarmBlockThreshold.BLOCK_NONE,
                    },
                    {
                        category: HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                        threshold: HarmBlockThreshold.BLOCK_NONE,
                    },
                    {
                        category: HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                        threshold: HarmBlockThreshold.BLOCK_NONE,
                    },
                ],
            },
        });

        const response = await chatSession.sendMessage({ message: userMessage }); 
        const reply = response.text;
        console.log(reply);

        const newChatHistory = chatSession.getHistory(); 

        return { reply, newChatHistory };

    } catch (error) {
        console.error("Error connecting with Gemini:", error);
        return "Error connecting with Gemini";
    }
};

// --- Helper Functions (from your original file) ---
function generateSystemPrompt(): Content[] {
    const modelInstructions = `You are a friendly chat bot. Have a casual conversation with the user. Answer questions based on the provided context if available. If the context does not contain the answer, state that you don't know.`;

    const modelAcknowledgment = "OK, I understand. I acknowledge I will act in a friendly manner and respond to user replies, using provided context when relevant.";

    const instructionHistory: Content[] = [
        {
            role: "user",
            parts: [{ text: modelInstructions }],
        },
        {
            role: "model",
            parts: [{ text: modelAcknowledgment }],
        }
    ];
    return instructionHistory;
}

function createGeminiHistoryParts(messageHistory: ChatHistory): Content[] {
    const historyParts = messageHistory.map((message) => {
        return {
            role: message.role,
            parts: [{ text: message.content }]
        };
    });
    return historyParts;
}

