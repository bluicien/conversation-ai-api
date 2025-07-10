import { GoogleGenAI } from '@google/genai';
import dotenv from 'dotenv';
dotenv.config();

const genAI = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY as string });

const websiteContentChunks = [
    { id: 'cv-summary', text: "My name is John Doe, and I am a software engineer with 5 years of experience in full-stack web development, specializing in .NET, C#, and TypeScript. I have a strong background in building scalable microservices and user-friendly web applications." },
    { id: 'cv-experience-companyA', text: "At Company A, I worked as a Senior Software Engineer for 3 years, leading the development of a new e-commerce platform using ASP.NET Core, React, and Azure. I improved system performance by 30%." },
    { id: 'cv-experience-companyB', text: "Prior to Company A, I was a Software Developer at Company B, where I developed RESTful APIs in Node.js and managed MongoDB databases for a logistics application." },
    { id: 'cv-skills', text: "My technical skills include: Languages (C#, TypeScript, JavaScript, Python), Frameworks (.NET, ASP.NET Core, Express.js, React, Angular), Databases (SQL Server, MongoDB, PostgreSQL), Cloud (Azure, AWS fundamentals), Tools (Docker, Git, Jira)." },
    { id: 'hidden-hobby', text: "Outside of work, I enjoy hiking, playing chess, and learning about quantum computing. I'm currently training for a marathon." },
    { id: 'hidden-future-plans', text: "I'm always looking for opportunities to contribute to open-source projects and am particularly interested in roles involving AI-powered applications." },
];

interface TextChunk {
    id: string;
    text: string;
    embedding?: number[] | null;
}

let embeddedChunks: TextChunk[] = websiteContentChunks.map(chunk => ({ ...chunk, embedding: null }));

export async function generateAndStoreEmbeddings() {
    console.log("Generating embeddings for website content...");
    for (let i = 0; i < embeddedChunks.length; i++) {
        const chunk = embeddedChunks[i];
        try {
            const result = await genAI.models.embedContent({
                model: "gemini-embedding-exp-03-07",
                contents: [chunk.text]
            });

            if (result.embeddings && result.embeddings.length > 0) {
                embeddedChunks[i].embedding = result.embeddings[0].values;
            } else {
                console.warn(`No embedding returned for chunk ${chunk.id}`);
                embeddedChunks[i].embedding = null;
            }
        } catch (error) {
            console.error(`Error embedding chunk ${chunk.id}: `, error);
            embeddedChunks[i].embedding = null;
        }
    }
    console.log(`Generated embeddings for ${embeddedChunks.length} chunks.`);
}

generateAndStoreEmbeddings();

export const getRelevantContext = async (query: string, topK: number = 3): Promise<string> => {
    if (embeddedChunks.length === 0 || embeddedChunks.every(c => c.embedding === null)) {
        console.warn("Embeddings not yet generated or failed for all chunks. Returning empty context.");
        return "No relevant context available.";
    }
    
    try {
        const queryEmbeddingResult = await genAI.models.embedContent({
            model: "gemini-embedding-exp-03-07",
            contents: [query],
        });

        if (!queryEmbeddingResult.embeddings || queryEmbeddingResult.embeddings.length === 0) {
            console.warn("No embedding generated for the query.");
            return "No relevant context available.";
        }

        const queryEmbedding = queryEmbeddingResult.embeddings[0].values;

        const similarities = embeddedChunks.filter(c => c.embedding !== null).map(chunk => {
            const dotProduct = (chunk.embedding as number[]).reduce((sum, val, i) => sum + val * queryEmbedding![i], 0);
            const magnitudeA = Math.sqrt((chunk.embedding as number[]).reduce((sum, val) => sum + val * val, 0));
            const magnitudeB = Math.sqrt(queryEmbedding!.reduce((sum, val) => sum + val * val, 0));
            // Handle division by zero if magnitude is zero (shouldn't happen with valid embeddings)
            const similarity = (magnitudeA === 0 || magnitudeB === 0) ? 0 : dotProduct / (magnitudeA * magnitudeB);
            return { similarity, text: chunk.text };
        });

        similarities.sort((a, b) => b.similarity - a.similarity);
        const topChunks = similarities.slice(0, topK);

        const context = topChunks.map(c => c.text).join('\n\n');
        return context;

    } catch (error) {
        console.error("Error retrieving context:", error);
        return "Error retrieving context.";
    }
}