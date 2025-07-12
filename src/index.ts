import dotenv from "dotenv";
import express from "express";
import cors from "cors";
import * as path from 'path';

import chatRouter from "./routes/chatRoutes";
import { loadAndEmbedDocuments } from "./services/chatServices";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

const corsOptions = {
    origin: process.env.CLIENT_URL || "http://localhost:5173",
    methods: ['GET', 'POST']
}

app.use(cors(corsOptions));
app.use(express.json());

startKnowledgeBase();

app.get("/", (req, res) => {
    res.send("Gemini Chatbot API is running!");
});

app.use("/api", chatRouter);

app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});


async function startKnowledgeBase() {
    const knowledgeBaseDirectory = path.join(__dirname, 'knowledge');
    await loadAndEmbedDocuments(knowledgeBaseDirectory);
}
