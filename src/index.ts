import dotenv from "dotenv";
import express from "express";
import cors from "cors";

import chatRouter from "./routes/chatRoutes";

dotenv.config();

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());

app.get("/", (req, res) => {
    res.send("Gemini Chatbot API is running!");
});

app.use("/api", chatRouter);

app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});