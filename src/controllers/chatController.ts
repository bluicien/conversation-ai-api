import { Request, Response } from "express";
import { sendGeminiMessage } from "../services/chatServices";

export const chatWithGemini = async (req: Request, res: Response) => {
    const { history } = req.body;
    console.log(history)
    if (!Array.isArray(history) || history.length < 1) {
        res.status(400).json({ error: "Message is required." });
        return ;
    }

    try {
        const aiResponse = await sendGeminiMessage(history);
        res.status(200).json(aiResponse);
        return 
    } catch (error) {
        console.log(error)
        res.status(500).json({ message: "Internal Server Error" });
        return;
    }
}