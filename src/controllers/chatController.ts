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
        if (typeof aiResponse !== "object" || aiResponse === null || !("reply" in aiResponse) || !("newChatHistory" in aiResponse)) {
            res.status(500).json({ message: "Unexpected response format from AI." });
            return;
        }

        if (!aiResponse.reply || !Array.isArray(aiResponse.newChatHistory) || aiResponse.newChatHistory.length === 0) {
            res.status(404).json({ message: "Failed to get response from AI."});
            return;
        }

        res.status(200).json(aiResponse);
        return 

    } catch (error) {
        console.log(error)
        res.status(500).json({ message: "Internal Server Error" });
        return;
    }
}