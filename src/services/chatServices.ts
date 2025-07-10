import { getRelevantContext } from './../data/data';
import dotenv from "dotenv";
import { GoogleGenAI, HarmCategory, HarmBlockThreshold, Content } from "@google/genai";
import { ChatHistory } from '../models/chat';

dotenv.config();

const genAI = new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY as string });


if (!genAI) {
  console.error("Failed to connect go Gemini API");
}

export const sendGeminiMessage = async (history: ChatHistory) => {
  const MODEL_NAME = "gemini-2.0-flash"; // Gemini Model to use.

  const instructionHistory = generateSystemPrompt();
  const chatHistory = createGeminiHistoryParts(history);

  const userMessage = chatHistory.pop()?.parts[0].text;
  if (typeof userMessage !== "string") 
    throw new Error("No message found")

  const fullHistory = [...instructionHistory, ...chatHistory]


  try {
    const chatSession = genAI.chats.create({
      model: MODEL_NAME,
      history: fullHistory,
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
        ]
      },
    });
  
    const response = await chatSession.sendMessage({ message: userMessage });
    const reply = response.text;
    console.log(reply);
  
    const newChatHistory = chatSession.getHistory();
    return { reply, newChatHistory}
    
  } catch (error) {
    console.error("Error connecting with Gemini: ", error);
    return "Error connecting with Gemini";
  }
}


function generateSystemPrompt() {
  const modelInstructions = `You are a friendly chat bot. Have a casual conversation with the user.`;

  const modelAcknowledgment = "OK, I understand. I acknowledge I will act in a friendly manner and respond to user replies.";

  const instructionHistory: Content[] = [
    {
      role: "user",
      parts: [{ text: modelInstructions }],
    },
    {
      role: "model",
      parts: [{ text: modelAcknowledgment }],
    }
  ]


  return instructionHistory;
}


function createGeminiHistoryParts(messageHistory: ChatHistory) {
  const historyParts = messageHistory.map((message) => {
      return {
          role: message.role,
          parts: [{ text: message.content}]
      }
  });

  return historyParts;
}

