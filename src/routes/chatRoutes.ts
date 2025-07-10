import { Router } from 'express';
import { chatWithGemini } from '../controllers/chatController';


const router = Router();

router.post("/chat", chatWithGemini);

export default router;