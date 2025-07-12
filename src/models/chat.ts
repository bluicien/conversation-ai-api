export type Message = {
    role: 'user' | 'model';
    content: string;
};

export type ChatHistory = Array<Message>;