import { use, useEffect, useRef } from "react"
import { ChatMessage } from "@/types/chat"
import MessageBubble from "./MessageBubble"

type ChatWindowProps = {
  messages: ChatMessage[]
  isloading?: boolean
}

export default function ChatWindow({ messages, isloading = false }: ChatWindowProps) {

    const bottomRef = useRef<HTMLDivElement | null>(null)
    useEffect(() => {
        bottomRef.current?.scrollIntoView({ behavior: "smooth" })
    }, [messages, isloading])

    return (
    <div className="flex max-h-[60vh] flex-col gap-4 overflow-y-auto">
        {messages.map((message) => (
        <MessageBubble
            key={message.id}
            role={message.role}
            content={message.content}
        />
        ))}

        {isloading && (
        <MessageBubble role="assistant" content="Thinking..." />
        )}
        <div ref={bottomRef} />
    </div>
  )
}