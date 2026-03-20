"use client"

import { useState } from "react"
import ChatWindow from "@/components/ChatWindow"
import ChatInput from "@/components/ChatInput"
import DebugDrawer from "@/components/DebugDrawer"
import { ChatMessage, ChatResponse, DebugInfo } from "@/types/chat"

function makeId() {
  return `${Date.now()}-${Math.random().toString(36).slice(2, 9)}`
}

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [debugInfo, setDebugInfo] = useState<DebugInfo | null>(null)
  const [isDebugOpen, setIsDebugOpen] = useState(false)

  const sendQuery = async () => {
    const trimmed = input.trim()

    if (!trimmed || isLoading) return

    const userMessage: ChatMessage = {
      id: makeId(),
      role: "user",
      content: trimmed,
    }

    setMessages((prev) => [...prev, userMessage])
    setInput("")
    setError(null)
    setIsLoading(true)

    try {
      const res = await fetch("http://localhost:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          query: trimmed,
          debug: true,
        }),
      })

      if (!res.ok) {
        throw new Error(`Request failed with status ${res.status}`)
      }

      const data: ChatResponse = await res.json()

      const assistantMessage: ChatMessage = {
        id: makeId(),
        role: "assistant",
        content: data.answer ?? "No answer returned.",
      }

      setMessages((prev) => [...prev, assistantMessage])
      setDebugInfo(data.debug ?? null)
    } catch (err) {
      console.error(err)
      setError("Something went wrong while contacting the inference server.")

      const assistantMessage: ChatMessage = {
        id: makeId(),
        role: "assistant",
        content: "I could not retrieve an answer right now.",
      }

      setMessages((prev) => [...prev, assistantMessage])
      setDebugInfo(null)
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <main className="min-h-screen bg-gray-50 px-4 py-8">
      <div className="mx-auto flex max-w-4xl flex-col gap-6">
        <header>
          <h1 className="text-2xl font-bold">Gameplay QA</h1>
          <p className="mt-1 text-sm text-gray-600">
            Ask questions about the processed videos.
          </p>
        </header>

        <section className="rounded-2xl border border-gray-200 bg-white p-4 shadow-sm">
          {messages.length === 0 ? (
            <p className="text-sm text-gray-500">
              Start by asking a question about a processed video.
            </p>
          ) : (
          <ChatWindow messages={messages} isloading={isLoading} />
          )}
        </section>

        {error && (
          <div className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">
            {error}
          </div>
        )}

        <ChatInput
          value={input}
          onChange={setInput}
          onSend={sendQuery}
          isLoading={isLoading}
        />

        <DebugDrawer
          isOpen={isDebugOpen}
          onToggle={() => setIsDebugOpen((prev) => !prev)}
          debugInfo={debugInfo}
        />
      </div>
    </main>
  )
}