type ChatInputProps = {
  value: string
  onChange: (value: string) => void
  onSend: () => void
  isLoading: boolean
}

export default function ChatInput({
  value,
  onChange,
  onSend,
  isLoading,
}: ChatInputProps) {
  return (
    <div className="flex items-end gap-3">
      <textarea
        value={value}
        onChange={(e) => onChange(e.target.value)}
        onKeyDown={(e) => {
          if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault()
            if (!isLoading) onSend()
          }
        }}
        placeholder="Ask something about the videos..."
        className="min-h-[56px] flex-1 resize-none rounded-xl border border-gray-300 px-4 py-3 outline-none focus:border-black"
        disabled={isLoading}
        rows={3}
      />

      <button
        onClick={onSend}
        disabled={isLoading}
        className="rounded-xl bg-black px-5 py-3 text-white disabled:opacity-50"
      >
        {isLoading ? "Thinking..." : "Send"}
      </button>
    </div>
  )
}