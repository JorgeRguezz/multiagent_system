import { DebugInfo } from "@/types/chat"

type DebugDrawerProps = {
  isOpen: boolean
  onToggle: () => void
  debugInfo: DebugInfo | null
}

export default function DebugDrawer({
  isOpen,
  onToggle,
  debugInfo,
}: DebugDrawerProps) {
  return (
    <div className="mt-4 rounded-2xl border border-gray-200 bg-white shadow-sm">
      <button
        onClick={onToggle}
        className="w-full px-4 py-3 text-left font-medium"
      >
        {isOpen ? "Hide debug details" : "Show debug details"}
      </button>

      {isOpen && (
        <div className="border-t border-gray-200 p-4 space-y-6 text-sm">
          {!debugInfo ? (
            <p className="text-gray-500">No debug information available yet.</p>
          ) : (
            <>
              <section>
                <h3 className="mb-2 font-semibold">Confidence</h3>
                <p>Score: {debugInfo.confidence ?? "N/A"}</p>
                <p>Band: {debugInfo.confidence_band ?? "N/A"}</p>
              </section>

              <section>
                <h3 className="mb-2 font-semibold">Intent</h3>
                <pre className="overflow-x-auto rounded-lg bg-gray-50 p-3">
                  {JSON.stringify(debugInfo.intent ?? {}, null, 2)}
                </pre>
              </section>

              <section>
                <h3 className="mb-2 font-semibold">Retrieval counts</h3>
                <pre className="overflow-x-auto rounded-lg bg-gray-50 p-3">
                  {JSON.stringify(debugInfo.retrieval_counts ?? {}, null, 2)}
                </pre>
              </section>

              <section>
                <h3 className="mb-2 font-semibold">Generation</h3>
                <p>
                  <strong>Final marker found:</strong>{" "}
                  {debugInfo.generation?.has_final_marker ? "Yes" : "No"}
                </p>
                <p>
                  <strong>Verifier enabled:</strong>{" "}
                  {debugInfo.generation?.verifier_enabled ? "Yes" : "No"}
                </p>
                <p>
                  <strong>Final answer source:</strong>{" "}
                  {debugInfo.generation?.final_answer_source ?? "N/A"}
                </p>
                <div className="mt-3 space-y-3">
                  <div>
                    <p className="mb-1 font-medium">Thoughts</p>
                    <pre className="overflow-x-auto rounded-lg bg-gray-50 p-3 whitespace-pre-wrap">
                      {debugInfo.generation?.thoughts || "No thought tokens captured."}
                    </pre>
                  </div>
                  <div>
                    <p className="mb-1 font-medium">Raw generated answer</p>
                    <pre className="overflow-x-auto rounded-lg bg-gray-50 p-3 whitespace-pre-wrap">
                      {debugInfo.generation?.answer_raw || "No generated answer captured."}
                    </pre>
                  </div>
                </div>
              </section>

              <section>
                <h3 className="mb-2 font-semibold">Evidence</h3>
                <div className="space-y-3">
                  {(debugInfo.evidence ?? []).length === 0 ? (
                    <p className="text-gray-500">No evidence returned.</p>
                  ) : (
                    debugInfo.evidence?.map((item, index) => (
                      <div
                        key={`${item.chunk_id ?? "evidence"}-${index}`}
                        className="rounded-xl border border-gray-200 p-3"
                      >
                        <p><strong>Video:</strong> {item.video_name ?? "N/A"}</p>
                        <p><strong>Time:</strong> {item.time_span ?? "N/A"}</p>
                        <p><strong>Source:</strong> {item.source ?? "N/A"}</p>
                        <p><strong>Chunk:</strong> {item.chunk_id ?? "N/A"}</p>
                        <p className="mt-2 whitespace-pre-wrap text-gray-700">
                          {item.text ?? ""}
                        </p>
                      </div>
                    ))
                  )}
                </div>
              </section>

              <section>
                <h3 className="mb-2 font-semibold">Verification</h3>
                <pre className="overflow-x-auto rounded-lg bg-gray-50 p-3">
                  {JSON.stringify(debugInfo.verification ?? {}, null, 2)}
                </pre>
              </section>
            </>
          )}
        </div>
      )}
    </div>
  )
}
