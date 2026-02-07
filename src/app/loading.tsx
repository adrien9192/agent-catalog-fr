export default function Loading() {
  return (
    <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
      <div className="animate-pulse space-y-6">
        <div className="h-4 w-48 rounded bg-muted" />
        <div className="h-10 w-96 max-w-full rounded bg-muted" />
        <div className="h-4 w-80 max-w-full rounded bg-muted" />
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3 mt-8">
          {Array.from({ length: 6 }).map((_, i) => (
            <div key={i} className="rounded-xl border p-6 space-y-3">
              <div className="h-4 w-20 rounded bg-muted" />
              <div className="h-5 w-full rounded bg-muted" />
              <div className="h-4 w-3/4 rounded bg-muted" />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
