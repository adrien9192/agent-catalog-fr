export default function UseCaseLoading() {
  return (
    <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
      <div className="animate-pulse">
        <div className="h-4 w-32 rounded bg-muted mb-6" />
        <div className="h-6 w-24 rounded bg-muted mb-3" />
        <div className="h-10 w-3/4 max-w-2xl rounded bg-muted mb-3" />
        <div className="h-5 w-2/3 max-w-xl rounded bg-muted mb-8" />

        <div className="grid gap-8 lg:grid-cols-3">
          <div className="lg:col-span-2 space-y-6">
            <div className="rounded-xl border p-6 space-y-3">
              <div className="h-6 w-48 rounded bg-muted" />
              <div className="h-4 w-full rounded bg-muted" />
              <div className="h-4 w-5/6 rounded bg-muted" />
              <div className="h-4 w-4/6 rounded bg-muted" />
            </div>
            <div className="rounded-xl border p-6 space-y-3">
              <div className="h-6 w-40 rounded bg-muted" />
              <div className="h-4 w-full rounded bg-muted" />
              <div className="h-4 w-3/4 rounded bg-muted" />
            </div>
          </div>
          <div className="space-y-4">
            <div className="rounded-xl border p-6 space-y-3">
              <div className="h-5 w-32 rounded bg-muted" />
              <div className="h-4 w-full rounded bg-muted" />
              <div className="h-4 w-2/3 rounded bg-muted" />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
