export default function CatalogueLoading() {
  return (
    <div className="mx-auto max-w-7xl px-4 py-16 sm:px-6 lg:px-8">
      <div className="animate-pulse">
        <div className="text-center mb-8 space-y-3">
          <div className="h-6 w-24 rounded bg-muted mx-auto" />
          <div className="h-10 w-80 max-w-full rounded bg-muted mx-auto" />
          <div className="h-4 w-64 max-w-full rounded bg-muted mx-auto" />
        </div>
        <div className="h-10 w-full max-w-md rounded bg-muted mx-auto mb-8" />
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 9 }).map((_, i) => (
            <div key={i} className="rounded-xl border p-6 space-y-3">
              <div className="flex gap-2">
                <div className="h-5 w-16 rounded bg-muted" />
                <div className="h-5 w-12 rounded bg-muted" />
              </div>
              <div className="h-5 w-full rounded bg-muted" />
              <div className="h-4 w-3/4 rounded bg-muted" />
              <div className="h-4 w-1/2 rounded bg-muted" />
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
