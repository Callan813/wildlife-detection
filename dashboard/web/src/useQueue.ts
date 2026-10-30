import { useCallback, useEffect, useRef, useState } from "react";

// Keeps a small batch of queue items in memory and refills it from the server, so the
// next item shows up instantly after each decision.
export function useQueue<T extends { id: number }>(load: () => Promise<T[]>) {
  const [items, setItems] = useState<T[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const done = useRef(new Set<number>());

  const refill = useCallback(async () => {
    setLoading(true);
    try {
      const fresh = await load();
      // The server may not have seen a just-submitted review yet; don't show it twice.
      setItems(fresh.filter((i) => !done.current.has(i.id)));
      setError(null);
    } catch (e) {
      setError((e as Error).message);
    } finally {
      setLoading(false);
    }
  }, [load]);

  useEffect(() => {
    void refill();
  }, [refill]);

  const next = useCallback(
    (id: number) => {
      done.current.add(id);
      setItems((prev) => {
        const rest = prev.filter((i) => i.id !== id);
        if (rest.length < 3) void refill();
        return rest;
      });
    },
    [refill],
  );

  return { current: items[0] as T | undefined, remaining: items.length, loading, error, next };
}
