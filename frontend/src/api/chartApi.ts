export async function fetchChart(symbol: string, interval: string) {
  const url = `/api/chart?symbol=${encodeURIComponent(symbol)}&interval=${encodeURIComponent(interval)}`;
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Chart request failed: ${response.status}`);
  }
  return response.json();
}
