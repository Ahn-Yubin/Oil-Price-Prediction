export async function fetchForecast(symbol: string, interval: string, models = "") {
  const modelQuery = models ? `&models=${encodeURIComponent(models)}` : "";
  const url = `/api/forecast?symbol=${encodeURIComponent(symbol)}&interval=${encodeURIComponent(interval)}${modelQuery}`;
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) {
    throw new Error(`Forecast request failed: ${response.status}`);
  }
  return response.json();
}
