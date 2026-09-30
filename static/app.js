const state = { location: 'Mangaluru', fish: 'Mackerel', demand_index: 650, catch_volume_kg: 150, buyers_count: 130, quantity_sold_kg: 105, festival: 0, buy_cost_per_kg: 0, operating_cost_per_kg: 35 };
const $ = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) throw new Error(await response.text());
  return response.json();
}

function setOptions(select, values) {
  select.innerHTML = values.map((value) => `<option value="${value}">${value}</option>`).join('');
}

function drawChart(history) {
  const svg = $('priceChart');
  if (!history.length) { svg.innerHTML = '<text x="30" y="130" class="chart-label">No historical rows for this selection</text>'; return; }
  const width = 900; const height = 260; const pad = 28;
  const prices = history.map((row) => row.price_per_kg); const min = Math.min(...prices) * 0.97; const max = Math.max(...prices) * 1.03;
  const points = prices.map((price, index) => `${pad + index * ((width - pad * 2) / Math.max(prices.length - 1, 1))},${height - pad - ((price - min) / (max - min || 1)) * (height - pad * 2)}`).join(' ');
  const area = `${pad},${height - pad} ${points} ${width - pad},${height - pad}`;
  const yLabels = [min, (min + max) / 2, max].map((value, index) => `<text x="0" y="${height - pad - index * ((height - pad * 2) / 2) + 4}" class="chart-label">₹${Math.round(value)}</text>`).join('');
  svg.innerHTML = `<defs><linearGradient id="chartFill" x1="0" y1="0" x2="0" y2="1"><stop offset="0" stop-color="#0d6872" stop-opacity=".2"/><stop offset="1" stop-color="#0d6872" stop-opacity="0"/></linearGradient></defs><line x1="${pad}" y1="${height - pad}" x2="${width}" y2="${height - pad}" class="chart-grid"/><line x1="${pad}" y1="${pad}" x2="${width}" y2="${pad}" class="chart-grid"/><line x1="${pad}" y1="${height / 2}" x2="${width}" y2="${height / 2}" class="chart-grid"/><polygon points="${area}" class="chart-area"/><polyline points="${points}" class="chart-line"/>${yLabels}`;
}

function updateWeather(weather) {
  $('weatherTemperature').textContent = weather.temperature_c.toFixed(1); $('temperatureValue').textContent = `${weather.temperature_c.toFixed(1)}°C`;
  $('rainfallValue').textContent = `${weather.rainfall_mm.toFixed(1)} mm`; $('windValue').textContent = `${weather.wind_speed_kmh.toFixed(1)} km/h`;
  $('humidityValue').textContent = `${weather.humidity_pct.toFixed(0)}%`; $('waveValue').textContent = `${weather.wave_height_m.toFixed(1)} m`;
}

async function refreshDashboard() {
  const data = await api(`/api/dashboard?location=${encodeURIComponent(state.location)}&fish=${encodeURIComponent(state.fish)}`);
  drawChart(data.history); updateWeather(data.weather);
}

function renderEconomics(economics, risk, supply = null) {
  $('revenueValue').textContent = `₹${Math.round(economics.revenue).toLocaleString()}`;
  $('costValue').textContent = `₹${Math.round(economics.total_cost).toLocaleString()}`;
  $('profitValue').textContent = `₹${Math.round(economics.profit).toLocaleString()}`;
  $('marginValue').textContent = `${economics.margin_pct.toFixed(1)}%`;
  $('actionBadge').textContent = economics.action;
  $('decisionText').textContent = `${risk.label} weather risk · ${economics.quantity_kg.toLocaleString()} kg scenario`;
  if (supply !== null) $('supplyValue').textContent = `${Math.round(supply).toLocaleString()} kg`;
}

async function loadInsights() {
  const result = await api('/api/insights', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(state) });
  renderEconomics(result.selected.economics, result.selected.weather_risk, result.selected.supply_prediction_kg);
  $('alertCount').textContent = result.alerts.length;
  $('alertsList').innerHTML = result.alerts.length ? result.alerts.map((alert) => `<div class="alert-item ${alert.severity}"><span>${alert.severity === 'positive' ? '↑' : '!'}</span><div><strong>${alert.title}</strong>${alert.message}</div></div>`).join('') : '<div class="empty-state">No active alerts for this scenario.</div>';
  $('marketCards').innerHTML = result.locations.map((item, index) => `<div class="market-card ${index === 0 ? 'best' : ''}"><div class="market-name">${item.location}<span>${index === 0 ? 'BEST' : `${item.heat_score} heat`}</span></div><div class="heat-bar"><i style="width:${item.heat_score}%"></i></div><div class="market-meta"><span>Price <strong>₹${Math.round(item.predicted_price)}</strong></span><span>Profit <strong>₹${Math.round(item.profit).toLocaleString()}</strong></span></div></div>`).join('');
  const maxImportance = Math.max(...result.explanation.map((item) => item.importance), 1);
  $('explanationList').innerHTML = result.explanation.map((item) => `<div class="explain-row"><span>${item.feature}</span><div class="explain-bar"><i style="width:${(item.importance / maxImportance) * 100}%"></i></div><strong>${item.importance.toFixed(1)}%</strong></div>`).join('');
}

async function predict() {
  $('predictButton').disabled = true; $('predictButton').textContent = 'Calculating...';
  try {
    const result = await api('/api/predict', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(state) });
    $('predictedPrice').textContent = `₹${Math.round(result.predicted_price).toLocaleString()}/kg`;
    $('predictionRange').textContent = `Expected ₹${Math.round(result.lower)}–₹${Math.round(result.upper)}/kg`;
    $('rangeValue').textContent = `₹${Math.round(result.lower)} – ₹${Math.round(result.upper)}/kg`;
    $('recommendationText').textContent = `${result.economics.action}: ${state.fish} in ${state.location} at ${state.demand_index}/1000 demand.`;
    $('modelMae').textContent = `₹${Math.round(result.metrics.mae)} MAE`; $('modelRows').textContent = `${result.metrics.train_rows.toLocaleString()} training rows`;
    updateWeather(result.weather); renderEconomics(result.economics, result.weather_risk); await loadInsights();
  } catch (error) { $('recommendationText').textContent = 'Prediction service unavailable. Check the Flask server.'; }
  finally { $('predictButton').disabled = false; $('predictButton').innerHTML = 'Run prediction <span>↗</span>'; }
}

async function init() {
  try {
    const metadata = await api('/api/metadata'); setOptions($('locationSelect'), metadata.locations); setOptions($('fishSelect'), metadata.fish);
    $('locationSelect').value = state.location; $('fishSelect').value = state.fish; await refreshDashboard(); await predict();
  } catch (error) { $('recommendationText').textContent = 'Connect the Flask server to load the model.'; }
}

$('locationSelect').addEventListener('change', (event) => { state.location = event.target.value; refreshDashboard(); predict(); });
$('fishSelect').addEventListener('change', (event) => { state.fish = event.target.value; refreshDashboard(); predict(); });
$('demandInput').addEventListener('input', (event) => { state.demand_index = Number(event.target.value); $('demandOutput').textContent = `${state.demand_index}/1000`; $('demandValue').textContent = `${state.demand_index}/1000`; });
$('volumeInput').addEventListener('input', (event) => { state.catch_volume_kg = Math.min(Math.max(Number(event.target.value), 10), 10000); event.target.value = state.catch_volume_kg; });
$('buyersInput').addEventListener('input', (event) => { state.buyers_count = Math.min(Math.max(Number(event.target.value), 0), 10000); });
$('soldInput').addEventListener('input', (event) => { state.quantity_sold_kg = Math.min(Math.max(Number(event.target.value), 0), state.catch_volume_kg); });
$('festivalInput').addEventListener('change', (event) => { state.festival = event.target.checked ? 1 : 0; });
$('buyCostInput').addEventListener('input', (event) => { state.buy_cost_per_kg = Number(event.target.value); });
$('operatingCostInput').addEventListener('input', (event) => { state.operating_cost_per_kg = Number(event.target.value); });
$('predictButton').addEventListener('click', predict); $('refreshButton').addEventListener('click', () => { refreshDashboard(); predict(); });
$('saveInputsButton').addEventListener('click', async () => { const button = $('saveInputsButton'); button.disabled = true; try { await api('/api/supply', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ location: state.location, fish: state.fish, available_volume_kg: state.catch_volume_kg, expected_arrival: new Date().toISOString().slice(0, 10) }) }); await api('/api/demand', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ location: state.location, fish: state.fish, buyers_count: state.buyers_count, requested_volume_kg: state.quantity_sold_kg, needed_by: new Date().toISOString().slice(0, 10), festival: Boolean(state.festival) }) }); button.textContent = 'Saved to database'; setTimeout(() => { button.textContent = 'Save supply + buyer inputs'; }, 1800); } catch (error) { button.textContent = 'Save failed'; } finally { button.disabled = false; } });
$('locationButton').addEventListener('click', () => { if (!navigator.geolocation) { $('recommendationText').textContent = 'Browser location is not available.'; return; } $('locationButton').textContent = 'Locating...'; navigator.geolocation.getCurrentPosition(async (position) => { try { const result = await api(`/api/nearest-location?lat=${position.coords.latitude}&lon=${position.coords.longitude}`); state.location = result.location; $('locationSelect').value = result.location; await refreshDashboard(); await predict(); } catch (error) { $('recommendationText').textContent = 'Could not resolve your nearest market.'; } finally { $('locationButton').textContent = '⌖ Use my location'; } }, () => { $('recommendationText').textContent = 'Location permission was not granted.'; $('locationButton').textContent = '⌖ Use my location'; }); });
document.querySelectorAll('[data-scroll]').forEach((button) => button.addEventListener('click', () => document.getElementById(button.dataset.scroll).scrollIntoView({behavior:'smooth'})));
document.querySelectorAll('.nav-item').forEach((button) => button.addEventListener('click', () => document.getElementById(button.dataset.view).scrollIntoView({behavior:'smooth'})));
init();
