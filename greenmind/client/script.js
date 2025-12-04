const API_URL = "http://localhost:3001";

function getParams() {
  return {
    layers: parseInt(document.getElementById("layers").value),
    training_time_hours: parseFloat(document.getElementById("trainingHours").value),
    flops_per_hour: parseFloat(document.getElementById("flopsHr").value)
  };
}

async function checkPrompt() {
  const role = document.getElementById("role").value;
  const context = document.getElementById("context").value;
  const expectations = document.getElementById("expectations").value;
  const params = getParams();

  document.getElementById("loading").style.display = "block";
  document.getElementById("loading").innerText = "Checking prompt…";
  document.getElementById("results").innerHTML = "";

  try {
    const res = await fetch(`${API_URL}/analyze`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ role, context, expectations, params })
    });
    const data = await res.json();

    document.getElementById("loading").style.display = "none";

    document.getElementById("results").innerHTML =
      `<h3>Check Prompt Result</h3>
       <p><b>Original prompt:</b><br>${data.original}</p>
       <p><b>Energy:</b> ${data.energy.toFixed(4)} kWh</p>
       <p><b>Features:</b> tokens ${data.features.tokens}, complexity ${data.features.complexity.toFixed(3)}, sections ${data.features.sections}</p>`;
  } catch (err) {
    document.getElementById("loading").style.display = "none";
    document.getElementById("results").innerHTML = `<p style="color:red;">Error: ${err.message}</p>`;
  }
}

async function makeItBetter() {
  const role = document.getElementById("role").value;
  const context = document.getElementById("context").value;
  const expectations = document.getElementById("expectations").value;
  const params = getParams();

  document.getElementById("loading").style.display = "block";
  document.getElementById("loading").innerText = "Making it better…";
  document.getElementById("results").innerHTML = "";

  try {
    const res = await fetch(`${API_URL}/improve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ role, context, expectations, params })
    });
    const data = await res.json();

    document.getElementById("loading").style.display = "none";

    document.getElementById("results").innerHTML =
      `<h3>Make It Better Result</h3>
       <p><b>Original prompt:</b><br>${data.original}</p>
       <p><b>Improved prompt:</b><br>${data.improved}</p>
       <p><b>Similarity:</b> ${data.similarity.toFixed(3)}</p>
       <p><b>Energy before:</b> ${data.predicted_kwh_before.toFixed(4)} kWh</p>
       <p><b>Energy after:</b> ${data.predicted_kwh_after.toFixed(4)} kWh</p>
       <p><b>Features (before → after):</b> tokens ${data.features_before.tokens} → ${data.features_after.tokens}, complexity ${data.features_before.complexity.toFixed(3)} → ${data.features_after.complexity.toFixed(3)}</p>`;
  } catch (err) {
    document.getElementById("loading").style.display = "none";
    document.getElementById("results").innerHTML = `<p style="color:red;">Error: ${err.message}</p>`;
  }
}