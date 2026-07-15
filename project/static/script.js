const fileInput = document.getElementById("csvFile");
const uploadBtn = document.getElementById("uploadBtn");
const analyzeBtn = document.getElementById("analyzeBtn");
const loadingEl = document.getElementById("loading");
const messageEl = document.getElementById("message");

const tableHead = document.querySelector("#resultsTable thead");
const tableBody = document.querySelector("#resultsTable tbody");

let clusterChart;
let scatterChart;
let importanceChart;

function setMessage(text, type = "success") {
  messageEl.textContent = text;
  messageEl.className = `message ${type}`;
}

function toggleLoading(show) {
  loadingEl.classList.toggle("hidden", !show);
}

async function parseJsonResponse(response) {
  const data = await response.json();
  if (!response.ok || data.success === false) {
    throw new Error(data.error || "Request failed.");
  }
  return data;
}

uploadBtn.addEventListener("click", async () => {
  if (!fileInput.files.length) {
    setMessage("Please select a CSV file first.", "error");
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  try {
    toggleLoading(true);
    const response = await fetch("/upload", {
      method: "POST",
      body: formData,
    });
    const data = await parseJsonResponse(response);
    setMessage(data.message, "success");
  } catch (error) {
    setMessage(error.message, "error");
  } finally {
    toggleLoading(false);
  }
});

analyzeBtn.addEventListener("click", async () => {
  try {
    toggleLoading(true);
    setMessage("Running analysis...", "success");

    const processRes = await fetch("/process", { method: "POST" });
    const processData = await parseJsonResponse(processRes);

    const [resultsRes, importanceRes] = await Promise.all([
      fetch("/results"),
      fetch("/importance"),
    ]);

    const resultsData = await parseJsonResponse(resultsRes);
    const importanceData = await parseJsonResponse(importanceRes);

    renderClusterChart(resultsData.cluster_distribution || []);
    renderScatterChart(resultsData.scatter_points || []);
    renderImportanceChart(importanceData.features || []);
    renderTable(resultsData.table_columns || [], resultsData.table_rows || []);

    const sil = processData.summary?.silhouette_score;
    setMessage(
      `Analysis complete. Rows: ${processData.summary?.rows ?? 0}, Silhouette Score: ${sil ?? "N/A"}`,
      "success"
    );
  } catch (error) {
    setMessage(error.message, "error");
  } finally {
    toggleLoading(false);
  }
});

function destroyChart(chartInstance) {
  if (chartInstance) {
    chartInstance.destroy();
  }
}

function renderClusterChart(clusterDistribution) {
  const labels = clusterDistribution.map((item) => item.cluster);
  const values = clusterDistribution.map((item) => item.count);

  destroyChart(clusterChart);
  clusterChart = new Chart(document.getElementById("clusterChart"), {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Users per Cluster",
        data: values,
        backgroundColor: "rgba(37, 99, 235, 0.7)",
      }],
    },
    options: {
      responsive: true,
      scales: { y: { beginAtZero: true } },
    },
  });
}

function renderScatterChart(scatterPoints) {
  const grouped = {};
  scatterPoints.forEach((point) => {
    if (!grouped[point.cluster]) grouped[point.cluster] = [];
    grouped[point.cluster].push({ x: point.x, y: point.y });
  });

  const colors = ["#2563eb", "#16a34a", "#ea580c", "#7c3aed", "#dc2626"];
  const datasets = Object.keys(grouped).map((cluster, index) => ({
    label: cluster,
    data: grouped[cluster],
    backgroundColor: colors[index % colors.length],
  }));

  destroyChart(scatterChart);
  scatterChart = new Chart(document.getElementById("scatterChart"), {
    type: "scatter",
    data: { datasets },
    options: {
      responsive: true,
      scales: {
        x: { title: { display: true, text: "Engagement Score" } },
        y: { title: { display: true, text: "Items Purchased" } },
      },
    },
  });
}

function renderImportanceChart(features) {
  const labels = features.map((item) => item.feature);
  const values = features.map((item) => item.importance);

  destroyChart(importanceChart);
  importanceChart = new Chart(document.getElementById("importanceChart"), {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Mean |SHAP| Importance",
        data: values,
        backgroundColor: "rgba(22, 163, 74, 0.7)",
      }],
    },
    options: {
      responsive: true,
      indexAxis: "y",
      scales: { x: { beginAtZero: true } },
    },
  });
}

function renderTable(columns, rows) {
  tableHead.innerHTML = "";
  tableBody.innerHTML = "";

  if (!columns.length || !rows.length) {
    return;
  }

  const headerRow = document.createElement("tr");
  columns.forEach((column) => {
    const th = document.createElement("th");
    th.textContent = column;
    headerRow.appendChild(th);
  });
  tableHead.appendChild(headerRow);

  rows.forEach((row) => {
    const tr = document.createElement("tr");
    columns.forEach((column) => {
      const td = document.createElement("td");
      const value = row[column];
      td.textContent = value === null || value === undefined ? "" : value;
      tr.appendChild(td);
    });
    tableBody.appendChild(tr);
  });
}
