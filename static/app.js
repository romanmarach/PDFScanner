const uploadForm = document.querySelector("#uploadForm");
const fileInput = document.querySelector("#fileInput");
const dropZone = document.querySelector("#dropZone");
const dropTitle = document.querySelector("#dropTitle");
const fileMeta = document.querySelector("#fileMeta");
const submitButton = document.querySelector("#submitButton");
const statusPill = document.querySelector("#statusPill");
const formMessage = document.querySelector("#formMessage");
const resultMeta = document.querySelector("#resultMeta");
const textOutput = document.querySelector("#textOutput");
const wordCount = document.querySelector("#wordCount");
const charCount = document.querySelector("#charCount");
const modeLabel = document.querySelector("#modeLabel");
const copyButton = document.querySelector("#copyButton");
const downloadLink = document.querySelector("#downloadLink");
const analysisGrid = document.querySelector("#analysisGrid");
const docType = document.querySelector("#docType");
const confidence = document.querySelector("#confidence");
const summaryBlock = document.querySelector("#summaryBlock");
const shortSummary = document.querySelector("#shortSummary");
const bulletSummary = document.querySelector("#bulletSummary");

let latestResult = null;
let latestText = "";

function setStatus(label, state = "") {
  statusPill.textContent = label;
  statusPill.className = `status-pill ${state}`.trim();
}

function setMessage(message, isError = false) {
  formMessage.textContent = message;
  formMessage.classList.toggle("error", isError);
}

function updateFileLabel() {
  const file = fileInput.files[0];

  if (!file) {
    dropTitle.textContent = "Drop file here or browse";
    fileMeta.textContent = "PDF, PNG, JPG, JPEG, DOCX up to 25 MB";
    return;
  }

  const sizeMb = file.size / (1024 * 1024);
  dropTitle.textContent = file.name;
  fileMeta.textContent = `${sizeMb.toFixed(2)} MB`;
}

function setLoading(isLoading) {
  submitButton.disabled = isLoading;
  submitButton.querySelector("span").textContent = isLoading ? "Processing..." : "Run OCR";
  setStatus(isLoading ? "Processing" : "Ready", isLoading ? "busy" : "");
}

function resetAnalysis() {
  analysisGrid.hidden = true;
  summaryBlock.hidden = true;
  docType.textContent = "-";
  confidence.textContent = "-";
  shortSummary.textContent = "";
  bulletSummary.innerHTML = "";
}

function renderSummary(summary) {
  if (!summary || typeof summary !== "object") {
    return;
  }

  summaryBlock.hidden = false;
  shortSummary.textContent = summary.short_summary || "";
  bulletSummary.innerHTML = "";

  const bullets = Array.isArray(summary.bullet_points) ? summary.bullet_points : [];
  for (const item of bullets) {
    const li = document.createElement("li");
    li.textContent = item;
    bulletSummary.appendChild(li);
  }
}

function renderAnalysis(result) {
  resetAnalysis();

  if (result.classification && typeof result.classification === "object") {
    analysisGrid.hidden = false;
    docType.textContent = result.classification.document_type || "-";
    const value = result.classification.confidence;
    confidence.textContent = value === undefined || value === null ? "-" : `${value}%`;
  }

  renderSummary(result.summary);
}

function renderResult(result) {
  latestResult = result;
  latestText = result.text || "";
  textOutput.value = latestText;
  resultMeta.textContent = result.fileName ? `Processed ${result.fileName}` : "Document processed.";
  wordCount.textContent = String(result.wordCount || 0);
  charCount.textContent = String(result.characterCount || 0);
  modeLabel.textContent = result.mode === "full" ? "Analyze" : "Extract";
  copyButton.disabled = !latestText;
  downloadLink.disabled = false;
  renderAnalysis(result);
}

function downloadJson(data, filename) {
  const blob = new Blob([JSON.stringify(data, null, 2)], {
    type: "application/json",
  });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

fileInput.addEventListener("change", updateFileLabel);

for (const eventName of ["dragenter", "dragover"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.add("drag-over");
  });
}

for (const eventName of ["dragleave", "drop"]) {
  dropZone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropZone.classList.remove("drag-over");
  });
}

dropZone.addEventListener("drop", (event) => {
  const file = event.dataTransfer.files[0];
  if (!file) {
    return;
  }

  const transfer = new DataTransfer();
  transfer.items.add(file);
  fileInput.files = transfer.files;
  updateFileLabel();
});

uploadForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!fileInput.files[0]) {
    setMessage("Choose a file before running OCR.", true);
    return;
  }

  const formData = new FormData(uploadForm);

  try {
    setLoading(true);
    setMessage("OCR can take a moment while PaddleOCR loads and processes the file.");

    const response = await fetch("/api/extract", {
      method: "POST",
      body: formData,
    });

    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.error || "The document could not be processed.");
    }

    renderResult(result);
    setMessage("Extraction complete.");
    setStatus("Complete");
  } catch (error) {
    setMessage(error.message, true);
    setStatus("Error", "error");
  } finally {
    setLoading(false);
  }
});

copyButton.addEventListener("click", async () => {
  if (!latestText) {
    return;
  }

  await navigator.clipboard.writeText(latestText);
  setMessage("Extracted text copied.");
});

downloadLink.addEventListener("click", () => {
  if (!latestResult) {
    return;
  }

  downloadJson(latestResult, "pdfscanner-result.json");
});
