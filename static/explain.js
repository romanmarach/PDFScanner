const explainForm = document.querySelector("#explainForm");
const fileInput = document.querySelector("#fileInput");
const dropZone = document.querySelector("#dropZone");
const dropTitle = document.querySelector("#dropTitle");
const fileMeta = document.querySelector("#fileMeta");
const languageSelect = document.querySelector("#languageSelect");
const submitButton = document.querySelector("#submitButton");
const statusPill = document.querySelector("#statusPill");
const formMessage = document.querySelector("#formMessage");
const resultTitle = document.querySelector("#resultTitle");
const resultMeta = document.querySelector("#resultMeta");
const languageTabs = document.querySelector("#languageTabs");
const translatedTab = document.querySelector("#translatedTab");
const explanationContent = document.querySelector("#explanationContent");
const summaryText = document.querySelector("#summaryText");
const explanationText = document.querySelector("#explanationText");
const warningSection = document.querySelector("#warningSection");
const pdfButton = document.querySelector("#pdfButton");
const copyButton = document.querySelector("#copyButton");
const downloadLink = document.querySelector("#downloadLink");

const TURNSTILE_ENABLED = explainForm.dataset.turnstileEnabled === "true";
const listFields = {
  important_points: document.querySelector("#importantPoints"),
  actions_required: document.querySelector("#actionsRequired"),
  important_dates: document.querySelector("#importantDates"),
  amounts: document.querySelector("#amounts"),
  warnings: document.querySelector("#warnings"),
};

const MAX_UPLOAD_BYTES = Number(explainForm.dataset.maxUpload) || 25 * 1024 * 1024;
const MAX_UPLOAD_MB = Math.round(MAX_UPLOAD_BYTES / (1024 * 1024));

let latestResult = null;
let activeView = "english";

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
    fileMeta.textContent = `PDF, PNG, JPG, JPEG, DOCX up to ${MAX_UPLOAD_MB} MB`;
    dropZone.classList.remove("error");
    submitButton.disabled = false;
    submitButton.title = "";
    return;
  }

  const sizeMb = file.size / (1024 * 1024);
  dropTitle.textContent = file.name;

  if (file.size > MAX_UPLOAD_BYTES) {
    fileMeta.textContent = `${sizeMb.toFixed(2)} MB — too large. Click here or drop a smaller file.`;
    dropZone.classList.add("error");
    setMessage(`This file is larger than the ${MAX_UPLOAD_MB} MB upload limit. Choose a smaller file.`, true);
    submitButton.disabled = true;
    submitButton.title = `File exceeds the ${MAX_UPLOAD_MB} MB upload limit`;
  } else {
    fileMeta.textContent = `${sizeMb.toFixed(2)} MB`;
    dropZone.classList.remove("error");
    setMessage("");
    submitButton.disabled = false;
    submitButton.title = "";
  }
}

function fileTooLarge() {
  const file = fileInput.files[0];
  return Boolean(file) && file.size > MAX_UPLOAD_BYTES;
}

function resetTurnstile() {
  if (TURNSTILE_ENABLED && window.turnstile) {
    window.turnstile.reset();
  }
}

function setLoading(isLoading) {
  submitButton.disabled = isLoading || fileTooLarge();
  submitButton.querySelector("span").textContent = isLoading
    ? "Explaining..."
    : "Explain document";
  setStatus(isLoading ? "Processing" : "Ready", isLoading ? "busy" : "");
}

function renderList(element, values) {
  element.innerHTML = "";
  const items = Array.isArray(values) && values.length ? values : ["None identified."];

  for (const value of items) {
    const item = document.createElement("li");
    item.textContent = value;
    element.appendChild(item);
  }
}

function currentExplanation() {
  if (!latestResult) {
    return null;
  }
  return activeView === "translated" ? latestResult.translated : latestResult.english;
}

function renderExplanation() {
  const explanation = currentExplanation();
  if (!explanation) {
    return;
  }

  explanationContent.classList.remove("empty");
  resultTitle.textContent = explanation.document_type || "Document explanation";
  summaryText.textContent = explanation.summary || "";
  explanationText.textContent = explanation.explanation || "";

  for (const [field, element] of Object.entries(listFields)) {
    renderList(element, explanation[field]);
  }

  warningSection.classList.toggle(
    "has-warnings",
    Array.isArray(explanation.warnings) && explanation.warnings.length > 0,
  );

  for (const button of languageTabs.querySelectorAll("button")) {
    button.classList.toggle("active", button.dataset.view === activeView);
  }
}

function renderResult(result) {
  latestResult = result;
  activeView = result.translated ? "translated" : "english";
  resultMeta.textContent = `Explained ${result.fileName} in ${result.languageName}.`;
  if (result.sourceWasTruncated) {
    resultMeta.textContent += " The source exceeded the current processing limit.";
  }

  languageTabs.hidden = !result.translated;
  translatedTab.textContent = result.languageName || "Translation";
  pdfButton.disabled = false;
  copyButton.disabled = false;
  downloadLink.disabled = false;
  renderExplanation();
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

function explanationAsText(explanation) {
  const sections = [
    explanation.document_type,
    explanation.summary,
    explanation.explanation,
  ];

  for (const [field, label] of [
    ["important_points", "Important points"],
    ["actions_required", "Actions required"],
    ["important_dates", "Important dates"],
    ["amounts", "Amounts"],
    ["warnings", "Warnings and risks"],
  ]) {
    if (Array.isArray(explanation[field]) && explanation[field].length) {
      sections.push(`${label}\n${explanation[field].map((item) => `- ${item}`).join("\n")}`);
    }
  }

  return sections.filter(Boolean).join("\n\n");
}

function escapeHtml(value) {
  const element = document.createElement("div");
  element.textContent = value || "";
  return element.innerHTML;
}

function printableList(values) {
  const items = Array.isArray(values) && values.length ? values : ["None identified."];
  return `<ul>${items.map((item) => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`;
}

function printExplanation(explanation) {
  const languageName = activeView === "translated"
    ? latestResult.languageName
    : "English";
  const sourceName = latestResult.fileName || "document";
  const title = explanation.document_type || "Document explanation";
  const iframe = document.createElement("iframe");
  iframe.hidden = true;
  iframe.title = "Printable document explanation";
  document.body.appendChild(iframe);

  iframe.srcdoc = `<!doctype html>
    <html lang="en">
      <head>
        <meta charset="utf-8">
        <title>${escapeHtml(sourceName)} - explanation</title>
        <style>
          @page { margin: 18mm; }
          body {
            color: #1a1714;
            font: 11pt/1.55 Arial, sans-serif;
            margin: 0;
          }
          h1 { font-size: 22pt; margin: 0 0 4px; }
          h2 {
            border-bottom: 1px solid #d9d3ca;
            font-size: 13pt;
            margin: 22px 0 8px;
            padding-bottom: 4px;
          }
          p { margin: 0 0 10px; white-space: pre-wrap; }
          ul { margin: 0; padding-left: 22px; }
          li { margin-bottom: 5px; }
          .meta { color: #6b645c; margin-bottom: 24px; }
          .summary {
            background: #f4f1eb;
            border-left: 4px solid #31543f;
            padding: 12px 14px;
          }
          .warning { border-left: 4px solid #a65d36; padding-left: 12px; }
          .disclaimer {
            border-top: 1px solid #d9d3ca;
            color: #6b645c;
            font-size: 9pt;
            margin-top: 28px;
            padding-top: 10px;
          }
        </style>
      </head>
      <body>
        <h1>${escapeHtml(title)}</h1>
        <p class="meta">${escapeHtml(sourceName)} | ${escapeHtml(languageName)}</p>
        <h2>Summary</h2>
        <p class="summary">${escapeHtml(explanation.summary)}</p>
        <h2>What this document means</h2>
        <p>${escapeHtml(explanation.explanation)}</p>
        <h2>Important points</h2>
        ${printableList(explanation.important_points)}
        <h2>Actions required</h2>
        ${printableList(explanation.actions_required)}
        <h2>Important dates</h2>
        ${printableList(explanation.important_dates)}
        <h2>Amounts</h2>
        ${printableList(explanation.amounts)}
        <section class="warning">
          <h2>Warnings and risks</h2>
          ${printableList(explanation.warnings)}
        </section>
        <p class="disclaimer">
          AI explanations can contain mistakes. Verify important legal, medical,
          financial, or government information with a qualified professional.
        </p>
      </body>
    </html>`;

  iframe.addEventListener("load", () => {
    const printWindow = iframe.contentWindow;
    printWindow.addEventListener("afterprint", () => iframe.remove(), { once: true });
    printWindow.focus();
    printWindow.print();
  }, { once: true });
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

languageTabs.addEventListener("click", (event) => {
  const button = event.target.closest("button[data-view]");
  if (!button) {
    return;
  }

  activeView = button.dataset.view;
  renderExplanation();
});

explainForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  if (!fileInput.files[0]) {
    setMessage("Choose a file before explaining the document.", true);
    return;
  }

  if (fileInput.files[0].size > MAX_UPLOAD_BYTES) {
    setMessage(`This file is larger than the ${MAX_UPLOAD_MB} MB upload limit. Choose a smaller file.`, true);
    return;
  }

  try {
    const formData = new FormData(explainForm);
    if (TURNSTILE_ENABLED && !formData.get("cf-turnstile-response")) {
      setMessage("Complete the bot verification before explaining the document.", true);
      return;
    }

    setLoading(true);
    setMessage("Reading and explaining the document. Scanned files may take longer.");

    const response = await fetch("/api/explain", {
      method: "POST",
      body: formData,
    });
    const result = await response.json();

    if (!response.ok) {
      throw new Error(result.error || "The document could not be explained.");
    }

    renderResult(result);
    setMessage("Explanation complete.");
    setStatus("Complete");
  } catch (error) {
    setMessage(error.message, true);
    setStatus("Error", "error");
  } finally {
    setLoading(false);
    resetTurnstile();
  }
});

copyButton.addEventListener("click", async () => {
  const explanation = currentExplanation();
  if (!explanation) {
    return;
  }

  await navigator.clipboard.writeText(explanationAsText(explanation));
  setMessage("Explanation copied.");
});

pdfButton.addEventListener("click", () => {
  const explanation = currentExplanation();
  if (!explanation) {
    return;
  }

  printExplanation(explanation);
  setMessage("Choose Save as PDF in the print dialog.");
});

downloadLink.addEventListener("click", () => {
  if (!latestResult) {
    return;
  }

  downloadJson(latestResult, "document-explanation.json");
});
