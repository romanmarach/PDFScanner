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
const copyButton = document.querySelector("#copyButton");
const downloadLink = document.querySelector("#downloadLink");

const listFields = {
  important_points: document.querySelector("#importantPoints"),
  actions_required: document.querySelector("#actionsRequired"),
  important_dates: document.querySelector("#importantDates"),
  amounts: document.querySelector("#amounts"),
  warnings: document.querySelector("#warnings"),
};

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
    fileMeta.textContent = "PDF, PNG, JPG, JPEG, DOCX up to 25 MB";
    return;
  }

  dropTitle.textContent = file.name;
  fileMeta.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;
}

function setLoading(isLoading) {
  submitButton.disabled = isLoading;
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
  copyButton.disabled = false;
  downloadLink.setAttribute("aria-disabled", "false");
  renderExplanation();
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

  try {
    setLoading(true);
    setMessage("Reading and explaining the document. Scanned files may take longer.");

    const response = await fetch("/api/explain", {
      method: "POST",
      body: new FormData(explainForm),
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
