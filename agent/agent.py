from .text_extraction import EXTRACTABLE_EXTENSIONS, extract_text
from .doc_classify import classify_document
from .doc_summarize import summarize_document
import argparse
import sys
import os
import json

def process_single_file(file_path, mode):
    print("📄 Extracting text from:", file_path)
    text = extract_text(file_path)

    result = {
        "file": file_path,
        "extracted_text": text
    }

    print("\n===== EXTRACTED TEXT =====\n")
    print(text)
    print("\n===========================\n")
    # with open("./output/extracted_text.txt", "w") as f:
    #     f.write(text)
    #     print("✅ Extracted text saved to ./output/extracted_text.txt")
#======================================================================
    if mode == "full":
        analysis_errors = {}

        print("\n Classifying document...")
        try:
            classification = classify_document(text)
            result["classification"] = classification
        except Exception as exc:
            print(f"Classification failed: {type(exc).__name__}: {exc}")
            analysis_errors["classification"] = "Classification could not be completed."

        print("\n summarizing document...")
        try:
            summary = summarize_document(text)
            result["summary"] = summary
        except Exception as exc:
            print(f"Summary failed: {type(exc).__name__}: {exc}")
            analysis_errors["summary"] = "Summary could not be completed."
        
        if analysis_errors:
            result["analysisErrors"] = analysis_errors


        print("document type and confidence")
        print(result.get("classification", "Classification unavailable."))
        print("Document summary:")
        print(result.get("summary", "Summary unavailable."))

    return result

#======================================================================

def main():
    parser = argparse.ArgumentParser(description="extract text from documents")
    parser.add_argument("path", help="File or directory to process")
    parser.add_argument("--mode", choices=["extract", "full"], default="extract", 
                        help="extract = text only, full = text +classify and summarize")
    args = parser.parse_args()
    results = []
    skipped = 0
    failed = 0

    file_path = args.path
    mode = args.mode

    if os.path.isfile(file_path):
        result = process_single_file(file_path, mode)
        results.append(result)

    elif os.path.isdir(file_path):
        for filename in sorted(os.listdir(file_path)):
            full_path = os.path.join(file_path, filename)
            if not os.path.isfile(full_path):
                continue

            extension = os.path.splitext(filename)[1].lower()
            if extension not in EXTRACTABLE_EXTENSIONS:
                print(f"Skipped {filename}: unsupported file type")
                skipped += 1
                continue

            try:
                result = process_single_file(full_path, mode)
                results.append(result)
            except Exception as exc:
                print(f"Failed {filename}: {type(exc).__name__}: {exc}")
                results.append({"file": full_path, "error": f"{type(exc).__name__}: {exc}"})
                failed += 1
    else:
        print("Path not found")
        return
    
    os.makedirs("./output", exist_ok=True)
    output_path = "./output/results.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
      
    print(f"\n✅ Results saved to {output_path}")
    
    txt_path = "./output/extracted_text.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for result in results:
            if "error" in result:
                continue
            f.write(f"===== File: {result['file']} =====\n")
            f.write(result["extracted_text"])
            f.write("\n\n")

    if os.path.isdir(file_path):
        print(
            f"\nBatch complete: {len(results) - failed} processed, "
            f"{skipped} skipped, {failed} failed."
        )

# run with python -m agent data/samples/invoice_test.png
# docker-compose run pdfscanner data/samples/invoice_test.png
# docker compose run --rm pdfscanner
# npx @anthropic-ai/claude-code .
