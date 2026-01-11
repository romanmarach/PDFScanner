from .text_extraction import extract_text
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
        print("\n Classifying document...")
        classification = classify_document(text)

        print("\n summarizing document...")
        summary = summarize_document(text)
        
        result["classification"] = classification
        result["summary"] = summary


        print("document type and confidence")
        print(classification)
        print("Document summary:")
        print(summary)

    return result

#======================================================================

def main():
    parser = argparse.ArgumentParser(description="extract text from documents")
    parser.add_argument("path", help="File or directory to process")
    parser.add_argument("--mode", choices=["extract", "full"], default="extract", 
                        help="extract = text only, full = text +classify and summarize")
    args = parser.parse_args()
    results = []

    file_path = args.path
    mode = args.mode

    if os.path.isfile(file_path):
        result = process_single_file(file_path, mode)
        results.append(result)

    elif os.path.isdir(file_path):
        for filename in os.listdir(file_path):
            full_path = os.path.join(file_path, filename)
            if os.path.isfile(full_path):
                result = process_single_file(full_path, mode)       
                results.append(result)
          # save all results
    else:
        print("Path not found")
        return
    
    output_path = "./output/results.json"

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
      
    print(f"\n✅ Results saved to {output_path}")
    
    txt_path = "./output/extracted_text.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for result in results:
            f.write(f"===== File: {result['file']} =====\n")
            f.write(result["extracted_text"])
            f.write("\n\n")



    
 
#======================================================================

   

#run with python -m agent data/samples/invoice_test.png

