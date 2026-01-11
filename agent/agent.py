from .text_extraction import extract_text
from .doc_classify import classify_document
from .doc_summarize import summarize_document
import argparse 
import sys
import os


def process_single_file(file_path, mode):
    print("📄 Extracting text from:", file_path)
    print("mode:",mode)
    text = extract_text(file_path)

    print("\n===== EXTRACTED TEXT =====\n")
    print(text)
    print("\n===========================\n")
    with open("./output/extracted_text.txt", "w") as f:
        f.write(text)
        print("✅ Extracted text saved to ./output/extracted_text.txt")
#======================================================================
    if mode == "full":
        print("\n Classifying document...")
        classification = classify_document(text)

        print("\n summarizing document...")
        summary = summarize_document(text)
        
        print("document type and confidence")
        print(classification)

        print("Document summary:")
        print(summary)

#======================================================================

def main():
    parser = argparse.ArgumentParser(description="extract text from documents")
    parser.add_argument("path", help="File or directory to process")
    parser.add_argument("--mode", choices=["extract", "full"], default="extract", help="extract = text only, full = text +classify and summarize")
    args = parser.parse_args()

    file_path = args.path
    mode = args.mode

    if os.path.isfile(file_path):
        process_single_file(file_path, mode)

    elif os.path.isdir(file_path):
        results = {}
        for filename in os.listdir(file_path):
              full_path = os.path.join(file_path, filename)
              if os.path.isfile(full_path):  # skip subdirectories
                  results[filename] = process_single_file(full_path, mode)
          # save all results
    else:
          print("Path not found")
    

    
 
#======================================================================

   

#run with python -m agent data/samples/invoice_test.png

