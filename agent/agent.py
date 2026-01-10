from .text_extraction import extract_text
from .doc_classify import classify_document
from .doc_summarize import summarize_document
import sys

def main():
    if len(sys.argv) < 2:
        print("Usage: python -m agent <file_path>")
        return

    file_path = sys.argv[1]

    print("📄 Extracting text from:", file_path)
    text = extract_text(file_path)

    print("\n===== EXTRACTED TEXT =====\n")
    print(text)
    print("\n===========================\n")
    with open("./output/extracted_text.txt", "w") as f:
        f.write(text)
        print("✅ Extracted text saved to ./output/extracted_text.txt")
#======================================================================

    # print("\n Classifying document...")
    # classification = classify_document(text)

    # print("\n summarizing document...")
    # summary = summarize_document(text)
    
    # print("document type and confidence")
    # print(classification)

    # print("Document summary:")
    # print(summary)
#uncomment this when open ai credit card is set up. 
#======================================================================

    if __name__ == "__main__":
        main()

