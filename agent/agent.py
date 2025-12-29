from .text_extraction import extract_text
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

    if __name__ == "__main__":
        main()

