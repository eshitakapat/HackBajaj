import PyPDF2
import os

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            for page in reader.pages:
                text += page.extract_text() + "\n\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
    return text

if __name__ == "__main__":
    pdf_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
                          "dataSamples", 
                          "Arogya Sanjeevani Policy - CIN - U10200WB1906GOI001713 1.pdf")
    print(f"Looking for PDF at: {pdf_path}")
    if not os.path.exists(pdf_path):
        print("Error: PDF file not found at the specified path.")
        print("Current working directory:", os.getcwd())
        print("Files in dataSamples directory:", os.listdir(os.path.dirname(pdf_path)) if os.path.exists(os.path.dirname(pdf_path)) else "dataSamples directory not found")
    content = extract_text_from_pdf(pdf_path)
    
    # Print first 1000 characters to understand the structure
    print("First 1000 characters of the PDF:")
    print("="*80)
    print(content[:1000])
    
    # Print section headers (lines in all caps or with colons)
    print("\nPossible section headers:")
    print("="*80)
    for line in content.split('\n'):
        line = line.strip()
        if (line.isupper() or ':' in line) and len(line) > 5 and len(line) < 100:
            print(f"- {line}")
