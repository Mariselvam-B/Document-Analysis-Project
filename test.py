import pdfplumber

with pdfplumber.open("uploads/AI_Test.pdf") as pdf:
    for page in pdf.pages:
        print(page.extract_text())