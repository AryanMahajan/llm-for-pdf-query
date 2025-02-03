import json
import ollama
import os
from PyPDF2 import PdfReader

PDF_PATH = 'pdfs'


def get_content_from_pdfs(pdf_paths):
    content = []
    for pdfs in pdf_paths:
        pdf = PdfReader(pdfs)
        for i,page in enumerate(pdf.pages):
            print("Extracting content from page: ", i+1)
            content.append(page.extract_text())
    return content

exceptions = []

def parsing_data(data):
    try:
        parts = data.strip().split("**Answer:**")
        question_part = parts[0].strip()
        answer_part = parts[1].strip()
        
        question_text = question_part.replace("**Question:**","").strip()
        
        question_json = {
            "user":question_text,
            "assistant":answer_part
        }
        
        question_list = [question_json]
        
        return question_list
    
    except Exception as e:
        exceptions.append(e)
        


text_data = get_content_from_pdfs([os.path.join(PDF_PATH, pdf) for pdf in os.listdir(PDF_PATH)])

QA_List = []
counter = 0

for page_num, page_text in enumerate(text_data):
    context = page_text
    response = ollama.chat(
        model="llama3.2",
        messages=[
            {
                'role':'user',
                'content':'Make up one question and a brief answer from this text:'+context+'Have standardised sections like **Question:** and **Answer:**'
            }
        ]
    )
    
    result = parsing_data(response['message']['content'])
    
    print("Page number: ", page_num+1)
    
    if result is not None:
        QA_List.append(result[0])
        counter += 1
    
    
#saving the data
print("Total number of questions generated: ", counter)
print("Writing the data to a file")
with open("QA_data.jsonl", "w") as f:
    for pair in QA_List:
        json_line = json.dumps(pair)
        f.write(json_line+"\n")
        
        
print("Data saved successfully")


def convert_jsonl_to_txt(jsonl_file, txt_file):
    with open(jsonl_file, 'r', encoding='utf-8') as infile, open(txt_file, 'w', encoding='utf-8') as outfile:
        for line in infile:
            data = json.loads(line.strip())
            question = data.get("user", "")
            answer = data.get("assistant", "")
            if question and answer:
                outfile.write(f"[Q] {question}\n[A] {answer}\n\n")

print("Converting jsonl to txt")
convert_jsonl_to_txt("QA_data.jsonl", "QA_data.txt")

print("Conversion completed")