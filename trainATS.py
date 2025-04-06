import os
import pickle
from ATS import ResumeAnalyzer
import docx 
import PyPDF2 

def load_resumes_from_directory(base_dir):
    """
    Load resumes from a directory structure where each subfolder name corresponds to a job category
    
    Directory structure:
    resumes_for_ATS/
    ├── DataScience/
    │   ├── resume1.docx
    │   ├── resume2.docx
    ├── Testing/
    │   ├── resume3.docx
    │   └── ...
    ├── hr/
    │   ├── resume4.docx
    │   └── ...
    """
    training_data = []
    
    # Map each folder names to the categories in the ResumeAnalyzer
    # Exact the categories defined in ResumeAnalyzer
    category_mapping = {
        'DataScience': 'data_scientist',
        'testing': 'testing',  
        'React Developer' : 'react_developer',
        'hr': 'hr',
        'CONSULTANT' : 'CONSULTANT',
        'designer' : 'designer',
        'DIGITAL-MEDIA' : 'DIGITAL-MEDIA',
        'FINANCE' : 'finance',
        'SALES' : 'SALES',
        'TEACHER' : 'TEACHER',  
    }
    
    # Count for statistics
    category_counts = {}
    
    # Go through each directory in the base directory
    for category_dir in os.listdir(base_dir):
        dir_path = os.path.join(base_dir, category_dir)
        
        # Skip if not a directory
        if not os.path.isdir(dir_path):
            continue
            
        # Get the mapped category name
        category = category_mapping.get(category_dir, category_dir)
        category_counts[category] = 0
        
        # Process each file in the directory
        for filename in os.listdir(dir_path):
            file_path = os.path.join(dir_path, filename)
            
            try:
                resume_text = ""
                
                # Handle different file types
                if filename.endswith('.txt'):
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                        resume_text = file.read()
                
                elif filename.endswith('.docx'):
                    # read DOCX files
                    try:
                        doc = docx.Document(file_path)
                        resume_text = "\n".join([paragraph.text for paragraph in doc.paragraphs])
                    except Exception as docx_error:
                        print(f"Error processing DOCX file {file_path}: {docx_error}")
                        continue
                
                elif filename.endswith('.pdf'):
                    try:
                        with open(file_path, 'rb') as pdf_file:
                            pdf_reader = PyPDF2.PdfReader(pdf_file)

                            #extract text
                            pdf_text = []
                            for page_num in range (len(pdf_reader.pages)):
                                page = pdf_reader.pages[page_num]
                                pdf_text.append(page.extract_text())
                            
                            resume_text = "\n".join(pdf_text)

                    except Exception as pdf_error:
                        print(f"Error in processing PDF file {file_path}: {pdf_error}")
                        continue
                
                else:
                    # Skip files that aren't .txt or .docx
                    print(f"Skipping unsupported file format: {filename}")
                    continue
                
                if resume_text.strip():  # Only add if we got some text
                    # Add to training data
                    training_data.append((resume_text, category))
                    category_counts[category] = category_counts.get(category, 0) + 1
                    print(f"Added resume from {filename} as category '{category}'")
                else:
                    print(f"Warning: No text extracted from {filename}")
            
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    
    # Print statistics
    print("\nResume count by category:")
    for category, count in category_counts.items():
        print(f"  {category}: {count}")
    
    return training_data

def main():
    # Path to the resume directory
    resume_dir = r"C:\Users\Harshvardhan Thakur\OneDrive\Desktop\datasets\NLP\resumes_for_ATS"  
    
    # Check if directory exists
    if not os.path.exists(resume_dir):
        print(f"Directory '{resume_dir}' does not exist. Please create it and add your resume files.")
        return
    
    # Load resumes
    print(f"Loading resumes from {resume_dir}...")
    training_data = load_resumes_from_directory(resume_dir)
    
    if not training_data:
        print("No training data found. Please check your directory structure and file formats.")
        return
    
    print(f"\nLoaded {len(training_data)} resumes for training")
    
    # Initialize the resume analyzer
    print("Initializing ResumeAnalyzer...")
    analyzer = ResumeAnalyzer()
    
    # available categories for reference
    print("\nAvailable categories in the model:")
    for idx, category in enumerate(analyzer.categories):
        print(f"  {idx+1}. {category}")
    
    # Train the model
    print("\nTraining model...")
    analyzer.train(training_data, epochs=50, batch_size=5, learning_rate=0.01)
    
    # Save the trained model
    print("Saving trained model...")
    with open('trained_resume_analyzer.pkl', 'wb') as f:
        pickle.dump(analyzer, f)
    
    print("Training complete! Model saved as 'trained_resume_analyzer.pkl'")
    
    # Optional: Test with a sample resume
    test_resume = """
    Python developer with 5 years experience. Proficient in Django, Flask, and FastAPI.
    Experience with AWS, Docker, and CI/CD pipelines. Database skills including PostgreSQL and MongoDB.
    """
    
    print("\nTesting model with a sample resume...")
    results = analyzer.analyze_resume(test_resume)
    
    print("Sample resume classification results:")
    for result in results:
        print(f"  Category: {result['category']}, Confidence: {result['confidence']:.2f}%")

if __name__ == "__main__":
    main()