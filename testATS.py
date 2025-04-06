import pickle
import sys
import os

def load_model(model_path='trained_resume_analyzer.pkl'):
    """Load the trained resume analyzer model"""
    if not os.path.exists(model_path):
        print(f"Error: Model file '{model_path}' not found.")
        print("Please train the model first using train_resume_analyzer.py")
        sys.exit(1)
        
    try:
        with open(model_path, 'rb') as f:
            analyzer = pickle.load(f)
        return analyzer
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

def analyze_resume_file(file_path, analyzer):
    """Analyze a resume file and print predictions"""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            resume_text = f.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return
        
    # Analyze the resume
    results = analyzer.analyze_resume(resume_text)
    
    # Print results
    print(f"\nResults for: {os.path.basename(file_path)}")
    print("-" * 50)
    for result in results:
        print(f"Category: {result['category']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print()
    
    # Print the top prediction
    top_prediction = results[0]['category']
    top_confidence = results[0]['confidence']
    print(f"Top prediction: {top_prediction} ({top_confidence:.2f}%)")
    
    return results

def analyze_resume_text(resume_text, analyzer):
    """Analyze a resume from text input"""
    results = analyzer.analyze_resume(resume_text)
    
    # Print results
    print("\nAnalysis Results")
    print("-" * 50)
    for result in results:
        print(f"Category: {result['category']}")
        print(f"Confidence: {result['confidence']:.2f}%")
        print()
        
    # Print the top prediction
    top_prediction = results[0]['category']
    top_confidence = results[0]['confidence']
    print(f"Top prediction: {top_prediction} ({top_confidence:.2f}%)")
    
    return results

def main():
    # Load the trained model
    print("Loading trained model...")
    analyzer = load_model()
    print("Model loaded successfully!")
    
    # Check if a file path was provided as an argument
    if len(sys.argv) > 1:
        file_path = sys.argv[1]
        if os.path.exists(file_path):
            analyze_resume_file(file_path, analyzer)
        else:
            print(f"Error: File '{file_path}' not found.")
            return
    else:
        # Interactive mode
        while True:
            print("\nResume Analyzer Menu:")
            print("1. Analyze a resume file")
            print("2. Paste resume text for analysis")
            print("3. Exit")
            
            choice = input("\nEnter your choice (1-3): ")
            
            if choice == '1':
                file_path = input("Enter the path to the resume file: ")
                if os.path.exists(file_path):
                    analyze_resume_file(file_path, analyzer)
                else:
                    print(f"Error: File '{file_path}' not found.")
            
            elif choice == '2':
                print("Enter or paste the resume text (type 'DONE' on a new line when finished):")
                lines = []
                while True:
                    line = input()
                    if line.strip() == 'DONE':
                        break
                    lines.append(line)
                
                resume_text = '\n'.join(lines)
                if resume_text.strip():
                    analyze_resume_text(resume_text, analyzer)
                else:
                    print("No resume text provided.")
            
            elif choice == '3':
                print("Exiting...")
                break
                
            else:
                print("Invalid choice. Please enter 1, 2, or 3.")

if __name__ == "__main__":
    main()