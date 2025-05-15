# ATS using Neural Network
The project is simple Applicant Tracking System using a neural network without any high-level framework like TensorFlow or PyTorch.
The system matches the keywords from Job description dictonaries to applicants' resume and predicts the job category with confidence score.

## Features
Clean and preprocess raw resume text.
Convert resumes to TF-IDF vectors (max 784 features)
Predict job category from resumes using a multi-layer neural network
Supports top-3 category suggestions with confidence levels
Includes training functionality on custom labeled resumes

## Installation
1. Clone the repository
<pre><code>
git clone https://github.com/hvrdhn/ATS.git
cd ATS
</code></pre>

2. Install dependencies
<pre><code>
pip install -r requirements.txt
</code></pre>

## Files
- "ATS.py" : Contains Neural Network class
- "trainATS.py" : Data preprocessing, text extraction and training neural network.
- "testATS.py" : loads the trained model and predicts job category with confidence score.
