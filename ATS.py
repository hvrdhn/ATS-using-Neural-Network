import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
import re
from collections import defaultdict


class Network:
    def __init__(self, sizes):
        self.num_layers = len(sizes)
        self.sizes = sizes
        self.biases = [np.random.randn(y, 1) * 0.1 for y in sizes[1:]]
        self.weights = [np.random.randn(y, x) * np.sqrt(2.0/x) for x, y in zip(sizes[:-1], sizes[1:])]
        
    @staticmethod
    def relu(z):
        return np.maximum(0, z)
    
    def softmax(self, z):
        z = z - np.max(z, axis=0, keepdims=True)
        e_z = np.exp(z)
        return e_z / np.sum(e_z, axis=0, keepdims=True)

    def feedforward(self, a):
        a = np.array(a, dtype=np.float64).reshape(-1, 1)
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(w, a) + b
            a = self.relu(z)
        return self.softmax(np.dot(self.weights[-1], a) + self.biases[-1])
    
    def SGD(self, training_data, epochs, mini_batch_size, eta, test_data=None):
        training_data = [(np.reshape(x, (784, 1)), np.reshape(y, (10, 1))) for x, y in training_data]
        if test_data:
            test_data = [(np.reshape(x, (784, 1)), np.reshape(y, (10, 1))) for x, y in test_data]
            
        n = len(training_data)
        for j in range(epochs):
            random.shuffle(training_data)
            mini_batches = [training_data[k:k+mini_batch_size] for k in range(0, n, mini_batch_size)]
            
            for mini_batch in mini_batches:
                self.update_mini_batch(mini_batch, eta)

            if test_data:
                accuracy = self.evaluate(test_data)
                print(f'Epoch {j}: {accuracy}/{len(test_data)} ({(accuracy/len(test_data))*100:.2f}%)')
            else:
                print(f'Epoch {j} complete')

    def update_mini_batch(self, mini_batch, eta):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self.backprop(x, y)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.weights = [w - (eta/len(mini_batch)) * nw for w, nw in zip(self.weights, nabla_w)]
        self.biases = [b - (eta/len(mini_batch)) * nb for b, nb in zip(self.biases, nabla_b)]

    def backprop(self, x, y):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []

        # Forward pass
        for w, b in zip(self.weights[:-1], self.biases[:-1]):
            z = np.dot(w, activation) + b
            zs.append(z)
            activation = self.relu(z)
            activations.append(activation)
        
        # Output layer
        z_final = np.dot(self.weights[-1], activation) + self.biases[-1]
        zs.append(z_final)
        activation = self.softmax(z_final)
        activations.append(activation)

        # Backward pass
        delta = self.cost_derivate(activations[-1], y)
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].T)

        for l in range(2, self.num_layers):
            z = zs[-l]
            sp = self.relu_prime(z)
            delta = np.dot(self.weights[-l+1].T, delta) * sp
            nabla_b[-l] = delta
            nabla_w[-l] = np.dot(delta, activations[-l-1].T)
            
        return nabla_b, nabla_w

    def evaluate(self, test_data):
        test_results = [(np.argmax(self.feedforward(x)), np.argmax(y)) for x, y in test_data]
        return sum(int(x == y) for x, y in test_results)

    def cost_derivate(self, output_activations, y):
        return output_activations - y

    @staticmethod
    def relu_prime(z):
        return (z > 0).astype(float)

class ResumeAnalyzer:
    def __init__(self):
        # Initialize vocabulary and categories
        self.vectorizer = TfidfVectorizer(
            max_features=784,  # Match input layer size
            stop_words='english',
            ngram_range=(1, 3),
            min_df=0.0,
            max_df=0.9,
            use_idf=True,
            smooth_idf=True,
            sublinear_tf=True
        )
        self.categories = [ #run on these categories & updatea PDF preprocessing
            'react_developer',
            'data_scientist',
            'testing',
            'designer',
            'SALES',
            'finance',
            'hr',
            'CONSULTANT',
            'DIGITAL-MEDIA',
            'TEACHER',

        ]
        self.category_terms = {
            'react_developer': ['react', 'developer', 'programming', 'mean', 'css', 'node js', 'java', 'angular', 'javascript', 'web', 'frontend', 'backend', 'mysql', 'api', 'database', 'sql', 'nosql'],
            'data_scientist': ['data', 'science', 'scientist', 'analytics', 'machine learning', 'statistics', 'algorithm', 'model', 'python', 'r', 'sql', 'tensorflow', 'pytorch', 'ai'],
            'testing': ['automation', 'monitoring', 'testing', 'rally', 'java', 'sql', 'javascript', 'html', 'xml', 'framework', 'api', 'quality'],
            'designer': ['design', 'designer', 'ui', 'ux', 'graphic', 'creative', 'adobe', 'photoshop', 'illustrator', 'figma', 'sketch', 'typography', 'visual'],
            #'marketing': ['marketing', 'brand', 'campaign', 'social media', 'content', 'seo', 'analytics', 'advertisement', 'pr', 'communication', 'audience'],
            'sales': ['sales', 'customer', 'client', 'account', 'revenue', 'quota', 'pipeline', 'crm', 'lead', 'prospect', 'negotiation', 'closing'],
            'finance': ['finance', 'financial', 'accounting', 'accountant', 'budget', 'audit', 'tax', 'investment', 'banking', 'cpa', 'cfa', 'analysis'],
            'hr': ['hr', 'human resources', 'recruiting', 'recruiter', 'talent', 'acquisition', 'onboarding', 'benefits', 'compensation', 'culture', 'training'],
            'CONSULTANT': ['operations', 'strategy development', 'logistics', 'process improvement', 'Stakeholder Engagement', 'inventory', 'quality', 'management', 'financial modeling', 'SWOT', 'communication', 'Data Collection'],
            'TEACHER': ['assistant', 'student support', 'support', 'coordinator', 'specialist', 'classroom management', 'Lesson Planning', 'Student Engagement'],
            'DIGITAL-MEDIA': ['Analysis', 'collaboration', 'Digital', 'Marketing', 'Media', 'Automation', 'Content', 'PPC', 'SEO'],
        }
        
        # Initialize neural network
        self.network = Network([784, 128, 64, 10])  # 10 job categories, reduce job categories
        
    def preprocess_resume(self, text):
        """Clean and standardize resume text"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and extra spaces
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        
        # Remove common resume words that don't add value
        stop_words = ['resume', 'curriculum', 'vitae', 'cv', 'name', 'email', 'phone']
        for word in stop_words:
            text = text.replace(word, '')
            
        return text.strip()
    
    def extract_features(self, resumes):
        """Convert resumes to TF-IDF vectors"""
        # Fit vectorizer if not already fit
        if not hasattr(self.vectorizer, 'vocabulary_'):
            self.vectorizer.fit(resumes)
        
        # Convert resumes to TF-IDF vectors
        features = self.vectorizer.transform(resumes).toarray()
        
        # 784 features
        if features.shape[1] < 784:
            padding = np.zeros((features.shape[0], 784 - features.shape[1]))
            features = np.hstack((features, padding))
        elif features.shape[1] > 784:
            features = features[:, :784]
            
        return features
    
    def prepare_training_data(self, resumes_with_labels):
        """Prepare data for neural network training"""
        processed_resumes = [self.preprocess_resume(text) for text, _ in resumes_with_labels]
        features = self.extract_features(processed_resumes)
        
        # Convert labels to one-hot encoding
        labels = []
        for _, category in resumes_with_labels:
            one_hot = np.zeros(len(self.categories))
            category_idx = self.categories.index(category)
            one_hot[category_idx] = 1.0
            labels.append(one_hot)
            
        return list(zip(features, labels))
    
    def train(self, training_data, epochs=30, batch_size=32, learning_rate=0.1):
        """Train the neural network"""
        prepared_data = self.prepare_training_data(training_data)
        self.network.SGD(prepared_data, epochs, batch_size, learning_rate)
    
    def analyze_resume(self, resume_text):
        """Analyze a single resume and return predicted category with confidence"""
        # Preprocess and vectorize
        processed_text = self.preprocess_resume(resume_text)
        features = self.extract_features([processed_text])
        
        # Get network prediction
        prediction = self.network.feedforward(features[0])
        
        # Get top 3 predictions
        top_indices = np.argsort(prediction.flatten())[-3:][::-1]
        results = []
        
        for idx in top_indices:
            category = self.categories[idx]
            confidence = float(prediction.flatten()[idx]) * 100
            results.append({
                'category': category,
                'confidence': confidence
            })
            
        return results

