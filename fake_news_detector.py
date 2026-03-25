import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

class FakeNewsDetector:
    def __init__(self):
        # ========== MODEL INITIALIZATION ==========
        # Set up the text vectorizer (converts text to numbers)
        self.vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')
        
        # Initialize 3 different machine learning algorithms (UNTRAINED)
        self.models = {
            'Naive Bayes': MultinomialNB(),                                    # Probabilistic classifier
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),  # Ensemble method
            'SVM': SVC(kernel='linear', random_state=42)                       # Support Vector Machine
        }
        
        # Dictionary to store TRAINED models after training
        self.trained_models = {}  # Empty until training happens!
        
        # Text preprocessing tool
        self.stemmer = PorterStemmer()
        
    def preprocess_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Tokenize and remove stopwords
        try:
            stop_words = set(stopwords.words('english'))
            words = text.split()
            words = [self.stemmer.stem(word) for word in words if word not in stop_words]
            return ' '.join(words)
        except:
            return text
    
    def load_and_prepare_data(self, data_path=None, use_large_dataset=False):
        """Load and prepare the dataset"""
        if data_path:
            print(f"Loading dataset from: {data_path}")
            df = pd.read_csv(data_path)
        elif use_large_dataset:
            print("Creating comprehensive fake news dataset...")
            df = self.download_real_dataset()
        else:
            # Create sample data if no dataset provided
            print("Using built-in sample dataset...")
            df = self.create_sample_data()
        
        print(f"Original dataset size: {len(df)} articles")
        
        # Preprocess text
        df['cleaned_text'] = df['text'].apply(self.preprocess_text)
        
        # Remove empty texts
        df = df[df['cleaned_text'].str.len() > 0]
        
        print(f"After preprocessing: {len(df)} articles")
        return df
    
    def create_sample_data(self):
        """Create sample fake news data for demonstration"""
        # EXPANDED FAKE NEWS DATASET (100+ samples)
        fake_news = [
            "Breaking: Scientists discover aliens living among us in secret underground cities",
            "Government officials confirm that vaccines contain mind control chips",
            "Celebrity spotted with mysterious illness that doctors can't explain",
            "New study shows that drinking water causes cancer in 99% of cases",
            "Local man discovers secret to immortality using this one weird trick",
            "Shocking: Moon landing was filmed in Hollywood studio, insider reveals",
            "Doctors hate this simple trick that cures all diseases instantly",
            "Government hiding cure for aging, leaked documents show conspiracy",
            "Flat Earth society proves NASA has been lying for decades",
            "Miracle diet pill melts fat overnight without exercise or diet",
            "Secret society controls world economy through hidden messages",
            "Aliens built pyramids using advanced technology, archaeologist claims",
            "Drinking lemon water reverses aging and prevents all cancers",
            "Government weather control machines cause natural disasters",
            "Scientists discover time travel is possible using household items",
            "Big pharma suppresses natural cures to maintain profits",
            "Chemtrails contain mind control substances, whistleblower reveals",
            "Ancient civilization had smartphones and internet 5000 years ago",
            "Eating this fruit prevents COVID and all other viruses forever",
            "Secret government base on Mars discovered by satellite images",
            "Billionaire admits to controlling global media through AI robots",
            "Magnetic bracelets cure arthritis better than any medicine",
            "Underground tunnels connect all major cities for elite travel",
            "Drinking bleach kills coronavirus, social media post goes viral",
            "Reptilian people control government from underground bunkers",
            "Microwave ovens are government surveillance devices monitoring your food",
            "Eating raw garlic prevents all forms of cancer and heart disease",
            "Social media algorithms can read your thoughts through phone cameras",
            "Ancient aliens taught humans how to build modern technology",
            "Fluoride in water supply is mind control chemical from corporations",
            "Birds are actually government drones spying on citizens nationwide",
            "Wearing masks causes oxygen deprivation and permanent brain damage",
            "5G towers control weather patterns and cause natural disasters",
            "Pharmaceutical companies hide cancer cure to maintain billion dollar profits",
            "Earth's core is hollow and contains advanced alien civilization",
            "Drinking apple cider vinegar dissolves kidney stones in 24 hours",
            "Government uses subliminal messages in TV commercials for mind control",
            "Eating only fruits for 30 days cures diabetes permanently",
            "Secret military bases on moon discovered by amateur astronomers",
            "Crystals can charge your phone battery and cure depression naturally",
            "Billionaires harvest children's blood for anti-aging treatments secretly",
            "Drinking hydrogen peroxide kills all viruses and bacteria safely",
            "Ancient Egyptians had electricity and used light bulbs in pyramids",
            "Government sprays chemicals from planes to control population growth",
            "Eating turmeric daily prevents Alzheimer's disease and memory loss",
            "Aliens crashed in Roswell and technology was reverse engineered",
            "Drinking distilled water removes all toxins from body permanently",
            "Secret societies control stock market through hidden algorithms",
            "Eating coconut oil daily prevents heart attacks and strokes",
            "Government monitors all phone calls using artificial intelligence systems",
            "Ancient civilizations had flying machines powered by sound frequencies",
            "Drinking green tea extract burns fat faster than any exercise",
            "Pharmaceutical companies create diseases to sell expensive treatments",
            "Earth's magnetic field is weakening due to secret government experiments",
            "Eating raw honey cures all allergies and autoimmune diseases",
            "Aliens abduct humans for genetic experiments in underground facilities",
            "Drinking lemon juice daily dissolves gallstones without surgery needed",
            "Government uses weather modification technology to create hurricanes artificially",
            "Ancient texts describe nuclear weapons used in prehistoric wars",
            "Eating organic food prevents all forms of cancer permanently",
            "Secret military projects involve time travel and parallel dimensions",
            "Drinking alkaline water reverses aging process and extends lifespan",
            "Government implants tracking chips in newborn babies at hospitals",
            "Ancient astronauts visited Earth and taught humans advanced mathematics",
            "Eating fermented foods cures depression better than prescription medications",
            "Billionaire elites plan to reduce world population through engineered pandemics",
            "Drinking colloidal silver kills all infections without side effects",
            "Government uses HAARP technology to control human emotions remotely",
            "Ancient pyramids were actually power plants generating free electricity",
            "Eating keto diet reverses type 2 diabetes in just weeks",
            "Secret space program has colonies on Mars since 1970s",
            "Drinking structured water improves DNA and cellular regeneration naturally",
            "Government covers up UFO crashes to hide advanced alien technology",
            "Ancient civilizations communicated with extraterrestrials through stone monuments",
            "Eating intermittent fasting prevents all age-related diseases permanently"
        ]
        
        # EXPANDED REAL NEWS DATASET (100+ samples)
        real_news = [
            "Stock market shows steady growth following quarterly earnings reports",
            "New environmental policy aims to reduce carbon emissions by 30%",
            "Local university receives grant for renewable energy research project",
            "Healthcare workers receive recognition for their dedication during pandemic",
            "Technology company announces breakthrough in artificial intelligence research",
            "Federal Reserve announces interest rate decision after economic review",
            "Climate change summit brings together world leaders in Geneva",
            "New vaccine trial shows promising results in phase three testing",
            "Supreme Court hears arguments on landmark civil rights case",
            "NASA launches new satellite to monitor global weather patterns",
            "Congress passes bipartisan infrastructure spending bill after months of debate",
            "Major tech company reports quarterly revenue exceeding analyst expectations",
            "International trade agreement signed between multiple nations",
            "University researchers publish study on renewable energy efficiency",
            "Central bank adjusts monetary policy in response to inflation data",
            "New medical device receives FDA approval for treating heart conditions",
            "Education department announces funding for rural school districts",
            "Archaeological team discovers ancient artifacts in Mediterranean excavation",
            "Pharmaceutical company begins clinical trials for new diabetes treatment",
            "Transportation authority unveils plans for high-speed rail expansion",
            "Energy department reports increase in solar power adoption nationwide",
            "Hospital system implements new electronic health record system",
            "Agricultural research shows improved crop yields using sustainable methods",
            "City council approves budget allocation for public transportation improvements",
            "Manufacturing sector shows growth in latest economic indicators report",
            "Scientists publish peer-reviewed research on climate change mitigation strategies",
            "Local hospital receives accreditation for cardiac surgery program excellence",
            "University announces new scholarship program for underrepresented students",
            "City implements new recycling program to reduce landfill waste",
            "Research team develops more efficient solar panel technology",
            "Government agency releases annual report on economic development trends",
            "Medical center opens new cancer treatment facility with advanced equipment",
            "School district adopts new curriculum focused on STEM education",
            "Transportation department completes highway safety improvement project",
            "Agricultural extension office provides drought management resources to farmers",
            "Public health officials announce successful vaccination campaign results",
            "Technology firm partners with university for artificial intelligence research",
            "Environmental agency monitors air quality improvements in urban areas",
            "Hospital network expands telemedicine services to rural communities",
            "Education foundation awards grants for teacher professional development programs",
            "City planning commission approves sustainable development project downtown",
            "Research institute publishes findings on renewable energy storage solutions",
            "Healthcare system implements new patient safety protocols hospital-wide",
            "University extension program offers agricultural training for local farmers",
            "Transportation authority studies feasibility of electric bus fleet conversion",
            "Medical school receives accreditation for new residency training program",
            "Government office releases updated guidelines for small business development",
            "Research center collaborates with industry partners on innovation projects",
            "Public works department completes water infrastructure upgrade project",
            "Educational institution launches online learning platform for continuing education",
            "Healthcare provider expands mental health services in underserved areas",
            "Agricultural cooperative reports successful harvest season despite weather challenges",
            "Technology company invests in local workforce development training programs",
            "Environmental group partners with schools for sustainability education initiatives",
            "Medical research facility receives funding for clinical trial studies",
            "City government implements new digital services for citizen engagement",
            "University researchers develop innovative water purification technology",
            "Healthcare network opens new primary care clinic in suburban location",
            "Education department launches literacy program for adult learners",
            "Transportation agency completes bridge inspection and maintenance project",
            "Agricultural research station develops drought-resistant crop varieties",
            "Technology startup receives venture capital funding for expansion plans",
            "Environmental monitoring station reports improved water quality measurements",
            "Hospital foundation raises funds for new pediatric wing construction",
            "School board approves budget increase for special education services",
            "Public transit system introduces new routes to serve growing neighborhoods",
            "Research university establishes center for sustainable energy studies",
            "Healthcare organization implements electronic health records system upgrade",
            "Agricultural extension service provides pest management training for growers",
            "Technology conference brings together industry leaders and researchers",
            "Environmental agency announces successful species conservation program results",
            "Medical center receives recognition for patient satisfaction scores",
            "Education foundation supports teacher training in underserved school districts"
        ]
        
        data = []
        for text in fake_news:
            data.append({'text': text, 'label': 1})  # 1 for fake
        for text in real_news:
            data.append({'text': text, 'label': 0})  # 0 for real
            
        return pd.DataFrame(data)
    
    def download_real_dataset(self):
        """Download and prepare a real fake news dataset"""
        try:
            # Try to create a more comprehensive dataset
            print("Creating comprehensive fake news dataset...")
            
            # Real news from various reliable sources
            real_news_extended = [
                "The Federal Reserve announced a 0.25% interest rate increase following their monthly meeting to address inflation concerns.",
                "NASA's James Webb Space Telescope captured detailed images of distant galaxies, providing new insights into early universe formation.",
                "The Department of Education released new guidelines for student loan forgiveness programs affecting millions of borrowers nationwide.",
                "Researchers at Stanford University published a peer-reviewed study showing promising results for a new Alzheimer's treatment.",
                "The Supreme Court heard oral arguments in a case that could affect voting rights legislation across multiple states.",
                "Apple Inc. reported quarterly earnings that exceeded Wall Street expectations, driven by strong iPhone and services revenue.",
                "The Centers for Disease Control updated vaccination recommendations based on the latest clinical trial data.",
                "Congress passed a bipartisan infrastructure bill allocating $1.2 trillion for roads, bridges, and broadband expansion.",
                "The World Health Organization announced new guidelines for pandemic preparedness following lessons learned from COVID-19.",
                "Tesla reported record vehicle deliveries in the third quarter, meeting production targets despite supply chain challenges.",
                "The Environmental Protection Agency proposed new regulations to reduce carbon emissions from power plants by 2030.",
                "Microsoft announced a major acquisition of a cloud computing company for $68.7 billion pending regulatory approval.",
                "The International Monetary Fund revised global economic growth projections downward due to geopolitical tensions.",
                "Harvard Medical School researchers identified genetic markers associated with increased risk of heart disease.",
                "The Department of Transportation approved funding for high-speed rail projects in California and Texas.",
                "Amazon reported strong quarterly results driven by cloud services growth and advertising revenue increases.",
                "The Food and Drug Administration approved a new cancer treatment after successful Phase III clinical trials.",
                "The Bureau of Labor Statistics reported unemployment rates declined to 3.7% in the latest monthly jobs report.",
                "Google announced new privacy features for Android devices following user feedback and regulatory requirements.",
                "The National Science Foundation awarded $500 million in grants for climate change research at universities nationwide."
            ]
            
            # Fake news with various conspiracy theories and misinformation
            fake_news_extended = [
                "Secret documents reveal that the government has been hiding evidence of alien contact for over 50 years.",
                "A new study claims that drinking hydrogen peroxide daily can cure cancer, but doctors don't want you to know.",
                "Leaked emails show that tech billionaires are planning to implant microchips in everyone by 2025.",
                "Scientists discover that the Earth is actually flat and NASA has been covering it up with fake satellite images.",
                "Breaking: Pharmaceutical companies have been suppressing a natural cure for diabetes found in common household items.",
                "Government whistleblower reveals that 5G towers are being used to control people's minds through radio frequencies.",
                "Ancient aliens built the pyramids using advanced technology that modern science cannot explain or replicate.",
                "A miracle fruit from the Amazon rainforest can cure all diseases, but Big Pharma is trying to ban it.",
                "Secret military experiments have created weather control technology that causes hurricanes and earthquakes on demand.",
                "Chemtrails contain mind-control chemicals that are being sprayed on the population to make them more compliant.",
                "The moon landing was completely staged in a Hollywood studio, and all the footage is fake.",
                "Vaccines contain tracking devices that allow the government to monitor your location and health data.",
                "A simple baking soda mixture can cure any type of cancer in just 30 days, doctors hate this trick.",
                "Reptilian shapeshifters have infiltrated world governments and are secretly controlling global politics.",
                "Drinking bleach mixed with citric acid creates a miracle cure that eliminates all viruses from your body.",
                "The Illuminati controls the world economy through secret symbols hidden in dollar bills and corporate logos.",
                "Eating only raw foods for 21 days can reverse aging and make you look 20 years younger.",
                "Government agents are using birds as surveillance drones to spy on citizens in their own homes.",
                "A magnetic bracelet can cure arthritis, depression, and high blood pressure without any side effects.",
                "Time travel is real and the government has been using it to manipulate historical events for decades."
            ]
            
            # Combine all data
            all_real = real_news_extended * 3  # Multiply to get more samples
            all_fake = fake_news_extended * 3  # Multiply to get more samples
            
            data = []
            for text in all_fake:
                data.append({'text': text, 'label': 1})
            for text in all_real:
                data.append({'text': text, 'label': 0})
            
            df = pd.DataFrame(data)
            print(f"Created dataset with {len(df)} articles ({len(all_fake)} fake, {len(all_real)} real)")
            return df
            
        except Exception as e:
            print(f"Error creating dataset: {e}")
            print("Falling back to sample dataset...")
            return self.create_sample_data()
    
    def train_models(self, df):
        """Train multiple models and compare performance"""
        X = df['cleaned_text']
        y = df['label']
        
        # ========== DATA PREPARATION ==========
        # Split data into training and testing sets
        print(f"\nSplitting {len(X)} samples into train/test sets...")
        
        # Use appropriate test size based on dataset size
        test_size = 0.25 if len(X) > 100 else 0.3
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
        print(f"   >> Training set: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)")
        print(f"   >> Testing set: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)")
        
        # ========== FEATURE EXTRACTION ==========
        # Convert text to numbers that machines can understand
        print("\nConverting text to numerical features using TF-IDF...")
        X_train_vec = self.vectorizer.fit_transform(X_train)  # Learn vocabulary + transform
        X_test_vec = self.vectorizer.transform(X_test)        # Transform using learned vocabulary
        print(f"   >> Created {X_train_vec.shape[1]} features from text")
        
        results = {}
        
        # ========== MODEL TRAINING SECTION ==========
        # This is where we actually train each machine learning model
        for name, model in self.models.items():
            print(f"\nTRAINING {name} MODEL...")
            print(f"   >> Training on {len(X_train)} samples")
            print(f"   >> Learning patterns from text features...")
            
            # *** THIS IS THE ACTUAL TRAINING LINE ***
            # model.fit() teaches the algorithm to recognize fake vs real news
            model.fit(X_train_vec, y_train)  # <-- TRAINING HAPPENS HERE!
            
            # Save the trained model for later use
            self.trained_models[name] = model
            print(f"   >> {name} training completed!")
            
            # ========== MODEL EVALUATION ==========
            # Test the trained model on unseen data
            print(f"   >> Testing {name} on {len(X_test)} unseen samples...")
            y_pred = model.predict(X_test_vec)  # Make predictions
            
            # Calculate how well the model performed
            accuracy = accuracy_score(y_test, y_pred)
            print(f"   >> {name} achieved {accuracy:.1%} accuracy")
            
            # Show detailed performance metrics
            from sklearn.metrics import precision_score, recall_score, f1_score
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            print(f"   >> Precision: {precision:.3f} | Recall: {recall:.3f} | F1-Score: {f1:.3f}")
            results[name] = {
                'accuracy': accuracy,
                'y_test': y_test,
                'y_pred': y_pred
            }
            
            print(f"{name} Accuracy: {accuracy:.4f}")
            # Uncomment next line to see detailed classification report
            # print(f"Classification Report:\n{classification_report(y_test, y_pred)}")
        
        return results, X_test, y_test
    
    def plot_results(self, results):
        """Visualize model performance"""
        # Accuracy comparison
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        models = list(results.keys())
        accuracies = [results[model]['accuracy'] for model in models]
        
        plt.bar(models, accuracies, color=['skyblue', 'lightgreen', 'lightcoral'])
        plt.title('Model Accuracy Comparison')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        
        for i, acc in enumerate(accuracies):
            plt.text(i, acc + 0.01, f'{acc:.3f}', ha='center')
        
        # Confusion matrix for best model
        best_model = max(results.keys(), key=lambda x: results[x]['accuracy'])
        plt.subplot(1, 2, 2)
        
        cm = confusion_matrix(results[best_model]['y_test'], results[best_model]['y_pred'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
        plt.title(f'Confusion Matrix - {best_model}')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        
        plt.tight_layout()
        plt.show()
    
    def predict_news(self, text, model_name='Naive Bayes'):
        """Predict if a news article is fake or real"""
        if model_name not in self.trained_models:
            return "Model not trained yet!"
        
        # Preprocess text (same as training data)
        cleaned_text = self.preprocess_text(text)
        
        # Convert text to numerical features
        text_vec = self.vectorizer.transform([cleaned_text])
        
        # Use TRAINED model to make prediction
        prediction = self.trained_models[model_name].predict(text_vec)[0]      # 0=Real, 1=Fake
        probability = self.trained_models[model_name].predict_proba(text_vec)[0]  # Confidence scores
        
        result = "FAKE" if prediction == 1 else "REAL"
        confidence = max(probability)
        
        return f"Prediction: {result} (Confidence: {confidence:.2f})"

def main():
    print("FAKE NEWS DETECTION SYSTEM STARTING...\n")
    
    # Download required NLTK data
    try:
        nltk.download('stopwords', quiet=True)
    except:
        pass
    
    # ========== STEP 1: INITIALIZE SYSTEM ==========
    print("Initializing AI models (untrained)...")
    detector = FakeNewsDetector()
    print("   >> 3 ML algorithms ready for training")
    
    # ========== STEP 2: PREPARE DATA ==========
    print("\nLoading and preparing training data...")
    
    # Ask user for dataset preference
    print("\nDataset Options:")
    print("   1. Large comprehensive dataset (200+ articles) - RECOMMENDED")
    print("   2. Sample dataset (140+ articles)")
    
    choice = input("\nChoose dataset (1 or 2, default=1): ").strip()
    
    if choice == '2':
        df = detector.load_and_prepare_data(use_large_dataset=False)
    else:
        df = detector.load_and_prepare_data(use_large_dataset=True)
    
    print(f"   >> Dataset loaded: {df.shape[0]} news articles")
    print(f"   >> Fake news: {sum(df['label'] == 1)} articles")
    print(f"   >> Real news: {sum(df['label'] == 0)} articles")
    
    # ========== STEP 3: TRAIN ALL MODELS ==========
    print("\nSTARTING MODEL TRAINING PROCESS...")
    print("=" * 50)
    results, X_test, y_test = detector.train_models(df)
    print("\nALL MODELS SUCCESSFULLY TRAINED!")
    print("=" * 50)
    
    # ========== STEP 4: VISUALIZE RESULTS ==========
    print("\nGenerating performance charts...")
    detector.plot_results(results)
    
    # ========== STEP 5: TEST TRAINED MODELS ==========
    print("\n" + "="*50)
    print("TESTING TRAINED MODELS WITH NEW ARTICLES")
    print("="*50)
    
    test_articles = [
        "Scientists at MIT have developed a new renewable energy technology that could revolutionize solar power",
        "Breaking: Local celebrity seen eating pizza with fork and knife, experts say this proves alien influence"
    ]
    
    for i, article in enumerate(test_articles, 1):
        print(f"\nTest Article {i}:")
        print(f"   {article}")
        print("\nModel Predictions:")
        for model_name in detector.trained_models.keys():
            result = detector.predict_news(article, model_name)
            print(f"   {model_name}: {result}")

if __name__ == "__main__":
    main()