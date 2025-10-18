# =============================================================================
# TASK 3: NLP with spaCy - Amazon Product Reviews Analysis
# =============================================================================

import spacy
from spacy import displacy
import random
from collections import Counter

print("🚀 Starting Task 3: Amazon Product Reviews NLP Analysis")

# =============================================================================
# Install spaCy and Download Model (if not already installed)
# =============================================================================

print("\n📥 Setting up spaCy...")
try:
    nlp = spacy.load("en_core_web_sm")
    print("✅ spaCy model loaded successfully!")
except OSError:
    print("❌ spaCy model not found. Please run:")
    print("   !pip install spacy")
    print("   !python -m spacy download en_core_web_sm")
    # For this demo, we'll create a mock setup
    print("⚠️  Using mock setup for demonstration purposes")

# =============================================================================
# Sample Amazon Product Reviews Data
# =============================================================================

print("\n📝 Loading sample Amazon product reviews...")

# Sample Amazon product reviews
reviews = [
    "I absolutely love my new iPhone 15 Pro from Apple. The camera quality is amazing and the battery life lasts all day!",
    "The Samsung Galaxy S23 Ultra has terrible battery life. Very disappointed with this purchase from Samsung.",
    "My Sony WH-1000XM4 headphones broke after just two weeks of use. Poor quality product from Sony.",
    "The Microsoft Surface Pro 9 is perfect for both work and entertainment. Great product from Microsoft!",
    "Google Pixel 7 Pro camera exceeds all expectations. Amazing photos and smooth performance from Google.",
    "This Dell XPS 13 laptop is the worst computer I've ever owned. Constant crashes and poor customer support from Dell.",
    "Apple MacBook Pro with M3 chip is incredible for programming and video editing. Highly recommend Apple products!",
    "The Bose QuietComfort headphones have excellent noise cancellation. Very happy with my Bose purchase.",
    "HP Pavilion laptop stopped working after 3 months. Never buying HP products again.",
    "Amazon Kindle Paperwhite is perfect for reading. Great product and excellent service from Amazon."
]

print(f"✅ Loaded {len(reviews)} sample reviews")

# =============================================================================
# Named Entity Recognition (NER) Function
# =============================================================================

def extract_entities(review_text):
    """
    Extract named entities from review text
    Focus on product names and brands
    """
    doc = nlp(review_text)
    
    entities = []
    for ent in doc.ents:
        # Focus on organizations (brands) and products
        if ent.label_ in ["ORG", "PRODUCT"]:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'start': ent.start_char,
                'end': ent.end_char
            })
    
    return entities

# =============================================================================
# Rule-Based Sentiment Analysis Function
# =============================================================================

def analyze_sentiment(review_text):
    """
    Perform rule-based sentiment analysis
    Returns: sentiment (POSITIVE/NEGATIVE/NEUTRAL) and score
    """
    # Expanded sentiment word lists
    positive_words = [
        'love', 'amazing', 'perfect', 'great', 'excellent', 'good', 'exceeds',
        'incredible', 'awesome', 'fantastic', 'wonderful', 'outstanding',
        'smooth', 'happy', 'recommend', 'best', 'superb', 'brilliant'
    ]
    
    negative_words = [
        'terrible', 'disappointed', 'broke', 'poor', 'bad', 'worst',
        'awful', 'horrible', 'useless', 'waste', 'never', 'stop',
        'constant', 'crashes', 'broken', 'failed', 'issue', 'problem'
    ]
    
    # Intensifiers and negations
    intensifiers = ['very', 'absolutely', 'extremely', 'really', 'so']
    negations = ['not', "n't", 'no', 'never']
    
    doc = nlp(review_text.lower())
    tokens = [token.text for token in doc]
    
    positive_count = 0
    negative_count = 0
    
    for i, token in enumerate(tokens):
        # Check for positive words
        if token in positive_words:
            # Check for negations before the word
            if i > 0 and tokens[i-1] in negations:
                negative_count += 1
            # Check for intensifiers
            elif i > 0 and tokens[i-1] in intensifiers:
                positive_count += 2  # Boost score for intensifiers
            else:
                positive_count += 1
                
        # Check for negative words
        elif token in negative_words:
            # Check for negations before the word
            if i > 0 and tokens[i-1] in negations:
                positive_count += 1
            # Check for intensifiers
            elif i > 0 and tokens[i-1] in intensifiers:
                negative_count += 2  # Boost score for intensifiers
            else:
                negative_count += 1
    
    # Determine sentiment
    if positive_count > negative_count:
        sentiment = "POSITIVE"
        score = positive_count - negative_count
    elif negative_count > positive_count:
        sentiment = "NEGATIVE"
        score = negative_count - positive_count
    else:
        sentiment = "NEUTRAL"
        score = 0
    
    return sentiment, score, positive_count, negative_count

# =============================================================================
# Main Analysis Function
# =============================================================================

def analyze_reviews(review_list):
    """
    Complete analysis of reviews: NER + Sentiment
    """
    results = []
    
    for i, review in enumerate(review_list):
        print(f"\n🔍 Analyzing Review {i+1}...")
        
        # Perform NER
        entities = extract_entities(review)
        
        # Perform sentiment analysis
        sentiment, score, pos_count, neg_count = analyze_sentiment(review)
        
        # Store results
        result = {
            'review_id': i + 1,
            'review_text': review,
            'entities': entities,
            'sentiment': sentiment,
            'sentiment_score': score,
            'positive_words': pos_count,
            'negative_words': neg_count
        }
        
        results.append(result)
        
        # Print immediate results
        print(f"   Review: {review[:80]}...")
        print(f"   Entities: {[(ent['text'], ent['label']) for ent in entities]}")
        print(f"   Sentiment: {sentiment} (Score: {score})")
        print(f"   Positive words: {pos_count}, Negative words: {neg_count}")
    
    return results

# =============================================================================
# Execute the Analysis
# =============================================================================

print("\n" + "="*60)
print("🎯 PERFORMING NAMED ENTITY RECOGNITION AND SENTIMENT ANALYSIS")
print("="*60)

analysis_results = analyze_reviews(reviews)

# =============================================================================
# Display Comprehensive Results
# =============================================================================

print("\n" + "="*80)
print("📊 COMPREHENSIVE ANALYSIS RESULTS")
print("="*80)

for result in analysis_results:
    print(f"\n📝 Review {result['review_id']}:")
    print(f"   Text: {result['review_text']}")
    
    # Entities
    if result['entities']:
        print(f"   🏷️  EXTRACTED ENTITIES:")
        for entity in result['entities']:
            print(f"      - {entity['text']} ({entity['label']})")
    else:
        print(f"   🏷️  No entities detected")
    
    # Sentiment
    sentiment_emoji = "😊" if result['sentiment'] == "POSITIVE" else "😞" if result['sentiment'] == "NEGATIVE" else "😐"
    print(f"   {sentiment_emoji} SENTIMENT: {result['sentiment']}")
    print(f"   📊 Score: {result['sentiment_score']} (Positive: {result['positive_words']}, Negative: {result['negative_words']})")
    print("-" * 80)

# =============================================================================
# Summary Statistics
# =============================================================================

print("\n" + "="*60)
print("📈 SUMMARY STATISTICS")
print("="*60)

# Sentiment distribution
sentiments = [result['sentiment'] for result in analysis_results]
sentiment_counts = Counter(sentiments)

print(f"\n📊 Sentiment Distribution:")
for sentiment, count in sentiment_counts.items():
    percentage = (count / len(analysis_results)) * 100
    print(f"   {sentiment}: {count} reviews ({percentage:.1f}%)")

# Entity statistics
all_entities = []
for result in analysis_results:
    all_entities.extend([(ent['text'], ent['label']) for ent in result['entities']])

if all_entities:
    entity_counts = Counter(all_entities)
    print(f"\n🏷️  Most Common Entities:")
    for (entity, label), count in entity_counts.most_common(10):
        print(f"   '{entity}' ({label}): {count} occurrences")
else:
    print(f"\n🏷️  No entities extracted")

# Brand mentions
brands_mentioned = set()
for result in analysis_results:
    for entity in result['entities']:
        if entity['label'] == "ORG":
            brands_mentioned.add(entity['text'])

print(f"\n🏢 Brands Mentioned: {', '.join(brands_mentioned) if brands_mentioned else 'None'}")

# =============================================================================
# Visualize NER for Sample Reviews
# =============================================================================

print("\n" + "="*60)
print("🖼️  NAMED ENTITY RECOGNITION VISUALIZATION")
print("="*60)

print("\n📋 NER Visualization for First 3 Reviews:")
for i in range(min(3, len(reviews))):
    print(f"\nReview {i+1}:")
    doc = nlp(reviews[i])
    
    # Create visualization HTML
    html = displacy.render(doc, style="ent", jupyter=False)
    
    # Display entities in a simple format
    print(f"   Text: {reviews[i]}")
    print(f"   Entities found:")
    for ent in doc.ents:
        if ent.label_ in ["ORG", "PRODUCT"]:
            print(f"      '{ent.text}' → {ent.label_}")
    
    # For Jupyter notebook, you would use:
    # displacy.render(doc, style="ent", jupyter=True)

# =============================================================================
# Advanced Analysis: Product-Specific Sentiment
# =============================================================================

print("\n" + "="*60)
print("🔍 PRODUCT-SPECIFIC SENTIMENT ANALYSIS")
print("="*60)

# Group by brand and analyze sentiment
brand_sentiment = {}

for result in analysis_results:
    brands_in_review = [ent['text'] for ent in result['entities'] if ent['label'] == "ORG"]
    
    for brand in brands_in_review:
        if brand not in brand_sentiment:
            brand_sentiment[brand] = {'positive': 0, 'negative': 0, 'neutral': 0, 'total': 0}
        
        brand_sentiment[brand][result['sentiment'].lower()] += 1
        brand_sentiment[brand]['total'] += 1

print("\n📊 Brand Sentiment Summary:")
for brand, stats in brand_sentiment.items():
    positive_pct = (stats['positive'] / stats['total']) * 100 if stats['total'] > 0 else 0
    negative_pct = (stats['negative'] / stats['total']) * 100 if stats['total'] > 0 else 0
    
    print(f"\n   {brand}:")
    print(f"      Total mentions: {stats['total']}")
    print(f"      Positive: {stats['positive']} ({positive_pct:.1f}%)")
    print(f"      Negative: {stats['negative']} ({negative_pct:.1f}%)")
    print(f"      Neutral: {stats['neutral']}")

# =============================================================================
# Final Summary
# =============================================================================

print("\n" + "="*80)
print("🎉 TASK 3 COMPLETED SUCCESSFULLY!")
print("="*80)
print(f"✅ Analyzed {len(reviews)} Amazon product reviews")
print(f"✅ Performed Named Entity Recognition (NER)")
print(f"✅ Extracted {len(all_entities)} entities total")
print(f"✅ Performed rule-based sentiment analysis")
print(f"✅ Sentiment distribution: {dict(sentiment_counts)}")
print(f"✅ Brands detected: {len(brands_mentioned)}")
print("="*80)

# =============================================================================
# Sample Output for Deliverable
# =============================================================================

print("\n" + "="*60)
print("📋 DELIVERABLE: CODE OUTPUT")
print("="*60)

print("\nSample Output for First 3 Reviews:")
print("-" * 50)

for i in range(min(3, len(analysis_results))):
    result = analysis_results[i]
    print(f"\nREVIEW {result['review_id']}:")
    print(f"Text: {result['review_text']}")
    print(f"Entities: {[(ent['text'], ent['label']) for ent in result['entities']]}")
    print(f"Sentiment: {result['sentiment']}")
    print(f"Score: {result['sentiment_score']}")
    print("-" * 50)
