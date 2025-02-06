import os
import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load class labels dynamically
train_dir = "images_split/train"
class_labels = ['Afghan Hound', 'African hunting dog', 'Airedale', 'American Staffordshire terrier', 'Appenzeller', 'Australian terrier', 'Basset', 'Beagle', 'Bedlington Terrier', 'Bernese mountain dog', 'Black and Tan Coonhound', 'Blenheim Spaniel', 'Bloodhound', 'Bluetick', 'Border Terrier', 'Border collie', 'Borzoi', 'Boston bull', 'Bouvier des Flandres', 'Brabancon griffon', 'Brittany spaniel', 'Cairn', 'Cardigan', 'Chesapeake Bay retriever', 'Chihuahua', 'Dandie Dinmont', 'Doberman', 'English Foxhound', 'English setter', 'English springer', 'EntleBucher', 'Eskimo dog', 'French bulldog', 'German shepherd', 'German short haired pointer', 'Gordon setter', 'Great Dane', 'Great Pyrenees', 'Greater Swiss Mountain dog', 'Ibizan Hound', 'Irish Wolfhound', 'Irish setter', 'Irish terrier', 'Irish water spaniel', 'Italian Greyhound', 'Japanese Spaniel', 'Kerry blue terrier', 'Labrador retriever', 'Lakeland terrier', 'Leonberg', 'Lhasa', 'Maltese', 'Mexican hairless', 'Newfoundland', 'Norfolk terrier', 'Norwegian elkhound', 'Norwich terrier', 'Old English sheepdog', 'Otterhound', 'Papillon', 'Pekinese', 'Pembroke', 'Pomeranian', 'Redbone', 'Rhodesian Ridgeback', 'Rottweiler', 'Saint Bernard', 'Saluki', 'Samoyed', 'Scotch terrier', 'Scottish Deerhound', 'Sealyham terrier', 'Shetland sheepdog', 'Shih Tzu', 'Siberian husky', 'Silky terrier', 'Soft coated wheaten terrier', 'Staffordshire Bullterrier', 'Sussex spaniel', 'Tibetan mastiff', 'Tibetan terrier', 'Toy Terrier', 'Walker Hound', 'Weimaraner', 'Welsh springer spaniel', 'West Highland white terrier', 'Whippet', 'Yorkshire terrier', 'affenpinscher', 'basenji', 'boxer', 'briard', 'bull mastiff', 'chow', 'clumber', 'cocker spaniel', 'collie', 'curly coated retriever', 'dhole', 'dingo', 'flat coated retriever', 'giant schnauzer', 'golden retriever', 'groenendael', 'keeshond', 'kelpie', 'komondor', 'kuvasz', 'malamute', 'malinois', 'miniature pinscher', 'miniature poodle', 'miniature schnauzer', 'pug', 'schipperke', 'standard poodle', 'standard schnauzer', 'toy poodle', 'vizsla', 'wire haired fox terrier']

# Load trained model
model = tf.keras.models.load_model("best_model_finetuned.h5")

# Function to preprocess image
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = image.img_to_array(img) / 255.0  
    img_array = np.expand_dims(img_array, axis=0)  
    return img_array

# Streamlit UI
st.title("🐶 Dog Breed Classification")
st.write("Upload an image of a dog to identify its breed.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    img = Image.open(uploaded_file)
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess and predict
    img_array = preprocess_image(img)
    predictions = model.predict(img_array)[0]

    # Get top prediction
    predicted_class = class_labels[np.argmax(predictions)]
    confidence = np.max(predictions)

    # Display results
    st.subheader(f"Prediction: {predicted_class}")
    st.write(f"Confidence: {confidence:.2%}")

    # Show top 5 probabilities
    st.subheader("Top 5 Predictions:")
    top_5_indices = np.argsort(predictions)[::-1][:5]
    for i in top_5_indices:
        st.write(f"{class_labels[i]}: {predictions[i]:.2%}")
