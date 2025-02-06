from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Define directories for the dataset
train_dir = 'images_split/train'
val_dir = 'images_split/validation'

# Define ImageDataGenerators for training and validation
train_datagen = ImageDataGenerator(
    rescale=1./255, 
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(224, 224), 
    batch_size=64,
    class_mode='categorical', 
    shuffle=True,
    seed=42
)

val_generator = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(224, 224),
    batch_size=64,
    class_mode='categorical',
    shuffle=False
)

# Load the best model saved during the initial training phase
model = load_model('best_model.h5')

# Unfreeze the last 30 layers of the base model for fine-tuning
for layer in model.layers[0]._layers[-50:]: 
    layer.trainable = True

# Recompile with a lower learning rate for fine-tuning
model.compile(
    optimizer=Adam(learning_rate=0.00005), 
    loss='categorical_crossentropy', 
    metrics=['accuracy']
)

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model_finetuned.h5', monitor='val_loss', save_best_only=True)

# Fine-tune model (Second Phase)
history_finetune = model.fit(
    train_generator, 
    validation_data=val_generator, 
    epochs=5,  
    callbacks=[early_stopping, checkpoint]
)

# Save the fine-tuned model
model.save('inceptionv3_dog_breed_detection_finetuned.h5')
print("Fine-tuning complete!")
