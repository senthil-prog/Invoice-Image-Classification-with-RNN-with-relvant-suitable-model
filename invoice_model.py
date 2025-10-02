import os
import numpy as np
from PIL import Image, ImageDraw
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt

def generate_images(num=200):
    categories = ['invoice', 'receipt', 'purchase_order', 'bill']
    images, labels = [], []
    
    # Create data folder
    os.makedirs('data', exist_ok=True)
    
    for cat_idx, cat in enumerate(categories):
        # Create category folders
        os.makedirs(f'data/{cat}', exist_ok=True)
        
        print(f"Generating {num} {cat} images...")
        for i in range(num):
            img = Image.new('RGB', (128, 128), 'white')
            draw = ImageDraw.Draw(img)
            
            if cat == 'invoice':
                draw.rectangle([10, 10, 118, 30], fill='blue')
                draw.text((40, 15), "INVOICE", fill='white')
                for j in range(3):
                    draw.text((15, 50+j*20), f"Item ${np.random.randint(10,99)}", fill='black')
            elif cat == 'receipt':
                draw.text((35, 15), "RECEIPT", fill='black')
                draw.line([10, 30, 118, 30], fill='black', width=2)
                for j in range(4):
                    draw.text((15, 40+j*15), f"Prod ${np.random.randint(5,50)}", fill='black')
            elif cat == 'purchase_order':
                draw.rectangle([10, 10, 118, 118], outline='green', width=3)
                draw.text((25, 20), "PURCHASE ORDER", fill='black')
                for j in range(3):
                    draw.text((15, 50+j*20), f"Qty {np.random.randint(1,20)}", fill='black')
            else:
                draw.rectangle([10, 10, 118, 40], fill='orange')
                draw.text((50, 20), "BILL", fill='white')
                for j in range(3):
                    draw.text((15, 55+j*20), f"Fee ${np.random.randint(10,99)}", fill='black')
            
            # Save image to disk
            img.save(f'data/{cat}/{cat}_{i:04d}.png')
            
            # Convert to array for training
            arr = np.array(img) / 255.0
            images.append(arr)
            labels.append(cat_idx)
    
    print(f"\n✓ Images saved in 'data/' folder")
    return np.array(images), np.array(labels), categories

def build_model(input_shape, num_classes):
    model = keras.Sequential([
        layers.Input(shape=input_shape),
        layers.Bidirectional(layers.LSTM(128, return_sequences=True)),
        layers.Dropout(0.3),
        layers.Bidirectional(layers.LSTM(64)),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train(X_train, y_train, X_val, y_val):
    X_train_seq = X_train.reshape(X_train.shape[0], X_train.shape[1], -1)
    X_val_seq = X_val.reshape(X_val.shape[0], X_val.shape[1], -1)
    
    model = build_model((X_train_seq.shape[1], X_train_seq.shape[2]), 4)
    model.summary()
    
    print("\nTraining...")
    history = model.fit(
        X_train_seq, y_train,
        validation_data=(X_val_seq, y_val),
        epochs=30, batch_size=32,
        callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)]
    )
    
    os.makedirs('model', exist_ok=True)
    model.save('model/invoice_rnn.keras')
    print("\n✓ Model saved to 'model/invoice_rnn.keras'")
    return model, history

def plot_results(model, X_test, y_test, history, categories):
    os.makedirs('results', exist_ok=True)
    X_test_seq = X_test.reshape(X_test.shape[0], X_test.shape[1], -1)
    y_pred = np.argmax(model.predict(X_test_seq, verbose=0), axis=1)
    
    # Graph 1: Training History
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train', linewidth=2)
    plt.plot(history.history['val_loss'], label='Validation', linewidth=2)
    plt.title('Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Train', linewidth=2)
    plt.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    plt.title('Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('results/1_history.png', dpi=150)
    print("✓ Saved: results/1_history.png")
    plt.show()
    plt.close()
    
    # Graph 2: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, cmap='Blues')
    for i in range(4):
        for j in range(4):
            plt.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=14, fontweight='bold')
    plt.xticks(range(4), categories, rotation=45)
    plt.yticks(range(4), categories)
    plt.xlabel('Predicted', fontsize=12)
    plt.ylabel('True', fontsize=12)
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.colorbar()
    plt.tight_layout()
    plt.savefig('results/2_confusion_matrix.png', dpi=150)
    print("✓ Saved: results/2_confusion_matrix.png")
    plt.show()
    plt.close()
    
    # Graph 3: Metrics
    report = classification_report(y_test, y_pred, target_names=categories, output_dict=True)
    metrics = ['precision', 'recall', 'f1-score']
    data = [[report[cat][m] for m in metrics] for cat in categories]
    
    plt.figure(figsize=(10, 6))
    im = plt.imshow(data, cmap='YlGn', vmin=0, vmax=1)
    plt.xticks(range(3), metrics, fontsize=12)
    plt.yticks(range(4), categories, fontsize=12)
    for i in range(4):
        for j in range(3):
            plt.text(j, i, f'{data[i][j]:.2f}', ha='center', va='center', fontsize=12, fontweight='bold')
    plt.colorbar(im)
    plt.title('Classification Metrics', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('results/3_metrics.png', dpi=150)
    print("✓ Saved: results/3_metrics.png")
    plt.show()
    plt.close()
    
    # Print Results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    test_loss, test_acc = model.evaluate(X_test_seq, y_test, verbose=0)
    print(f"Test Accuracy: {test_acc:.2%}")
    print(f"Test Loss: {test_loss:.4f}")
    print("\n" + classification_report(y_test, y_pred, target_names=categories))

def main():
    print("="*60)
    print("INVOICE RNN CLASSIFIER")
    print("="*60 + "\n")
    
    print("[1/4] Generating images...")
    X, y, cats = generate_images(200)
    print(f"✓ Total: {len(X)} images\n")
    
    print("[2/4] Splitting data...")
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
    print(f"✓ Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}\n")
    
    print("[3/4] Training model...")
    model, history = train(X_train, y_train, X_val, y_val)
    
    print("\n[4/4] Creating graphs...")
    plot_results(model, X_test, y_test, history, cats)
    
    print("\n" + "="*60)
    print("✓ COMPLETE!")
    print("="*60)
    print("\nGenerated Files:")
    print("  📁 data/          - 800 invoice images")
    print("  📁 model/         - Trained RNN model")
    print("  📁 results/       - 3 performance graphs")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()