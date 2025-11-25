# phase4_training.py
import tensorflow as tf
import numpy as np
import json
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os

class PostProcessorTrainer:
    def __init__(self):
        self.vocab = {'<PAD>': 0, '<UNK>': 1}
        self.max_seq_length = 12
        self.model_path = 'asl_post_processor.h5'
        self.vocab_path = 'asl_vocab.pkl'
        
    def load_dataset(self, filename='asl_sequences_dataset.json'):
        """Carga y prepara el dataset"""
        with open(filename, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        # Construir vocabulario
        all_chars = set()
        for sample in dataset:
            for char in sample['input_sequence'] + list(sample['output_word']):
                all_chars.add(char)
        
        for i, char in enumerate(sorted(all_chars), start=2):
            self.vocab[char] = i
        
        self.vocab_size = len(self.vocab)
        
        # Guardar vocabulario inmediatamente
        self.save_vocabulary()
        
        return dataset
    
    def save_vocabulary(self):
        """Guarda el vocabulario para uso futuro"""
        with open(self.vocab_path, 'wb') as f:
            pickle.dump(self.vocab, f)
        print(f"✅ Vocabulario guardado: {self.vocab_path}")
        print(f"   - Tamaño del vocabulario: {self.vocab_size}")
    
    def load_vocabulary(self):
        """Carga el vocabulario desde archivo"""
        if os.path.exists(self.vocab_path):
            with open(self.vocab_path, 'rb') as f:
                self.vocab = pickle.load(f)
            self.vocab_size = len(self.vocab)
            print(f"✅ Vocabulario cargado: {self.vocab_size} caracteres")
            return True
        return False
    
    def prepare_data(self, dataset):
        """Prepara datos para entrenamiento"""
        X, y = [], []
        
        for sample in dataset:
            # Convertir secuencia de entrada a índices
            input_seq = [self.vocab.get(char, self.vocab['<UNK>']) 
                        for char in sample['input_sequence']]
            
            # Padding
            if len(input_seq) < self.max_seq_length:
                input_seq += [self.vocab['<PAD>']] * (self.max_seq_length - len(input_seq))
            else:
                input_seq = input_seq[:self.max_seq_length]
            
            # Convertir palabra de salida a índices
            output_word = sample['output_word']
            # Para simplificar, usamos la primera letra como clase (puedes expandir esto)
            target_class = self.vocab.get(output_word[0], self.vocab['<UNK>'])
            
            X.append(input_seq)
            y.append(target_class)
        
        return np.array(X), np.array(y)
    
    def build_model(self):
        """Construye el modelo RNN para post-procesamiento"""
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=64,
                input_length=self.max_seq_length
            ),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
            tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(self.vocab_size, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train(self, X_train, y_train, X_val, y_val):
        """Entrena el modelo y guarda automáticamente"""
        model = self.build_model()
        
        print("🧠 Entrenando post-procesador RNN...")
        
        # Callback para guardar el mejor modelo
        checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            self.model_path,
            monitor='val_accuracy',
            save_best_only=True,
            save_weights_only=False,
            mode='max',
            verbose=1
        )
        
        # Callback para early stopping
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True,
            verbose=1
        )
        
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=20,
            batch_size=32,
            verbose=1,
            callbacks=[checkpoint_callback, early_stopping]
        )
        
        # Guardar modelo final por si acaso
        model.save(self.model_path)
        print(f"✅ Modelo guardado automáticamente: {self.model_path}")
        
        return model, history
    
    def load_trained_model(self):
        """Carga el modelo entrenado previamente"""
        if os.path.exists(self.model_path):
            model = tf.keras.models.load_model(self.model_path)
            print(f"✅ Modelo cargado: {self.model_path}")
            return model
        else:
            print("❌ No se encontró modelo entrenado")
            return None
    
    def plot_training_history(self, history):
        """Grafica el historial de entrenamiento"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        ax1.plot(history.history['accuracy'], label='Precisión Entrenamiento')
        ax1.plot(history.history['val_accuracy'], label='Precisión Validación')
        ax1.set_title('Precisión del Modelo')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Precisión')
        ax1.legend()
        
        ax2.plot(history.history['loss'], label='Pérdida Entrenamiento')
        ax2.plot(history.history['val_loss'], label='Pérdida Validación')
        ax2.set_title('Pérdida del Modelo')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Pérdida')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    print("🎯 FASE 4: ENTRENAMIENTO DEL POST-PROCESADOR")
    print("=" * 50)
    
    trainer = PostProcessorTrainer()
    
    # Verificar si ya existe modelo entrenado
    existing_model = trainer.load_trained_model()
    if existing_model:
        print("📁 Modelo ya existe. ¿Quieres reentrenar? (s/n)")
        response = input().strip().lower()
        if response != 's':
            print("✅ Usando modelo existente")
            return existing_model, trainer.vocab
    
    # Si no existe o se quiere reentrenar, proceder con entrenamiento
    dataset = trainer.load_dataset()
    X, y = trainer.prepare_data(dataset)
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    print(f"📊 Datos de entrenamiento: {X_train.shape[0]} muestras")
    print(f"📊 Datos de validación: {X_val.shape[0]} muestras")
    print(f"📊 Datos de prueba: {X_test.shape[0]} muestras")
    print(f"🔤 Tamaño del vocabulario: {trainer.vocab_size}")
    
    # Entrenar modelo
    model, history = trainer.train(X_train, y_train, X_val, y_val)
    
    # Graficar historial
    trainer.plot_training_history(history)
    
    # Evaluación inicial
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"📈 Precisión en prueba: {test_accuracy:.4f}")
    
    print(f"\n🎉 ENTRENAMIENTO COMPLETADO")
    print(f"📁 Modelo guardado: {trainer.model_path}")
    print(f"📁 Vocabulario guardado: {trainer.vocab_path}")
    
    return model, trainer.vocab

if __name__ == "__main__":
    model, vocab = main()