# phase5_evaluation.py
import tensorflow as tf
import numpy as np
import json
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd
from sklearn.model_selection import train_test_split

class ModelEvaluator:
    def __init__(self):
        self.vocab = {}
        self.reverse_vocab = {}
        
    def load_trained_system(self):
        """Carga el modelo y vocabulario entrenados"""
        try:
            # Cargar modelo
            model = tf.keras.models.load_model('asl_post_processor.h5')
            print("✅ Modelo post-procesador cargado")
            
            # Cargar vocabulario
            with open('asl_vocab.pkl', 'rb') as f:
                self.vocab = pickle.load(f)
            self.reverse_vocab = {v: k for k, v in self.vocab.items()}
            print(f"✅ Vocabulario cargado: {len(self.vocab)} caracteres")
            
            return model
        except Exception as e:
            print(f"❌ Error cargando sistema entrenado: {e}")
            return None
    
    def load_and_prepare_data(self, filename='asl_sequences_dataset.json'):
        """Carga y prepara datos para evaluación"""
        with open(filename, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        X, y = [], []
        max_seq_length = 12
        
        for sample in dataset:
            # Convertir secuencia de entrada a índices
            input_seq = [self.vocab.get(char, self.vocab['<UNK>']) 
                        for char in sample['input_sequence']]
            
            # Padding
            if len(input_seq) < max_seq_length:
                input_seq += [self.vocab['<PAD>']] * (max_seq_length - len(input_seq))
            else:
                input_seq = input_seq[:max_seq_length]
            
            # Convertir palabra de salida
            output_word = sample['output_word']
            target_class = self.vocab.get(output_word[0], self.vocab['<UNK>'])
            
            X.append(input_seq)
            y.append(target_class)
        
        return np.array(X), np.array(y)
    
    def evaluate_model(self, model, X_test, y_test):
        """Evalúa el modelo y genera métricas"""
        # Predicciones
        y_pred_proba = model.predict(X_test, verbose=0)
        y_pred = np.argmax(y_pred_proba, axis=1)
        
        # Métricas
        accuracy = np.mean(y_pred == y_test)
        
        print("🎯 FASE 5: EVALUACIÓN DEL MODELO")
        print("=" * 50)
        print(f"📊 Precisión General: {accuracy:.4f}")
        print(f"📊 Total de muestras: {len(X_test)}")
        
        return y_pred, y_pred_proba
    
    def plot_confusion_matrix(self, y_true, y_pred, title="Matriz de Confusión"):
        """Genera y muestra matriz de confusión"""
        # Filtrar clases que aparecen en los datos de prueba
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        class_labels = [self.reverse_vocab.get(cls, 'UNK') for cls in unique_classes]
        
        cm = confusion_matrix(y_true, y_pred, labels=unique_classes)
        
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_labels, yticklabels=class_labels)
        plt.title(title)
        plt.xlabel('Predicción')
        plt.ylabel('Real')
        plt.xticks(rotation=45)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return cm

def main():
    print("🎯 FASE 5: EVALUACIÓN COMPLETA")
    print("=" * 50)
    
    # Cargar sistema entrenado
    evaluator = ModelEvaluator()
    model = evaluator.load_trained_system()
    
    if model is None:
        print("❌ No se pudo cargar el modelo. Ejecuta Fase 4 primero.")
        return
    
    # Cargar y preparar datos
    X, y = evaluator.load_and_prepare_data()
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Evaluar
    y_pred, y_pred_proba = evaluator.evaluate_model(model, X_test, y_test)
    
    # Matriz de confusión
    cm = evaluator.plot_confusion_matrix(y_test, y_pred)
    
    print("✅ Evaluación completada. Matriz de confusión guardada.")

if __name__ == "__main__":
    main()