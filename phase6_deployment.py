# phase6_deployment.py
import tensorflow as tf
import numpy as np
import streamlit as st
import cv2
import mediapipe as mp
import pickle
import os
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode
import warnings
warnings.filterwarnings('ignore')

class ASLSystem:
    def __init__(self):
        self.letter_model = None
        self.post_processor = None
        self.hands = None
        self.vocab = {}
        self.reverse_vocab = {}
        self.max_seq_length = 12
        self.letter_buffer = []
        self.buffer_size = 5
        
        # Cargar modelos automáticamente
        self.load_models()
    
    def load_models(self):
        """Carga todos los modelos necesarios"""
        # Modelo de detección de letras
        try:
            self.letter_model = tf.keras.models.load_model('ASLmodelF.h5', compile=False)
            st.success("✅ Modelo de letras cargado")
        except Exception as e:
            st.error(f"❌ Error cargando modelo de letras: {e}")
        
        # Modelo post-procesador (si existe)
        try:
            if os.path.exists('asl_post_processor.h5'):
                self.post_processor = tf.keras.models.load_model('asl_post_processor.h5', compile=False)
                
                # Cargar vocabulario
                if os.path.exists('asl_vocab.pkl'):
                    with open('asl_vocab.pkl', 'rb') as f:
                        self.vocab = pickle.load(f)
                    self.reverse_vocab = {v: k for k, v in self.vocab.items()}
                    st.success("✅ Post-procesador y vocabulario cargados")
                else:
                    st.warning("⚠️ Vocabulario no encontrado, usando básico")
                    self.setup_basic_vocab()
            else:
                st.warning("⚠️ Post-procesador no entrenado aún")
                self.setup_basic_vocab()
                
        except Exception as e:
            st.warning(f"⚠️ Post-procesador no disponible: {e}")
            self.setup_basic_vocab()
        
        # Mediapipe Hands
        try:
            mp_hands = mp.solutions.hands
            self.hands = mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
        except Exception as e:
            st.error(f"❌ Error cargando Mediapipe: {e}")
    
    def setup_basic_vocab(self):
        """Configura vocabulario básico como fallback"""
        chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZ012345 "
        self.vocab = {char: i for i, char in enumerate(chars, 2)}
        self.vocab['<PAD>'] = 0
        self.vocab['<UNK>'] = 1
        self.reverse_vocab = {v: k for k, v in self.vocab.items()}

    
    def process_landmarks(self, landmarks):
        """Procesa landmarks y devuelve letra detectada"""
        if self.letter_model is None:
            return None, 0.0
        
        landmarks_array = np.array([landmarks])
        prediction = self.letter_model.predict(landmarks_array, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = np.max(prediction[0])
        
        LABELS = ['A','B','C','D','E','F','G','H','I','J','K','L','M',
                 'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
                 'del','space']
        
        if predicted_class < len(LABELS):
            return LABELS[predicted_class], confidence
        return None, 0.0
    
    def post_process_sequence(self, sequence):
        """Aplica post-procesamiento a secuencia de letras"""
        if self.post_processor is None or len(sequence) == 0:
            return " ".join(sequence)  # Fallback: unir con espacios
        
        # Convertir a formato del modelo
        input_seq = [self.vocab.get(char, self.vocab['<UNK>']) for char in sequence]
        
        # Padding
        if len(input_seq) < self.max_seq_length:
            input_seq += [self.vocab['<PAD>']] * (self.max_seq_length - len(input_seq))
        else:
            input_seq = input_seq[:self.max_seq_length]
        
        # Predecir
        prediction = self.post_processor.predict(np.array([input_seq]), verbose=0)
        predicted_class = np.argmax(prediction[0])
        
        # Convertir de vuelta a caracter
        corrected_char = self.reverse_vocab.get(predicted_class, '?')
        
        return corrected_char
    
    def update_buffer(self, new_letter, confidence):
        """Actualiza el buffer de letras con la nueva detección"""
        if confidence > 0.7:  # Solo añadir si la confianza es alta
            self.letter_buffer.append(new_letter)
            
            # Mantener tamaño máximo del buffer
            if len(self.letter_buffer) > self.buffer_size:
                self.letter_buffer.pop(0)

class EnhancedSignDetector(VideoTransformerBase):
    def __init__(self, asl_system):
        self.asl_system = asl_system
        self.current_word = ""
    
    def transform(self, frame):
        try:
            img = frame.to_ndarray(format="bgr24")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.asl_system.hands.process(img_rgb)
            
            current_letter = None
            confidence = 0.0
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # Dibujar landmarks
                    mp.solutions.drawing_utils.draw_landmarks(
                        img,
                        hand_landmarks,
                        mp.solutions.hands.HAND_CONNECTIONS,
                        mp.solutions.drawing_styles.get_default_hand_landmarks_style(),
                        mp.solutions.drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # Extraer coordenadas
                    landmarks = []
                    for landmark in hand_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y])
                    
                    # Procesar si tenemos 42 landmarks
                    if len(landmarks) == 42:
                        letter, conf = self.asl_system.process_landmarks(landmarks)
                        if letter:
                            current_letter = letter
                            confidence = conf
                            
                            # Actualizar buffer
                            self.asl_system.update_buffer(letter, confidence)
                            
                            # Aplicar post-procesamiento si hay suficiente contexto
                            if len(self.asl_system.letter_buffer) >= 3:
                                processed_word = self.asl_system.post_process_sequence(
                                    self.asl_system.letter_buffer
                                )
                                self.current_word = processed_word
                            
                # Mostrar información en pantalla
                color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
                
                # Letra actual
                if current_letter:
                    cv2.putText(img, f'Letra: {current_letter} ({confidence:.2f})', 
                               (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                
                # Palabra procesada
                if self.current_word:
                    cv2.putText(img, f'Palabra: {self.current_word}', 
                               (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                # Buffer actual
                buffer_text = f'Buffer: {" ".join(self.asl_system.letter_buffer)}'
                cv2.putText(img, buffer_text, 
                           (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            
            return img
            
        except Exception as e:
            return frame.to_ndarray(format="bgr24")

# Configuración de la página
st.set_page_config(
    page_title="Traductor ASL Avanzado",
    page_icon="🧏",
    layout="wide"
)

# Configuración WebRTC
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

def main():
    st.title("🧏 Traductor ASL Avanzado con Post-Procesador")
    
    # Inicializar sistema
    asl_system = ASLSystem()
    
    if asl_system.letter_model is None:
        st.error("No se pudo cargar el modelo principal")
        return
    
    st.success("✅ Sistema ASL cargado completamente")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.header("🎥 Cámara en Vivo - Sistema Avanzado")
        st.info("""
        **Instrucciones:**
        1. Haz clic en **START** para activar la cámara
        2. Muestra **señas con tu mano** para formar palabras
        3. El sistema detectará letras y formará palabras automáticamente
        4. **Post-procesador** corregirá errores y formará palabras completas
        """)
        
        # Widget de cámara
        webrtc_ctx = webrtc_streamer(
            key="enhanced-asl-detector",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIGURATION,
            video_processor_factory=lambda: EnhancedSignDetector(asl_system),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
    
    with col2:
        st.header("📊 Sistema Avanzado")
        
        st.subheader("Componentes Activos")
        if asl_system.letter_model:
            st.write("✅ **Detección de Letras**")
        if asl_system.post_processor:
            st.write("✅ **Post-procesador RNN**")
        if asl_system.hands:
            st.write("✅ **MediaPipe Hands**")
        
        st.subheader("Funcionalidades")
        st.write("• 🧠 **Detección en tiempo real**")
        st.write("• 🔄 **Buffer de letras**")
        st.write("• ✨ **Corrección automática**")
        st.write("• 📝 **Formación de palabras**")
        
        st.markdown("---")
        st.info("**Fases Completadas:**")
        st.write("1. ✅ Entrenamiento Post-procesador")
        st.write("2. ✅ Evaluación con Métricas")
        st.write("3. ✅ Despliegue Integrado")

if __name__ == "__main__":
    main()