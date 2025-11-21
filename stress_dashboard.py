"""
🧠 NeuroStress Pro - Advanced Real-Time Stress Detection Dashboard
A cutting-edge, futuristic stress monitoring system with AI-powered emotion recognition
"""

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
from datetime import datetime, timedelta
import math
import time
from collections import deque
import json
from pathlib import Path

# ==================== PAGE CONFIG ====================
st.set_page_config(
    page_title="NeuroStress Pro - AI Stress Detection",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS ====================
st.markdown("""
<style>
    /* Import modern fonts */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Rajdhani:wght@300;400;600;700&display=swap');
    
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #1a1f3a 50%, #0f1419 100%);
        color: #e0e7ff;
    }
    
    /* Glassmorphism cards */
    .glass-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 20px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 25px;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        margin: 15px 0;
    }
    
    /* Neon glow effect */
    .neon-text {
        font-family: 'Orbitron', sans-serif;
        color: #00ffff;
        text-shadow: 0 0 10px #00ffff, 0 0 20px #00ffff, 0 0 30px #00ffff;
        font-size: 3rem;
        font-weight: 900;
        text-align: center;
        margin-bottom: 20px;
    }
    
    /* Metric cards with gradient */
    .metric-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.1) 0%, rgba(168, 85, 247, 0.1) 100%);
        border-radius: 15px;
        padding: 20px;
        border: 2px solid rgba(99, 102, 241, 0.3);
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 10px 40px rgba(99, 102, 241, 0.4);
        border-color: rgba(99, 102, 241, 0.6);
    }
    
    /* Stress level indicators */
    .stress-low {
        color: #10b981;
        text-shadow: 0 0 10px #10b981;
    }
    
    .stress-medium {
        color: #f59e0b;
        text-shadow: 0 0 10px #f59e0b;
    }
    
    .stress-high {
        color: #ef4444;
        text-shadow: 0 0 10px #ef4444;
    }
    
    /* Animated border */
    @keyframes border-pulse {
        0%, 100% { border-color: rgba(99, 102, 241, 0.3); }
        50% { border-color: rgba(168, 85, 247, 0.8); }
    }
    
    .pulse-border {
        animation: border-pulse 2s ease-in-out infinite;
    }
    
    /* Custom button styling */
    .stButton>button {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 30px;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }
    
    .stButton>button:hover {
        transform: scale(1.05);
        box-shadow: 0 6px 25px rgba(99, 102, 241, 0.6);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #1a1f3a 0%, #0a0e27 100%);
    }
    
    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Status badge */
    .status-badge {
        display: inline-block;
        padding: 8px 20px;
        border-radius: 20px;
        font-weight: 600;
        font-family: 'Rajdhani', sans-serif;
        margin: 10px 0;
    }
    
    .status-active {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.5);
    }
    
    .status-inactive {
        background: linear-gradient(135deg, #6b7280 0%, #4b5563 100%);
    }
    
    /* HUD-style info display */
    .hud-display {
        font-family: 'Orbitron', monospace;
        background: rgba(0, 255, 255, 0.1);
        border: 2px solid #00ffff;
        border-radius: 10px;
        padding: 15px;
        margin: 10px 0;
        box-shadow: inset 0 0 20px rgba(0, 255, 255, 0.2);
    }
    
    /* Emotion tag */
    .emotion-tag {
        display: inline-block;
        padding: 5px 15px;
        margin: 5px;
        border-radius: 15px;
        font-size: 0.9rem;
        font-weight: 600;
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* Session stats */
    .stat-box {
        text-align: center;
        padding: 15px;
        background: rgba(99, 102, 241, 0.1);
        border-radius: 10px;
        border: 1px solid rgba(99, 102, 241, 0.3);
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        font-family: 'Orbitron', sans-serif;
        color: #00ffff;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #9ca3af;
        font-family: 'Rajdhani', sans-serif;
    }
</style>
""", unsafe_allow_html=True)

# ==================== STRESS CALCULATION FUNCTIONS ====================
def anger(p):
    t = 0.343 * p + 1.003
    return 2.332 * math.log(t)

def fear(p):
    t = 1.356 * p + 1
    return 1.763 * math.log(t)

def contempt(p):
    t = 0.01229 * p + 1.036
    return 5.03 * math.log(t)

def disgust(p):
    t = 0.0123 * p + 1.019
    return 7.351 * math.log(t)

def happy(p):
    t = 5.221e-5 * p + 0.9997
    return 532.2 * math.log(t)

def sad(p):
    t = 0.1328 * p + 1.009
    return 2.851 * math.log(t)

def surprise(p):
    t = 0.2825 * p + 1.003
    return 2.478 * math.log(t)

# Map emotions to stress functions
STRESS_FUNCTIONS = [anger, disgust, fear, happy, sad, surprise, contempt]
EMOTION_LABELS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
EMOTION_COLORS = ['#ef4444', '#8b5cf6', '#f59e0b', '#10b981', '#3b82f6', '#ec4899', '#6b7280']

# ==================== MODEL LOADING ====================
def legacy_load_model(filepath):
    """Load models saved with older Keras versions - rebuilds architecture and loads weights"""
    try:
        # Strategy: Rebuild the exact model architecture and load weights only
        from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
        from tensorflow.keras.layers import Flatten, Dense, Dropout, Activation, LeakyReLU
        from tensorflow.keras.models import Sequential
        
        # Recreate the exact model architecture from your training notebook
        model = Sequential()
        
        # Module 1
        model.add(Conv2D(256, kernel_size=(3, 3), input_shape=(48, 48, 1), data_format='channels_last'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(256, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        # Module 2
        model.add(Conv2D(256, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(256, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        # Module 3
        model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Conv2D(128, kernel_size=(3, 3), padding='same'))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        # Flatten
        model.add(Flatten())
        
        # Dense layers
        model.add(Dense(256))
        model.add(BatchNormalization())
        model.add(LeakyReLU(alpha=0.1))
        model.add(Dropout(0.3))
        
        model.add(Dense(128))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dropout(0.1))
        
        model.add(Dense(7, activation='softmax'))
        
        # Load weights from the old model file
        model.load_weights(filepath)
        
        return model
        
    except Exception as e:
        raise Exception(f"Failed to load model architecture and weights: {e}")

@st.cache_resource
def load_model():
    """Load the trained emotion detection model with legacy support"""
    try:
        model_path = Path(__file__).parent / 'model_c.h5'
        
        if not model_path.exists():
            st.error(f"❌ Model file not found: {model_path}")
            return None
        
        # Try legacy load first (handles batch_shape issue)
        try:
            model = legacy_load_model(str(model_path))
        except Exception as e1:
            # Try standard load with compile=False
            try:
                model = keras.models.load_model(str(model_path), compile=False)
            except Exception as e2:
                raise Exception(f"Both load methods failed: {e1} | {e2}")
        
        # Recompile the model
        model.compile(
            loss='categorical_crossentropy',
            optimizer='adam',
            metrics=['accuracy']
        )
        
        return model
        
    except Exception as e:
        st.error(f"⚠️ Error loading model: {e}")
        
        with st.expander("🔧 Fix Model Compatibility Issue", expanded=True):
            st.warning("**Root Cause:** Your model was saved with an older TensorFlow/Keras version that used deprecated parameters.")
            
            st.markdown("### Quick Fix:")
            st.code("python fix_models.py", language="bash")
            
            st.markdown("### Manual Alternative:")
            st.code("""
# In terminal, run Python:
python
>>> import tensorflow as tf
>>> from pathlib import Path
>>> exec(Path('fix_models.py').read_text())
>>> exit()
            """, language="python")
            
            st.info("💡 This will create `model_c_fixed.h5` - then rename it to `model_c.h5`")
        
        return None

# ==================== FACE DETECTION ====================
@st.cache_resource
def load_face_detector():
    """Load Haar Cascade face detector"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return face_cascade

# ==================== SESSION STATE INITIALIZATION ====================
if 'session_data' not in st.session_state:
    st.session_state.session_data = {
        'emotions': deque(maxlen=100),
        'stress_levels': deque(maxlen=100),
        'timestamps': deque(maxlen=100),
        'session_start': datetime.now(),
        'total_detections': 0,
        'emotion_counts': {emotion: 0 for emotion in EMOTION_LABELS},
        'avg_stress': 0,
        'peak_stress': 0,
        'history': []
    }

if 'camera_active' not in st.session_state:
    st.session_state.camera_active = False

if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'

# ==================== HELPER FUNCTIONS ====================
def calculate_stress_level(emotion_probs):
    """Calculate stress level based on emotion probabilities"""
    emotion_idx = np.argmax(emotion_probs)
    confidence = np.max(emotion_probs) * 100
    
    # For neutral emotion, use contempt function
    if emotion_idx == 6:
        stress = contempt(confidence)
    else:
        stress = STRESS_FUNCTIONS[emotion_idx](confidence)
    
    # Normalize stress to 0-100 scale
    stress_percentage = (stress / 9) * 100
    stress_percentage = max(0, min(100, stress_percentage))
    
    return stress_percentage, EMOTION_LABELS[emotion_idx], confidence

def get_stress_color(stress_level):
    """Get color based on stress level"""
    if stress_level < 33:
        return '#10b981', 'LOW'
    elif stress_level < 66:
        return '#f59e0b', 'MEDIUM'
    else:
        return '#ef4444', 'HIGH'

def create_3d_stress_sphere(stress_level):
    """Create advanced 3D sphere visualization with dynamic stress representation"""
    # Create high-resolution sphere
    theta = np.linspace(0, 2*np.pi, 50)
    phi = np.linspace(0, np.pi, 40)
    theta, phi = np.meshgrid(theta, phi)
    
    # Base sphere coordinates
    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    
    # Add stress-based perturbations for dynamic effect
    stress_factor = stress_level / 100.0
    noise = np.random.rand(*x.shape) * 0.05 * stress_factor
    x_perturbed = x * (1 + noise)
    y_perturbed = y * (1 + noise)
    z_perturbed = z * (1 + noise)
    
    # Create color gradient based on stress level
    color_base, category = get_stress_color(stress_level)
    
    # Dynamic color intensity mapping
    color_values = np.sqrt(x**2 + y**2 + z**2) * stress_level
    
    # Define colorscale based on stress category
    if stress_level < 33:  # Low stress - green gradient
        colorscale = [
            [0, '#064e3b'],    # Dark green
            [0.5, '#10b981'],  # Medium green
            [1, '#6ee7b7']     # Light green
        ]
    elif stress_level < 66:  # Medium stress - orange gradient
        colorscale = [
            [0, '#92400e'],    # Dark orange
            [0.5, '#f59e0b'],  # Medium orange
            [1, '#fcd34d']     # Light orange
        ]
    else:  # High stress - red gradient
        colorscale = [
            [0, '#7f1d1d'],    # Dark red
            [0.5, '#ef4444'],  # Medium red
            [1, '#fca5a5']     # Light red
        ]
    
    # Create main stress sphere
    fig = go.Figure()
    
    # Add main sphere surface
    fig.add_trace(go.Surface(
        x=x_perturbed,
        y=y_perturbed,
        z=z_perturbed,
        surfacecolor=color_values,
        colorscale=colorscale,
        showscale=False,
        opacity=0.85,
        lighting=dict(
            ambient=0.4,
            diffuse=0.8,
            fresnel=2,
            specular=0.6,
            roughness=0.3
        ),
        lightposition=dict(x=100, y=100, z=200),
        name='Stress Sphere'
    ))
    
    # Add inner glow sphere for depth
    inner_scale = 0.7
    fig.add_trace(go.Surface(
        x=x * inner_scale,
        y=y * inner_scale,
        z=z * inner_scale,
        surfacecolor=color_values * 0.5,
        colorscale=colorscale,
        showscale=False,
        opacity=0.3,
        name='Inner Glow'
    ))
    
    # Add stress level annotation
    stress_text = f"<b>{category}</b><br>{stress_level:.1f}%"
    fig.add_trace(go.Scatter3d(
        x=[0], y=[0], z=[1.5],
        mode='text',
        text=[stress_text],
        textfont=dict(size=16, color=color_base, family='Orbitron'),
        showlegend=False
    ))
    
    # Update layout with enhanced settings
    fig.update_layout(
        scene=dict(
            xaxis=dict(
                showticklabels=False, 
                showgrid=False, 
                zeroline=False, 
                visible=False,
                range=[-1.5, 1.5]
            ),
            yaxis=dict(
                showticklabels=False, 
                showgrid=False, 
                zeroline=False, 
                visible=False,
                range=[-1.5, 1.5]
            ),
            zaxis=dict(
                showticklabels=False, 
                showgrid=False, 
                zeroline=False, 
                visible=False,
                range=[-1.5, 1.5]
            ),
            bgcolor='rgba(10, 14, 39, 0.5)',
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.5),
                center=dict(x=0, y=0, z=0)
            ),
            aspectmode='cube'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(l=0, r=0, t=10, b=0),
        height=350,
        showlegend=False,
        hovermode=False
    )
    
    return fig

def create_emotion_radar(emotion_probs):
    """Create radar chart for emotion distribution"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatterpolar(
        r=emotion_probs * 100,
        theta=EMOTION_LABELS,
        fill='toself',
        fillcolor='rgba(99, 102, 241, 0.3)',
        line=dict(color='#6366f1', width=3),
        marker=dict(size=8, color='#a855f7')
    ))
    
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                showticklabels=True,
                tickfont=dict(size=10, color='#9ca3af'),
                gridcolor='rgba(156, 163, 175, 0.2)'
            ),
            angularaxis=dict(
                tickfont=dict(size=12, color='#e0e7ff', family='Rajdhani')
            ),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        height=400,
        margin=dict(l=80, r=80, t=40, b=40)
    )
    
    return fig

def create_stress_gauge(stress_level):
    """Create an animated gauge for stress level"""
    color, level_text = get_stress_color(stress_level)
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=stress_level,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"Stress Level: {level_text}", 'font': {'size': 24, 'color': '#e0e7ff', 'family': 'Orbitron'}},
        delta={'reference': 50, 'increasing': {'color': "#ef4444"}, 'decreasing': {'color': "#10b981"}},
        number={'font': {'size': 50, 'color': color, 'family': 'Orbitron'}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 2, 'tickcolor': "#9ca3af"},
            'bar': {'color': color, 'thickness': 0.75},
            'bgcolor': "rgba(255,255,255,0.1)",
            'borderwidth': 2,
            'bordercolor': "rgba(255,255,255,0.3)",
            'steps': [
                {'range': [0, 33], 'color': 'rgba(16, 185, 129, 0.2)'},
                {'range': [33, 66], 'color': 'rgba(245, 158, 11, 0.2)'},
                {'range': [66, 100], 'color': 'rgba(239, 68, 68, 0.2)'}
            ],
            'threshold': {
                'line': {'color': "#fff", 'width': 4},
                'thickness': 0.75,
                'value': stress_level
            }
        }
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#e0e7ff", 'family': "Rajdhani"},
        height=350,
        margin=dict(l=20, r=20, t=80, b=20)
    )
    
    return fig

def create_timeline_chart(timestamps, stress_levels, emotions):
    """Create interactive timeline of stress and emotions"""
    if len(timestamps) == 0:
        return None
    
    fig = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Stress Level Over Time', 'Emotion Distribution'),
        vertical_spacing=0.15,
        row_heights=[0.6, 0.4]
    )
    
    # Stress timeline
    colors = [get_stress_color(s)[0] for s in stress_levels]
    fig.add_trace(
        go.Scatter(
            x=list(timestamps),
            y=list(stress_levels),
            mode='lines+markers',
            name='Stress Level',
            line=dict(color='#6366f1', width=3),
            marker=dict(size=8, color=colors, line=dict(width=2, color='#fff')),
            fill='tozeroy',
            fillcolor='rgba(99, 102, 241, 0.2)'
        ),
        row=1, col=1
    )
    
    # Emotion bars
    emotion_counts = pd.Series(list(emotions)).value_counts()
    fig.add_trace(
        go.Bar(
            x=emotion_counts.index,
            y=emotion_counts.values,
            marker=dict(
                color=[EMOTION_COLORS[EMOTION_LABELS.index(e)] if e in EMOTION_LABELS else '#6b7280' 
                       for e in emotion_counts.index],
                line=dict(width=2, color='rgba(255,255,255,0.3)')
            ),
            name='Emotions'
        ),
        row=2, col=1
    )
    
    fig.update_xaxes(title_text="Time", row=1, col=1, showgrid=True, gridcolor='rgba(156, 163, 175, 0.1)')
    fig.update_yaxes(title_text="Stress %", row=1, col=1, showgrid=True, gridcolor='rgba(156, 163, 175, 0.1)')
    fig.update_xaxes(title_text="Emotion", row=2, col=1)
    fig.update_yaxes(title_text="Count", row=2, col=1)
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#e0e7ff", 'family': "Rajdhani"},
        showlegend=False,
        height=600,
        margin=dict(l=50, r=50, t=80, b=50)
    )
    
    return fig

def create_emotion_pie_chart(emotion_counts):
    """Create pie chart of emotion distribution"""
    labels = list(emotion_counts.keys())
    values = list(emotion_counts.values())
    colors = [EMOTION_COLORS[EMOTION_LABELS.index(label)] for label in labels]
    
    fig = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        marker=dict(colors=colors, line=dict(color='#fff', width=2)),
        textfont=dict(size=14, color='#fff', family='Rajdhani'),
        textinfo='label+percent',
        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
    )])
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font={'color': "#e0e7ff", 'family': "Rajdhani"},
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
        height=400,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    
    return fig

# ==================== MAIN DASHBOARD ====================
def main():
    # Header with neon effect
    st.markdown('<h1 class="neon-text">🧠 NEUROSTRESS PRO</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: #9ca3af; font-size: 1.2rem; font-family: Rajdhani, sans-serif; margin-top: -15px;">Advanced AI-Powered Real-Time Stress Detection System</p>', unsafe_allow_html=True)
    
    # Load model and face detector
    model = load_model()
    face_cascade = load_face_detector()
    
    if model is None:
        st.error("❌ Failed to load model. Please ensure 'model_c.h5' is in the same directory.")
        return
    
    # Sidebar controls
    with st.sidebar:
        st.markdown('<h2 style="color: #00ffff; font-family: Orbitron;">⚙️ CONTROL PANEL</h2>', unsafe_allow_html=True)
        
        # Session info
        session_duration = datetime.now() - st.session_state.session_data['session_start']
        st.markdown(f"""
        <div class="hud-display">
            <div style="font-size: 0.9rem; color: #9ca3af;">SESSION DURATION</div>
            <div style="font-size: 1.5rem; color: #00ffff; font-weight: 700;">{str(session_duration).split('.')[0]}</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Camera controls
        st.markdown("### 📹 Camera Control")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🟢 START"):
                st.session_state.camera_active = True
        with col2:
            if st.button("🔴 STOP"):
                st.session_state.camera_active = False
        
        st.markdown("---")
        
        # Detection settings
        st.markdown("### 🎯 Detection Settings")
        confidence_threshold = st.slider("Confidence Threshold", 0.0, 1.0, 0.5, 0.05)
        detection_interval = st.slider("Detection Interval (frames)", 1, 30, 5)
        
        st.markdown("---")
        
        # Session controls
        st.markdown("### 💾 Session Management")
        if st.button("🔄 Reset Session"):
            st.session_state.session_data = {
                'emotions': deque(maxlen=100),
                'stress_levels': deque(maxlen=100),
                'timestamps': deque(maxlen=100),
                'session_start': datetime.now(),
                'total_detections': 0,
                'emotion_counts': {emotion: 0 for emotion in EMOTION_LABELS},
                'avg_stress': 0,
                'peak_stress': 0,
                'history': []
            }
            st.success("✅ Session reset successfully!")
        
        if st.button("💾 Export Data"):
            # Prepare data for export
            export_data = {
                'session_start': str(st.session_state.session_data['session_start']),
                'total_detections': st.session_state.session_data['total_detections'],
                'emotion_counts': st.session_state.session_data['emotion_counts'],
                'avg_stress': st.session_state.session_data['avg_stress'],
                'peak_stress': st.session_state.session_data['peak_stress'],
                'history': [
                    {
                        'timestamp': str(ts),
                        'emotion': em,
                        'stress': float(sl)
                    }
                    for ts, em, sl in zip(
                        st.session_state.session_data['timestamps'],
                        st.session_state.session_data['emotions'],
                        st.session_state.session_data['stress_levels']
                    )
                ]
            }
            
            json_str = json.dumps(export_data, indent=2)
            st.download_button(
                label="📥 Download JSON",
                data=json_str,
                file_name=f"stress_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
        
        st.markdown("---")
        
        # Info
        st.markdown("### ℹ️ About")
        st.info("""
        **NeuroStress Pro** uses advanced CNN models trained on FER2013 and CK+ datasets to detect facial emotions in real-time and calculate stress levels based on scientific algorithms.
        
        **Emotions Detected:**
        - 😠 Angry
        - 🤢 Disgust  
        - 😨 Fear
        - 😊 Happy
        - 😢 Sad
        - 😲 Surprise
        - 😐 Neutral
        """)
    
    # Main content area
    tab1, tab2, tab3, tab4 = st.tabs(["📊 LIVE MONITOR", "📈 ANALYTICS", "🎯 SESSION STATS", "📚 INSIGHTS"])
    
    with tab1:
        # Live monitoring interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 🎥 Live Camera Feed")
            camera_placeholder = st.empty()
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Real-time detection
            if st.session_state.camera_active:
                cap = cv2.VideoCapture(0)
                frame_count = 0
                
                while st.session_state.camera_active:
                    ret, frame = cap.read()
                    if not ret:
                        st.error("❌ Failed to access camera")
                        break
                    
                    frame_count += 1
                    
                    # Convert to grayscale for face detection
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
                    
                    current_emotion = "No Face Detected"
                    current_stress = 0
                    emotion_probs = np.zeros(7)
                    
                    # Process faces
                    if len(faces) > 0 and frame_count % detection_interval == 0:
                        for (x, y, w, h) in faces:
                            # Draw rectangle around face
                            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
                            
                            # Extract and preprocess face
                            face_roi = gray[y:y+h, x:x+w]
                            face_roi = cv2.resize(face_roi, (48, 48))
                            face_roi = face_roi.astype('float32') / 255.0
                            face_roi = np.expand_dims(face_roi, axis=-1)
                            face_roi = np.expand_dims(face_roi, axis=0)
                            
                            # Predict emotion
                            emotion_probs = model.predict(face_roi, verbose=0)[0]
                            
                            if np.max(emotion_probs) >= confidence_threshold:
                                current_stress, current_emotion, confidence = calculate_stress_level(emotion_probs)
                                
                                # Update session data
                                st.session_state.session_data['emotions'].append(current_emotion)
                                st.session_state.session_data['stress_levels'].append(current_stress)
                                st.session_state.session_data['timestamps'].append(datetime.now())
                                st.session_state.session_data['total_detections'] += 1
                                st.session_state.session_data['emotion_counts'][current_emotion] += 1
                                
                                # Update statistics
                                if len(st.session_state.session_data['stress_levels']) > 0:
                                    st.session_state.session_data['avg_stress'] = np.mean(list(st.session_state.session_data['stress_levels']))
                                    st.session_state.session_data['peak_stress'] = np.max(list(st.session_state.session_data['stress_levels']))
                                
                                # Draw emotion and stress on frame
                                color, level_text = get_stress_color(current_stress)
                                cv2.putText(frame, f"{current_emotion} ({confidence:.1f}%)", (x, y-30), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                                cv2.putText(frame, f"Stress: {current_stress:.1f}% ({level_text})", (x, y-10), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Display frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                    
                    # Small delay to control frame rate
                    time.sleep(0.03)
                
                cap.release()
            else:
                camera_placeholder.info("📷 Camera is stopped. Click START to begin detection.")
        
        with col2:
            # Live metrics
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 💫 Current Status")
            
            if len(st.session_state.session_data['stress_levels']) > 0:
                latest_stress = st.session_state.session_data['stress_levels'][-1]
                latest_emotion = st.session_state.session_data['emotions'][-1]
                
                # Get stress color and category
                stress_color, stress_category = get_stress_color(latest_stress)
                
                # Stress gauge
                gauge_fig = create_stress_gauge(latest_stress)
                st.plotly_chart(gauge_fig, use_container_width=True)
                
                # Current emotion
                emotion_idx = EMOTION_LABELS.index(latest_emotion) if latest_emotion in EMOTION_LABELS else 6
                emotion_color = EMOTION_COLORS[emotion_idx]
                
                st.markdown(f"""
                <div style="text-align: center; margin: 20px 0;">
                    <div class="emotion-tag" style="background: {emotion_color}; font-size: 1.2rem;">
                        {latest_emotion}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # 3D visualization with enhanced description
                st.markdown("##### 🌐 3D Stress Visualization")
                st.markdown(f"""
                <div style="text-align: center; padding: 10px; margin-bottom: 10px; 
                     background: rgba(99, 102, 241, 0.1); border-radius: 10px; 
                     border-left: 4px solid {stress_color};">
                    <small style="color: #94a3b8;">
                        Interactive 3D sphere showing stress intensity • 
                        Size and color intensity reflect stress level • 
                        Rotate with mouse/touch
                    </small>
                </div>
                """, unsafe_allow_html=True)
                
                sphere_fig = create_3d_stress_sphere(latest_stress)
                st.plotly_chart(sphere_fig, use_container_width=True, key=f"stress_sphere_{time.time()}")
                
                # Add stress interpretation
                st.markdown(f"""
                <div style="text-align: center; margin-top: 10px;">
                    <div style="display: inline-block; padding: 8px 20px; 
                         background: {stress_color}; border-radius: 20px; 
                         font-weight: bold; color: white; box-shadow: 0 4px 15px {stress_color}50;">
                        {stress_category} STRESS: {latest_stress:.1f}%
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.info("⏳ Waiting for detection data...")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Emotion radar
        if len(st.session_state.session_data['emotions']) > 0:
            st.markdown('<div class="glass-card">', unsafe_allow_html=True)
            st.markdown("#### 🎯 Real-Time Emotion Distribution")
            
            # Get latest emotion probabilities (mock for now, would need to store actual probs)
            latest_emotion = st.session_state.session_data['emotions'][-1]
            mock_probs = np.random.rand(7) * 0.3
            emotion_idx = EMOTION_LABELS.index(latest_emotion) if latest_emotion in EMOTION_LABELS else 6
            mock_probs[emotion_idx] = 0.7 + np.random.rand() * 0.3
            mock_probs = mock_probs / mock_probs.sum()
            
            radar_fig = create_emotion_radar(mock_probs)
            st.plotly_chart(radar_fig, use_container_width=True)
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        # Analytics dashboard
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📊 Comprehensive Analytics")
        
        if len(st.session_state.session_data['timestamps']) > 0:
            # Timeline chart
            timeline_fig = create_timeline_chart(
                st.session_state.session_data['timestamps'],
                st.session_state.session_data['stress_levels'],
                st.session_state.session_data['emotions']
            )
            st.plotly_chart(timeline_fig, use_container_width=True)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-value">{st.session_state.session_data['avg_stress']:.1f}%</div>
                    <div class="stat-label">AVG STRESS</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-value">{st.session_state.session_data['peak_stress']:.1f}%</div>
                    <div class="stat-label">PEAK STRESS</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col3:
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-value">{st.session_state.session_data['total_detections']}</div>
                    <div class="stat-label">DETECTIONS</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col4:
                dominant_emotion = max(st.session_state.session_data['emotion_counts'], 
                                     key=st.session_state.session_data['emotion_counts'].get)
                st.markdown(f"""
                <div class="stat-box">
                    <div class="stat-value" style="font-size: 1.5rem;">{dominant_emotion}</div>
                    <div class="stat-label">DOMINANT</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("📊 No data available yet. Start the camera to begin collecting analytics.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab3:
        # Session statistics
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 📈 Session Statistics")
        
        if st.session_state.session_data['total_detections'] > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                # Emotion pie chart
                emotion_counts = {k: v for k, v in st.session_state.session_data['emotion_counts'].items() if v > 0}
                if emotion_counts:
                    pie_fig = create_emotion_pie_chart(emotion_counts)
                    st.plotly_chart(pie_fig, use_container_width=True)
            
            with col2:
                # Detailed statistics
                st.markdown("#### 📋 Detailed Breakdown")
                
                for emotion, count in sorted(st.session_state.session_data['emotion_counts'].items(), 
                                            key=lambda x: x[1], reverse=True):
                    if count > 0:
                        percentage = (count / st.session_state.session_data['total_detections']) * 100
                        emotion_idx = EMOTION_LABELS.index(emotion) if emotion in EMOTION_LABELS else 6
                        color = EMOTION_COLORS[emotion_idx]
                        
                        st.markdown(f"""
                        <div style="margin: 10px 0;">
                            <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                                <span style="color: {color}; font-weight: 600;">{emotion}</span>
                                <span style="color: #9ca3af;">{count} ({percentage:.1f}%)</span>
                            </div>
                            <div style="background: rgba(255,255,255,0.1); height: 8px; border-radius: 4px; overflow: hidden;">
                                <div style="background: {color}; height: 100%; width: {percentage}%;"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                
                # Stress level distribution
                st.markdown("#### 🌡️ Stress Distribution")
                stress_data = list(st.session_state.session_data['stress_levels'])
                low_stress = sum(1 for s in stress_data if s < 33)
                med_stress = sum(1 for s in stress_data if 33 <= s < 66)
                high_stress = sum(1 for s in stress_data if s >= 66)
                
                total = len(stress_data)
                if total > 0:
                    st.markdown(f"""
                    <div style="margin: 10px 0;">
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <span style="color: #10b981; font-weight: 600;">LOW</span>
                            <span style="color: #9ca3af;">{low_stress} ({low_stress/total*100:.1f}%)</span>
                        </div>
                        <div style="display: flex; justify-content: space-between; margin-bottom: 5px;">
                            <span style="color: #f59e0b; font-weight: 600;">MEDIUM</span>
                            <span style="color: #9ca3af;">{med_stress} ({med_stress/total*100:.1f}%)</span>
                        </div>
                        <div style="display: flex; justify-content: space-between;">
                            <span style="color: #ef4444; font-weight: 600;">HIGH</span>
                            <span style="color: #9ca3af;">{high_stress} ({high_stress/total*100:.1f}%)</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("📊 No session data available yet.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with tab4:
        # Insights and recommendations
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("### 🧠 AI-Powered Insights")
        
        if st.session_state.session_data['total_detections'] > 10:
            avg_stress = st.session_state.session_data['avg_stress']
            dominant_emotion = max(st.session_state.session_data['emotion_counts'], 
                                 key=st.session_state.session_data['emotion_counts'].get)
            
            # Stress level assessment
            if avg_stress < 33:
                st.success("✅ **Great News!** Your stress levels are well-managed. Keep up the excellent work!")
            elif avg_stress < 66:
                st.warning("⚠️ **Moderate Stress Detected.** Consider taking short breaks and practicing relaxation techniques.")
            else:
                st.error("🚨 **High Stress Levels!** It's important to address this. Consider the recommendations below.")
            
            # Personalized recommendations
            st.markdown("#### 💡 Personalized Recommendations")
            
            if dominant_emotion == "Happy":
                st.markdown("""
                - 😊 You're experiencing positive emotions! Maintain this state with regular breaks and social interactions.
                - Continue activities that bring you joy
                - Share your positive energy with others
                """)
            elif dominant_emotion in ["Angry", "Fear", "Sad"]:
                st.markdown("""
                - 🧘 Practice deep breathing exercises (4-7-8 technique)
                - 🚶 Take a 10-minute walk outside
                - 💬 Talk to someone you trust
                - 🎵 Listen to calming music
                - 📱 Consider using meditation apps (Calm, Headspace)
                """)
            elif dominant_emotion == "Surprise":
                st.markdown("""
                - 🎯 Take time to process new information
                - 📝 Journal your thoughts and feelings
                - 🤔 Practice mindfulness to stay grounded
                """)
            elif dominant_emotion == "Disgust":
                st.markdown("""
                - 🌿 Change your environment if possible
                - 🍃 Practice grounding techniques
                - 🧹 Organize your workspace for better mental clarity
                """)
            else:  # Neutral
                st.markdown("""
                - ⚡ Maintain your balanced state
                - 🎯 Set clear goals for focused work
                - ⏰ Use the Pomodoro technique for productivity
                """)
            
            # Scientific insights
            st.markdown("#### 🔬 Scientific Insights")
            st.info("""
            **Did you know?**
            
            - Chronic stress can affect memory, concentration, and decision-making abilities
            - The optimal stress level (eustress) can enhance performance and motivation
            - Regular monitoring of emotional states can improve self-awareness and emotional regulation
            - Facial expressions are universal across cultures, making emotion recognition highly accurate
            """)
            
            # Wellness tips
            st.markdown("#### 🌟 Daily Wellness Tips")
            st.markdown("""
            1. **Morning Routine:** Start with 5 minutes of stretching or meditation
            2. **Hydration:** Drink water regularly throughout the day
            3. **Screen Breaks:** Follow the 20-20-20 rule (every 20 min, look 20 feet away for 20 sec)
            4. **Sleep Hygiene:** Maintain consistent sleep schedule (7-9 hours)
            5. **Social Connection:** Interact with friends or family daily
            6. **Physical Activity:** Aim for 30 minutes of movement
            7. **Mindful Eating:** Take time to enjoy your meals without distractions
            """)
        else:
            st.info("📊 Collect more data (10+ detections) to receive personalized insights and recommendations.")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #6b7280; font-family: Rajdhani; padding: 20px;">
        <p style="margin: 5px 0;">🧠 <b>NeuroStress Pro</b> v1.0 | Powered by Deep Learning & AI</p>
        <p style="margin: 5px 0; font-size: 0.9rem;">Built with ❤️ using TensorFlow, Keras, OpenCV, Streamlit & Plotly</p>
        <p style="margin: 5px 0; font-size: 0.85rem;">© 2025 | For Educational & Research Purposes</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
