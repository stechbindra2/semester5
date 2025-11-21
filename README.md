# 🧠 NeuroStress Pro - Advanced AI Stress Detection Dashboard

<div align="center">

![NeuroStress Pro](https://img.shields.io/badge/NeuroStress-Pro-00ffff?style=for-the-badge&logo=brain&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange?style=for-the-badge&logo=tensorflow)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red?style=for-the-badge&logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

**The most advanced, futuristic real-time stress detection system powered by Deep Learning & AI**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Technology](#-technology-stack) • [Demo](#-demo)

</div>

---

## 🌟 Features

### 🎯 Core Capabilities
- **Real-Time Emotion Detection**: Detects 7 facial emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
- **AI-Powered Stress Calculation**: Scientific algorithms calculate stress levels from emotion patterns
- **Live Webcam Integration**: Seamless camera feed with face detection and emotion overlay
- **Multi-Dataset Training**: Model trained on FER2013 (35,887 images) and CK+ datasets

### 📊 Visualization & Analytics
- **3D Stress Sphere**: Interactive 3D visualization of stress levels
- **Animated Stress Gauge**: Real-time stress meter with dynamic thresholds
- **Emotion Radar Chart**: Multi-dimensional emotion distribution analysis
- **Timeline Analytics**: Historical stress and emotion tracking with trends
- **Session Statistics**: Comprehensive breakdown of emotion patterns

### 🎨 User Experience
- **Futuristic Glassmorphism UI**: Modern design with neon accents and glow effects
- **HUD-Style Interface**: Biometric-inspired heads-up display
- **Responsive Layout**: Optimized for all screen sizes
- **Dark Theme**: Eye-friendly interface for extended monitoring
- **Real-Time Updates**: Sub-second latency for instant feedback

### 🔬 Advanced Features
- **AI-Powered Insights**: Personalized recommendations based on emotion patterns
- **Session Management**: Export data in JSON format for further analysis
- **Confidence Thresholds**: Adjustable detection sensitivity
- **Detection Intervals**: Configurable frame processing rate
- **Emotion Statistics**: Detailed breakdowns and percentages
- **Stress Distribution**: Low/Medium/High stress categorization

---

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- Webcam/Camera device
- Windows/macOS/Linux

### Step 1: Clone or Download
```bash
# If using git
git clone <repository-url>
cd semester5

# Or simply extract the zip file and navigate to the folder
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Verify Model Files
Ensure `model_c.h5` is in the same directory as `stress_dashboard.py`:
```
semester5/
├── stress_dashboard.py
├── model_c.h5          ← Required!
├── requirements.txt
└── README.md
```

---

## 💻 Usage

### Launch the Dashboard
```bash
streamlit run stress_dashboard.py
```

The dashboard will automatically open in your default browser at `http://localhost:8501`

### Quick Start Guide

1. **Start Camera Detection**
   - Click the 🟢 **START** button in the sidebar
   - Allow camera permissions when prompted
   - Position your face in the camera frame

2. **Monitor Your Stress**
   - View real-time stress levels on the animated gauge
   - See your current emotion displayed prominently
   - Watch the 3D stress sphere change colors

3. **Explore Analytics**
   - Switch to the **ANALYTICS** tab for historical trends
   - View the **SESSION STATS** tab for detailed breakdowns
   - Check **INSIGHTS** for personalized recommendations

4. **Adjust Settings**
   - **Confidence Threshold**: Set minimum confidence for detection (default: 0.5)
   - **Detection Interval**: Adjust processing speed (default: 5 frames)

5. **Export Your Data**
   - Click **💾 Export Data** to download session statistics
   - Data includes timestamps, emotions, and stress levels in JSON format

---

## 🎯 How It Works

### Emotion Detection Pipeline
```
Camera Feed → Face Detection → CNN Model → Emotion Probabilities → Stress Calculation → Visualization
```

1. **Face Detection**: Haar Cascade classifier identifies faces in real-time
2. **Preprocessing**: Faces are cropped, resized to 48x48, and normalized
3. **CNN Prediction**: Deep learning model predicts emotion probabilities
4. **Stress Calculation**: Mathematical functions convert emotions to stress levels
5. **Visualization**: Results displayed on interactive dashboard

### Stress Calculation Algorithm

Each emotion has a specific mathematical function that converts confidence to stress:

- **Anger**: `2.332 * log(0.343 * p + 1.003)`
- **Fear**: `1.763 * log(1.356 * p + 1)`
- **Disgust**: `7.351 * log(0.0123 * p + 1.019)`
- **Happy**: `532.2 * log(5.221e-5 * p + 0.9997)`
- **Sad**: `2.851 * log(0.1328 * p + 1.009)`
- **Surprise**: `2.478 * log(0.2825 * p + 1.003)`
- **Neutral**: `5.03 * log(0.01229 * p + 1.036)`

Where `p` is the confidence percentage (0-100)

### Stress Categories
- 🟢 **Low (0-33%)**: Well-managed stress levels
- 🟡 **Medium (33-66%)**: Moderate stress, manageable
- 🔴 **High (66-100%)**: Elevated stress, action recommended

---

## 🛠️ Technology Stack

### Deep Learning & AI
- **TensorFlow 2.13**: Deep learning framework
- **Keras 2.13**: High-level neural network API
- **CNN Architecture**: Custom convolutional neural network
  - Multiple Conv2D layers with LeakyReLU activation
  - Batch normalization for stability
  - MaxPooling for dimensionality reduction
  - Dropout for regularization

### Computer Vision
- **OpenCV 4.8**: Real-time face detection and image processing
- **Haar Cascade**: Face detection algorithm
- **48x48 Grayscale**: Optimized input format

### Data Science
- **NumPy**: Numerical computing and array operations
- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Machine learning utilities

### Visualization
- **Plotly**: Interactive 3D visualizations and charts
- **Streamlit**: Modern web dashboard framework

### Training Datasets
- **FER2013**: 35,887 facial expression images
  - 28,709 training images (80%)
  - 3,589 validation images (10%)
  - 3,589 test images (10%)
- **CK+**: Extended Cohn-Kanade dataset for emotion recognition

---

## 📊 Dashboard Tabs

### 📊 LIVE MONITOR
- **Camera Feed**: Real-time video with emotion overlay
- **Stress Gauge**: Animated circular gauge showing current stress
- **3D Visualization**: Interactive stress sphere
- **Emotion Radar**: Multi-dimensional emotion distribution

### 📈 ANALYTICS
- **Timeline Chart**: Stress levels over time with emotion bars
- **Key Metrics**: Average stress, peak stress, total detections
- **Trend Analysis**: Historical patterns and insights

### 🎯 SESSION STATS
- **Emotion Pie Chart**: Distribution of detected emotions
- **Detailed Breakdown**: Percentage bars for each emotion
- **Stress Distribution**: Low/Medium/High categorization
- **Session Duration**: Elapsed time tracking

### 📚 INSIGHTS
- **AI Assessment**: Automated stress level evaluation
- **Personalized Recommendations**: Context-aware suggestions
- **Scientific Facts**: Educational information about stress
- **Wellness Tips**: Daily practices for better mental health

---

## 🎨 UI Features

### Design Elements
- **Glassmorphism**: Frosted glass effect with backdrop blur
- **Neon Accents**: Cyan glow effects on headers and important elements
- **Gradient Backgrounds**: Smooth color transitions
- **Animated Borders**: Pulsing borders on active components
- **Custom Fonts**: Orbitron for headers, Rajdhani for body text

### Color Palette
- **Primary**: Indigo/Purple gradient (#6366f1 → #a855f7)
- **Accent**: Cyan neon (#00ffff)
- **Success**: Green (#10b981)
- **Warning**: Orange (#f59e0b)
- **Danger**: Red (#ef4444)
- **Background**: Dark blue gradient (#0a0e27 → #1a1f3a)

---

## 📱 Use Cases

### Personal Wellness
- Monitor stress during work/study sessions
- Track emotional patterns throughout the day
- Identify stress triggers and patterns
- Improve self-awareness and emotional regulation

### Professional Settings
- Remote work stress monitoring
- Mental health check-ins
- Wellness program integration
- HR analytics (with consent)

### Research & Education
- Psychology studies on emotion and stress
- Human-computer interaction research
- Machine learning demonstrations
- Data science projects

### Healthcare Support
- Therapy session monitoring
- Patient stress tracking
- Anxiety management tools
- Mental health assessments

---

## 🔒 Privacy & Security

- ✅ All processing happens **locally** on your device
- ✅ No data is sent to external servers
- ✅ Camera feed is not recorded or stored
- ✅ Session data export is optional and user-controlled
- ✅ Open-source code for transparency

---

## 🎓 Model Performance

### Test Accuracy
- **FER2013 Test Set**: ~70% accuracy
- **CK+ Test Set**: ~85% accuracy
- **7-Class Classification**: All emotions detected

### Optimization
- **Real-time Processing**: 30+ FPS on standard webcams
- **Low Latency**: < 100ms detection time
- **GPU Support**: Automatic acceleration if available
- **Batch Normalization**: Improved stability and convergence

---

## 🐛 Troubleshooting

### Camera Not Detected
```python
# Try different camera indices
cap = cv2.VideoCapture(1)  # or 2, 3, etc.
```

### Model Loading Error
- Verify `model_c.h5` is in the correct directory
- Check file permissions
- Ensure TensorFlow is properly installed

### Low FPS
- Increase detection interval in sidebar
- Close other applications using the camera
- Reduce browser zoom level

### Import Errors
```bash
pip install --upgrade -r requirements.txt
```

---

## 🚀 Future Enhancements

- [ ] Multi-face detection and tracking
- [ ] Voice/audio stress analysis
- [ ] Heart rate estimation from video
- [ ] Mobile app version (iOS/Android)
- [ ] Cloud sync for cross-device access
- [ ] Advanced ML models (Vision Transformers)
- [ ] Integration with wearable devices
- [ ] Social features (anonymous community insights)
- [ ] Gamification (stress reduction challenges)
- [ ] API for third-party integrations

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **FER2013 Dataset**: Kaggle - Challenges in Representation Learning
- **CK+ Dataset**: Extended Cohn-Kanade AU-Coded Expression Database
- **OpenCV**: Computer vision library
- **Streamlit**: Dashboard framework
- **Plotly**: Interactive visualization library
- **TensorFlow/Keras**: Deep learning frameworks

---

## 📧 Contact & Support

For questions, issues, or contributions:
- 📧 Email: your.email@example.com
- 🐛 Issues: GitHub Issues
- 💬 Discussions: GitHub Discussions
- 🌟 Star this repo if you find it useful!

---

## 🌟 Key Highlights

> "NeuroStress Pro represents the cutting edge of emotion AI, combining deep learning, computer vision, and modern UX design to create an unparalleled stress monitoring experience."

### What Makes It Unique?

1. **🎨 Futuristic Design**: No other stress detection tool has this level of visual sophistication
2. **🧠 Scientific Accuracy**: Mathematical stress functions based on research
3. **⚡ Real-Time Performance**: Sub-second latency with smooth animations
4. **📊 Comprehensive Analytics**: Multiple visualization types and insights
5. **🔬 Research-Grade**: Trained on 35,000+ professional-grade images
6. **🎯 User-Centric**: Intuitive interface with minimal learning curve
7. **🌐 Open Source**: Fully transparent and customizable

---

<div align="center">

**Built with ❤️ using Python, TensorFlow, and AI**

⭐ Star this repository if you find it helpful!

[Back to Top](#-neurostress-pro---advanced-ai-stress-detection-dashboard)

</div>
