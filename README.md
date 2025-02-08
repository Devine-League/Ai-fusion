# ENVIRON- "Play. Learn. Save the Planet. 🌍🎮"
# 📌 Project Overview
This AI-powered application integrates TrashNet (waste classification), Google Earth Engine (climate analysis), and a carbon footprint calculator into a seamless, user-friendly interface using Streamlit. The tool helps users classify waste, analyze climate impact, and calculate their carbon footprint in a single web-based platform.
# 🚀 Features
1.	Waste Classification (TrashNet)
o	Users can upload an image of waste.
o	The model classifies it into categories like plastic, paper, metal, glass, or organic.
o	Displays the classification result with an image preview.
2.	Climate Analysis (Google Earth Engine)
o	Users select a region for climate-related data analysis.
o	Retrieves deforestation, pollution, and temperature change data.
o	Displays the results using an interactive map and visualizations.
3.	Carbon Footprint Calculator
o	Users input details about daily activities (transport, diet, energy use).
o	Calculates the carbon footprint based on emission factors.
o	Displays the results using interactive graphs with reduction tips.
# 🛠️ Technology Stack
•	Frontend: Streamlit
•	Backend: Python (FastAPI/Flask optional for API management)
•	AI Models: TensorFlow/Keras (TrashNet), Hugging Face (NLP for analysis)
•	Google Earth Engine API: Climate data retrieval
•	Visualization: Matplotlib, Plotly, Streamlit-Folium (maps)
•	Data Handling: Pandas, NumPy
# 📦 Installation
1️⃣ Set Up Environment
conda create --name ai_env python=3.10.6
conda activate ai_env
2️⃣ Install Dependencies
conda install -c conda-forge streamlit tensorflow pillow matplotlib plotly openai numpy pandas opencv
pip install earthengine-api folium streamlit-folium
3️⃣ Authenticate Google Earth Engine
•	Run: 
•	earthengine authenticate
•	Follow the instructions to log in and authenticate.
🔧 Running the Application
streamlit run app.py
🏗️ Project Structure
📂 ai-environment-dashboard
├── 📄 app.py  # Main Streamlit app
├── 📂 models  # AI models (TrashNet, Carbon Calculator)
├── 📂 data  # Sample datasets
├── 📄 requirements.txt  # Dependencies
├── 📄 README.md  # Project documentation
# 📌 How It Works
1.	Waste Classification
o	Uses a pre-trained CNN model (TrashNet) to classify waste images.
o	Takes input via Streamlit and displays predictions.
2.	Climate Analysis with Google Earth Engine
o	Fetches real-time environmental data based on the user’s selected region.
o	Uses Earth Engine API and visualizes maps with Streamlit-Folium.
3.	Carbon Footprint Calculation
o	Computes footprint based on user activities.
o	Generates graphical reports using Matplotlib/Plotly.
# 🎯 Future Enhancements
•	AI Chatbot for personalized sustainability tips.
•	Multi-language support for wider accessibility.
•	More waste categories with improved classification accuracy.
📜 License
This project is open-source under the MIT License.
________________________________________
💡 Contributions Welcome! If you have ideas or improvements, feel free to contribute. 🚀

