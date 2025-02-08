import streamlit as st

# Set page configuration (must be the first Streamlit command)
st.set_page_config(
    page_title='EcoSmart Analyzer',
    page_icon='♻',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Initialize points in session state
if 'points' not in st.session_state:
    st.session_state.points = 0

# Function to add points
def add_points(points):
    st.session_state.points += points

# Display points at the top right corner
st.markdown(f"""
    <div style='position: fixed; top: 10px; right: 10px; background-color: #2E7D32; color: white; padding: 10px; border-radius: 5px;'>
        Points: {st.session_state.points}
    </div>
""", unsafe_allow_html=True)

# Import other dependencies after page config
import plotly.express as px
import pandas as pd
import numpy as np
from PIL import Image
from transformers import pipeline
import torch

# Initialize the image-to-text pipeline
@st.cache_resource(show_spinner="Loading image analysis model...")
def load_image_captioning_model():
    try:
        return pipeline("image-to-text", model="Salesforce/blip-image-captioning-base", device="cpu")
    except Exception as e:
        st.error("Error loading the image analysis model. Please ensure you have installed all requirements.")
        st.error(f"Error details: {str(e)}")
        return None

# Load the model
pipe = load_image_captioning_model()

# Custom CSS for styling
st.markdown("""
    <style>
        .analysis-card {
            background-color: white;
            padding: 1.5rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 1rem;
        }
        .chat-message {
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 0.5rem;
            max-width: 80%;
        }
        .user-message {
            background-color: #E8F5E9;
            margin-left: auto;
        }
        .assistant-message {
            background-color: #F5F5F5;
            margin-right: auto;
        }
        .upload-section {
            border: 2px dashed #2E7D32;
            border-radius: 10px;
            padding: 2rem;
            text-align: center;
        }
        .metric-container {
            background-color: #F5F5F5;
            padding: 1rem;
            border-radius: 8px;
            margin-bottom: 1rem;
        }
        .chart-container {
            margin-top: 2rem;
            margin-bottom: 1rem;
        }
    </style>
""", unsafe_allow_html=True)

# Left Sidebar - Model Hub
with st.sidebar:
    st.header("🤖 Model Hub")
    with st.container():
        st.markdown('<div class="sidebar-box">', unsafe_allow_html=True)
        
        model_type = st.selectbox(
            'Model Type',
            ('Waste Classification', 'Climate Analysis', 'Carbon Footprint'),
            on_change=add_points, args=(1,)
        )
        
        model_name = st.selectbox(
            'Model Name',
            ('TrashNet-v1', 'EarthEngine-v2', 'CarbonCalc-v1'),
            on_change=add_points, args=(1,)
        )
        
        model_version = st.selectbox(
            'Version',
            ('Latest', 'v2.1', 'v2.0', 'v1.0'),
            on_change=add_points, args=(1,)
        )
        
        st.info(f'Selected Model: {model_name}\nVersion: {model_version}')
        st.markdown('</div>', unsafe_allow_html=True)

# Right Sidebar - Parameter Tuner
with st.sidebar.container():
    st.header("⚙ Parameter Tuner")
    st.markdown('<div class="sidebar-box">', unsafe_allow_html=True)
    
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.1,
        on_change=add_points, args=(1,)
    )
    
    processing_mode = st.selectbox(
        "Processing Mode",
        ("Fast", "Balanced", "Accurate"),
        on_change=add_points, args=(1,)
    )
    
    batch_size = st.select_slider(
        "Batch Size",
        options=[1, 2, 4, 8, 16, 32],
        on_change=add_points, args=(1,)
    )
    
    st.markdown('</div>', unsafe_allow_html=True)

# Main content with tabs
tab1, tab2, tab3 = st.tabs(["🗑 Waste Classification", "🌡 Climate Analysis", "👣 Carbon Footprint"])

# Tab 1: Waste Classification
with tab1:
    st.markdown("<h1 style='text-align: center; color: #2E7D32; margin-bottom: 2rem;'>🔍 Waste Analysis Hub</h1>", unsafe_allow_html=True)
    
    # Create tabs for different analysis methods
    analysis_tab1, analysis_tab2 = st.tabs(["💬 Chat Analysis", "📸 Visual Analysis"])
    
    # Chat Analysis Tab
    with analysis_tab1:
        st.markdown("""
            <div class='analysis-card'>
                <h3 style='color: #2E7D32; margin-bottom: 1rem;'>Interactive Waste Analyzer</h3>
                <p style='color: #666; margin-bottom: 1.5rem;'>
                    Chat with our AI assistant to get detailed waste analysis, composition information, 
                    and proper disposal recommendations.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Initialize chat history
        if 'chat_history' not in st.session_state:
            st.session_state.chat_history = []
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': """
                👋 Welcome to the Waste Analysis Chat!
                
                I can help you with:
                🔍 Detailed waste classification
                🧪 Material composition analysis
                ♻ Recycling guidance
                🌍 Environmental impact assessment
                
                Describe any waste item, and I'll provide a comprehensive analysis!
                """
            })
        
        # Chat container with custom styling
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                message_class = "user-message" if message['role'] == 'user' else "assistant-message"
                st.markdown(f"""
                    <div class='chat-message {message_class}'>
                        <b>{'You' if message['role'] == 'user' else '🤖 Assistant'}:</b><br>
                        {message['content']}
                    </div>
                """, unsafe_allow_html=True)
        
        # Input area with custom styling
        st.markdown("<div class='analysis-card' style='background-color: #f8f9fa;'>", unsafe_allow_html=True)
        user_input = st.text_input(
            "Describe the waste for analysis:",
            key="user_input",
            placeholder="e.g., 'What type of waste is a plastic water bottle?'"
        )
        
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            analyze_button = st.button("🔍 Analyze", key="chat_analyze")
        st.markdown("</div>", unsafe_allow_html=True)
        
        if analyze_button and user_input:
            # Add user message to chat history
            st.session_state.chat_history.append({
                'role': 'user',
                'content': user_input
            })
            
            # Waste analysis responses based on keywords
            waste_analysis = {
                'plastic bottle': {
                    'type': 'Plastic Waste - PET (Polyethylene Terephthalate)',
                    'composition': 'Single-use plastic, typically made from PET',
                    'biodegradability': 'Non-biodegradable (450+ years)',
                    'recycling': 'Highly recyclable - Category 1 plastic',
                    'impact': 'High environmental impact if not recycled, can harm marine life'
                },
                'food': {
                    'type': 'Organic Waste',
                    'composition': 'Biodegradable organic matter',
                    'biodegradability': 'Fully biodegradable (1-6 months)',
                    'recycling': 'Compostable',
                    'impact': 'Can produce methane in landfills if not composted properly'
                },
                'electronics': {
                    'type': 'E-Waste',
                    'composition': 'Mixed materials (metals, plastics, rare earth elements)',
                    'biodegradability': 'Non-biodegradable',
                    'recycling': 'Requires specialized recycling',
                    'impact': 'Contains hazardous materials, needs proper disposal'
                },
                'paper': {
                    'type': 'Paper Waste',
                    'composition': 'Cellulose fibers',
                    'biodegradability': 'Biodegradable (2-6 weeks)',
                    'recycling': 'Highly recyclable',
                    'impact': 'Low environmental impact if recycled properly'
                }
            }
            
            # Generate response based on waste type
            response = "I'll help analyze this waste. "
            for waste_type, details in waste_analysis.items():
                if waste_type in user_input.lower():
                    response += f"\n\n*Waste Analysis Results:*\n"
                    response += f"• Type: {details['type']}\n"
                    response += f"• Composition: {details['composition']}\n"
                    response += f"• Biodegradability: {details['biodegradability']}\n"
                    response += f"• Recycling: {details['recycling']}\n"
                    response += f"• Environmental Impact: {details['impact']}\n"
                    break
            else:
                response += ("I need more specific information about the waste to provide an accurate analysis. "
                           "You can describe the material or upload an image for visual analysis.")
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({
                'role': 'assistant',
                'content': response
            })
            
            # Add points for chat analysis
            add_points(5)
            
            # Rerun to update chat display
            st.experimental_rerun()
    
    # Visual Analysis Tab
    with analysis_tab2:
        st.markdown("""
            <div class='analysis-card'>
                <h3 style='color: #2E7D32; margin-bottom: 1rem;'>Visual Waste Classifier</h3>
                <p style='color: #666; margin-bottom: 1.5rem;'>
                    Upload an image of waste items for instant classification, hazard assessment, 
                    and recycling recommendations.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
        # Create two columns for upload and results
        upload_col, result_col = st.columns([1, 1])
        
        with upload_col:
            st.markdown("""
                <div class='upload-section'>
                    <h4 style='color: #2E7D32; margin-bottom: 1rem;'>Upload Image</h4>
                    <p style='color: #666;'>Supported formats: JPG, JPEG, PNG</p>
                </div>
            """, unsafe_allow_html=True)
            
            uploaded_file = st.file_uploader("", type=['jpg', 'jpeg', 'png'], key="visual_analysis")
        
        if uploaded_file:
            # Display image and analysis in result column
            with result_col:
                st.markdown("<div class='analysis-card'>", unsafe_allow_html=True)
                image = Image.open(uploaded_file)
                st.image(image, caption="Analyzing...", use_container_width=True)
                
                # Generate image caption
                if pipe is not None:
                    with st.spinner("Analyzing image content..."):
                        try:
                            caption = pipe(image)[0]['generated_text']
                            st.markdown(f"""
                                <div style='background-color: #E8F5E9; padding: 1rem; border-radius: 8px; margin-top: 1rem;'>
                                    <h4 style='color: #2E7D32; margin-bottom: 0.5rem;'>📝 Image Analysis</h4>
                                    <p style='color: #1B5E20; margin-bottom: 0;'>{caption}</p>
                                </div>
                            """, unsafe_allow_html=True)
                        except Exception as e:
                            st.error("Error analyzing the image. Please try again.")
                            st.error(f"Error details: {str(e)}")
                            caption = ""
                else:
                    st.error("Image analysis is currently unavailable. Please ensure all requirements are installed.")
                    caption = ""
                
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Add points for visual analysis
            add_points(5)
            
            # Analysis results in a separate card below
            st.markdown("<div class='analysis-card'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: #2E7D32; margin-bottom: 1rem;'>Analysis Results</h4>", unsafe_allow_html=True)
            
            # Initialize waste types with default values if no caption is available
            if not caption:
                waste_types = {
                    "Plastic": {"confidence": 0.2, "recyclable": True, "hazard_level": "Medium"},
                    "Paper": {"confidence": 0.2, "recyclable": True, "hazard_level": "Low"},
                    "Metal": {"confidence": 0.2, "recyclable": True, "hazard_level": "Low"},
                    "Glass": {"confidence": 0.2, "recyclable": True, "hazard_level": "Medium"},
                    "Organic": {"confidence": 0.2, "recyclable": True, "hazard_level": "Low"}
                }
            else:
                # Use the caption to determine waste type probabilities
                waste_types = {
                    "Plastic": {"confidence": 0.0, "recyclable": True, "hazard_level": "Medium"},
                    "Paper": {"confidence": 0.0, "recyclable": True, "hazard_level": "Low"},
                    "Metal": {"confidence": 0.0, "recyclable": True, "hazard_level": "Low"},
                    "Glass": {"confidence": 0.0, "recyclable": True, "hazard_level": "Medium"},
                    "Organic": {"confidence": 0.0, "recyclable": True, "hazard_level": "Low"}
                }
                
                # Simple keyword matching to adjust confidence scores
                caption_lower = caption.lower()
                keywords = {
                    "Plastic": ["plastic", "bottle", "container", "packaging"],
                    "Paper": ["paper", "cardboard", "box", "newspaper"],
                    "Metal": ["metal", "can", "aluminum", "steel"],
                    "Glass": ["glass", "bottle", "jar"],
                    "Organic": ["food", "waste", "organic", "vegetable", "fruit"]
                }
                
                # Calculate confidence scores based on keywords
                for waste_type, related_words in keywords.items():
                    confidence = sum([1 for word in related_words if word in caption_lower]) / len(related_words)
                    waste_types[waste_type]["confidence"] = min(confidence, 0.95)  # Cap at 95% confidence
                
                # Normalize confidence scores
                total_confidence = sum(type_info["confidence"] for type_info in waste_types.values())
                if total_confidence > 0:
                    for waste_type in waste_types:
                        waste_types[waste_type]["confidence"] /= total_confidence
                else:
                    # If no matches, set a small baseline confidence for each type
                    for waste_type in waste_types:
                        waste_types[waste_type]["confidence"] = 0.2
            
            # Display key metrics in a grid
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            
            top_waste = max(waste_types.items(), key=lambda x: x[1]['confidence'])
            
            with metric_col1:
                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                st.metric("Primary Type", top_waste[0])
                st.markdown("</div>", unsafe_allow_html=True)
            
            with metric_col2:
                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                st.metric("Confidence", f"{top_waste[1]['confidence']:.1%}")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with metric_col3:
                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                st.metric("Hazard Level", top_waste[1]['hazard_level'])
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Create detailed analysis chart
            st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
            analysis_data = []
            for waste_type, details in waste_types.items():
                analysis_data.append({
                    'Type': waste_type,
                    'Confidence': details['confidence'],
                    'Recyclable': 'Yes' if details['recyclable'] else 'No',
                    'Hazard Level': details['hazard_level']
                })
            
            df = pd.DataFrame(analysis_data)
            fig = px.bar(df, x='Type', y='Confidence',
                        title='Waste Classification Analysis',
                        color='Hazard Level',
                        color_discrete_map={'Low': '#4CAF50', 'Medium': '#FFA726', 'High': '#EF5350'})
            
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                title_font_size=16,
                title_font_color='#2E7D32',
                showlegend=True,
                legend_title_text='Hazard Level',
                xaxis_title="Waste Type",
                yaxis_title="Confidence Score"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Recommendations section
            st.markdown("""
                <div style='background-color: #f8f9fa; padding: 1rem; border-radius: 8px; margin-top: 1rem;'>
                    <h4 style='color: #2E7D32; margin-bottom: 0.5rem;'>📋 Recommendations</h4>
                    <ul style='margin-bottom: 0;'>
                        <li>Ensure proper segregation before disposal</li>
                        <li>Check local recycling guidelines</li>
                        <li>Consider reuse options if applicable</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

# Tab 2: Climate Analysis
with tab2:
    st.header("Climate Analysis")
    
    # Region selector
    region = st.selectbox(
        "Select Region",
        ["North America", "South America", "Europe", "Asia", "Africa", "Oceania"]
    )
    
    # Date range
    col1, col2 = st.columns(2)
    with col1:
        start_date = st.date_input("Start Date")
    with col2:
        end_date = st.date_input("End Date")
    
    # Add points for climate analysis
    if region and start_date and end_date:
        add_points(5)
    
    # Dummy data for visualization
    dates = pd.date_range(start='2020-01-01', end='2023-12-31', freq='M')
    temperature = np.random.normal(25, 5, len(dates))
    deforestation = np.cumsum(np.random.normal(0.1, 0.02, len(dates)))
    
    # Create interactive plots
    fig1 = px.line(x=dates, y=temperature, title='Temperature Changes Over Time')
    fig2 = px.line(x=dates, y=deforestation, title='Cumulative Deforestation')
    
    st.plotly_chart(fig1)
    st.plotly_chart(fig2)

# Tab 3: Carbon Footprint
with tab3:
    st.header("Carbon Footprint Calculator")
    
    # Input form
    with st.form("carbon_calculator"):
        st.subheader("Daily Activities")
        
        transport_km = st.number_input("Daily commute distance (km)", min_value=0.0)
        energy_kwh = st.number_input("Monthly energy consumption (kWh)", min_value=0.0)
        diet_type = st.selectbox("Diet type", ["Vegan", "Vegetarian", "Pescatarian", "Meat-eater"])
        
        submitted = st.form_submit_button("Calculate Footprint")
        
        if submitted:
            # Dummy calculation
            transport_emissions = transport_km * 0.2  # kg CO2
            energy_emissions = energy_kwh * 0.5  # kg CO2
            diet_emissions = {"Vegan": 1.5, "Vegetarian": 2.5, "Pescatarian": 3.5, "Meat-eater": 4.5}
            
            total_emissions = transport_emissions + energy_emissions + diet_emissions[diet_type]
            
            # Display results
            st.success(f"Your estimated annual carbon footprint: {total_emissions:.2f} tonnes CO2e")
            
            # Add points for carbon footprint calculation
            add_points(5)
            
            # Create pie chart
            emissions_breakdown = {
                "Transport": transport_emissions,
                "Energy": energy_emissions,
                "Diet": diet_emissions[diet_type]
            }
            
            df = pd.DataFrame(list(emissions_breakdown.items()), columns=['Category', 'Emissions'])
            fig = px.pie(df, values='Emissions', names='Category',
                        title='Carbon Footprint Breakdown')
            st.plotly_chart(fig)
            
            # Eco-friendly suggestions
            st.subheader("💡 Suggestions to Reduce Your Footprint")
            suggestions = [
                "Consider using public transport or cycling",
                "Switch to energy-efficient appliances",
                "Reduce meat consumption",
                "Install solar panels",
                "Use a smart thermostat"
            ]
            for suggestion in suggestions:
                st.markdown(f"- {suggestion}")
