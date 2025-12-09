import streamlit as st
import requests
from PIL import Image
import json
import io

# Configure page
st.set_page_config(
    page_title="Product Review Analyzer",
    page_icon="🛍️",
    layout="wide"
)

# API URL
API_URL = "http://localhost:8000"

st.title("🛍️ Multimodel Product Review Analyzer")
st.markdown("Analyze product images and reviews using AI.")

# Check API status
try:
    requests.get(f"{API_URL}/docs")
    st.sidebar.success("✅ API Connected")
except:
    st.sidebar.error("❌ API Not Connected. Please run `python main.py server`")

# Tabs
tab1, tab2, tab3 = st.tabs(["📸 Image Classification", "📝 Text Analysis", "🚀 Full Recommendation"])

# --- Tab 1: Image Classification ---
with tab1:
    st.header("Image Classification")
    uploaded_file = st.file_uploader("Upload Product Image", type=["jpg", "jpeg", "png"], key="img_cls")
    
    if uploaded_file:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", width=300)
        
        if st.button("Classify Image"):
            with st.spinner("Classifying..."):
                try:
                    # Reset file pointer
                    uploaded_file.seek(0)
                    files = {"file": uploaded_file}
                    response = requests.post(f"{API_URL}/classify-image", files=files)
                    
                    if response.status_code == 200:
                        result = response.json()
                        st.success(f"**Label:** {result['label']}")
                        st.info(f"**Score:** {result['score']:.4f}")
                        
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

# --- Tab 2: Text Analysis ---
with tab2:
    st.header("Sentiment Analysis")
    text_input = st.text_area("Enter Review Text", "This product is amazing! I love it.", height=150)
    
    if st.button("Analyze Sentiment"):
        with st.spinner("Analyzing..."):
            try:
                payload = {"text": text_input}
                response = requests.post(f"{API_URL}/analyze-text", json=payload)
                
                if response.status_code == 200:
                    result = response.json()
                    st.success(f"**Sentiment:** {result['label']}")
                    st.info(f"**Score:** {result['score']:.4f}")
                else:
                    st.error(f"Error: {response.text}")
            except Exception as e:
                st.error(f"Connection Error: {e}")

# --- Tab 3: Full Recommendation ---
with tab3:
    st.header("Full Product Recommendation")
    
    col1, col2 = st.columns(2)
    
    with col1:
        rec_image = st.file_uploader("1. Upload Product Image", type=["jpg", "jpeg", "png"], key="rec_img")
        if rec_image:
            st.image(rec_image, width=300)
            
    with col2:
        st.subheader("2. Enter Reviews")
        
        # Initialize session state for reviews if not exists
        if 'reviews' not in st.session_state:
            st.session_state.reviews = [""]

        # Function to add a new review input
        def add_review():
            st.session_state.reviews.append("")

        # Function to remove a review input
        def remove_review(index):
            st.session_state.reviews.pop(index)

        # Display review inputs
        for i, review in enumerate(st.session_state.reviews):
            col_input, col_remove = st.columns([0.9, 0.1])
            with col_input:
                st.session_state.reviews[i] = st.text_area(f"Review {i+1}", value=review, height=100, key=f"review_{i}")
            with col_remove:
                if len(st.session_state.reviews) > 1:
                    if st.button("🗑️", key=f"remove_{i}"):
                        remove_review(i)
                        st.rerun()

        if st.button("➕ Add Another Review"):
            add_review()
            st.rerun()
        
    if st.button("Generate Recommendation", type="primary"):
        # Filter out empty reviews
        valid_reviews = [r.strip() for r in st.session_state.reviews if r.strip()]
        
        if not rec_image or not valid_reviews:
            st.warning("Please upload an image and enter at least one review.")
        else:
            with st.spinner("Analyzing multi-modal data..."):
                try:
                    # Prepare data
                    rec_image.seek(0)
                    files = {"file": rec_image}
                    
                    # Send as JSON string as expected by the endpoint
                    data = {"reviews": json.dumps(valid_reviews)}
                    
                    response = requests.post(f"{API_URL}/recommend", files=files, data=data)
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Display Final Result
                        st.divider()
                        final_score = result['final_score']
                        recommendation = result['recommendation']
                        
                        # Check relevance
                        relevance = result['components'].get('relevance', {})
                        is_relevant = relevance.get('is_relevant', True)
                        
                        # Color coding based on score and relevance
                        if not is_relevant:
                            color = "gray"
                        elif final_score >= 3.5:
                            color = "green"
                        elif final_score >= 2:
                            color = "orange"
                        else:
                            color = "red"
                        
                        st.markdown(f"<h2 style='text-align: center; color: {color};'>{recommendation}</h2>", unsafe_allow_html=True)
                        st.markdown(f"<h3 style='text-align: center;'>Score: {final_score:.1f} / 5</h3>", unsafe_allow_html=True)
                        
                        st.divider()
                        
                        # Component Breakdown
                        c1, c2, c3 = st.columns(3)
                        
                        components = result['components']
                        
                        with c1:
                            st.subheader("Sentiment")
                            sent = components['sentiment']
                            st.metric("Label", sent['label'])
                            norm_score = sent.get('normalized_score', 0.5)
                            st.progress(float(norm_score))
                            
                        with c2:
                            st.subheader("Image Label")
                            img = components['image']
                            label_text = img['label']
                            if len(label_text) > 30:
                                label_text = ", ".join(l.strip() for l in label_text.split(",")[:2])
                            st.metric("Label", label_text)
                            st.progress(float(img['confidence_score']))
                            
                        with c3:
                            st.subheader("Relevance")
                            rel = components.get('relevance', {'score': 0.5})
                            rel_score = rel.get('score', 0.5)
                            is_rel = rel.get('is_relevant', True)
                            
                            # Show relevance status
                            status = "✓ Match" if is_rel else "⚠ Low"
                            st.metric("Status", status, delta=f"{rel_score:.2f}")
                            st.progress(float(rel_score))
                            
                            # Show details in expander
                            with st.expander("Details"):
                                st.write(f"Alignment: {rel.get('alignment_score', 0):.3f}")
                                st.write(f"Shift: {rel.get('shift_score', 0):.3f}")
                                st.write(f"Threshold: {rel.get('threshold', 0.55):.2f}")
                                
                    else:
                        st.error(f"Error: {response.text}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")
