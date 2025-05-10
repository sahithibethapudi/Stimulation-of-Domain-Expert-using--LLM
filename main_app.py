# Triggering rebuild to clear old cache
import streamlit as st
import asyncio
# Setting the page config as the very first Streamlit command
st.set_page_config(
    page_title="Knowledge Transfer System",
    layout="wide",
    initial_sidebar_state="expanded"
)

import os
import time
import logging
import tempfile
from datetime import datetime

# Setting up logging after Streamlit page config commands
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Importing all required libraries
from dotenv import load_dotenv
import plotly.graph_objects as go 
import plotly.express as px
import json
from typing import Dict, List, Any, Optional, Union
from agentic_modeling_classes import DocumentProcessor, KnowledgeSystem
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from metrics_tracker import StreamlitMetricsTracker,log_rag_question_metrics
from session_manager import SessionManager
import pandas as pd
from langchain_anthropic import ChatAnthropic
from huggingface_hub import InferenceClient
from together import Together
import boto3

# Main function for the Streamlit application.

# This function initializes the application, sets up AWS resources, creates the UI components, and handles user interactions.
# The application flow:
    # 1. User uploads a document
    # 2. Document is processed and stored in S3
    # 3. An expert agent generates questions based on the document
    # 4. User answers questions to transfer knowledge
    # 5. Responses are stored in DynamoDB

# Updating the model evaluation leaderboard with new metrics summary
def update_model_eval_leaderboard(summary):
    if "all_eval_summaries" not in st.session_state:
        st.session_state.all_eval_summaries = []

    model_name = summary["model"]
    total = summary["total_questions"]
    duplicates = summary["duplicates"]

    avg_relevance = round(sum(summary["relevance_scores"]) / total, 3) if total else 0
    avg_diversity = round(sum(summary["diversity_scores"]) / total, 3) if total else 0
    avg_readability = round(sum(summary["readability_scores"]) / total, 1) if total else 0
    # Calculating user rating average if available
    avg_user_rating = 0
    if "user_ratings" in summary and summary["user_ratings"]:
        avg_user_rating = round(sum(summary["user_ratings"]) / len(summary["user_ratings"]), 2)
    
    # Calculate final score with user feedback
    # Normalize readability score to 0-1 scale
    normalized_readability = min(avg_readability / 100.0, 1.0)
    # Base score (80% weight)
    base_score = round(0.4 * avg_relevance + 0.3 * avg_diversity + 0.3 * normalized_readability, 3)

    # If we have user ratings, they get 20% weight
    if "user_ratings" in summary and summary["user_ratings"]:
        # Convert user rating from [-1,1] to [0,1] scale for consistency
        user_score_normalized = (avg_user_rating + 1) / 2
        avg_score = round(0.8 * base_score + 0.2 * user_score_normalized, 3)
    else:
        avg_score = base_score

    # Checking if already exists
    existing = next((entry for entry in st.session_state.all_eval_summaries if entry["Model"] == model_name), None)

    if existing:
        existing["Total Questions"] = total
        existing["Repeated Questions"] = duplicates
        existing["Avg Context Relevance"] = avg_relevance
        existing["Avg Diversity"] = avg_diversity
        existing["Avg Readability"] = avg_readability
        existing["Avg User Rating"] = avg_user_rating  # Added user rating
        existing["Avg Score"] = avg_score
    else:
        # Adding new entry
        st.session_state.all_eval_summaries.append({
            "Model": model_name,
            "Total Questions": total,
            "Repeated Questions": duplicates,
            "Avg Context Relevance": avg_relevance,
            "Avg Diversity": avg_diversity,
            "Avg Readability": avg_readability,
            "Avg User Rating": avg_user_rating,  # Added user rating
            "Avg Score": avg_score
        })
    logger.info(f"Updated leaderboard for {model_name}: score={avg_score:.3f}, user_rating={avg_user_rating:.2f}")


def main():

    # Loading environment variables
    load_dotenv()

    # Fixing asyncio event loop issue
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.set_event_loop(asyncio.new_event_loop())

    # Creating AWS clients with credentials first
    creds = get_streamlit_secrets()
    if creds is None:
        st.error("Error accessing secrets")
        st.warning("Using environment variables instead of secrets.")

    # ===== AWS UTILITY FUNCTIONS ===== #
    # Function for getting S3 client with credentials if provided
    def get_s3_client(credentials=None):
        if credentials and credentials.get("aws_access_key_id") and credentials.get("aws_secret_access_key"):
            return boto3.client(
                's3',
                region_name=credentials.get("region_name", "us-east-1"),
                aws_access_key_id=credentials["aws_access_key_id"],
                aws_secret_access_key=credentials["aws_secret_access_key"]
            )
        else:
            # Using default credentials (from environment variables)
            return boto3.client('s3', region_name="us-east-1")

    # Function for getting DynamoDB resource with credentials if provided
    def get_dynamodb_resource(credentials=None):
        if credentials and credentials.get("aws_access_key_id") and credentials.get("aws_secret_access_key"):
            return boto3.resource(
                'dynamodb',
                region_name=credentials.get("region_name", "us-east-1"),
                aws_access_key_id=credentials["aws_access_key_id"],
                aws_secret_access_key=credentials["aws_secret_access_key"]
            )
        else:
            # Trying to use Streamlit secrets if environment variables are missing
            try:
                secrets = st.secrets["aws"]
                return boto3.resource(
                    'dynamodb',
                    region_name=secrets["region_name"],
                    aws_access_key_id=secrets["aws_access_key_id"],
                    aws_secret_access_key=secrets["aws_secret_access_key"]
                )
            except:
                logger.error("❌ AWS credentials missing for DynamoDB!")
                return None  # Prevents crashing

    # Function for getting CloudWatch client with credentials if provided
    def get_cloudwatch_client(credentials=None):
        if credentials and credentials.get("aws_access_key_id") and credentials.get("aws_secret_access_key"):
            return boto3.client(
                'cloudwatch',
                region_name=credentials.get("region_name", "us-east-1"),
                aws_access_key_id=credentials["aws_access_key_id"],
                aws_secret_access_key=credentials["aws_secret_access_key"]
            )
        else:
            # Using default credentials (from environment variables)
            return boto3.client('cloudwatch', region_name="us-east-1")
    
    # Creating AWS clients with credentials
    s3_client = get_s3_client(creds)
    dynamodb_resource = get_dynamodb_resource(creds)
    cloudwatch_client = get_cloudwatch_client(creds)
    
    # Initializing session state
    initialize_session_state(dynamodb_resource, cloudwatch_client)

    # Initializing session manager
    session_manager = SessionManager()
    
    # Defining modern color palette
    colors = {
        'primary': '#3b7ea1',       # Main blue color
        'primary_dark': '#2c5f7c',  # Darker blue for hover states
        'secondary': '#4CAF50',     # Green for success/confirm actions
        'secondary_dark': '#3d8b40',# Darker green for hover
        'accent': '#ff9800',        # Orange for attention/warning
        'accent_dark': '#e68a00',   # Darker orange
        'danger': '#f44336',        # Red for dangerous actions
        'danger_dark': '#d32f2f',   # Darker red
        'background': '#f8f9fa',    # Light background
        'surface': '#ffffff',       # White surface
        'text': '#212529',          # Dark text
        'text_secondary': '#5f6369',# Secondary text
        'border': '#e9ecef'         # Border color
    }
    
    # Adding custom CSS for improved styling
    st.markdown(f"""
        <style>
        /* Base styles */
        .main-header {{
            font-size: 2.2rem;
            margin-bottom: 1rem;
        }}
        
        /* Modern tab styling */
        .stTabs {{
            margin-top: 1rem;
            margin-bottom: 2rem;
        }}
        
        .stTabs [data-baseweb="tab-list"] {{
            gap: 8px;
            border-bottom: 2px solid {colors['border']};
            padding-bottom: 0;
        }}
        
        .stTabs [data-baseweb="tab"] {{
            height: 50px;
            border-radius: 8px 8px 0 0;
            padding: 10px 16px;
            background-color: {colors['background']};
            border: none;
            font-weight: 500;
            font-size: 1rem;
            color: {colors['text_secondary']};
            transition: all 0.3s ease;
        }}
        
        .stTabs [data-baseweb="tab"]:hover {{
            background-color: {colors['border']};
            color: {colors['primary']};
        }}
        
        .stTabs [aria-selected="true"] {{
            background-color: {colors['primary']} !important;
            color: white !important;
            border-bottom: none !important;
            font-weight: 600;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }}
        
        /* Tab content container */
        .stTabs [data-testid="stTabContent"] {{
            padding: 20px 0;
        }}
        
        /* Modern button styling */
        .stButton > button {{
            border-radius: 8px;
            font-weight: 500;
            border: none;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
        }}
        
        .stButton > button:hover {{
            box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            transform: translateY(-2px);
        }}
        
        .stButton > button:active {{
            transform: translateY(0px);
        }}
        
        /* Custom process button */
        [data-testid="element-container"]:has([data-testid="stButton"] > button:contains("Process Document")) button {{
            background-color: {colors['secondary']};
            color: white;
        }}
        
        [data-testid="element-container"]:has([data-testid="stButton"] > button:contains("Process Document")) button:hover {{
            background-color: {colors['secondary_dark']};
        }}
        
        /* Submit button */
        [data-testid="element-container"]:has([data-testid="stButton"] > button:contains("Submit Response")) button {{
            background-color: {colors['primary']};
            color: white;
        }}
        
        [data-testid="element-container"]:has([data-testid="stButton"] > button:contains("Submit Response")) button:hover {{
            background-color: {colors['primary_dark']};
        }}
        
        /* Stop button */
        [data-testid="element-container"]:has([data-testid="stButton"] > button:contains("Stop Session")) button {{
            background-color: {colors['danger']};
            color: white;
        }}
        
        [data-testid="element-container"]:has([data-testid="stButton"] > button:contains("Stop Session")) button:hover {{
            background-color: {colors['danger_dark']};
        }}
        
        /* For uploading file area */
        [data-testid="stFileUploader"] {{
            border-radius: 10px;
            padding: 20px;
            border: 2px dashed {colors['primary']};
            background-color: {colors['background']};
        }}
        
        [data-testid="stFileUploader"]:hover {{
            border-color: {colors['primary_dark']};
            background-color: #f1f3f5;
        }}
        
        /* Custom progress bar */
        .stProgress > div > div > div {{
            background-color: {colors['primary']};
            background-image: linear-gradient(45deg, 
                            rgba(255, 255, 255, 0.15) 25%, 
                            transparent 25%, 
                            transparent 50%, 
                            rgba(255, 255, 255, 0.15) 50%, 
                            rgba(255, 255, 255, 0.15) 75%, 
                            transparent 75%, 
                            transparent);
            background-size: 1rem 1rem;
            animation: progress-bar-stripes 1s linear infinite;
        }}
        
        @keyframes progress-bar-stripes {{
            0% {{ background-position: 1rem 0; }}
            100% {{ background-position: 0 0; }}
        }}
        
        /* Question box styling */
        .question-box {{
            background-color: #f0f2f6;
            padding: 1.5rem;
            border-radius: 0.75rem;
            border-left: 5px solid {colors['primary']};
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        }}
        
        /* Response area styling */
        .response-area {{
            background-color: #ffffff;
            padding: 1.5rem;
            border-radius: 0.75rem;
            border: 1px solid #e0e0e0;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        }}
        
        /* Info box styling */
        .info-box {{
            background-color: #e8f4f8; 
            padding: 1.25rem;
            border-radius: 0.75rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
            border-left: 4px solid {colors['primary']};
        }}
        </style>
    """, unsafe_allow_html=True)
    
    # Main title with icon
    st.title("📚 Interactive Knowledge Transfer System")
    
    
    # ===== SIDEBAR CONFIGURATION ===== #
    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        
        # Domain selection
        domain_options = ["Data Science", "Software Engineering", "Electrical Engineering"]
        # Default domain set to Data Science
        if 'selected_domain' not in st.session_state:
            st.session_state.selected_domain = "Data Science"
            
        selected_domain = st.selectbox(
            "Select Knowledge Domain",
            options=domain_options,
            index=domain_options.index(st.session_state.selected_domain),
            help="Choose your domain"
        )
        
        # Updating session state if domain changed
        if selected_domain != st.session_state.selected_domain:
            st.session_state.selected_domain = selected_domain
            # Reset of relevant session state for the new domain
            st.session_state.uploaded_file = None
            st.session_state.processing_complete = False
            st.session_state.knowledge_system = None
            st.session_state.inquiry_started = False
            st.rerun()
        
        # Model selection for expert agent
        model_options = [
            "claude-3-haiku-20240307", 
            "claude-3-sonnet-20240229", 
            "google/gemma-2-9b-it",
            "gemma-finetuned-dora",
            "Qwen/Qwen2.5-7B-Instruct-Turbo",
            "qwen-finetuned-dora", 
            "deepseek-chat",
            "deepseekcoder-qlora8bit-finetuned",
            "llama-3.2-base",
            "llama-lora-finetuned"
            ]
        if 'expert_model' not in st.session_state:
            st.session_state.expert_model = model_options[0]
            
        expert_model = st.selectbox(
            "Select Expert Model",
            options=model_options,
            index=model_options.index(st.session_state.expert_model),
            help="Choose the expert agent model"
        )
        
        # Storing the selected model in session state
        if st.session_state.expert_model != expert_model:
            st.session_state.expert_model = expert_model
            # Reset of knowledge system if it exists to use the new model
            if 'knowledge_system' in st.session_state and st.session_state.knowledge_system:
                st.session_state.knowledge_system = None
                st.session_state.processing_complete = False
        
        st.divider()
        
        # Session controls
        st.header("Session Controls")
        
        # Session controls
        if st.button("Reset Session", use_container_width=True):
            # Logging final metrics before resetting
            if 'metrics_tracker' in st.session_state and st.session_state.metrics_tracker:
                st.session_state.metrics_tracker.log_metrics()
            
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            initialize_session_state()
            st.success("Session reset successfully!")
            st.rerun()
        
        # Knowledge base viewer button
        if st.button("View Knowledge Base", use_container_width=True):
            view_knowledge_base()
        
        # Displaying metrics if available
        if 'metrics_tracker' in st.session_state and st.session_state.metrics_tracker is not None:
            st.session_state.metrics_tracker.display_metrics_dashboard()
            
            # Displaying historical metrics
            st.session_state.metrics_tracker.load_historical_metrics()
    
    # Main content area with tabs for better navigation
    # tab1, tab2, tab3 = st.tabs(["📄 Document Upload", "💬 Knowledge Transfer", "📊 Results"])
    tab1, tab2, tab3, tab4 = st.tabs(["📄 Document Upload", "💬 Knowledge Transfer", "📊 Qualitative Metrics", "📊 Quantitative Metrics"])
    
    # ===== DOCUMENT UPLOAD TAB ===== #
    with tab1:
        # Document upload section
        st.header("Document Upload")
        st.markdown("Upload a knowledge transfer document (PDF) to begin the process.")
        
        # Two-column layout for upload and processing
        col1, col2 = st.columns([3, 1])
        
        with col1:
            uploaded_file = st.file_uploader(
                "Upload Knowledge Transfer Document (PDF)",
                type=["pdf"],
                help="Upload the knowledge transition document",
                key="doc_uploader"
            )
            
            # Showing file details if uploaded
            if uploaded_file:
                file_details = {
                    "Filename": uploaded_file.name,
                    "File size": f"{uploaded_file.size / 1024:.2f} KB",
                    "File type": uploaded_file.type
                }
                st.json(file_details)
        
        with col2:
            if uploaded_file:
                st.success("✅ Document uploaded!")
                process_button = st.button("🚀 Process Document", use_container_width=True)

                # Button to process all S3 documents
                st.write("")  # Adding some spacing
                process_all_button = st.button("🔄 Process All S3 Documents", use_container_width=True)

                if process_button:
                    with st.spinner("Processing document..."):
                        # Saving temporary file
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                            tmp_file.write(uploaded_file.getvalue())
                            temp_path = tmp_file.name

                        # Defining domain-specific bucket names
                        domain_buckets = {
                            "Data Science": ("knowledge-raw-team1", "knowledge-processed-team1"),
                            "Software Engineering": ("se-knowledge-raw", "se-knowledge-processed"),
                            "Electrical Engineering": ("ee-knowledge-raw", "ee-knowledge-processed")
                        }

                        # Getting appropriate S3 buckets for the selected domain
                        raw_bucket, processed_bucket = domain_buckets[st.session_state.selected_domain]

                        # Initializing the document processor
                        processor = DocumentProcessor(raw_bucket, processed_bucket, s3_client=s3_client)

                        # Uploading document to S3 before processing
                        s3_client.upload_file(temp_path, raw_bucket, uploaded_file.name)
                        st.info(f"📤 Uploaded {uploaded_file.name} to {raw_bucket} bucket.")

                        # Adding a progress bar for visual feedback
                        progress_bar = st.progress(0)

                        # Helper function for updating progress
                        def update_progress(progress, status=""):
                            progress_bar.progress(progress)
                            # Only displaying major milestone messages, not every document status
                            if "Building vector store" in status or "All documents processed" in status:
                                st.info(f"Status: {status}")
                            # Logging all messages without displaying them
                            logger.info(f"Progress: {progress:.2f} - {status}")

                        # Determining file size and processing method
                        file_size_mb = uploaded_file.size / (1024 * 1024)
                        try:
                            if file_size_mb > 10:
                                # Large document processing
                                st.info(f"📖 Large document detected ({file_size_mb:.1f} MB). Processing in batches...")

                                chunks = processor.process_large_document(
                                    temp_path, uploaded_file.name, max_pages_per_chunk=20, progress_callback=update_progress
                                )
                            else:
                                # Standard document processing
                                chunks = processor.process_with_progress(
                                    temp_path, uploaded_file.name, progress_callback=update_progress
                                )

                            # After processing, trying to load or update FAISS vector store
                            try:
                                update_progress(0.8, "Loading vector store")
                                vector_store = processor.get_vector_store(new_documents=chunks)
                            except Exception as e:
                                st.warning(f"⚠️ Creating new vector store: {str(e)}")
                                # Ensure embeddings model is loaded
                                if 'embeddings_model' not in st.session_state:
                                    with st.spinner("🔍 Loading embeddings model..."):
                                        st.session_state.embeddings_model = HuggingFaceEmbeddings(
                                            model_name="sentence-transformers/all-mpnet-base-v2"
                                        )
                                # Creating vector store from processed chunks
                                vector_store = FAISS.from_documents(chunks, st.session_state.embeddings_model)

                            # Saving updated FAISS index to S3
                            update_progress(0.9, "Saving FAISS index to S3...")
                            data = vector_store.serialize_to_bytes()
                            s3_client.put_object(
                                Bucket=processed_bucket,
                                Key="vector_store/faiss_index.pickle",
                                Body=data
                            )
                            st.success("✅ FAISS index updated successfully!")

                            # Loading API keys for LLM interaction
                            ANTHROPIC_API_KEY = creds.get("anthropic_api_key") if creds else os.getenv('ANTHROPIC_API_KEY')
                            HUGGINGFACE_API_KEY = creds.get("huggingface_api_key") if creds else os.getenv('HUGGINGFACE_API_KEY')
                            TOGETHER_API_KEY = creds.get("together_api_key") if creds else os.getenv('TOGETHER_API_KEY')

                            # Ensuring required API keys are available
                            if not ANTHROPIC_API_KEY:
                                st.error("❌ Anthropic API key missing. Please add it to your secrets or environment variables.")
                                os.unlink(temp_path)
                                st.stop()

                            # Initializing knowledge system
                            st.session_state.knowledge_system = KnowledgeSystem(
                                vector_store=vector_store,
                                anthropic_api_key=ANTHROPIC_API_KEY,
                                huggingface_api_key=HUGGINGFACE_API_KEY,
                                together_api_key=TOGETHER_API_KEY,
                                dynamodb_resource=dynamodb_resource
                            )

                            # Assigning the selected model to the expert agent
                            model_name = st.session_state.expert_model
                            st.session_state.knowledge_system.expert.set_model(
                                model_name=model_name,
                                anthropic_api_key=ANTHROPIC_API_KEY,
                                huggingface_api_key=HUGGINGFACE_API_KEY,
                                together_api_key=TOGETHER_API_KEY
                            )

                            # Cleaning up temporary file
                            os.unlink(temp_path)

                            update_progress(1.0, "✅ Processing complete")
                            st.session_state.processing_complete = True

                            # Logging the successful processing event
                            session_manager.log_interaction("📄 Document processed successfully!", "success")

                            # Displaying success message and instructions
                            st.success("✅ Document processed successfully!")
                            st.info("Go to the **Knowledge Transfer** tab to begin the session.")

                            # Button to navigate to the next tab
                            if st.button("➡️ Go to Knowledge Transfer"):
                                st.session_state.active_tab = "Knowledge Transfer"
                                st.rerun()

                        except Exception as e:
                            st.error(f"⚠️ Error processing document: {str(e)}")
                            os.unlink(temp_path)  # Cleanup on error
                            st.stop()

                if process_all_button:
                    if process_all_raw_documents(s3_client, dynamodb_resource, creds):
                        st.info("Go to the **Knowledge Transfer** tab to begin the session.")
                        if st.button("➡️ Go to Knowledge Transfer", key="goto_kt_all"):
                            st.session_state.active_tab = "Knowledge Transfer"
                            st.rerun()

                        
    
    # ===== KNOWLEDGE TRANSFER TAB ===== #
    with tab2:
        # Knowledge transfer session
        st.header("Knowledge Transfer Session")
        
        if not st.session_state.processing_complete:
            # Showing informative message if document not processed yet
            st.info("Please upload and process a document in the Document Upload tab first.")
            col1, col2 = st.columns([3, 1])
            with col2:
                # Adding convenience button to go to upload tab
                if st.button("Go to Document Upload", use_container_width=True):
                    st.session_state.active_tab = "Document Upload"
                    st.rerun()
        else:
            # Section for starting knowledge transfer or continuing session
            if not st.session_state.inquiry_started:
                # Session start section with some explanation
                st.markdown("""
                    ### How Knowledge transfer works
                    
                    The knowledge transfer session will help extract critical information from your document.
                    The AI expert agent will ask targeted questions based on the document content, and your responses
                    will be stored in the knowledge base.
                    
                    - You'll answer up to 10 questions about the document
                    - Your responses will be saved to the knowledge base
                    - You can stop the session at any time by clicking on the stop button or typing 'stop and exit'
                """)
                
                # Displaying domain and model in a nice info box
                st.markdown(f"""
                <div class="info-box">
                    <h4>Session Configuration</h4>
                    <p><strong>Domain:</strong> {st.session_state.selected_domain}<br>
                    <strong>Expert Model:</strong> {st.session_state.expert_model}</p>
                </div>
                """, unsafe_allow_html=True)

                # Styling the start button
                st.markdown("""
                <style>
                [data-testid="element-container"]:has([data-testid="stButton"] > button:contains("Start Knowledge Transfer")) button {
                    background-color: #4CAF50;
                    color: white;
                    font-size: 1.1rem;
                    padding: 0.6rem 1.2rem;
                }
                [data-testid="element-container"]:has([data-testid="stButton"] > button:contains("Start Knowledge Transfer")) button:hover {
                    background-color: #3d8b40;
                }
                </style>
                """, unsafe_allow_html=True)
                
                # Start button with clear visual emphasis
                start_col1, start_col2, start_col3 = st.columns([1, 2, 1])
                with start_col2:
                    if st.button("🚀 Start Knowledge Transfer Session", use_container_width=True):
                        start_time = time.time()
                        initial_query = "Identify knowledge gaps based on transition document."
                        
                        try:
                            first_question = st.session_state.knowledge_system.expert.ask_questions(
                                initial_query,
                                st.session_state.known_info
                            )
                            retrieval_time = time.time() - start_time
                            
                            st.session_state.last_question = first_question
                            st.session_state.inquiry_started = True
                            
                            # Updating metrics
                            if 'metrics_tracker' not in st.session_state:
                                st.session_state.metrics_tracker = StreamlitMetricsTracker()

                            # Initialize qa_pairs if not present
                            if 'qa_pairs' not in st.session_state:
                                st.session_state.qa_pairs = []
                                
                            st.session_state.metrics_tracker.update_metrics(
                                initial_query,
                                first_question,
                                retrieval_time,
                                0
                            )
                            
                            session_manager.log_interaction(
                                "Knowledge transfer session started",
                                "success"
                            )
                            
                            st.rerun()
                        except Exception as e:
                            session_manager.handle_error(e, "starting_session")
            
            # Active knowledge transfer session
            if st.session_state.inquiry_started:
                # Displaying question counter and progress bar
                # question_count = len(st.session_state.known_info) + 1
                question_count =  len(st.session_state.get("qa_pairs", []))+1 
                st.markdown(f"""
                <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
                    <h3>Knowledge Transfer Progress</h3>
                    <span style="font-size: 1.2rem; font-weight: bold;">Question {question_count}/10</span>
                </div>
                """, unsafe_allow_html=True)
                
                progress_value = min(question_count / 10, 1.0)
                st.progress(progress_value)
                
                # Displaying the question in a nice formatted box
                st.markdown(f"""
                <div class="question-box">
                    <h3>Expert Question:</h3>
                    <p style="font-size: 1.1rem;">{st.session_state.last_question}</p>
                </div>
                """, unsafe_allow_html=True)

                # Adding custom CSS for the feedback buttons
                st.markdown("""
                <style>
                /* Styling for upvote button */
                [data-testid="element-container"]:has(button[key*="upvote_"]) button {
                    background-color: #4CAF50 !important;  /* Green color */
                    color: white !important;
                    min-height: 50px !important;
                    width: 100% !important;
                    font-size: 1.5rem !important;
                    padding: 0.75rem !important;
                    border-radius: 8px !important;
                    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
                }

                [data-testid="element-container"]:has(button[key*="upvote_"]) button:hover {
                    transform: translateY(-2px) !important;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2) !important;
                }

                /* Styling for downvote button */
                [data-testid="element-container"]:has(button[key*="downvote_"]) button {
                    background-color: #F44336 !important;  /* Red color */
                    color: white !important;
                    min-height: 50px !important;
                    width: 100% !important;
                    font-size: 1.5rem !important;
                    padding: 0.75rem !important;
                    border-radius: 8px !important;
                    transition: transform 0.2s ease, box-shadow 0.2s ease !important;
                }

                [data-testid="element-container"]:has(button[key*="downvote_"]) button:hover {
                    transform: translateY(-2px) !important;
                    box-shadow: 0 4px 8px rgba(0, 0, 0, 0.2) !important;
                }
                </style>
                """, unsafe_allow_html=True)

                # Adding user feedback buttons and adjusting column widths for bigger buttons and better layout
                feedback_col1, feedback_col2, feedback_col3 = st.columns([0.15, 0.15, 0.7])

                with feedback_col1:
                    if st.button("👍", key=f"upvote_{st.session_state.response_counter}"):
                        # Finding the current question in qa_pairs
                        if len(st.session_state.get("qa_pairs", [])) > 0:
                            current_idx = len(st.session_state.qa_pairs) - 1
                            if current_idx >= 0:
                                st.session_state.qa_pairs[current_idx]["User Rating"] = 1
                                st.session_state.qa_pairs[current_idx]["Has User Feedback"] = True
                                st.success("Question upvoted!")

                with feedback_col2:
                    if st.button("👎", key=f"downvote_{st.session_state.response_counter}"):
                        # Finding the current question in qa_pairs
                        if len(st.session_state.get("qa_pairs", [])) > 0:
                            current_idx = len(st.session_state.qa_pairs) - 1
                            if current_idx >= 0:
                                st.session_state.qa_pairs[current_idx]["User Rating"] = -1
                                st.session_state.qa_pairs[current_idx]["Has User Feedback"] = True
                                st.success("Question downvoted!")
                 
                # Adding help text for stop command
                st.caption("Type 'stop and exit' to end the session")
                
                # User response section
                response_col1, response_col2 = st.columns([3, 1])
                
                with response_col1:
                    # Adding a unique key for the text input that changes with each response
                    input_key = f"user_response_{st.session_state.response_counter}"
                    user_response = st.text_area(
                        "Your response:",
                        key=input_key,
                        height=150,
                        placeholder="Enter your response here..."
                    )
                
                with response_col2:
                    # Submit button with clear visual emphasis
                    submit_response = st.button("📤 Submit Response", use_container_width=True)
                    
                    st.write("")  # Spacer
                    
                    # Adding a stop button
                    if st.button("⏹️ Stop Session", use_container_width=True):
                        st.session_state.inquiry_started = False
                        st.session_state.session_stopped = True
                        st.session_state.last_question = None
                        st.success("Session ended by user.")
                        
                        # Logging final metrics
                        if 'metrics_tracker' in st.session_state and st.session_state.metrics_tracker:
                            st.session_state.metrics_tracker.log_metrics()
                        
                        session_manager.log_interaction(
                            "Session ended by user command",
                            "info"
                        )
                        
                        time.sleep(1)
                        st.rerun()
                
                # Processing response when submitted
                if submit_response:
                    if user_response.strip():
                        # Checking for stop command
                        if user_response.lower().strip() == 'stop and exit':
                            st.success("🛑 Session ended by user command.")
                            st.session_state.inquiry_started = False
                            st.session_state.session_stopped = True  # ✅ Flag to prevent further submission

                            if len(st.session_state.get("qa_pairs", [])) >= 10:
                                st.warning("You have reached the maximum of 10 questions for this session.")
                                st.session_state.session_stopped = True
                                return
                            
                            # Logging final metrics
                            if 'metrics_tracker' in st.session_state and st.session_state.metrics_tracker:
                                st.session_state.metrics_tracker.log_metrics()
                            
                            # Display final metrics
                            st.subheader("Final Session Summary")
                            st.write(f"Total questions answered: {len(st.session_state.known_info)}")
                            
                            if 'metrics_tracker' in st.session_state and st.session_state.metrics_tracker:
                                st.session_state.metrics_tracker.display_metrics_dashboard()
                            
                            session_manager.log_interaction(
                                "Session ended by user command",
                                "info"
                            )
                            
                            # Adding option to start new session
                            if st.button("Start New Session"):
                                st.session_state.inquiry_started = False
                                st.session_state.known_info = []
                                st.session_state.last_question = None
                                st.session_state.response_counter = 0
                                st.rerun()
                        
                        else:
                            try:
                                with st.spinner("Processing your response..."):
                                    response_start_time = time.time()
                                    
                                    # ✅ Prevent continuing after 10 questions
                                    if len(st.session_state.known_info) >= 10:
                                        st.success("✅ You've completed all 10 questions. Knowledge transfer session ended.")
                                        st.session_state.inquiry_started = False

                                        # Log final metrics
                                        if 'metrics_tracker' in st.session_state and st.session_state.metrics_tracker:
                                            st.session_state.metrics_tracker.log_metrics()
                                            st.session_state.metrics_tracker.display_metrics_dashboard()

                                        # Optional restart
                                        if st.button("Start New Session"):
                                            st.session_state.inquiry_started = False
                                            st.session_state.known_info = []
                                            st.session_state.last_question = None
                                            st.session_state.response_counter = 0
                                            st.rerun()

                                        st.stop()  # ✅ clean halt without breaking Streamlit
                                    
                                    # Using this approach for more efficient batch processing:
                                    try:
                                        # Creating a batch of items to update
                                        current_model = st.session_state.expert_model
                                        batch_items = [(st.session_state.last_question, user_response)]
                                        
                                        # Adding any pending items from previous sessions if available
                                        if 'pending_kb_updates' in st.session_state:
                                            batch_items.extend(st.session_state.pending_kb_updates)
                                            st.session_state.pending_kb_updates = []
                                        
                                        # Batch update the knowledge base
                                        kb_update_status = st.session_state.knowledge_system.knowledge_manager.batch_update_knowledge_base(batch_items)
                                    except Exception as e:
                                        # Fall back to single update if batch fails
                                        kb_update_status = st.session_state.knowledge_system.knowledge_manager.update_knowledge_base(
                                            st.session_state.last_question,
                                            user_response, 
                                            current_model
                                        )
                                        logger.warning(f"Falling back to single item update: {str(e)}")

                                    # === Initialize eval_summary before first metric logging ===
                                    if "eval_summary" not in st.session_state and "knowledge_system" in st.session_state and st.session_state.knowledge_system:
                                        selected_model_name = st.session_state.knowledge_system.expert.model_name
                                        st.session_state.eval_summary = {
                                            "model": selected_model_name,
                                            "total_questions": 0,
                                            "duplicates": 0,
                                            "relevance_scores": [],
                                            "diversity_scores": [],
                                            "readability_scores": [], 
                                            "user_ratings": []  # Added for user feedback
                                        }
                                    # # ⬇️ Append current Q&A and model
                                    # st.session_state.known_info.append(st.session_state.last_question)
                                    # st.session_state.known_info.append(user_response)
                                    # st.session_state.models_used_per_step.append(current_model)

                                    # 🔍 Debug log
                                    print("🔍 known_info:", st.session_state.known_info)
                                    print("🔍 models_used_per_step:", st.session_state.models_used_per_step)
                                    
                                    # Getting next question from the knowledge system
                                    next_question = st.session_state.knowledge_system.expert.ask_questions(
                                        st.session_state.last_question,
                                        st.session_state.known_info + [user_response]
                                    )
                                    
                                    # # ADD THE EVALUATION DATA RETRIEVAL HERE
                                    # if hasattr(st.session_state.knowledge_system.expert, "last_question_evaluation"):
                                    #     evaluation_data = st.session_state.knowledge_system.expert.last_question_evaluation
                                    #     print(f"Retrieved evaluation data: {evaluation_data}")
                                    # else:
                                    #     evaluation_data = {"score": 0}
                                    #     print("No last_question_evaluation attribute found")
                                        
                                    evaluation_data = getattr(st.session_state.knowledge_system.expert, "last_question_evaluation", {"score": 0})
                                    print(f"Retrieved evaluation data: {evaluation_data}")
                                    
                                    # Get the context used by the expert agent for this question
                                    context_used = "\n".join(st.session_state.knowledge_system.expert.last_retrieved_context)
                                    model_name = st.session_state.knowledge_system.expert.model_name if st.session_state.knowledge_system else "Unknown"

                                    # Evaluation of RAG-generated questions: wandb log metrics
                                    from metrics_tracker import log_rag_question_metrics

                                    if 'rag_question_embeddings' not in st.session_state:
                                        st.session_state.rag_question_embeddings = []

                                    # Get the context used by the expert agent for this question
                                    context_used = "\n".join(st.session_state.knowledge_system.expert.last_retrieved_context)
                                    model_name = st.session_state.knowledge_system.expert.model_name if st.session_state.knowledge_system else "Unknown"
                                    # Log evaluation and store embedding for diversity computation
                                    metrics = log_rag_question_metrics(
                                    question=next_question,
                                    context=context_used,
                                    step=st.session_state.metrics['questions_asked'],
                                    previous_embeddings=st.session_state.rag_question_embeddings,
                                    #previous_questions=st.session_state.known_info # to track all the asked questions
                                    #model_used=st.session_state.model_name
                                    previous_questions=[qap["Question"] for qap in st.session_state.qa_pairs] if "qa_pairs" in st.session_state else []
                                    )   
                                    # update_model_eval_leaderboard(summary)
                                    # Initialize qa_pairs if not present
                                    if 'qa_pairs' not in st.session_state:
                                        st.session_state.qa_pairs = []

                                    # Appending full Q&A with metadata and metrics
                                    st.session_state.qa_pairs.append({
                                        "Step": len(st.session_state.qa_pairs) + 1, 
                                        "Model Used": current_model,
                                        "Question": st.session_state.last_question,
                                        "Answer": user_response,
                                        "Context Relevance": metrics["context_relevance"],
                                        "Diversity": metrics["diversity"],
                                        "Readability": metrics["readability"],
                                        "Final Score": round(
                                            0.4 * metrics["context_relevance"] +
                                            0.3 * metrics["diversity"] +
                                            0.3 * metrics["readability"], 3
                                        ), 
                                        "User Rating": 0,  # New field to store the rating (-1, 0, +1) 
                                        "Has User Feedback": False  # Flag to track if user has provided feedback

                                    })
                                    st.session_state.known_info.append(user_response)
                                    # === Update or Reset eval_summary if model switched ===
                                    current_model = st.session_state.knowledge_system.expert.model_name
                                    existing_summary_model = st.session_state.eval_summary.get("model") if "eval_summary" in st.session_state else None

                                    if current_model != existing_summary_model:
                                        st.session_state.eval_summary = {
                                            "model": current_model,
                                            "total_questions": 0,
                                            "duplicates": 0,
                                            "relevance_scores": [],
                                            "diversity_scores": [],
                                            "readability_scores": [], 
                                            "user_ratings": []  # Added for user feedback
                                        }


                                    # Update evaluation summary
                                    summary = st.session_state.eval_summary
                                    summary["total_questions"] += 1
                                    summary["relevance_scores"].append(metrics["context_relevance"])
                                    summary["diversity_scores"].append(metrics["diversity"])
                                    summary["readability_scores"].append(metrics["readability"])

                                    # Collecting any user rating from the previous question
                                    if len(st.session_state.qa_pairs) > 0 and st.session_state.qa_pairs[-1].get("Has User Feedback", False):
                                        summary["user_ratings"].append(st.session_state.qa_pairs[-1].get("User Rating", 0))

                                    if metrics["is_duplicate"]:
                                        summary["duplicates"] += 1
                                        # st.warning(f"⚠️ Repeated question detected:\n**{next_question}**")
                                    
                                    # === Push to leaderboard
                                    update_model_eval_leaderboard(summary)

                                    # Storing embedding for diversity in next steps
                                    st.session_state.rag_question_embeddings.append(metrics["embedding"])
                                    
                                    # Calculating response time
                                    response_time = time.time() - response_start_time
                                    
                                    #Including evaluation data in metrics
                                    evaluation_data = getattr(st.session_state.knowledge_system.expert, "last_question_evaluation", None)
                                    if evaluation_data is None:  
                                        evaluation_data = {"score": 0, "feedback": "No evaluation available"}
                                    
                                    # Updating metrics
                                    if 'metrics_tracker' in st.session_state and st.session_state.metrics_tracker:
                                        # Get context data if available
                                        if hasattr(st.session_state.knowledge_system.expert, "last_retrieved_context"):
                                            context_data = {
                                                "relevant_chunks": st.session_state.knowledge_system.expert.last_retrieved_context
                                            }
                                        else:
                                            context_data = None
                                                
                                        st.session_state.metrics_tracker.update_metrics(
                                            st.session_state.last_question,
                                            user_response,
                                            retrieval_time=0.0,
                                            response_time=response_time,
                                            evaluation_data=evaluation_data,  # Use the evaluation data we retrieved
                                            context_data=context_data, 
                                            user_rating=st.session_state.qa_pairs[-1].get("User Rating") if len(st.session_state.qa_pairs) > 0 and st.session_state.qa_pairs[-1].get("Has User Feedback", False) else None
                                        )
                                    
                                    # Incrementing response counter
                                    if 'response_counter' in st.session_state:
                                        st.session_state.response_counter += 1
                                    else:
                                        st.session_state.response_counter = 1
                                    
                                    # Checking if knowledge transfer is complete
                                    if next_question == "No further questions." or "No further questions" in next_question:
                                        # Visual celebration
                                        st.balloons()  
                                        st.success("Knowledge transfer complete!")
                                        st.session_state.inquiry_started = False
                                        
                                        # Logging final metrics
                                        if 'metrics_tracker' in st.session_state and st.session_state.metrics_tracker:
                                            st.session_state.metrics_tracker.log_metrics()
                                        
                                        session_manager.log_interaction(
                                            "Knowledge transfer session completed successfully",
                                            "success"
                                        )
                                        
                                        # Displaying final metrics
                                        st.subheader("Final Session Metrics")
                                        if 'metrics_tracker' in st.session_state and st.session_state.metrics_tracker:
                                            st.session_state.metrics_tracker.display_metrics_dashboard()
                                        
                                        # Add option to start new session
                                        if st.button("Start New Session"):
                                            st.session_state.inquiry_started = False
                                            st.session_state.known_info = []
                                            st.session_state.last_question = None
                                            st.session_state.response_counter = 0
                                            st.rerun()
                                    else:
                                        # Updating session state for next iteration
                                        st.session_state.last_question = next_question
                                        st.session_state.known_info.append(user_response)
                                        
                                        session_manager.log_interaction(
                                            f"Response recorded and new question generated. {kb_update_status}",
                                            "info"
                                        )
                                        
                                        # Forcing streamlit to rerun to show the new question
                                        st.rerun()
                                
                            except Exception as e:
                                session_manager.handle_error(e, "processing_response")
                    else:
                        st.warning("Please provide a response before submitting.")
    
    with tab3:
        # Results and metrics tab
        st.header("Qualitative Metrics")

        # Section for question evaluation metrics
        if ('metrics' in st.session_state and 
            'evaluation_scores' in st.session_state.metrics and 
            st.session_state.metrics['evaluation_scores']):
            
            # Calculating average score
            scores = st.session_state.metrics['evaluation_scores']  
            avg_score = sum(scores) / len(scores) if scores else 0
            
            print(f"Scores: {scores}")
            print(f"Average: {avg_score}")
            
            st.metric("Average Question Quality", f"{avg_score:.1f}/10")
        else:
            # Default message when no evaluations
            st.metric("Average Question Quality", "N/A", "No evaluations yet")
            
            # Summary statistics in cards
            summary_col1, summary_col2, summary_col3 = st.columns(3)
            
            with summary_col1:
                questions_asked = len(st.session_state.known_info) if 'known_info' in st.session_state else 0
                st.metric("Questions Answered", questions_asked, f"{10 - questions_asked} remaining")
            
            with summary_col2:
                if 'metrics_tracker' in st.session_state and st.session_state.metrics_tracker:
                    metrics = st.session_state.metrics
                    if 'retrieval_times' in metrics and metrics['retrieval_times']:
                        avg_retrieval = sum(metrics['retrieval_times']) / len(metrics['retrieval_times'])
                        st.metric("Avg Retrieval Time", f"{avg_retrieval:.2f}s")
            
            with summary_col3:
                if 'metrics_tracker' in st.session_state and st.session_state.metrics_tracker:
                    metrics = st.session_state.metrics
                    if 'response_times' in metrics and metrics['response_times']:
                        avg_response = sum(metrics['response_times']) / len(metrics['response_times'])
                        st.metric("Avg Response Time", f"{avg_response:.2f}s")
            
            # Displaying detailed metrics
            st.subheader("Detailed Metrics")
            
            # Displaying metrics dashboard
            if 'metrics_tracker' in st.session_state and st.session_state.metrics_tracker:
                st.session_state.metrics_tracker.display_metrics_dashboard()
            
            # Viewing knowledge base section
            st.subheader("Knowledge Base")
            st.markdown("View the knowledge captured during this session and previous sessions.")
            
            # View knowledge base button
            if st.button("View Knowledge Base", key="view_kb_results"):
                view_knowledge_base()

    # === Tab 4: Evaluation Dashboard ===
    with tab4:
        st.header("Quantitative Metrics")

        if 'metrics_tracker' in st.session_state and st.session_state.metrics_tracker:
            st.session_state.metrics_tracker.display_metrics_dashboard()
        else:
            st.warning("No evaluation metrics available yet. Run a knowledge transfer session first.")

        if "all_eval_summaries" in st.session_state:
            st.subheader("📋 Model Leaderboard")
            df_leaderboard = pd.DataFrame(st.session_state.all_eval_summaries)
            st.dataframe(df_leaderboard)

        if "qa_pairs" in st.session_state and st.session_state.qa_pairs:
            # Converting qa_pairs to a dataframe for display
            question_logs = []
            for qa_pair in st.session_state.qa_pairs:
                # Formatting data for display
                entry = {
                    "Step": qa_pair.get("Step", 0),
                    "Model Used": qa_pair.get("Model Used", "Unknown"),
                    "Question": qa_pair.get("Question", ""),
                    "Context Relevance %": qa_pair.get("Context Relevance", 0) * 100,
                    "Diversity %": qa_pair.get("Diversity", 0) * 100,
                    "Readability": qa_pair.get("Readability", 0),
                    "Specificity": qa_pair.get("Specificity", 0) * 100 if "Specificity" in qa_pair else 0,
                    "Question Length": len(qa_pair.get("Question", "").split()),
                    "Final Score %": qa_pair.get("Final Score", 0) * 100, 
                    "User Rating": qa_pair.get("User Rating", 0)
                }
                question_logs.append(entry)
                
            # Storing for potential future use
            st.session_state.question_logs = question_logs
            
            df = pd.DataFrame(question_logs)
            
            st.subheader("📈 Session-Level Metric Averages")
            
            if not df.empty:
                # Calculating averages
                avg_context = df["Context Relevance %"].mean()
                avg_diversity = df["Diversity %"].mean()
                avg_readability = df["Readability"].mean()
                avg_specificity = df["Specificity"].mean()
                avg_q_len = df["Question Length"].mean()
                avg_final_score = df["Final Score %"].mean()
                avg_user_rating = df["User Rating"].mean() if "User Rating" in df.columns else 0

                # Displaying metrics in columns
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Context Relevance", f"{avg_context:.2f}%")
                    st.metric("Diversity", f"{avg_diversity:.2f}%")
                with col2:
                    st.metric("Readability", f"{avg_readability:.2f}")
                    st.metric("Specificity", f"{avg_specificity:.2f}%")
                with col3:
                    st.metric("Question Length", f"{avg_q_len:.1f} words")
                    st.metric("Final Score", f"{avg_final_score:.2f}%")
                    st.metric("User Rating", f"{avg_user_rating:.2f} (-1 to +1)")
            
            # User Feedback Section
            st.subheader("📊 User Feedback on Question Quality")
            
            # Count ratings
            ratings = [qa_pair.get("User Rating", 0) for qa_pair in st.session_state.qa_pairs 
                    if qa_pair.get("Has User Feedback", False)]
            
            if ratings:
                positive = sum(1 for r in ratings if r > 0)
                negative = sum(1 for r in ratings if r < 0)
                neutral = sum(1 for r in ratings if r == 0)
                total = len(ratings)

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Upvoted Questions", f"{positive} ({positive/total*100:.1f}%)" if total > 0 else "0 (0%)")
                with col2:
                    st.metric("Downvoted Questions", f"{negative} ({negative/total*100:.1f}%)" if total > 0 else "0 (0%)")
                with col3:
                    avg_rating = sum(ratings) / total if total > 0 else 0
                    st.metric("Average Rating", f"{avg_rating:.2f}")

                # Add visualization of user ratings per model
                st.subheader("User Ratings by Model")
                
                # Group data by model
                model_ratings = {}
                for qa_pair in st.session_state.qa_pairs:
                    if qa_pair.get("Has User Feedback", False):
                        model = qa_pair.get("Model Used", "Unknown")
                        if model not in model_ratings:
                            model_ratings[model] = []
                        model_ratings[model].append(qa_pair.get("User Rating", 0))
                
                # Calculate average ratings per model
                model_avg_ratings = {model: sum(ratings)/len(ratings) for model, ratings in model_ratings.items() if ratings}
                
                # Display as bar chart
                if model_avg_ratings:
                    df_model_ratings = pd.DataFrame({
                        "Model": list(model_avg_ratings.keys()),
                        "Average User Rating": list(model_avg_ratings.values())
                    })
                    
                    fig = px.bar(df_model_ratings, x="Model", y="Average User Rating", 
                                color="Average User Rating", 
                                color_continuous_scale=["red", "gray", "green"],
                                range_color=[-1, 1])
                    fig.update_layout(title="Average User Rating by Model")
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No model-specific ratings available yet.")
            else:
                st.info("No user feedback has been provided yet.")

            # Displaying per-question details
            st.subheader("🧾 Per-Question Evaluation Summary")
            st.dataframe(df, use_container_width=True)

            # Showing models used
            st.markdown("### 📌 Models used")
            unique_models = df["Model Used"].unique().tolist()
            for model in unique_models:
                st.markdown(f"- `{model}`")

            # Final Score Trend visualization
            st.markdown("### 📈 Final Score Trend")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["Step"],
                y=df["Final Score %"],
                mode='lines+markers+text',
                text=df["Model Used"],
                textposition="top center",
                name="Final Score"
            ))
            fig.update_layout(
                title="Final Score by Question Step",
                xaxis_title="Step",
                yaxis_title="Score (%)"
            )
            st.plotly_chart(fig, use_container_width=True)

            # Displaying bar charts for key metrics in a 2-column grid
            metric_cols = ["Context Relevance %", "Diversity %", "Readability", "Specificity", "Question Length", "User Rating"]
            col_pairs = zip(metric_cols[::2], metric_cols[1::2])  # Pairing up metrics for 2-column layout

            for left_col, right_col in col_pairs:
                # Skipping pairs with metrics not in the DataFrame
                if left_col not in df.columns and right_col not in df.columns:
                    continue

                col1, col2 = st.columns(2)

                # Left column chart
                with col1:
                    st.markdown(f"#### {left_col}")
                    fig1 = go.Figure()
                    fig1.add_trace(go.Bar(x=df["Step"], y=df[left_col], marker_color="teal"))
                    fig1.update_layout(
                        height=300,
                        margin=dict(l=20, r=20, t=30, b=20),
                        xaxis_title="Step",
                        yaxis_title=left_col
                    )
                    st.plotly_chart(fig1, use_container_width=True)

                # Right column chart (only if it exists)
                with col2:
                    if right_col in metric_cols:
                        st.markdown(f"#### {right_col}")
                        fig2 = go.Figure()
                        # Special coloring for User Rating
                        if right_col == "User Rating":
                            colors = []
                            for val in df[right_col]:
                                if val > 0:
                                    colors.append("green")
                                elif val < 0:
                                    colors.append("red")
                                else:
                                    colors.append("gray")
                            fig2.add_trace(go.Bar(x=df["Step"], y=df[right_col], marker_color=colors))
                        else:
                            fig2.add_trace(go.Bar(x=df["Step"], y=df[right_col], marker_color="slateblue"))
                            
                        fig2.update_layout(
                            height=300,
                            margin=dict(l=20, r=20, t=30, b=20),
                            xaxis_title="Step",
                            yaxis_title=right_col
                        )
                        st.plotly_chart(fig2, use_container_width=True)

            
            # Combined metrics chart
            st.subheader("🔄 Combined Metrics Visualization")
            
            # Creating a radar chart with all metrics
            if not df.empty:
                # Create normalized versions of metrics for consistent visualization
                metrics_to_include = ["Context Relevance %", "Diversity %"]

                # Normalizing readability to 0-100 scale
                df["Normalized Readability %"] = df["Readability"].apply(lambda x: min(x, 100))
                metrics_to_include.append("Normalized Readability %")

                # Final score should already be 0-100 
                metrics_to_include.append("Final Score %")
                
                if "User Rating" in df.columns:
                    # Scaling user rating from [-1,1] to [0,100] for radar chart
                    df["User Rating %"] = ((df["User Rating"] + 1) / 2) * 100
                    metrics_to_include.append("User Rating %")

                # Ensuring metrics are present in df
                metrics_to_include = [m for m in metrics_to_include if m in df.columns]
                
                if metrics_to_include:
                    # Calculating average by model
                    model_avg = df.groupby("Model Used")[metrics_to_include].mean().reset_index()
                    
                    # Creating radar chart
                    fig = go.Figure()
                    
                    for i, model in enumerate(model_avg["Model Used"]):
                        fig.add_trace(go.Scatterpolar(
                            r=model_avg.loc[i, metrics_to_include].values,
                            theta=metrics_to_include,
                            fill='toself',
                            name=model
                        ))
                    
                    fig.update_layout(
                        polar=dict(
                            radialaxis=dict(
                                visible=True,
                                range=[0, 100]
                            )),
                        showlegend=True,
                        title="Model Performance Comparison"
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)


            # Tag showing total answered and total models used
            st.markdown("---")
            st.markdown(f"✅ **Total Questions Answered**: `{len(df)}`")
            st.markdown(f"🔁 **Unique Models Used**: {', '.join([f'`{m}`' for m in unique_models])}")
            if st.button("🧹 Clear Evaluation Logs"):
                st.session_state.question_logs = []
                st.session_state.qa_pairs = []
                st.success("Cleared question evaluations")
                time.sleep(1)
                st.rerun()
        else:
            st.info("⚠️ No evaluation data available. Run a session first.")

    # Displaying interaction logs
    if 'agent_logs' in st.session_state and st.session_state.agent_logs:
        with st.expander("Session Logs", expanded=False):
            for log in st.session_state.agent_logs:
                st.markdown(log)
    
    # Displaying error logs if any
    if 'error_log' in st.session_state and st.session_state.error_log:
        with st.expander("Error Logs", expanded=False):
            for error in st.session_state.error_log:
                st.error(error)

# Getting secrets from Streamlit if available
def get_streamlit_secrets():
    try:
        return {
            "aws_access_key_id": st.secrets["aws"]["aws_access_key_id"],
            "aws_secret_access_key": st.secrets["aws"]["aws_secret_access_key"],
            "region_name": st.secrets["aws"]["region_name"],
            "anthropic_api_key": st.secrets["apis"]["anthropic_api_key"],
            "huggingface_api_key": st.secrets["apis"]["huggingface_api_key"],
            "together_api_key": st.secrets["apis"]["together_api_key"]
        }
    except Exception as e:
        # Not using st.error or st.warning here to avoid error
        logger.warning(f"Error accessing secrets: {e}")
        return None
    
# Function to process all documents in the raw bucket
def process_all_raw_documents(s3_client, dynamodb_resource, creds):
    with st.spinner("Processing all documents in bucket..."):
        domain_buckets = {
            "Data Science": ("knowledge-raw-team1", "knowledge-processed-team1"),
            "Software Engineering": ("se-knowledge-raw", "se-knowledge-processed"),
            "Electrical Engineering": ("ee-knowledge-raw", "ee-knowledge-processed")
        }
        
        # Get appropriate S3 buckets for the selected domain
        raw_bucket, processed_bucket = domain_buckets[st.session_state.selected_domain]
        
        # Initialize the document processor
        processor = DocumentProcessor(raw_bucket, processed_bucket, s3_client=s3_client)
        
        # Adding a progress bar for visual feedback
        progress_bar = st.progress(0)
        
        # Helper function for updating progress
        def update_progress(progress, status=""):
            progress_bar.progress(progress)
            # Only showing important milestone messages
            if progress > 0.7 and ("Loading vector store" in status or "Processing complete" in status):
                st.info(f"Status: {status}")
        
        # Process all documents
        try:
            chunks = processor.process_all_raw_documents(progress_callback=update_progress)
            st.success(f"✅ Successfully processed {len(chunks)} chunks from all documents")
            
            # After processing, ensure we have the vector store loaded
            vector_store = processor.get_vector_store()
            
            # Initialize the knowledge system with this vector store
            ANTHROPIC_API_KEY = creds.get("anthropic_api_key") if creds else os.getenv('ANTHROPIC_API_KEY')
            HUGGINGFACE_API_KEY = creds.get("huggingface_api_key") if creds else os.getenv('HUGGINGFACE_API_KEY')
            TOGETHER_API_KEY = creds.get("together_api_key") if creds else os.getenv('TOGETHER_API_KEY')
            
            st.session_state.knowledge_system = KnowledgeSystem(
                vector_store=vector_store,
                anthropic_api_key=ANTHROPIC_API_KEY,
                huggingface_api_key=HUGGINGFACE_API_KEY,
                together_api_key=TOGETHER_API_KEY,
                dynamodb_resource=dynamodb_resource
            )
            
            # Assign the selected model to the expert agent
            model_name = st.session_state.expert_model
            st.session_state.knowledge_system.expert.set_model(
                model_name=model_name,
                anthropic_api_key=ANTHROPIC_API_KEY,
                huggingface_api_key=HUGGINGFACE_API_KEY,
                together_api_key=TOGETHER_API_KEY
            )
            
            st.session_state.processing_complete = True
            return True
            
        except Exception as e:
            st.error(f"⚠️ Error processing documents: {str(e)}")
            return False

# ===== SESSION STATE MANAGEMENT ===== #
# Initializing all session state variables
def initialize_session_state(dynamodb_resource=None, cloudwatch_client=None):
    if 'metrics_tracker' not in st.session_state:
        st.session_state.metrics_tracker = StreamlitMetricsTracker(
            dynamodb_resource=dynamodb_resource, 
            cloudwatch_client=cloudwatch_client
        )

    # Initializing session state variables with defaults
    session_vars = {
        'uploaded_file': None,
        'knowledge_system': None,
        'inquiry_started': False,
        'processing_complete': False,
        'last_question': None,
        'known_info': [],
        'agent_logs': [],
        'error_log': [],
        'response_counter': 0,
        'selected_domain': "Data Science",
        'models_used_per_step': [],
        'pending_kb_updates': [],
        'embeddings_model': None,
        'session_stopped': False,
        'qa_pairs': [],
        'rag_question_embeddings': []
    }
    
    # Setting all variables that don't exist yet
    for var, default in session_vars.items():
        if var not in st.session_state:
            st.session_state[var] = default
            
    logger.info("Session state initialized")


# ===== DOCUMENT PROCESSING FUNCTIONS ===== #
# Caching results for 5 minutes
@st.cache_data(ttl=300)  
# Getting knowledge base items with caching for better performance
def get_knowledge_base_items(table_name, credentials=None):
    try:
        # Using credentials if provided, otherwise use environment variables
        if credentials:
            kb_table = boto3.resource(
                'dynamodb', 
                region_name='us-east-1',
                aws_access_key_id=credentials.get("aws_access_key_id"),
                aws_secret_access_key=credentials.get("aws_secret_access_key")
            ).Table(table_name)
        else:
            kb_table = boto3.resource(
                'dynamodb', 
                region_name='us-east-1',
                aws_access_key_id=os.getenv('AWS_ACCESS_KEY_ID'),
                aws_secret_access_key=os.getenv('AWS_SECRET_ACCESS_KEY')
            ).Table(table_name)
        
        response = kb_table.scan()
        return response.get('Items', [])
    except Exception as e:
        logger.error(f"Failed to access table {table_name}: {str(e)}")
        return []

# Displaying contents of the knowledge base
def view_knowledge_base():
    try:
        # Getting domain-specific table name based on the selected domain
        domain_tables = {
            "Data Science": "knowledge_base",
            "Software Engineering": "se_knowledge_base",
            "Electrical Engineering": "ee_knowledge_base"
        }

        # Defaulting to the base table if none found
        table_name = domain_tables.get(st.session_state.selected_domain, "knowledge_base")

        # Getting credentials
        creds = None
        try:
            creds = {
                "aws_access_key_id": st.secrets["aws"]["aws_access_key_id"],
                "aws_secret_access_key": st.secrets["aws"]["aws_secret_access_key"], 
                "region_name": os.getenv('AWS_REGION', 'us-east-1')
            }
        except:
            # Fallback to environment variables
            pass
        
        # Using cached function to get items
        with st.spinner("Loading knowledge base..."):
            items = get_knowledge_base_items(table_name, creds)
            
            # Trying default table if domain table is empty
            if not items and table_name != "knowledge_base":
                st.info(f"No entries found in {table_name}. Checking default knowledge base...")
                items = get_knowledge_base_items("knowledge_base", creds)
    
        if items:
            st.subheader(f"Knowledge Base: {st.session_state.selected_domain}")
            # Creating a df for easier viewing
            kb_data = []
            for item in items: 
                kb_data.append({
                    "ID": item.get('id','N/A'), 
                    "Question": item.get('question', item.get('content', 'N/A')), 
                    "Answer": item.get('answer', item.get('analysis', 'N/A')), 
                    "Timestamp": item.get('timestamp', 'N/A')
                })
            
            # Showing as a dataframe 
            kb_df = pd.DataFrame(kb_data) 
            st.dataframe(kb_df, use_container_width=True)

            # Also showing detailed view with expandable sections
            st.subheader("Detailed View")
            # Limiting number of items shown for better performance
            max_items = 50
            if len(items) > max_items:
                st.info(f"Showing {max_items} most recent entries out of {len(items)} total entries")
                items = sorted(items, key=lambda x: x.get('timestamp', ''), reverse=True)[:max_items]
                
            for item in items:
                question_text = item.get('question', item.get('content', 'N/A'))
                # Truncating long questions for the expander header
                display_text = question_text[:50] + "..." if len(question_text) > 50 else question_text
                
                with st.expander(f"{item.get('id', 'Entry')}: {display_text}", expanded=False):
                    st.markdown("#### Question:")
                    st.write(question_text)
                    
                    st.markdown("#### Answer:")
                    st.write(item.get('answer', item.get('analysis', 'N/A')))
                    
                    st.markdown("#### Metadata:")
                    st.write(f"Timestamp: {item.get('timestamp', 'N/A')}")
                    if 'metadata' in item:
                        st.json(item['metadata'])

        else:
            st.info(f"Knowledge base for {st.session_state.selected_domain} is empty")
    except Exception as e:
        st.error(f"Error accessing knowledge base: {str(e)}")
        # Showing detailed error for debugging
        st.exception(e)  

if __name__ == "__main__":
    main()
