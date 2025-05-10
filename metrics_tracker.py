import streamlit as st
import plotly.graph_objects as go
from datetime import datetime
import boto3
from decimal import Decimal
import logging
import uuid
import os
import textstat
from sentence_transformers import SentenceTransformer, util

logger = logging.getLogger(__name__)

# Loading sentence transformer model - doing this once at module level
try:
    embedder = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    logger.error(f"Error loading SentenceTransformer model: {str(e)}")
    embedder = None


# Evaluation

# Calculating semantic similarity between context and question
def compute_context_relevance(context, question):
    if not embedder:
        logger.warning("Embedder not available for context relevance computation")
        return 0.5
        
    try:
        embeddings = embedder.encode([context, question], convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(embeddings[0], embeddings[1]).item()
        return round(similarity, 4)
    except Exception as e:
        logger.error(f"Error computing context relevance: {str(e)}")
        return 0.0

# Calculating readability score for text
def compute_readability(question):
    try:
        return textstat.flesch_reading_ease(question)
    except Exception as e:
        logger.error(f"Error computing readability: {str(e)}")
        return 50.0  # Default middle value

# Calculating specificity ratio of unique words to total words
def compute_specificity(question):
    try:
        words = question.split()
        if not words:
            return 0.0
        return min(1.0, len(set(words)) / len(words))
    except Exception as e:
        logger.error(f"Error computing specificity: {str(e)}")
        return 0.0

# Counting words in question
def compute_question_length(question):
    return len(question.split())

# Calculating semantic diversity compared to previous questions
def compute_diversity(current_embedding, previous_embeddings):
    if not previous_embeddings:
        return 1.0
    try:
        similarities = [util.pytorch_cos_sim(current_embedding, prev).item() for prev in previous_embeddings]
        return 1 - max(similarities) if similarities else 1.0
    except Exception as e:
        logger.error(f"Error computing diversity: {str(e)}")
        return 1.0

# Detecting if question is semantically similar to previous ones
def is_semantically_duplicate(current_embedding, previous_embeddings, threshold=0.92):
    if not previous_embeddings:
        return False
    try:
        similarities = [util.pytorch_cos_sim(current_embedding, prev).item() for prev in previous_embeddings]
        return max(similarities) > threshold if similarities else False
    except Exception as e:
        logger.error(f"Error checking semantic duplicates: {str(e)}")
        return False
    
# Logging comprehensive metrics for a RAG-generated question
def log_rag_question_metrics(question, context, step, previous_embeddings=[], previous_questions=[]):
    try:
        if not embedder:
            logger.warning("Embedder not available for question metrics")
            return {
                "embedding": None,
                "context_relevance": 0.5,
                "diversity": 1.0,
                "readability": 50.0,
                "is_duplicate": False,
                "question_length": len(question.split()),
                "specificity": 0.5,
                "final_score": 0.5
            }
            
        current_embedding = embedder.encode(question, convert_to_tensor=True)

        # Computing all metrics
        context_relevance = compute_context_relevance(context, question)
        readability = compute_readability(question)
        question_length = compute_question_length(question)
        specificity = compute_specificity(question)
        diversity = compute_diversity(current_embedding, previous_embeddings)

        # Checking for duplicate questions (exact and semantic)
        is_exact_duplicate = question in previous_questions
        is_semantic_duplicate_result = is_semantically_duplicate(current_embedding, previous_embeddings)
        is_duplicate = is_exact_duplicate or is_semantic_duplicate_result

        # Normalizing readability to 0-1 scale
        normalized_readability = min(readability / 100.0, 1.0)

        # Calculating final weighted score
        final_score = (
            0.4 * context_relevance +
            0.3 * diversity +
            0.3 * normalized_readability
        )

        # Returning comprehensive metrics
        return {
            "embedding": current_embedding,
            "context_relevance": context_relevance,
            "diversity": diversity,
            "readability": normalized_readability,
            "is_duplicate": is_duplicate,
            "question_length": question_length,
            "specificity": specificity,
            "final_score": final_score
        }
    except Exception as e:
        logger.error(f"Error logging RAG question metrics: {str(e)}")
        return {
            "embedding": None,
            "context_relevance": 0,
            "diversity": 0,
            "readability": 0,
            "is_duplicate": False,
            "question_length": 0,
            "specificity": 0,
            "final_score": 0
        }

# Function to initialize and return a DynamoDB resource 
# Using AWS credentials from environment variables or Streamlit secrets
def get_dynamodb_resource():
    try:
        # If using Streamlit secrets (recommended)
        import streamlit as st
        if "aws" in st.secrets:
            return boto3.resource(
                "dynamodb",
                region_name=st.secrets["aws"]["region_name"],
                aws_access_key_id=st.secrets["aws"]["aws_access_key_id"],
                aws_secret_access_key=st.secrets["aws"]["aws_secret_access_key"]
            )

        # Fallback: Use environment variables
        return boto3.resource("dynamodb", region_name="us-east-1")
    
    except Exception as e:
        logger.error(f"❌ Failed to initialize DynamoDB resource: {str(e)}")
        return None

# StreamlitMetricsTracker: Tracks, logs, and visualizes knowledge transfer metrics

# This module handles:
    # 1. Tracking knowledge transfer performance metrics
    # 2. Persisting metrics to DynamoDB and CloudWatch
    # 3. Visualizing metrics in the Streamlit interface
    # 4. Comparing current metrics with historical performance

class StreamlitMetricsTracker:
    def __init__(self, dynamodb_resource=None, cloudwatch_client=None):
        try:
            self._initialize_session_metrics()
            
            # Using provided clients or create default ones
            self.dynamodb = dynamodb_resource or get_dynamodb_resource()
            self.cloudwatch = cloudwatch_client or boto3.client('cloudwatch', region_name='us-east-1')
            self.metrics_table = None  # Lazy initialization
            self.session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        except Exception as e:
                logger.error(f"❌ Error initializing MetricsTracker: {str(e)}")
                raise

    # Initializing metrics in session state
    def _initialize_session_metrics(self):
        try:
            if 'metrics' not in st.session_state:
                st.session_state.metrics = {
                'retrieval_times': [],
                'response_times': [],
                'context_utilization': [],
                'questions_asked': 0,
                'info_density': [],
                'response_specificity': [],
                'response_lengths': [],
                'evaluation_scores': [],
                'evaluations': [],
                # Question quality metrics
                'context_relevance': [],
                'readability': [],
                'question_specificity': [], 
                'question_length': [],
                'diversity': [],
                'duplicate_count': 0,
                'question_quality_scores': [],
                'evaluation_metrics': {
                    'questions': [],
                    'context_relevance': [],
                    'answer_quality': []
                }
            }
            logger.info("Session metrics initialized")
        except Exception as e:
            logger.error(f"Error initializing session metrics: {str(e)}")
            raise

    # Function to get or create DynamoDB table for metrics
    def get_or_create_metrics_table(self):
        if self.metrics_table:
            return self.metrics_table
        try:
            table_name = 'rag_evaluation_metrics'
            existing_tables = self.dynamodb.meta.client.list_tables()["TableNames"]
            if table_name not in existing_tables:
                self.metrics_table = self.dynamodb.create_table(
                    TableName=table_name,
                    KeySchema=[
                        {'AttributeName': 'session_id', 'KeyType': 'HASH'},
                        {'AttributeName': 'timestamp', 'KeyType': 'RANGE'}
                    ],
                    AttributeDefinitions=[
                        {'AttributeName': 'session_id', 'AttributeType': 'S'},
                        {'AttributeName': 'timestamp', 'AttributeType': 'S'}
                    ],
                    ProvisionedThroughput={
                        'ReadCapacityUnits': 5,
                        'WriteCapacityUnits': 5
                    }
                )
                self.metrics_table.wait_until_exists()
            else:
                self.metrics_table = self.dynamodb.Table(table_name)
            return self.metrics_table
        except Exception as e:
            logger.error(f"⚠️ DynamoDB table issue: {str(e)}")
            return None

    # Logging metrics to CloudWatch
    def log_to_cloudwatch(self, metrics_data):
        try:
            metric_items = [
                {
                    'MetricName': key, 
                    'Value': float(value), 
                    'Unit': 'Count' if key == 'total_questions' else 'Seconds',  
                    'Dimensions': [{'Name': 'SessionId', 'Value': self.session_id}]
                }
                for key, value in metrics_data['interaction_metrics'].items()
            ]
            self.cloudwatch.put_metric_data(Namespace='RAGSystem', MetricData=metric_items)
            logger.info("✅ Metrics logged to CloudWatch.")
        except Exception as e:
            logger.error(f"❌ Failed to log to CloudWatch: {str(e)}")

    # Logging metrics to DynamoDB and CloudWatch
    def log_metrics(self):
        # Getting metrics table, creating it if needed
        self.metrics_table = self.get_or_create_metrics_table()
        
        # Handling case where DynamoDB is unavailable
        if not self.metrics_table:
            logger.warning("DynamoDB unavailable. Metrics will not be persisted.")
            return False
            
        try:
            # Safety checks for metrics data
            retrieval_times = st.session_state.metrics.get('retrieval_times', [])
            response_times = st.session_state.metrics.get('response_times', [])
            
            # Preparing metrics item with safe calculations
            metrics_item = {
                'session_id': self.session_id,
                'timestamp': datetime.now().isoformat(),
                'interaction_metrics': {
                    'average_retrieval_time': Decimal(str(sum(retrieval_times) / max(len(retrieval_times), 1))),
                    'average_response_time': Decimal(str(sum(response_times) / max(len(response_times), 1))),
                    'total_questions': Decimal(str(st.session_state.metrics['questions_asked']))
                }
            }
            
            # Addng additional content metrics if available
            if 'info_density' in st.session_state.metrics and st.session_state.metrics['info_density']:
                density_values = st.session_state.metrics['info_density']
                metrics_item['interaction_metrics']['average_info_density'] = Decimal(str(sum(density_values) / len(density_values)))
                
            if 'context_utilization' in st.session_state.metrics and st.session_state.metrics['context_utilization']:
                util_values = st.session_state.metrics['context_utilization']
                metrics_item['interaction_metrics']['average_context_utilization'] = Decimal(str(sum(util_values) / len(util_values)))
            
            # Adding question quality metrics if available
            if 'context_relevance' in st.session_state.metrics and st.session_state.metrics['context_relevance']:
                relevance_values = st.session_state.metrics['context_relevance']
                metrics_item['interaction_metrics']['average_context_relevance'] = Decimal(str(sum(relevance_values) / len(relevance_values)))
            
            if 'readability' in st.session_state.metrics and st.session_state.metrics['readability']:
                readability_values = st.session_state.metrics['readability']
                metrics_item['interaction_metrics']['average_readability'] = Decimal(str(sum(readability_values) / len(readability_values)))
                
            if 'diversity' in st.session_state.metrics and st.session_state.metrics['diversity']:
                diversity_values = st.session_state.metrics['diversity']
                metrics_item['interaction_metrics']['average_diversity'] = Decimal(str(sum(diversity_values) / len(diversity_values)))
            
            if 'question_quality_scores' in st.session_state.metrics and st.session_state.metrics['question_quality_scores']:
                quality_scores = st.session_state.metrics['question_quality_scores']
                metrics_item['interaction_metrics']['average_question_quality'] = Decimal(str(sum(quality_scores) / len(quality_scores)))
                
            # Include duplicate question count
            if 'duplicate_count' in st.session_state.metrics:
                metrics_item['interaction_metrics']['duplicate_questions'] = Decimal(str(st.session_state.metrics['duplicate_count']))

            # Writing to DynamoDB
            response = self.metrics_table.put_item(Item=metrics_item)
            logger.info(f"Metrics logged to DynamoDB for session {self.session_id}")
            
            # Also logging to CloudWatch if available
            self.log_to_cloudwatch(metrics_item)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to log metrics: {str(e)}")
            return False

    # Updating metrics with new interaction data
    def update_metrics(self, question, response, retrieval_time, response_time, context_data=None, evaluation_data=None, user_rating=None):
        # Initializing metrics if not present
        if 'metrics' not in st.session_state:
            self._initialize_session_metrics()
        try:
            # Updating basic metrics
            st.session_state.metrics['retrieval_times'].append(retrieval_time)
            st.session_state.metrics['response_times'].append(response_time)
            st.session_state.metrics['questions_asked'] += 1

            # Syncing with known_info if available
            if 'known_info' in st.session_state:
                st.session_state.metrics['questions_asked'] = len(st.session_state.known_info)
            
            # Calculating information density (response length vs information content)
            if response:
                response_words = response.lower().split()
                unique_words = set(response_words)
                info_density = len(unique_words) / len(response_words) if response_words else 0
            else:
                info_density = 0
            
            # Calculating response specificity (how specific vs generic the response is)
                # Specificity calculation: 
                # 1. Remove common English words that don't carry domain-specific meaning
                # 2. Calculate ratio of specific words to total unique words
                # Higher values indicate more specific, technical responses
            common_words = {'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but', 'for', 'with'}
            if response:
                unique_words = set(response.lower().split())
                specific_words = len(unique_words - common_words)
                total_unique_words = len(unique_words)
                specificity = specific_words / total_unique_words if total_unique_words > 0 else 0
            else:
                specificity = 0

            # Storing calculated metrics
            st.session_state.metrics['info_density'].append(info_density)
            st.session_state.metrics['response_specificity'].append(specificity)
            st.session_state.metrics['response_lengths'].append(len(response.split()))
            
            # Calculating context utilization if context data available
            if context_data and 'relevant_chunks' in context_data and response:
                # Get context words
                context_text = ' '.join(context_data['relevant_chunks'])
                context_words = set(context_text.lower().split())
                
                # Getting response words
                response_words = set(response.lower().split())
                
                # Calculating overlap
                if context_words:
                    utilization = len(context_words.intersection(response_words)) / len(context_words)
                    st.session_state.metrics['context_utilization'].append(utilization)
            
            # Add evaluation metrics if available
            if evaluation_data is None:
                print("No evaluation data received")
                evaluation_data = {"score": 0}

            # Extract and store the score
            print(f"Raw evaluation data: {evaluation_data}")
            try:
                if isinstance(evaluation_data, dict) and "score" in evaluation_data:
                    score_value = evaluation_data["score"]
                    print(f"Successfully extracted score value: {score_value}")
                else:
                    print(f"Invalid evaluation data format: {evaluation_data}")
                    score_value = 0
                    
            except Exception as e:
                print(f"Error accessing score: {e}")
                score_value = 0

            # Store the score in the metrics
            if 'evaluation_scores' not in st.session_state.metrics:
                st.session_state.metrics['evaluation_scores'] = []

            st.session_state.metrics['evaluation_scores'].append(score_value)
            print(f"Added score: {score_value}")
            print(f"Current scores: {st.session_state.metrics['evaluation_scores']}")

            # Store minimal data for average calculation
            if 'evaluations' not in st.session_state.metrics:
                st.session_state.metrics['evaluations'] = []

            st.session_state.metrics['evaluations'].append({
                'score': score_value
            })
            
            # Calculating question quality metrics
            if context_data:
                context_text = ' '.join(context_data.get('relevant_chunks', []))
                
                # Calculating question quality metrics
                question_metrics = log_rag_question_metrics(
                    question, 
                    context_text, 
                    st.session_state.metrics['questions_asked'],
                    st.session_state.get('previous_question_embeddings', []),
                    st.session_state.get('previous_questions', [])
                )
                
                # Storing individual metrics
                st.session_state.metrics['context_relevance'].append(question_metrics['context_relevance'])
                st.session_state.metrics['readability'].append(question_metrics['readability'])
                st.session_state.metrics['question_specificity'].append(question_metrics['specificity'])
                st.session_state.metrics['question_length'].append(question_metrics['question_length'])
                st.session_state.metrics['diversity'].append(question_metrics['diversity'])
                st.session_state.metrics['question_quality_scores'].append(question_metrics['final_score'])
                
                # Tracking duplicate questions
                if question_metrics['is_duplicate']:
                    st.session_state.metrics['duplicate_count'] += 1
                
                # Storing embedding and question for future comparison
                if question_metrics['embedding'] is not None:
                    if 'previous_question_embeddings' not in st.session_state:
                        st.session_state.previous_question_embeddings = []
                    st.session_state.previous_question_embeddings.append(question_metrics['embedding'])
                
                if 'previous_questions' not in st.session_state:
                    st.session_state.previous_questions = []
                st.session_state.previous_questions.append(question)

            # Adding user feedback metrics if available
            if user_rating is not None:
                if 'user_ratings' not in st.session_state.metrics:
                    st.session_state.metrics['user_ratings'] = []
                
                st.session_state.metrics['user_ratings'].append(user_rating)
                
                # Calculate percentage of positively rated questions
                positive_ratings = sum(1 for rating in st.session_state.metrics['user_ratings'] if rating > 0)
                total_ratings = len(st.session_state.metrics['user_ratings'])
                
                if total_ratings > 0:
                    st.session_state.metrics['positive_rating_percentage'] = positive_ratings / total_ratings
                else:
                    st.session_state.metrics['positive_rating_percentage'] = 0
            
            # Logging metrics to persistent storage
            self.log_metrics()
            
            logger.info(f"Metrics updated for question: '{question[:30]}...'")
            return True
            
        except Exception as e:
            logger.error(f"Error updating metrics: {str(e)}")
            return False
           
    # Displaying metrics dashboard in Streamlit
    # Dashboard layout:
        # - Top row: Basic metrics (questions, retrieval time, response time)
        # - Middle row: Quality metrics (density, specificity, utilization)
        # - Bottom: Trend visualization showing response length over time
    def display_metrics_dashboard(self):
        try:
            if 'metrics' not in st.session_state:
                self._initialize_session_metrics()
            st.subheader("Knowledge Transfer Metrics")
            
            # Basic metrics row
            col1, col2, col3 = st.columns(3)
            
            # Questions asked
            with col1:
                st.metric(
                    "Questions Asked", 
                    st.session_state.metrics['questions_asked'], 
                    f"{10 - st.session_state.metrics['questions_asked']} remaining"
                )
            
            # Retrieval time
            with col2:
                retrieval_times = st.session_state.metrics['retrieval_times']
                if retrieval_times:
                    avg_retrieval = sum(retrieval_times) / len(retrieval_times)
                    st.metric("Avg Retrieval Time (s)", f"{avg_retrieval:.2f}")
                else:
                    st.metric("Avg Retrieval Time (s)", "N/A")
            
            # Response time
            with col3:
                response_times = st.session_state.metrics['response_times']
                if response_times:
                    avg_response = sum(response_times) / len(response_times)
                    st.metric("Avg Response Time (s)", f"{avg_response:.2f}")
                else:
                    st.metric("Avg Response Time (s)", "N/A")
            
            
            # Response quality metrics row
            if 'info_density' in st.session_state.metrics and st.session_state.metrics['info_density']:
                col1, col2 = st.columns(2)
                
                # Information density
                with col1:
                    density_values = st.session_state.metrics['info_density']
                    avg_density = sum(density_values) / len(density_values)
                    st.metric("Avg Information Density", f"{avg_density:.1%}")
                
                # Response specificity
                with col2:
                    if 'response_specificity' in st.session_state.metrics and st.session_state.metrics['response_specificity']:
                        specificity_values = st.session_state.metrics['response_specificity']
                        avg_specificity = sum(specificity_values) / len(specificity_values)
                        st.metric("Response Specificity", f"{avg_specificity:.1%}")
            
            # Context utilization
            utilization_values = st.session_state.metrics.get('context_utilization', [])
            if utilization_values:
                avg_utilization = sum(utilization_values) / len(utilization_values)
                st.metric("Context Utilization", f"{avg_utilization:.1%}")
            
            # Question quality metrics
            if ('context_relevance' in st.session_state.metrics and 
                st.session_state.metrics['context_relevance'] and 
                'readability' in st.session_state.metrics and 
                'diversity' in st.session_state.metrics):
                
                st.subheader("Question Quality Metrics")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    relevance_values = st.session_state.metrics['context_relevance']
                    avg_relevance = sum(relevance_values) / len(relevance_values)
                    st.metric("Context Relevance", f"{avg_relevance:.2f}")
                
                with col2:
                    readability_values = st.session_state.metrics['readability']
                    avg_readability = sum(readability_values) / len(readability_values)
                    st.metric("Readability", f"{avg_readability:.1f}")
                
                with col3:
                    diversity_values = st.session_state.metrics['diversity']
                    avg_diversity = sum(diversity_values) / len(diversity_values)
                    st.metric("Question Diversity", f"{avg_diversity:.2f}")
                
                # Second row of question metrics
                col1, col2 = st.columns(2)
                
                with col1:
                    if 'question_specificity' in st.session_state.metrics and st.session_state.metrics['question_specificity']:
                        q_specificity_values = st.session_state.metrics['question_specificity']
                        avg_q_specificity = sum(q_specificity_values) / len(q_specificity_values)
                        st.metric("Question Specificity", f"{avg_q_specificity:.2f}")
                
                with col2:
                    if 'duplicate_count' in st.session_state.metrics:
                        duplicate_count = st.session_state.metrics['duplicate_count']
                        questions_asked = st.session_state.metrics['questions_asked']
                        if questions_asked > 0:
                            duplicate_pct = (duplicate_count / questions_asked) * 100
                            st.metric("Duplicate Questions", f"{duplicate_count} ({duplicate_pct:.1f}%)")
                        else:
                            st.metric("Duplicate Questions", "0 (0.0%)")
                            
            # Overall question quality score
            if 'question_quality_scores' in st.session_state.metrics and st.session_state.metrics['question_quality_scores']:
                quality_scores = st.session_state.metrics['question_quality_scores']
                avg_quality = sum(quality_scores) / len(quality_scores)
                st.metric("Overall Question Quality", f"{avg_quality:.2f}/1.0")
                
            # Response length trend visualization
            if 'response_lengths' in st.session_state.metrics and st.session_state.metrics['response_lengths']:
                # Create unique key for chart to prevent re-rendering issues
                chart_key = f"length_trend_{self.session_id}_{uuid.uuid4().hex[:6]}"
                
                # Creating figure
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=st.session_state.metrics['response_lengths'],
                    mode='lines+markers',
                    name='Response Length (words)'
                ))
                
                # Setting layout
                fig.update_layout(
                    title='Response Length Trend',
                    xaxis_title='Question Number',
                    yaxis_title='Word Count',
                    height=300
                )
                
                # Displaying chart
                st.plotly_chart(fig, use_container_width=True, key=chart_key)
                
            # Question quality trend visualization
            if ('context_relevance' in st.session_state.metrics and 
                st.session_state.metrics['context_relevance'] and
                'readability' in st.session_state.metrics and
                'diversity' in st.session_state.metrics):
                
                # Creating unique key for chart
                chart_key = f"quality_trend_{self.session_id}_{uuid.uuid4().hex[:6]}"
                
                # Creating figure with multiple metrics
                fig = go.Figure()
                
                # Adding traces for each metric
                fig.add_trace(go.Scatter(
                    y=st.session_state.metrics['context_relevance'],
                    mode='lines+markers',
                    name='Context Relevance'
                ))
                
                fig.add_trace(go.Scatter(
                    y=[r/100 for r in st.session_state.metrics['readability']],  # Scale readability to 0-1
                    mode='lines+markers',
                    name='Readability (scaled)'
                ))
                
                fig.add_trace(go.Scatter(
                    y=st.session_state.metrics['diversity'],
                    mode='lines+markers',
                    name='Diversity'
                ))
                
                # Setting layout
                fig.update_layout(
                    title='Question Quality Trends',
                    xaxis_title='Question Number',
                    yaxis_title='Score (0-1)',
                    height=400,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                # Displaying chart
                st.plotly_chart(fig, use_container_width=True, key=chart_key)
                
        except Exception as e:
            logger.error(f"Error displaying metrics dashboard: {str(e)}")
            st.error("Error displaying metrics dashboard")


    # Returning current session metrics for external use
    def get_session_metrics(self):
        try:
            return {
                'session_id': self.session_id,
                'metrics': st.session_state.metrics.copy() if hasattr(st.session_state, 'metrics') else {}
            }
        except Exception as e:
            logger.error(f"Error getting session metrics: {str(e)}")
            return {'session_id': self.session_id, 'metrics': {}, 'error': str(e)}

    # Loading historical metrics from DynamoDB for comparison
    # Retrieves past metrics from DynamoDB to provide performance benchmarks
    # This helps track system improvement over time and identify anomalies
    def load_historical_metrics(self):
        # Ensure metrics table is initialized
        if not self.metrics_table:
            self.metrics_table = self.get_or_create_metrics_table()
            
        if not self.metrics_table:
            st.warning("Cannot access historical metrics: DynamoDB unavailable")
            return None
                
        try:
            # Scan table for historical records
            response = self.metrics_table.scan()
            historical_metrics = response.get('Items', [])
            
            if not historical_metrics:
                st.info("No historical metrics available yet")
                return []
                
            st.subheader("Historical Performance")
            
            # Initialize counters
            total_retrieval = 0
            total_response = 0
            total_context_relevance = 0
            total_readability = 0
            total_diversity = 0
            total_duplicate_pct = 0
            valid_entries = 0
            valid_question_entries = 0
            
            # Process historical data
            for item in historical_metrics:
                try:
                    # Skip invalid entries
                    if 'interaction_metrics' not in item or not isinstance(item['interaction_metrics'], dict):
                        continue
                        
                    # Extract metrics with safe defaults
                    metrics = item['interaction_metrics']
                    retrieval_time = float(metrics.get('average_retrieval_time', 0))
                    response_time = float(metrics.get('average_response_time', 0))
                    
                    # Accumulate basic totals
                    total_retrieval += retrieval_time
                    total_response += response_time
                    valid_entries += 1
                    
                    # Extract question quality metrics if available
                    if 'average_context_relevance' in metrics:
                        total_context_relevance += float(metrics['average_context_relevance'])
                        valid_question_entries += 1
                        
                    if 'average_readability' in metrics:
                        total_readability += float(metrics['average_readability'])
                        
                    if 'average_diversity' in metrics:
                        total_diversity += float(metrics['average_diversity'])
                        
                    if 'duplicate_questions' in metrics and 'total_questions' in metrics:
                        duplicate_count = float(metrics['duplicate_questions'])
                        total_questions = float(metrics['total_questions'])
                        if total_questions > 0:
                            total_duplicate_pct += (duplicate_count / total_questions) * 100
                    
                except (KeyError, TypeError, ValueError) as e:
                    logger.warning(f"Skipping malformed historical entry: {str(e)}")
                    continue
            
            # Calculate and display averages
            if valid_entries > 0:
                avg_retrieval = total_retrieval / valid_entries
                avg_response = total_response / valid_entries
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Historical Avg Retrieval Time", f"{avg_retrieval:.2f}s")
                with col2:
                    st.metric("Historical Avg Response Time", f"{avg_response:.2f}s")
                
                # Show additional information
                st.caption(f"Based on {valid_entries} historical sessions")
                
                # Display question quality metrics if available
                if valid_question_entries > 0:
                    st.subheader("Historical Question Quality")
                    
                    avg_context_relevance = total_context_relevance / valid_question_entries
                    avg_readability = total_readability / valid_question_entries
                    avg_diversity = total_diversity / valid_question_entries
                    avg_duplicate_pct = total_duplicate_pct / valid_question_entries
                    
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Avg Context Relevance", f"{avg_context_relevance:.2f}")
                    
                    with col2:
                        st.metric("Avg Readability", f"{avg_readability:.1f}")
                    
                    with col3:
                        st.metric("Avg Diversity", f"{avg_diversity:.2f}")
                    
                    if avg_duplicate_pct > 0:
                        st.metric("Avg Duplicate Questions", f"{avg_duplicate_pct:.1f}%")
                
                # Option to view raw data
                if st.checkbox("Show Raw Historical Data"):
                    # Format data for display
                    display_data = [{
                        'session_id': item.get('session_id', 'unknown'),
                        'timestamp': item.get('timestamp', 'unknown'),
                        'retrieval_time': float(item.get('interaction_metrics', {}).get('average_retrieval_time', 0)),
                        'response_time': float(item.get('interaction_metrics', {}).get('average_response_time', 0)),
                        'questions': int(float(item.get('interaction_metrics', {}).get('total_questions', 0)))
                    } for item in historical_metrics]
                    
                    st.dataframe(display_data)
            else:
                st.info("No valid historical metrics found")
                
            return historical_metrics
                
        except Exception as e:
            logger.error(f"Error loading historical metrics: {str(e)}")
            st.warning("Unable to load historical metrics")
            return None
