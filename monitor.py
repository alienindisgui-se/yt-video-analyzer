import json
import os
import requests
import time
import random
import sys
import logging
import hashlib
import subprocess
from datetime import datetime
from dotenv import load_dotenv
from groq import Groq
from playwright.sync_api import sync_playwright, TimeoutError
import yt_dlp
import google.genai as genai

# Load environment variables from .env file
load_dotenv()

# Validate required environment variables
WEBHOOK = os.getenv("DISCORD_WEBHOOK")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GROQ_API_KEY:
    logging.error("GROQ_API_KEY environment variable is required but not set")
    sys.exit(1)

if not WEBHOOK:
    logging.warning("DISCORD_WEBHOOK environment variable not set - Discord notifications will be disabled")

if not GEMINI_API_KEY:
    logging.warning("GEMINI_API_KEY environment variable not set - Gemini fallback will be unavailable")

# Setup logging and encoding
sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def setup_run_logging():
    """Setup file-based logging for individual runs with sequential numbering"""
    # Find existing run logs to determine next number
    run_logs = [f for f in os.listdir('.') if f.startswith('runlog') and f.endswith('.log')]
    run_numbers = []
    
    for log_file in run_logs:
        try:
            # Extract number from filename like "runlog1.log", "runlog2.log", etc.
            number = int(log_file.replace('runlog', '').replace('.log', ''))
            run_numbers.append(number)
        except ValueError:
            continue
    
    next_number = max(run_numbers) + 1 if run_numbers else 1
    run_log_file = f"runlog{next_number}.log"
    
    # Create file handler for run logging
    file_handler = logging.FileHandler(run_log_file, mode='w', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Add file handler to root logger
    logging.getLogger().addHandler(file_handler)
    
    logging.info(f"=== RUN {next_number} STARTED at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    return run_log_file

# Configuration
CHANNELS = ['ANJO1', 'CarlFredrikAlexanderRask']
# CHANNELS = ['CarlFredrikAlexanderRask', 'ANJO1', 'MotVikten', 'Skuldis']
STATE_FILE = "comment_state.json"
ANALYSIS_STATS_FILE = "analysis_stats.json"  # Track video analysis counts per channel
CONFIG_FILE = "config.json"


def estimate_tokens(text, content_type="general"):
    """
    Estimate token count based on content type and character count.
    Using conservative ratios to avoid 413 errors.
    """
    if not text:
        return 0
    
    char_count = len(text)
    
    # Conservative token-to-character ratios (updated based on actual API behavior)
    ratios = {
        "general": 3.0,      # More conservative: 1 token ≈ 3 chars
        "transcript": 2.8,    # Transcripts: dense, 1 token ≈ 2.8 chars  
        "comments": 3.2,      # Comments: informal, 1 token ≈ 3.2 chars
        "prompt": 2.9         # Instructions/prompts: 1 token ≈ 2.9 chars
    }
    
    ratio = ratios.get(content_type, ratios["general"])
    return int(char_count / ratio)

def validate_and_trim_content(content, max_tokens, content_type="general", priority="beginning"):
    """
    Validate content size and trim if necessary to stay within token limits.
    
    Args:
        content: Text content to validate
        max_tokens: Maximum allowed tokens
        content_type: Type of content for accurate token estimation
        priority: Trimming strategy - "beginning", "balanced", or "end"
    
    Returns:
        Tuple of (trimmed_content, estimated_tokens, was_trimmed)
    """
    if not content:
        return "", 0, False
    
    estimated_tokens = estimate_tokens(content, content_type)
    
    if estimated_tokens <= max_tokens:
        return content, estimated_tokens, False
    
    logging.warning(f"Content exceeds token limit: {estimated_tokens} > {max_tokens} (type: {content_type})")
    
    # Calculate target character count based on token ratio - use 75% for safety margin
    ratios = {"general": 3.0, "transcript": 2.8, "comments": 3.2, "prompt": 2.9}
    ratio = ratios.get(content_type, 3.0)
    target_chars = int(max_tokens * ratio * 0.75)  # 75% to be very safe
    
    if priority == "beginning":
        # Keep beginning - most important info usually at start
        trimmed = content[:target_chars] + "... [truncated]"
    elif priority == "end":
        # Keep end - for cases where conclusion matters most
        trimmed = "... [truncated] " + content[-target_chars:]
    else:  # balanced
        # Keep beginning and end, sample middle
        if target_chars < 200:
            # Too short for balanced approach, just keep beginning
            trimmed = content[:target_chars] + "... [truncated]"
        else:
            third = target_chars // 3
            beginning = content[:third]
            ending = content[-third:]
            trimmed = beginning + "... [middle truncated] ..." + ending
    
    new_tokens = estimate_tokens(trimmed, content_type)
    logging.info(f"Trimmed content from {estimated_tokens} to {new_tokens} tokens ({len(content)} -> {len(trimmed)} chars) - using 75% safety margin")
    
    return trimmed, new_tokens, True

def log_payload_size(content, model_name, max_tokens, content_type="general"):
    """Log payload size information for debugging"""
    estimated_tokens = estimate_tokens(content, content_type)
    percentage = (estimated_tokens / max_tokens * 100) if max_tokens > 0 else 0
    
    logging.info(f"Payload size for {model_name}: {estimated_tokens}/{max_tokens} tokens ({percentage:.1f}%) - {content_type}")
    
    if estimated_tokens > max_tokens:
        logging.warning(f"Payload EXCEEDS limit by {estimated_tokens - max_tokens} tokens!")
    
    return estimated_tokens

class ModelManager:
    """Manages AI model selection, fallbacks, and rate limiting"""
    
    def __init__(self, config):
        self.config = config
        self.ai_config = config.get('ai_models', {})
        self.primary_model = self.ai_config.get('primary', 'llama-3.3-70b-versatile')
        self.fallback_models = self.ai_config.get('fallback', [])
        self.available_models = self.ai_config.get('models', {})
        self.client = Groq(api_key=GROQ_API_KEY)
        
        # Conservative model token limits based on actual API behavior and safety margins
        self.model_limits = {
            "llama-3.3-70b-versatile": 10000,  # Reduced from 12k for safety
            "qwen/qwen3-32b": 4500,        # Reduced from 6k due to 413 errors
            "llama-3.1-8b-instant": 4500,  # Reduced from 6k due to 413 errors
            "gemini-2.0-flash": 7000,        # Reduced from 8k for safety
        }
        
        # Initialize Gemini client if API key is available
        self.gemini_client = None
        if GEMINI_API_KEY:
            try:
                self.gemini_client = genai.Client(api_key=GEMINI_API_KEY)
                # logging.info("Gemini client initialized successfully")
            except Exception as e:
                logging.warning(f"Failed to initialize Gemini client: {e}")
        else:
            logging.warning("No GEMINI_API_KEY found - Gemini fallback unavailable")
        
    def get_model_for_request(self):
        """Select the best model for current request"""
        # Try primary model first
        if self.primary_model in self.available_models:
            return self.primary_model
        
        # Fallback to available models
        for model in self.fallback_models:
            if model in self.available_models:
                return model
                
        # Ultimate fallback to legacy model
        logging.warning("No configured models available, falling back to llama-3.3-70b-versatile")
        return "llama-3.3-70b-versatile"
    
    def try_gemini_fallback(self, prompt):
        """Try to generate response using Gemini API as final fallback"""
        if not self.gemini_client:
            logging.warning("Gemini client not available for fallback")
            return None, None
            
        try:
            logging.info("Attempting Gemini API fallback...")
            response = self.gemini_client.models.generate_content(
                model="gemini-2.0-flash",
                contents=prompt
            )
            summary = response.text.strip()
            logging.info("Successfully generated summary using Gemini API")
            return summary, "gemini-2.0-flash"
        except Exception as e:
            logging.error(f"Gemini API fallback failed: {e}")
            return None, None
    
    def try_model_with_fallback(self, prompt, max_retries=3):
        """Try to generate summary with fallback models, prioritizing higher token limits"""
        # Sort models by estimated token capacity (higher limits first)
        models_by_capacity = [
            ("llama-3.3-70b-versatile", 10000),  # Highest capacity
            ("gemini-2.0-flash", 7000),  # Gemini second highest
            ("qwen/qwen3-32b", 4500),
            ("llama-3.1-8b-instant", 4500),
        ]
        
        # Add primary and fallback models in capacity order
        models_to_try = []
        seen = set()
        
        for model_name, capacity in models_by_capacity:
            if model_name in self.available_models and model_name not in seen:
                models_to_try.append(model_name)
                seen.add(model_name)
        
        # Add any remaining available models
        for model in self.fallback_models:
            if model not in seen and model in self.available_models:
                models_to_try.append(model)
                seen.add(model)
        
        # Ultimate fallback
        if "llama-3.3-70b-versatile" not in models_to_try and "llama-3.3-70b-versatile" in self.available_models:
            models_to_try.append("llama-3.3-70b-versatile")
        
        groq_failures = 0
        
        for attempt, model_name in enumerate(models_to_try):
            try:
                # Get token limit for this model
                max_tokens = self.model_limits.get(model_name, 4500)
                
                # Progressive content reduction if we've had 413 errors before
                current_prompt = prompt
                if attempt > 0:
                    # Reduce content size progressively for retry attempts
                    reduction_factor = 0.6 - (attempt * 0.1)  # 60%, 50%, 40%
                    if reduction_factor < 0.3:
                        reduction_factor = 0.3
                    
                    # Apply reduction to prompt content
                    lines = current_prompt.split('\n')
                    if len(lines) > 4:
                        # Keep first few lines and reduce middle content
                        keep_lines = max(2, len(lines) // 3)
                        reduced_lines = lines[:keep_lines] + ["\n... [content reduced for retry]...\n"] + lines[-2:]
                        current_prompt = '\n'.join(reduced_lines)
                        logging.info(f"Retry {attempt+1}: Reduced prompt content by {int((1-reduction_factor)*100)}% due to previous 413 errors")
                
                # Log payload size before validation
                log_payload_size(current_prompt, model_name, max_tokens, "prompt")
                
                # Validate and trim content if necessary
                validated_prompt, final_tokens, was_trimmed = validate_and_trim_content(
                    current_prompt, max_tokens, "prompt", "beginning"
                )
                
                if was_trimmed:
                    logging.info(f"Content was trimmed for {model_name} to fit token limits")
                
                logging.info(f"Attempting to use model: {model_name} (attempt {attempt + 1}/{len(models_to_try)})")
                
                # Choose API client based on model provider
                if model_name.startswith("gemini"):
                    # Use Gemini API
                    response = self.gemini_client.models.generate_content(
                        model=model_name,
                        contents=validated_prompt
                    )
                    summary = response.text.strip()
                else:
                    # Use Groq API
                    chat_completion = self.client.chat.completions.create(
                        messages=[{"role": "user", "content": validated_prompt}],
                        model=model_name,
                    )
                    summary = chat_completion.choices[0].message.content.strip()
                
                logging.info(f"Successfully generated summary using model: {model_name} ({final_tokens} tokens)")
                return summary, model_name
                
            except Exception as e:
                if not model_name.startswith("gemini"):
                    groq_failures += 1
                    
                error_msg = str(e).lower()
                if "413" in error_msg or "payload too large" in error_msg:
                    logging.error(f"Model {model_name} failed: 413 Payload Too Large. This indicates our token estimation still needs adjustment.")
                    # Continue to next model with reduced content
                elif "429" in error_msg or "rate limit" in error_msg:
                    logging.warning(f"Model {model_name} failed: Rate limit reached - {str(e)[:100]}...")
                else:
                    logging.warning(f"Model {model_name} failed: {str(e)}")
                
                # After 2 Groq failures, try Gemini fallback
                if groq_failures >= 2 and self.gemini_client and "gemini-2.0-flash" not in models_to_try[:attempt+1]:
                    logging.info("2 Groq models failed, attempting Gemini fallback...")
                    gemini_summary, gemini_model = self.try_gemini_fallback(prompt)
                    if gemini_summary:
                        return gemini_summary, gemini_model
                    else:
                        logging.warning("Gemini fallback also failed, continuing with remaining Groq models...")
                
                if attempt < len(models_to_try) - 1:
                    logging.info(f"Trying next model...")
                    time.sleep(2)  # Longer delay for rate limits
                else:
                    logging.error("All models failed to generate summary")
                    # Final Gemini attempt if not already tried
                    if groq_failures < 2 and self.gemini_client:
                        logging.info("Final attempt: trying Gemini fallback...")
                        gemini_summary, gemini_model = self.try_gemini_fallback(prompt)
                        if gemini_summary:
                            return gemini_summary, gemini_model
                    raise e
        
        return None, None

# Load configuration
def load_config():
    try:
        with open(CONFIG_FILE, "r", encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logging.warning(f"Config file {CONFIG_FILE} not found. Using default English settings.")
        return {
            "language": "en",
            "prompts": {
                "en": {
                    "ai_summary": "Analyze the following comments from the YouTube video '{title}'. Write a concise summary (around 3-4 sentences) of the general viewer sentiment, the main topics being discussed, and the overall opinions of the audience.\n\nComments:\n{comments_string}\n\nPlease provide a summary that is neutral and does not take a stance on the topic.",
                    "discord_title": "🤖 AI Comment Analysis {date}",
                    "channel_field": "Channel",
                    "video_field": "Video"
                }
            },
            "settings": {
                "max_comments": 150,
                "date_format": "%Y-%m-%d"
            },
            "ai_models": {
                "primary": "llama-3.3-70b-versatile",
                "fallback": [],
                "models": {}
            }
        }

CONFIG = load_config()
CURRENT_LANGUAGE = CONFIG["language"]
PROMPTS = CONFIG["prompts"][CURRENT_LANGUAGE]
SETTINGS = CONFIG["settings"]

# Initialize Model Manager
model_manager = ModelManager(CONFIG)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
]

def clean_ai_output(text):
    """Clean AI output to remove unwanted formatting and think blocks"""
    if not text:
        return text
    
    # Remove code blocks completely
    import re
    # Remove all content between triple backticks
    text = re.sub(r'```.*?```', '', text, flags=re.DOTALL)
    # Remove all content between single backticks
    text = re.sub(r'`.*?`', '', text, flags=re.DOTALL)
    
    # Remove malformed "" blocks that appear in AI output
    text = re.sub(r'""', '', text)
    text = re.sub(r'""\s*$', '', text, flags=re.MULTILINE)
    
    # Remove qwen think blocks specifically
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
    text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL)
    
    # Remove other common AI reasoning patterns
    text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL)
    text = re.sub(r'<analysis>.*?</analysis>', '', text, flags=re.DOTALL)
    
    # Remove lines that look like AI reasoning/thinking indicators
    lines = text.split('\n')
    filtered_lines = []
    
    for line in lines:
        # Skip lines that are just AI reasoning indicators
        if re.match(r'^\s*<.*?>\s*$', line.strip()) or \
           re.match(r'^\s*</.*?>\s*$', line.strip()):
            continue
        
        # Skip lines that start with thinking indicators
        if any(line.strip().startswith(prefix) for prefix in [
            'Let me think', 'Let me analyze', 'Okay, let me', 
            'First, I need to', 'I should consider', 'Let me start',
            'Alright, let me', 'Let me work through', 'I need to',
            'Let me consider', 'Let me break down', 'Let me examine'
        ]):
            continue
        
        filtered_lines.append(line)
    
    # Rejoin and clean up
    text = '\n'.join(filtered_lines)
    
    # Remove unwanted AI prefixes and signatures - preserve original structure
    # First, remove prefixes at the start of the text
    text = re.sub(r'^🤖\s*AI-?Analys:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^🤖\s*AI\s*Analysis:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^AI\s*Analysis:\s*', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^AI-?Analys:\s*', '', text, flags=re.IGNORECASE)
    
    # Remove lines that are just AI signatures
    lines = text.split('\n')
    filtered_lines = []
    
    for line in lines:
        # Skip lines that are just AI signatures or generated by messages
        if re.match(r'^🤖.*$', line.strip()) or \
           re.match(r'^AI\s*generated.*$', line.strip(), re.IGNORECASE) or \
           re.match(r'^Generated\s*by\s*AI.*$', line.strip(), re.IGNORECASE):
            continue
        
        # Skip lines that look like AI reasoning/thinking
        if any(keyword in line.lower() for keyword in [
            'okay, let\'s start', 'alright, let\'s tackle', 'let me think', 
            'first, i need to', 'next, i should', 'finally', 'in conclusion',
            'step by step', 'breaking this down', 'analysis:', 'thinking:',
            '🤖', 'ai analysis', 'ai-analys', 'ai generated', 'generated by ai'
        ]):
            continue
        
        filtered_lines.append(line)
    
    # Rejoin exactly as they were (preserve original spacing)
    cleaned_text = '\n'.join(filtered_lines)
    
    # Clean up leading/trailing whitespace only
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text

def summarize_transcript(transcript_text, title):
    """Summarize long transcript to key points for analysis (saves tokens)"""
    try:
        # Get appropriate token limit for transcript summarization
        max_tokens = min(model_manager.model_limits.values())  # Use most conservative limit
        # Reserve 70% for transcript content, 30% for instructions
        transcript_tokens_available = int(max_tokens * 0.7)
        
        # Validate and trim transcript content
        validated_transcript, transcript_tokens, was_trimmed = validate_and_trim_content(
            transcript_text, transcript_tokens_available, "transcript", "balanced"
        )
        
        if was_trimmed:
            logging.info(f"Transcript trimmed from {len(transcript_text)} to {len(validated_transcript)} chars for summarization")
        
        # Create summarization prompt
        prompt = f"""Summarize this YouTube video transcript titled '{title}' into a very concise summary (max 300 words):

Key elements to capture:
- Main topic and central argument
- 2-3 key claims or points
- Overall tone/sentiment
- Any controversial elements

Keep it extremely brief!

Transcript excerpt:
{validated_transcript}"""
        
        # Log prompt size
        final_tokens = estimate_tokens(prompt, "prompt")
        logging.info(f"Transcript summarization prompt size: {final_tokens}/{max_tokens} tokens ({final_tokens/max_tokens*100:.1f}%)")
        
        # Use lightweight model for summarization with fallback
        summary, used_model = model_manager.try_model_with_fallback(prompt)
        if summary:
            logging.info(f"Transcript summarized from {len(transcript_text)} to {len(summary)} characters using model '{used_model}'")
            return summary
        else:
            logging.warning("Transcript summarization failed with all models. Using aggressive truncation.")
            # More aggressive fallback: extract key sentences
            sentences = transcript_text.split('.')[:10]  # First 10 sentences
            return '. '.join(sentences) + "... (severely truncated for token limits)"
        
    except Exception as e:
        logging.warning(f"Transcript summarization failed: {e}. Using aggressive truncation.")
        # More aggressive fallback: extract key sentences
        sentences = transcript_text.split('.')[:10]  # First 10 sentences
        return '. '.join(sentences) + "... (severely truncated for token limits)"

def get_gradient_color(like_ratio):
    """Generate a color gradient from red to green based on like ratio (0-100)"""
    # Ensure like_ratio is within 0-100 range
    ratio = max(0, min(100, like_ratio))
    
    # Red (255, 0, 0) at 0% to Green (0, 255, 0) at 100%
    red = int(255 * (1 - ratio / 100))
    green = int(255 * (ratio / 100))
    blue = 0
    
    # Convert to hex color string
    hex_color = f"{red:02x}{green:02x}{blue:02x}"
    return int(hex_color, 16)

def get_thumbnail_url(v_id):
    """Get YouTube thumbnail URL for a video"""
    return f"https://img.youtube.com/vi/{v_id}/maxresdefault.jpg"

def get_video_stats(v_id):
    """Fetch likes and dislikes using returnyoutubedislike.com API"""
    try:
        api_url = f"https://returnyoutubedislikeapi.com/votes?videoId={v_id}"
        response = requests.get(api_url, timeout=10000)
        if response.status_code == 200:
            data = response.json()
            likes = data.get('likes', 0)
            dislikes = data.get('dislikes', 0)
            views = data.get('viewCount', 0)
            
            # Calculate engagement ratio
            total_reactions = likes + dislikes
            like_ratio = (likes / total_reactions * 100) if total_reactions > 0 else 0
            
            logging.info(f"Fetched video stats - Likes: {likes:,}, Dislikes: {dislikes:,}, Like Ratio: {like_ratio:.1f}%")
            return {
                'likes': likes,
                'dislikes': dislikes,
                'views': views,
                'like_ratio': like_ratio
            }
    except Exception as e:
        logging.warning(f"Failed to fetch video stats from API: {e}")
    
    return {'likes': 0, 'dislikes': 0, 'views': 0, 'like_ratio': 0}

def fetch_latest_videos(channels):
    """Fetch latest videos from channels and return only new ones not analyzed"""
    latest_videos = []
    video_to_channel = {}
    user_agent = random.choice(USER_AGENTS)
    
    # Load existing analyzed videos
    analysis_stats = load_analysis_stats()
    analyzed_videos = set()
    for channel_data in analysis_stats.values():
        if isinstance(channel_data, dict) and "videos" in channel_data:
            for video in channel_data["videos"]:
                if isinstance(video, dict) and "video_id" in video:
                    analyzed_videos.add(video["video_id"])
    
    # logging.info(f"Found {len(analyzed_videos)} previously analyzed videos")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=user_agent,
            viewport={'width': 1920, 'height': 1080},
            locale='en-US',
            extra_http_headers={'Accept-Language': 'en-US,en;q=0.9'}
        )
        page = context.new_page()
        
        for channel in channels:
            try:
                url = f"https://www.youtube.com/@{channel}/videos"
                page.goto(url, timeout=60000)
                page.wait_for_load_state('networkidle')
                
                # Handle YouTube consent page
                if "Before you continue to YouTube" in page.title():
                    try:
                        accept_button = page.locator('button').filter(has_text="Accept all").first
                        if accept_button.count() > 0:
                            accept_button.click()
                            page.wait_for_timeout(2000)
                            page.wait_for_load_state('networkidle')
                    except Exception as e:
                        logging.error(f"Failed to accept consent: {e}")
                
                # Scroll to load videos
                page.evaluate("window.scrollBy(0, 1000)")
                page.wait_for_timeout(5000)
                
                # Find the first video link
                video_locator = page.locator('ytd-rich-item-renderer a[href*="/watch?v="]').first
                if video_locator.count() > 0:
                    href = video_locator.get_attribute('href')
                    if href and 'v=' in href:
                        v_id = href.split('v=')[1].split('&')[0]
                        
                        # Check if video is already analyzed
                        if v_id not in analyzed_videos:
                            latest_videos.append(v_id)
                            video_to_channel[v_id] = channel
                            # logging.info(f"Found new video {v_id} for channel {channel}")
                        else:
                            logging.info(f"Video {v_id} already analyzed, skipping")
                else:
                    # Fallback
                    video_locator = page.locator('ytd-rich-grid-media a[href*="/watch?v="]').first
                    if video_locator.count() > 0:
                        href = video_locator.get_attribute('href')
                        if href and 'v=' in href:
                            v_id = href.split('v=')[1].split('&')[0]
                            
                            # Check if video is already analyzed
                            if v_id not in analyzed_videos:
                                latest_videos.append(v_id)
                                video_to_channel[v_id] = channel
                                logging.info(f"Found new video {v_id} for channel {channel} using fallback")
                            else:
                                logging.info(f"Video {v_id} already analyzed, skipping")
            except Exception as e:
                logging.error(f"Failed to fetch latest video for {channel}: {e}")
        
        browser.close()
    
    return latest_videos, video_to_channel

def generate_persistent_id(author, text):
    raw_str = f"{author}|{text}"
    return hashlib.md5(raw_str.encode('utf-8')).hexdigest()

def load_analysis_stats():
    """Load analysis statistics from JSON file with simple retry mechanism"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with open(ANALYSIS_STATS_FILE, "r", encoding='utf-8') as f:
                data = json.load(f)
                # Ensure new structure exists
                for channel in CHANNELS:
                    if channel not in data:
                        data[channel] = {"videos": []}
                    elif "videos" not in data[channel]:
                        # Migrate old structure to new
                        old_data = data[channel]
                        data[channel] = {"videos": []}
                        if "videos_analyzed" in old_data:
                            # Create a placeholder video entry for migration
                            data[channel]["videos"].append({
                                "video_id": old_data.get("last_video_id", ""),
                                "title": "Migrated Analysis",
                                "analysis_date": old_data.get("last_checked", time.time()),
                                "analyses": {
                                    "comment_review": {
                                        "input_prompt": old_data.get("last_prompt", ""),
                                        "output": "[Migrated output]",
                                        "model": old_data.get("last_model", ""),
                                        "timestamp": old_data.get("last_checked", time.time())
                                    }
                                }
                            })
                return data
        except FileNotFoundError:
            # Return empty structure with all channels
            return {channel: {"videos": []} for channel in CHANNELS}
        except (json.JSONDecodeError, IOError) as e:
            if attempt == max_retries - 1:
                logging.warning(f"Failed to load analysis stats after {max_retries} attempts: {e}")
                return {channel: {"videos": []} for channel in CHANNELS}
            logging.debug(f"Retry {attempt + 1} for loading analysis stats: {e}")
            time.sleep(0.1 * (attempt + 1))
    return {channel: {"videos": []} for channel in CHANNELS}

def save_analysis_stats(stats):
    """Save analysis statistics to JSON file with retry mechanism"""
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with open(ANALYSIS_STATS_FILE, "w", encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            return
        except (IOError, OSError) as e:
            if attempt == max_retries - 1:
                logging.error(f"Failed to save analysis stats after {max_retries} attempts: {e}")
                return
            logging.debug(f"Retry {attempt + 1} for saving analysis stats: {e}")
            time.sleep(0.1 * (attempt + 1))

def find_or_create_video(stats, channel_name, video_id, title):
    """Find existing video or create new entry in the videos array"""
    channel_data = stats.get(channel_name, {"videos": []})
    
    # Find existing video
    for video in channel_data["videos"]:
        if video["video_id"] == video_id:
            return video
    
    # Create new video entry
    new_video = {
        "video_id": video_id,
        "title": title,
        "analysis_date": time.time(),
        "analyses": {}
    }
    channel_data["videos"].append(new_video)
    return new_video

def add_analysis_to_video(video, analysis_type, input_prompt, output, model):
    """Add analysis entry to a video's analyses"""
    if "analyses" not in video:
        video["analyses"] = {}
    
    video["analyses"][analysis_type] = {
        "input_prompt": input_prompt,
        "output": output,
        "model": model,
        "timestamp": time.time()
    }

def get_yt_data(v_id, deep_scrape=False, video_to_channel=None):
    user_agent = random.choice(USER_AGENTS)
    # logging.info(f"Selected user agent for scraping comments: {user_agent}")
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=user_agent,
            viewport={'width': 1920, 'height': 1080},
            locale='en-US',
            extra_http_headers={'Accept-Language': 'en-US,en;q=0.9'}
        )
        page = context.new_page()
        
        try:
            page.goto(f"https://www.youtube.com/watch?v={v_id}", timeout=60000)
            page.wait_for_load_state('networkidle')
            page.evaluate("window.scrollBy(0, 800)")
            
            # Check for private/members-only content indicators in page content
            page_content = page.content()
            page_title = page.title()
            
            # Check for private video indicators
            private_indicators = [
                "This video is private",
                "Private video", 
                "Video unavailable",
                "This video is not available"
            ]
            
            # Check for members-only content indicators (more specific patterns)
            members_indicators = [
                "Join this channel to get access to members-only content like this video",
                "Become a member to watch this video",
                "This video is only available to members"
            ]
            
            # Check page title for private indicators (more strict)
            page_title_lower = page_title.lower()
            if any(indicator.lower() == page_title_lower or indicator.lower() in page_title_lower and "private" in page_title_lower for indicator in private_indicators):
                logging.warning(f"Private video detected in page title for {v_id}: {page_title}")
                return "MEMBERS_ONLY", None, None, None, None, None
                
            # Check page content for members-only indicators (more specific matching)
            page_content_lower = page_content.lower()
            if any(indicator.lower() in page_content_lower for indicator in members_indicators):
                logging.warning(f"Members-only content detected in page content for {v_id}")
                return "MEMBERS_ONLY", None, None, None, None, None
            
            try:
                page.wait_for_selector('ytd-comments#comments', state='attached', timeout=15000)
                page.wait_for_timeout(3000)
            except TimeoutError:
                logging.warning("Comments section did not attach in time. Video might have comments disabled.")
                return None, None, None, None, None, None
            
            title_elem = page.locator('h1.ytd-watch-metadata yt-formatted-string')
            title = title_elem.text_content().strip() if title_elem.count() > 0 else 'Unknown'
            
            # Fetch video stats
            video_stats = get_video_stats(v_id)
            
            # Early detection for members-only/private content
            # Check for indicators: zero engagement + title available suggests restricted content
            if (video_stats['likes'] == 0 and 
                video_stats['dislikes'] == 0 and 
                video_stats['like_ratio'] == 0.0 and
                title != 'Unknown'):
                logging.warning(f"Early detection: Video {v_id} shows signs of members-only/private content (zero engagement but title available)")
                return "MEMBERS_ONLY", None, None, None, None, None
            
            ui_count = 0
            # [Count extraction logic preserved for brevity]
            count_locators = [('#count .yt-core-attributed-string', 'yt-core-attributed-string'), ('h2#count yt-formatted-string', 'yt-formatted-string'), ('yt-formatted-string.count-text', 'count-text')]
            for selector, desc in count_locators:
                try:
                    loc = page.locator(selector)
                    loc.wait_for(state='visible', timeout=10000)
                    digits = ''.join(filter(str.isdigit, loc.first.text_content().strip()))
                    if digits:
                        ui_count = int(digits)
                        break
                except TimeoutError:
                    pass

            # LANGUAGE-INDEPENDENT SORT TO "NEWEST FIRST"
            if deep_scrape:
                try:
                    page.evaluate("""() => {
                        const btn = document.querySelector('ytd-comments-header-renderer #sort-menu');
                        if (btn) btn.click();
                    }""")
                    page.wait_for_timeout(1000)
                    
                    page.evaluate("""() => {
                        const items = document.querySelectorAll('ytd-menu-service-item-renderer');
                        if (items.length > 1) items[1].click();
                    }""")
                    page.wait_for_timeout(3000)
                    # logging.info("Sorted comments to 'Newest first' (language-independent).")
                except Exception as e:
                    logging.warning(f"Failed to sort comments: {e}. Proceeding with default sort.")

            comments = {}
            if deep_scrape:
                logging.info(f"Starting deep scrape for '{title}'. UI reports ~{ui_count} comments.")
                
                # Phase 1: Load all top-level threads by scrolling
                # logging.info("Loading all top-level comment threads...")
                last_thread_count = 0
                no_change = 0
                while True:
                    thread_nodes = page.locator('ytd-comment-thread-renderer')
                    current_thread = thread_nodes.count()
                    # logging.info(f"Current top-level threads: {current_thread}")
                    if current_thread == last_thread_count:
                        no_change += 1
                        if no_change >= 3:
                            # logging.info(f"Loaded {current_thread} top-level threads.")
                            break
                    else:
                        no_change = 0
                        last_thread_count = current_thread
                    page.evaluate("document.scrollingElement.scrollTop = document.scrollingElement.scrollHeight")
                    # logging.info("Scrolled to bottom for thread loading.")
                    page.wait_for_timeout(5000)

                # Phase 2: Expand replies using JavaScript to bypass actionability checks
                # logging.info("Expanding nested replies via JavaScript injection...")
                max_iterations = 3  # Run a few times in case clicking reveals nested "show more" buttons
                
                for i in range(max_iterations):
                    try:
                        # Inject JS to find all reply buttons and click them directly in the DOM
                        clicks_dispatched = page.evaluate("""() => {
                            const buttons = Array.from(document.querySelectorAll('ytd-button-renderer#more-replies button'));
                            let count = 0;
                            for (let btn of buttons) {
                                // Basic check to ensure the button is actually rendered in the DOM
                                if (btn.offsetParent !== null) { 
                                    btn.click();
                                    count++;
                                }
                            }
                            return count;
                        }""")
                        
                        # logging.info(f"Iteration {i+1}: Dispatched {clicks_dispatched} clicks via JS.")
                        
                        if clicks_dispatched == 0:
                            # logging.info("No more expansion buttons found. Expansion complete.")
                            break
                            
                        # Wait a moment for the requested replies to render in the DOM
                        page.wait_for_timeout(4000)
                        
                        # Scroll down to ensure we trigger any lazy-loaded elements
                        page.evaluate("document.scrollingElement.scrollTop = document.scrollingElement.scrollHeight")
                        page.wait_for_timeout(2000)
                        
                    except Exception as e:
                        logging.warning(f"Iteration {i+1} JS click failed: {str(e).splitlines()[0]}")
                        break

                # logging.info("Proceeding to final extraction.")

                # Phase 3: Final scroll to ensure all loaded
                # logging.info("Performing final scroll to load any remaining content...")
                page.evaluate("document.scrollingElement.scrollTop = document.scrollingElement.scrollHeight")
                page.wait_for_timeout(5000)

                # Extract all loaded comments
                # logging.info("Extracting all loaded comments...")
                author_locs = page.locator('#author-text')
                text_locs = page.locator('#content-text')
                extracted_count = text_locs.count()
                # logging.info(f"Found {extracted_count} comment texts for extraction (including replies).")
                for i in range(extracted_count):
                    try:
                        author = author_locs.nth(i).text_content().strip()
                        text = text_locs.nth(i).text_content().strip()
                        c_id = generate_persistent_id(author, text)
                        if c_id in comments:
                            logging.debug(f"Duplicate comment detected at index {i}: {text[:50]}...")
                        else:
                            comments[c_id] = {
                                'a': author,
                                't': text,
                                'ts_posted': int(time.time()),  # Approximate posted time, since scraping doesn't provide exact
                                'created_at': int(time.time()),
                                'lastSeen': int(time.time()),
                                'deleted': False,
                                'notFoundCounter': 0
                            }
                    except Exception as e:
                        logging.warning(f"Failed to extract comment {i}: {e}")
                
                logging.info(f"Extracted {len(comments)} unique comments after deduplication.")
                if ui_count == 0 and len(comments) > 0:
                    ui_count = len(comments)
                    
            # Get transcript and analysis
            transcript_text, ai_analysis = get_transcript_and_analysis(v_id, title)
            
            # Also run comprehensive analysis
            # comprehensive_analysis = analyze_video_comprehensive(v_id, title, comments, video_stats, transcript_text, video_to_channel)
            
            return ui_count, comments, title, video_stats, transcript_text, ai_analysis
            
        except Exception as e:
            logging.error(f"Scrape failed for {v_id}: {e}")
            return None, None, None, None, None, None
        finally:
            browser.close()

def get_transcript_and_analysis(v_id, title):
    """Downloads video audio and transcribes it using Groq's Whisper API."""
    import tempfile
    import os
    
    full_text = None
    
    # Use secure temporary file with random name
    with tempfile.NamedTemporaryFile(suffix='.m4a', delete=False) as temp_file:
        audio_filepath = temp_file.name
    
    # We grab the worst audio quality to ensure blazing fast downloads 
    # and to stay under Groq's 25MB audio file limit.
    ydl_opts = {
        'format': 'worstaudio/worst',  # More flexible - any audio format, lowest quality
        'outtmpl': audio_filepath,
        'quiet': True,
        'no_warnings': True,
        'extract_audio': True,  # Ensure we get audio
    }

    try:
        logging.info(f"Ripping audio for {v_id}...")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([f"https://www.youtube.com/watch?v={v_id}"])
            
        logging.info(f"Audio downloaded successfully for {v_id}. File size: {os.path.getsize(audio_filepath):,} bytes")
        
        # Dynamically trim audio based on duration to respect Groq's ASPH rate limits (7200s/hour)
        # Trim to first 45 minutes for safety, as longer videos would exceed limits.
        try:
            # Use yt-dlp to download only first 45 minutes
            trimmed_filepath = audio_filepath.replace('.m4a', '_trimmed.m4a')
            trim_opts = ydl_opts.copy()
            trim_opts['outtmpl'] = trimmed_filepath
            trim_opts['download_sections'] = '*0:2700'  # Correct parameter for time ranges
            
            with yt_dlp.YoutubeDL(trim_opts) as trim_ydl:
                trim_ydl.download([f"https://www.youtube.com/watch?v={v_id}"])
            
            # Replace original with trimmed version
            os.remove(audio_filepath)
            audio_filepath = trimmed_filepath
            logging.info(f"Audio trimmed successfully using yt-dlp. New file size: {os.path.getsize(audio_filepath):,} bytes")
        except Exception as ydl_trim_e:
            logging.warning(f"Audio trimming failed: {ydl_trim_e}. Proceeding with full audio - may hit rate limits.")
        
        logging.info(f"Sending audio to Groq Whisper API for transcription...")
        
        # Using the Groq client you already initialized in ModelManager
        try:
            with open(audio_filepath, "rb") as file:
                transcription = model_manager.client.audio.transcriptions.create(
                    file=(os.path.basename(audio_filepath), file.read()),
                    model="whisper-large-v3",  # Updated to correct model name
                    response_format="text",
                    language="sv" # Forcing Swedish context for channels like Anjo/Rask. Remove this line for auto-detect.
                )
                
            full_text = transcription
            logging.info(f"Audio successfully transcribed! ({len(full_text)} characters)")
            
        except Exception as transcribe_e:
            logging.error(f"Transcription failed: {transcribe_e}")
            # Return empty transcription but continue with analysis
            full_text = ""
            logging.info("Continuing without transcription due to API error")
        
        # Save transcription to JSON file along with video ID
        current_timestamp = time.time()
        transcription_data = {
            "video_id": v_id,
            "title": title,
            "transcription": full_text,
            "character_count": len(full_text),
            "timestamp": current_timestamp
        }
        
        # Save to separate transcription file
        transcription_file = "transcriptions.json"
        try:
            if os.path.exists(transcription_file):
                with open(transcription_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            else:
                existing_data = {}
            
            existing_data[v_id] = transcription_data
            
            with open(transcription_file, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)
            
            logging.info(f"Transcription saved to {transcription_file} for video {v_id}")
            
        except Exception as save_e:
            logging.warning(f"Failed to save transcription to JSON: {save_e}")
        
    except Exception as e:
        error_msg = str(e).lower()
        # Check for members-only/private content restrictions
        if ("join this channel to get access to members-only content" in error_msg or 
            "members-only content" in error_msg or 
            "private video" in error_msg or 
            "this video is private" in error_msg):
            logging.warning(f"Skipping members-only/private video {v_id}: {e}")
            return "MEMBERS_ONLY", None
        
        if "429" in error_msg or "rate limit" in error_msg or "rate_limit_exceeded" in error_msg:
            logging.error(f"Groq rate limit reached during transcription for {v_id}: {e}")
            logging.error("Aborting entire script run to prevent further violations.")
            sys.exit(1)  # Immediate abortion as requested
        else:
            logging.error(f"Audio ripping or transcription failed for {v_id}: {e}")
            return None, None
        
    finally:
        # Always clean up the evidence (temp file)
        if os.path.exists(audio_filepath):
            os.remove(audio_filepath)

    if not full_text:
        return None, None

    # Summarize long transcripts to fit token limits
    summarized_transcript = summarize_transcript(full_text, title)

    # AI analysis via Groq (using summarized transcript to fit token limits)
    try:
        prompt = (
            f"Analyze the following YouTube video transcript summary titled '{title}'. "
            f"Provide a concise professional summary (max 400 words) covering:\n"
            f"- Main topics and key arguments\n"
            f"- Overall sentiment and tone\n"
            f"- Any controversial, political, or sensitive elements that might provoke comment deletions\n"
            f"- Suggested tags or themes for monitoring purposes\n\n"
            f"Transcript Summary:\n{summarized_transcript}"
        )
        
        # Store prompt and analysis in new structure
        current_timestamp = time.time()
        
        analysis_stats = load_analysis_stats()
        # Use a generic channel name since we don't have channel context here
        channel_name = "transcript_analysis"
        
        # Find or create video entry
        video_entry = find_or_create_video(analysis_stats, channel_name, v_id, title)
        
        # Add transcription analysis
        add_analysis_to_video(video_entry, "transcription", prompt, full_text if full_text else "No transcription available", "whisper-large-v3")
        
        # Save updated stats
        save_analysis_stats(analysis_stats)
        
        logging.info(f"Saved transcript analysis to JSON for video {v_id}")
        
        analysis, used_model = model_manager.try_model_with_fallback(prompt)
        
        if analysis:
            cleaned_analysis = clean_ai_output(analysis)
            logging.info(f"AI analysis completed for {v_id} using model '{used_model}'.")
            
            # Update the video entry with actual analysis output
            add_analysis_to_video(video_entry, "transcription", prompt, cleaned_analysis, used_model)
            save_analysis_stats(analysis_stats)
            
            return full_text, cleaned_analysis
        else:
            logging.warning(f"AI analysis failed for {v_id} with all available models.")
            return full_text, None
            
    except Exception as e:
        logging.error(f"Groq analysis failed for {v_id}: {e}")
        return full_text, None

# def analyze_video_comprehensive(v_id, title, comments_dict, video_stats, transcript_text=None, video_to_channel=None):
#     """Unified analysis combining transcript content and comment sentiment in one AI call."""
#     if not GROQ_API_KEY:
#         logging.warning("No GROQ_API_KEY found. Skipping comprehensive AI analysis.")
#         return None

#     logging.info(f"Generating comprehensive AI analysis for video: {title}")
    
#     # Get the model with highest capacity for this analysis
#     max_tokens = max(model_manager.model_limits.values())
#     logging.info(f"Using max token limit for comprehensive analysis: {max_tokens}")
    
#     # Prepare transcript section with size validation
#     transcript_section = ""
#     if transcript_text:
#         # Reserve tokens for other sections (stats, comments, instructions)
#         transcript_tokens_available = int(max_tokens * 0.4)  # 40% for transcript
#         validated_transcript, transcript_tokens, was_trimmed = validate_and_trim_content(
#             transcript_text, transcript_tokens_available, "transcript", "balanced"
#         )
#         transcript_section = f"**VIDEO TRANSCRIPT:**\n{validated_transcript}"
#         if was_trimmed:
#             logging.info(f"Transcript trimmed from {len(transcript_text)} to {len(validated_transcript)} chars")
#     else:
#         transcript_section = "**VIDEO TRANSCRIPT:**\n(No transcript available - captions disabled)"
    
#     # Prepare video stats section
#     stats_section = f"**VIDEO ENGAGEMENT:**\n{video_stats['likes']:,} likes, {video_stats['dislikes']:,} dislikes ({video_stats['like_ratio']:.1f}% like ratio)"
    
#     # Prepare comments section with size validation
#     comments_section = ""
#     if comments_dict:
#         texts = [c['t'] for c in comments_dict.values() if not c.get('deleted', False)]
#         if texts:
#             max_comments = SETTINGS.get("max_comments", 150)
#             sample_texts = texts[:max_comments]
#             comments_string = "\n".join([f"- {t}" for t in sample_texts])
            
#             # Reserve tokens for comments (30% of total)
#             comments_tokens_available = int(max_tokens * 0.3)
#             validated_comments, comment_tokens, was_trimmed = validate_and_trim_content(
#                 comments_string, comments_tokens_available, "comments", "balanced"
#             )
#             comments_section = f"**AUDIENCE COMMENTS:**\n{validated_comments}"
#             if was_trimmed:
#                 logging.info(f"Comments trimmed from {len(comments_string)} to {len(validated_comments)} chars")
#         else:
#             comments_section = "**AUDIENCE COMMENTS:**\n(No comments available)"
#     else:
#         comments_section = "**AUDIENCE COMMENTS:**\n(No comments available)"
    
#     # Create comprehensive prompt
#     prompt = f"""Analyze this YouTube video comprehensively:

# {transcript_section}

# {stats_section}

# {comments_section}

# **ANALYSIS REQUEST:**
# Provide a unified analysis covering:

# **CONTENT SUMMARY:** Key topics, arguments, and narrative structure from the video transcript

# **AUDIENCE REACTION:** Comment sentiment, dominant themes, and public reception patterns  

# **CONTENT-RECEPTION ALIGNMENT:** How well the video content aligns with audience reactions

# **LEGAL ASSESSMENT:** Defamation risk assessment (always start with 'Sannolikheten är [hög/låg] för förtal')

# **MONITORING INSIGHTS:** Suggested tags and themes for monitoring purposes

# **FORMATTING:** 
# - Structure clearly with bold headings
# - Be concise but comprehensive
# - Only mention like ratio if below 90%
# - Include a blank line (\n\n) between each analysis section
# - Provide direct analysis without thinking blocks or meta-commentary
# """
    
    # # Log final prompt size
    # final_tokens = estimate_tokens(prompt, "prompt")
    # logging.info(f"Final comprehensive prompt size: {final_tokens}/{max_tokens} tokens ({final_tokens/max_tokens*100:.1f}%)")
    
    # # Store prompt in analysis stats immediately after creation
    # import time
    # current_timestamp = time.time()
    
    # analysis_stats = load_analysis_stats()
    # # Determine channel name from available data or use default
    # channel_name = "comprehensive_analysis"  # Default if we can't determine channel
    
    # # Use provided video_to_channel mapping
    # if video_to_channel:
    #     channel_name = video_to_channel.get(v_id, "Unknown Channel")
    #     if channel_name != "Unknown Channel":
    #         logging.info(f"Using provided channel name '{channel_name}' for video {v_id}")
    #     else:
    #         logging.warning(f"Video {v_id} not found in provided channel mapping")
    # else:
    #     logging.warning(f"No video_to_channel mapping provided for video {v_id}")
    
    # # Find or create video entry
    # video_entry = find_or_create_video(analysis_stats, channel_name, v_id, title)
    
    # # Save transcript section to analysis
    # add_analysis_to_video(video_entry, "transcript", transcript_section, "transcript_data", "system")
    
    # # Save updated stats
    # save_analysis_stats(analysis_stats)
    
    # # Add comprehensive analysis
    # add_analysis_to_video(video_entry, "comprehensive", prompt, "[Analysis pending]", "pending")
    
    # # Save updated stats
    # save_analysis_stats(analysis_stats)
    
    # logging.info(f"Saved comprehensive analysis prompt to JSON for video {v_id} (channel: {channel_name})")
    
    # try:
    #     # Use ModelManager for intelligent model selection and fallback
    #     analysis, used_model = model_manager.try_model_with_fallback(prompt)
        
    #     if not analysis:
    #         logging.error("Failed to generate comprehensive AI analysis with all available models")
    #         return None
        
    #     # Clean AI output to remove any unwanted formatting
    #     cleaned_analysis = clean_ai_output(analysis)
        
    #     # Update the video entry with actual analysis output
    #     add_analysis_to_video(video_entry, "comprehensive", prompt, cleaned_analysis, used_model)
    #     save_analysis_stats(analysis_stats)
        
    #     logging.info(f"Comprehensive AI analysis completed using model '{used_model}'")
        
    #     # Generate embed color based on like ratio gradient
    #     embed_color = get_gradient_color(video_stats['like_ratio'])
    #     logging.info(f"Generated gradient color: {hex(embed_color)} for like ratio: {video_stats['like_ratio']:.1f}%")
        
    #     # Send comprehensive analysis to Discord
    #     if WEBHOOK:
    #         # Use Discord title from configuration with video title and ID
    #         discord_title = PROMPTS["discord_title"].format(title=title, v_id=v_id)
            
    #         # Get thumbnail URL
    #         thumbnail_url = get_thumbnail_url(v_id)
            
    #         payload = {
    #             "embeds": [{
    #                 "title": discord_title,
    #                 "color": embed_color,
    #                 "image": {
    #                     "url": thumbnail_url
    #                 },
    #                 "fields": [
    #                     {"name": PROMPTS["channel_field"], "value": channel_name, "inline": True},
    #                     {"name": PROMPTS["video_field"], "value": f"[{title}](https://www.youtube.com/watch?v={v_id})", "inline": False},
    #                 ],
    #                 "description": cleaned_analysis
    #             }]
    #         }
    #         requests.post(WEBHOOK, json=payload)
            
    # except Exception as e:
    #     logging.error(f"Failed to generate comprehensive AI analysis: {e}")

def summarize_comments_with_ai(title, comments_dict, v_id, video_stats, video_to_channel=None):
    """Generate AI summary of comments and send to Discord webhook"""
    # Load analysis stats
    analysis_stats = load_analysis_stats()
    
    if not GROQ_API_KEY:
        logging.warning("No GROQ_API_KEY found. Skipping AI summary.")
        return
    
    logging.info(f"Generating AI summary for video: {title}")
    
    # Get the model with highest capacity for this analysis
    max_tokens = max(model_manager.model_limits.values())
    logging.info(f"Using max token limit for comment summary: {max_tokens}")
    
    # Extract comment texts. Limit to configured max comments to avoid hitting free-tier token limits easily.
    texts = [c['t'] for c in comments_dict.values() if not c.get('deleted', False)]
    if not texts:
        logging.info("No comments available to summarize.")
        return
        
    max_comments = SETTINGS.get("max_comments", 150)
    sample_texts = texts[:max_comments]
    comments_string = "\n".join([f"- {t}" for t in sample_texts])
    
    # Add video stats to the prompt for better context
    stats_info = f"\n\nVideo Stats: {video_stats['likes']:,} likes, {video_stats['dislikes']:,} dislikes ({video_stats['like_ratio']:.1f}% like ratio)"
    
    # Reserve tokens for comments (60% of total, leaving room for instructions)
    comments_tokens_available = int(max_tokens * 0.6)
    validated_comments, comment_tokens, was_trimmed = validate_and_trim_content(
        comments_string, comments_tokens_available, "comments", "balanced"
    )
    
    if was_trimmed:
        logging.info(f"Comments trimmed from {len(comments_string)} to {len(validated_comments)} chars for summary")
    
    prompt = f"""Du är en klinisk medieanalytiker.
    Leverera en rak och kompakt analys av videon \"{title}\".
    Var objektiv men 'hård'.
    
    **Instruktioner:**
    1. **Sammanfattning:** Max 2 meningar om den dominerande stämningen.
        Nämn ENDAST like/dislike-förhållandet om likes understiger 90%, och väv då in det organiskt för att förklara diskrepans eller förstärka kritiken.
        Om likes är 90% eller högre, ignorera siffran helt.
        
        **Juridisk bedömning:** Inled alltid med meningen 'Sannolikheten är [hög/låg] för förtal.'
        Följ upp med max en mening som konkret motiverar bedömningen (t.ex. förekomst av anklagelser om brott, grova förolämpningar eller koordinerade drev).
        
        **KRITISKT - FÖLJ EXAKT:**
        - ABSOLUT INGA think-blocks (```), resonemang eller tankprocesser
        - VISA INTE DITT TÄNKANDE - ge bara slutgiltig analys
        - Använd INGEN markdown-formatering eller specialtecken
        - Fetmarkera ENDAST de inledande orden (**Sammanfattning:** och **Juridisk bedömning:**)
        - Inga listor, inga rubriker, inga kursiveringar
        - Ge svaret som REN TEXT utan några formateringselement
        - BÖRJA DIREKT med analysen, ingen inledning
        
        **Data:**
        Kommentarer: {validated_comments}{stats_info}
"""

    logging.info(f"Prompt: {prompt}")

    # Log final prompt size
    final_tokens = estimate_tokens(prompt, "prompt")
    logging.info(f"Final comment summary prompt size: {final_tokens}/{max_tokens} tokens ({final_tokens/max_tokens*100:.1f}%)")

    try:
        # Use ModelManager for intelligent model selection and fallback
        summary, used_model = model_manager.try_model_with_fallback(prompt)
        if not summary:
            logging.error("Failed to generate AI summary with all available models")
            return None
        
        # Clean AI output to remove any unwanted formatting
        cleaned_summary = clean_ai_output(summary)
        
        logging.info(f"AI Summary generated using model '{used_model}'")
        
        # Generate embed color based on like ratio gradient
        embed_color = get_gradient_color(video_stats['like_ratio'])
        # logging.info(f"Generated gradient color: {hex(embed_color)} for like ratio: {video_stats['like_ratio']:.1f}%")
        
        # Determine channel name (used for tracking even without webhook)
        channel_name = "Unknown Channel"
    
        # Use provided video_to_channel mapping
        if video_to_channel:
            channel_name = video_to_channel.get(v_id, "Unknown Channel")
            if channel_name != "Unknown Channel":
                logging.info(f"Using provided channel name '{channel_name}' for video {v_id}")
            else:
                logging.warning(f"Video {v_id} not found in provided channel mapping")
        else:
            logging.warning(f"No video_to_channel mapping provided for video {v_id}")
        
        # Find or create video entry
        video_entry = find_or_create_video(analysis_stats, channel_name, v_id, title)
        
        # Add comment review analysis
        add_analysis_to_video(video_entry, "comment_review", prompt, cleaned_summary, used_model)
        
        # Save updated stats
        save_analysis_stats(analysis_stats)
        
        # Create footer with analysis count and model
        channel_data = analysis_stats.get(channel_name, {"videos": []})
        videos_count = len(channel_data["videos"])
        footer_text = f"{videos_count} videos from @{channel_name} has been analysed. This analysis was made using {used_model}"
        
        # Send summary to Discord
        if WEBHOOK:
            
            # Use Discord title from configuration with video title and ID
            discord_title = PROMPTS["discord_title"].format(title=title, v_id=v_id)
            
            # Get thumbnail URL
            thumbnail_url = get_thumbnail_url(v_id)
            
            payload = {
                "embeds": [{
                    "title": discord_title,
                    "color": embed_color,
                    "image": {
                        "url": thumbnail_url
                    },
                    "fields": [
                        {"name": PROMPTS["channel_field"], "value": channel_name, "inline": True},
                        {"name": PROMPTS["video_field"], "value": f"[{title}](https://www.youtube.com/watch?v={v_id})", "inline": False},
                    ],
                    "description": cleaned_summary,
                    "footer": {
                        "text": footer_text
                    }
                }]
            }
            requests.post(WEBHOOK, json=payload)
    except Exception as e:
        logging.error(f"Failed to generate AI summary: {e}")
        sys.exit(1)  # Abort script on ERROR logs as requested

# --- MAIN LOGIC ---
if __name__ == "__main__":
    # Setup run logging with sequential numbering
    run_log_file = setup_run_logging()
    
    logging.info("Starting YouTube video analyzer...")
    
    # 1. Fetch new videos from channels
    video_ids, video_to_channel = fetch_latest_videos(CHANNELS)
    
    if not video_ids:
        logging.info("No new videos found to analyze.")
    else:
        logging.info(f"Found {len(video_ids)} new videos to analyze.")
    
    # 2. Load analysis stats
    analysis_stats = load_analysis_stats()
    
    # 3. Process each video
    for v_id in video_ids:
        channel_name = video_to_channel.get(v_id, "Unknown Channel")
        logging.info(f"Processing video {v_id} from channel {channel_name}.")
        
        # Get video data
        ui_count, comments, title, video_stats, transcript_text, ai_analysis = get_yt_data(v_id, deep_scrape=True, video_to_channel=video_to_channel)
        
        # Check for early members-only/private content detection
        if title == "MEMBERS_ONLY":
            logging.warning(f"Skipping members-only/private video {v_id} - detected during initial fetch (zero engagement but title available)")
            continue
        
        if title is None:
            logging.warning(f"Skipping video {v_id} due to scraping failure.")
            continue
        
        # Check for members-only/private content (late detection from transcription)
        if transcript_text == "MEMBERS_ONLY":
            logging.warning(f"Skipping members-only/private video {v_id} - content access restricted")
            continue
        
        # Find or create video entry
        video_entry = find_or_create_video(analysis_stats, channel_name, v_id, title)
        
        # Save video metadata
        video_entry["video_stats"] = video_stats
        video_entry["ui_comment_count"] = ui_count
        
        # Save transcript if available
        if transcript_text:
            video_entry["transcript"] = transcript_text
            logging.info(f"Transcript available for {v_id} ({len(transcript_text)} characters)")
        else:
            logging.info(f"No transcript available for {v_id}")
        
        # Save AI analysis if available
        if ai_analysis:
            video_entry["ai_analysis"] = ai_analysis
            logging.info(f"AI analysis available for {v_id}")
        
        # Trigger comment analysis if comments are available
        if comments:
            summarize_comments_with_ai(title, comments, v_id, video_stats, video_to_channel)
        
        # Save analysis stats after each video
        save_analysis_stats(analysis_stats)
        time.sleep(2)  # Brief pause between videos
    
    logging.info(f"=== RUN COMPLETE at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
    logging.info(f"Run log saved to: {run_log_file}")