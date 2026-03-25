import json
import os
import requests
import time
import random
import sys
import logging
import tempfile
import re
from datetime import datetime, timedelta
from dotenv import load_dotenv
from groq import Groq
from playwright.sync_api import sync_playwright, TimeoutError
import yt_dlp
import assemblyai as aai

# Load environment variables from .env file
load_dotenv()

# Validate required environment variables
WEBHOOK = os.getenv("DISCORD_WEBHOOK")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
ASSEMBLYAI_API_KEY = os.getenv("ASSEMBLYAI_API_KEY")

if not GROQ_API_KEY:
    logging.error("GROQ_API_KEY environment variable is required but not set")
    sys.exit(1)

if not WEBHOOK:
    logging.warning(
        "DISCORD_WEBHOOK environment variable not set - Discord notifications will be disabled"
    )

if not GEMINI_API_KEY:
    logging.warning(
        "GEMINI_API_KEY environment variable not set - Gemini fallback will be unavailable"
    )

if not ASSEMBLYAI_API_KEY:
    logging.warning(
        "ASSEMBLYAI_API_KEY environment variable not set - AssemblyAI transcription fallback will be unavailable"
    )

# Setup logging and encoding
sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# WORKING 2026-03-16 17:40
def get_gradient_color(like_ratio):
    """Generate a color gradient from red to green based on like ratio (0-100)"""
    # logging.info("=== FUNCTION START: get_gradient_color ===")
    # Ensure like_ratio is within 0-100 range
    ratio = max(0, min(100, like_ratio))

    # Red (255, 0, 0) at 0% to Green (0, 255, 0) at 100%
    red = int(255 * (1 - ratio / 100))
    green = int(255 * (ratio / 100))
    blue = 0

    # Convert to hex color string
    hex_color = f"{red:02x}{green:02x}{blue:02x}"
    return int(hex_color, 16)


# WORKING 2026-03-16 17:40
def get_thumbnail_url(v_id):
    """Get YouTube thumbnail URL for a video"""
    # logging.info("=== FUNCTION START: get_thumbnail_url ===")
    return f"https://img.youtube.com/vi/{v_id}/maxresdefault.jpg"


def setup_run_logging():
    """Setup file-based logging for individual runs with sequential numbering"""
    logging.info("=== FUNCTION START: setup_run_logging ===")
    # Find existing run logs to determine next number
    run_logs = [
        f for f in os.listdir(".") if f.startswith("runlog") and f.endswith(".log")
    ]
    run_numbers = []

    for log_file in run_logs:
        try:
            # Extract number from filename like "runlog1.log", "runlog2.log", etc.
            number = int(log_file.replace("runlog", "").replace(".log", ""))
            run_numbers.append(number)
        except ValueError:
            continue

    next_number = max(run_numbers) + 1 if run_numbers else 1
    run_log_file = f"runlog{next_number}.log"

    # Create file handler for run logging
    file_handler = logging.FileHandler(run_log_file, mode="w", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_formatter)

    # Add file handler to root logger
    logging.getLogger().addHandler(file_handler)

    logging.info(
        f"=== RUN {next_number} STARTED at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ==="
    )
    return run_log_file


# Configuration
def load_channels_from_env():
    """Load YouTube channels from environment variable"""
    channels_str = os.getenv("CHANNELS_LIST", "")
    
    if not channels_str:
        # Fallback to default channels if environment variable is not set
        logging.warning("CHANNELS_LIST environment variable not set, using default channels")
        return ["CarlFredrikAlexanderRask", "ANJO1", "MotVikten", "Skuldis"]
    
    # Parse comma-separated list and clean up whitespace
    channels = [channel.strip() for channel in channels_str.split(",") if channel.strip()]
    
    if not channels:
        logging.error("CHANNELS_LIST is empty after parsing, using default channels")
        return ["CarlFredrikAlexanderRask", "ANJO1", "MotVikten", "Skuldis"]
    
    logging.info(f"Loaded {len(channels)} channels from environment: {channels}")
    return channels

CHANNELS = load_channels_from_env()
STATE_FILE = "comment_state.json"
ANALYSIS_STATS_FILE = "analysis_stats.json"  # Track video analysis counts per channel
CONFIG_FILE = "config.json"

# Channel name mapping to normalize extracted names to expected identifiers
CHANNEL_NAME_MAPPING = {
    "RASK": "CarlFredrikAlexanderRask",
    "ANJO": "ANJO1", 
    "MOTVIKTEN": "MotVikten",
    "SKULDIS": "Skuldis"
}

def normalize_channel_name(extracted_name):
    """Normalize extracted channel name to expected channel identifier"""
    if not extracted_name:
        return None
    
    # Clean up the name (remove @ prefix, convert to uppercase)
    clean_name = extracted_name.lstrip("@").upper()
    
    # Direct mapping
    if clean_name in CHANNEL_NAME_MAPPING:
        return CHANNEL_NAME_MAPPING[clean_name]
    
    # Fuzzy matching for partial matches
    for expected_channel in CHANNELS:
        if clean_name in expected_channel.upper() or expected_channel.upper() in clean_name:
            return expected_channel
    
    # If no match found, return the cleaned name for debugging
    logging.warning(f"No channel mapping found for extracted name: '{extracted_name}' (cleaned: '{clean_name}')")
    return clean_name


def estimate_tokens(text, content_type="general"):
    """
    Estimate token count based on content type and character count.
    Using conservative ratios to avoid 413 errors.
    """
    logging.info("=== FUNCTION START: estimate_tokens ===")
    if not text:
        return 0

    char_count = len(text)

    # Conservative token-to-character ratios (updated based on actual API behavior)
    ratios = {
        "general": 3.0,  # More conservative: 1 token ≈ 3 chars
        "transcript": 2.8,  # Transcripts: dense, 1 token ≈ 2.8 chars
        "comments": 3.2,  # Comments: informal, 1 token ≈ 3.2 chars
        "prompt": 2.9,  # Instructions/prompts: 1 token ≈ 2.9 chars
    }

    ratio = ratios.get(content_type, ratios["general"])
    return int(char_count / ratio)


def validate_and_trim_content(
    content, max_tokens, content_type="general", priority="beginning"
):
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
    logging.info("=== FUNCTION START: validate_and_trim_content ===")
    if not content:
        return "", 0, False

    estimated_tokens = estimate_tokens(content, content_type)

    if estimated_tokens <= max_tokens:
        return content, estimated_tokens, False

    logging.warning(
        f"Content exceeds token limit: {estimated_tokens} > {max_tokens} (type: {content_type})"
    )

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
    logging.info("=== FUNCTION START: log_payload_size ===")
    estimated_tokens = estimate_tokens(content, content_type)
    percentage = (estimated_tokens / max_tokens * 100) if max_tokens > 0 else 0

    logging.info(
        f"Payload size for {model_name}: {estimated_tokens}/{max_tokens} tokens ({percentage:.1f}%) - {content_type}"
    )

    if estimated_tokens > max_tokens:
        logging.warning(
            f"Payload EXCEEDS limit by {estimated_tokens - max_tokens} tokens!"
        )

    return estimated_tokens


class ModelManager:
    def __init__(self, config):
        logging.info("=== FUNCTION START: ModelManager.__init__ ===")
        self.config = config
        self.ai_config = config.get("ai_models", {})
        self.primary_model = self.ai_config.get("primary", DEFAULT_MODEL)
        self.fallback_models = self.ai_config.get("fallback", [])
        self.available_models = self.ai_config.get("models", {})
        self.client = Groq(api_key=GROQ_API_KEY)

        # Removed Gemini from model limits
        self.model_limits = {
            DEFAULT_MODEL: 128000,
            "qwen/qwen3-32b": 32000,
            AI_ANALYSIS_MODEL: 32000,
        }

        self.gemini_client = None  # Explicitly disabled for general use

    def get_model_for_request(self):
        """Select the best model for current request"""
        logging.info("=== FUNCTION START: get_model_for_request ===")
        # Try primary model first
        if self.primary_model in self.available_models:
            return self.primary_model

        # Fallback to available models
        for model in self.fallback_models:
            if model in self.available_models:
                return model

        # Ultimate fallback to legacy model
        logging.warning(
            f"No configured models available, falling back to {DEFAULT_MODEL}"
        )
        return DEFAULT_MODEL

    def _build_model_priority_list(self):
        """Build ordered list of models to try, prioritized by capacity"""
        logging.info("=== FUNCTION START: _build_model_priority_list ===")
        
        models_by_capacity = [
            (DEFAULT_MODEL, 10000),  # Highest capacity
            ("qwen/qwen3-32b", 4500),
            (AI_ANALYSIS_MODEL, 4500),
        ]
        
        models_to_try = []
        seen = set()
        
        # Add models by capacity order
        for model_name, capacity in models_by_capacity:
            if model_name in self.available_models and model_name not in seen:
                models_to_try.append(model_name)
                seen.add(model_name)
        
        # Add any remaining available fallback models
        for model in self.fallback_models:
            if model not in seen and model in self.available_models:
                models_to_try.append(model)
                seen.add(model)
        
        # Ultimate fallback to default model
        if DEFAULT_MODEL not in models_to_try and DEFAULT_MODEL in self.available_models:
            models_to_try.append(DEFAULT_MODEL)
        
        return models_to_try

    def _reduce_prompt_content(self, prompt, attempt):
        """Progressively reduce prompt content for retry attempts"""
        logging.info("=== FUNCTION START: _reduce_prompt_content ===")
        
        reduction_factor = 0.6 - (attempt * 0.1)  # 60%, 50%, 40%
        if reduction_factor < 0.3:
            reduction_factor = 0.3
        
        lines = prompt.split("\n")
        if len(lines) > 4:
            keep_lines = max(2, len(lines) // 3)
            reduced_lines = (
                lines[:keep_lines]
                + ["\n... [content reduced for retry]...\n"]
                + lines[-2:]
            )
            reduced_prompt = "\n".join(reduced_lines)
            logging.info(f"Retry {attempt + 1}: Reduced prompt content by {int((1 - reduction_factor) * 100)}% due to previous 413 errors")
            return reduced_prompt
        
        return prompt

    def _prepare_prompt_for_attempt(self, prompt, model_name, attempt):
        """Validate and prepare prompt for specific model attempt"""
        logging.info("=== FUNCTION START: _prepare_prompt_for_attempt ===")
        
        # Get token limit for this model
        max_tokens = self.model_limits.get(model_name, 4500)
        
        # Reduce content if this is a retry attempt
        current_prompt = prompt
        if attempt > 0:
            current_prompt = self._reduce_prompt_content(prompt, attempt)
        
        # Log payload size before validation
        log_payload_size(current_prompt, model_name, max_tokens, "prompt")
        
        # Validate and trim content if necessary
        validated_prompt, final_tokens, was_trimmed = validate_and_trim_content(
            current_prompt, max_tokens, "prompt", "beginning"
        )
        
        if was_trimmed:
            logging.info(f"Content was trimmed for {model_name} to fit token limits")
        
        return validated_prompt, final_tokens

    def _log_api_details(self, content, content_type="input"):
        """Log API input/output details with proper formatting"""
        logging.info("=== FUNCTION START: _log_api_details ===")
        
        content_json = json.dumps(content, indent=2, ensure_ascii=False)
        if len(content_json) > 1000:
            logging.info(f"{content_type.capitalize()} (first 1000 chars):")
            for line in content_json.split("\n")[:10]:
                logging.info(line)
            if len(content_json.split("\n")) > 10:
                logging.info("... (truncated)")
            logging.info(f"[Full {content_type} length: {len(content_json)} chars]")
        else:
            for line in content_json.split("\n"):
                logging.info(line)

    def _call_model_api(self, validated_prompt, model_name):
        """Make API call to specified model"""
        logging.info("=== FUNCTION START: _call_model_api ===")
        
        chat_completion = self.client.chat.completions.create(
            messages=[{"role": "user", "content": validated_prompt}],
            model=model_name,
        )
        return chat_completion.choices[0].message.content.strip()

    def _handle_api_error(self, error, model_name):
        """Handle different types of API errors"""
        logging.info("=== FUNCTION START: _handle_api_error ===")
        
        error_msg = str(error).lower()
        if "413" in error_msg or "payload too large" in error_msg:
            logging.error(
                f"Model {model_name} failed: 413 Payload Too Large. This indicates our token estimation still needs adjustment."
            )
            return "payload_too_large"
        elif "429" in error_msg or "rate limit" in error_msg:
            logging.warning(f"Model {model_name} failed: Rate limit reached - {str(error)[:100]}...")
            return "rate_limit"
        else:
            logging.warning(f"Model {model_name} failed: {str(error)}")
            return "other_error"

    def try_model_with_fallback(self, prompt):
        """Try to generate summary with fallback models, prioritizing higher token limits"""
        logging.info("=== FUNCTION START: try_model_with_fallback ===")

        # Log initial state and prompt
        logging.info("=== MODEL MANAGER STATE ===")
        logging.info(f"Available models: {self.available_models}")
        logging.info(f"Primary model: {self.primary_model}")
        logging.info(f"Fallback models: {self.fallback_models}")
        logging.info(f"Model limits: {self.model_limits}")
        logging.info("=== END MODEL MANAGER STATE ===")

        logging.info("=== PROMPT CONTENT ===")
        logging.info(f"Prompt length: {len(prompt)} characters")
        logging.info(f"Prompt preview (first 500 chars): {prompt[:500]}...")
        logging.info("=== END PROMPT CONTENT ===")

        # Get ordered list of models to try
        models_to_try = self._build_model_priority_list()

        # Try each model in order
        for attempt, model_name in enumerate(models_to_try):
            try:
                logging.info(f"Attempting to use model: {model_name} (attempt {attempt + 1}/{len(models_to_try)})")
                
                # Prepare prompt for this attempt
                validated_prompt, final_tokens = self._prepare_prompt_for_attempt(prompt, model_name, attempt)
                
                # Log input details
                logging.info("=== AI MODEL INPUT ===")
                logging.info(f"Model: {model_name}")
                self._log_api_details(validated_prompt, "input")
                logging.info("=== END AI MODEL INPUT ===")
                
                # Make API call
                summary = self._call_model_api(validated_prompt, model_name)
                
                # Log output details
                logging.info("=== AI MODEL OUTPUT ===")
                logging.info(f"Model: {model_name}")
                self._log_api_details(summary, "output")
                logging.info("=== END AI MODEL OUTPUT ===")
                
                logging.info(f"Successfully generated summary using model: {model_name} ({final_tokens} tokens)")
                return summary, model_name
                
            except Exception as e:
                self._handle_api_error(e, model_name)
                # Continue to next model

        return None, None


# Load configuration
def load_config():
    logging.info("=== FUNCTION START: load_config ===")
    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logging.error(
            f"Config file {CONFIG_FILE} not found. Script cannot continue without configuration."
        )
        sys.exit(1)


CONFIG = load_config()
CURRENT_LANGUAGE = CONFIG["language"]
SETTINGS = CONFIG["settings"]
PROMPTS = CONFIG["prompts"][CURRENT_LANGUAGE]


def save_config(config):
    """Save configuration to file"""
    logging.info("=== FUNCTION START: save_config ===")
    try:
        with open(CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        logging.info("Configuration saved successfully")
    except Exception as e:
        logging.error(f"Failed to save configuration: {e}")
        return None

def _find_time_pattern_match(error_message):
    """Find and return the first matching time pattern with its groups"""
    logging.info("=== FUNCTION START: _find_time_pattern_match ===")
    
    time_patterns = [
        r"(\d+)h\s*(\d+)m\s*(\d+)s",  # hours minutes seconds
        r"(\d+)h\s*(\d+)m",  # hours minutes
        r"(\d+)h\s*(\d+)s",  # hours seconds
        r"(\d+)m\s*(\d+)s",  # minutes seconds
        r"(\d+)h",  # just hours
        r"(\d+)m",  # just minutes
        r"(\d+)s",  # just seconds
    ]
    
    for pattern in time_patterns:
        match = re.search(pattern, error_message, re.IGNORECASE)
        if match:
            groups = match.groups()
            logging.info(f"Pattern matched: {pattern}, Groups: {groups}")
            return pattern, groups
    
    logging.warning("No duration pattern found in error message")
    return None, None

def _convert_groups_to_seconds(pattern, groups):
    """Convert matched groups to seconds based on pattern"""
    logging.info("=== FUNCTION START: _convert_groups_to_seconds ===")
    
    if len(groups) == 3:  # h m s
        return _convert_hms_to_seconds(groups)
    elif len(groups) == 2:
        if "h" in pattern and "m" in pattern:  # h m
            return _convert_hm_to_seconds(groups)
        elif "h" in pattern and "s" in pattern:  # h s
            return _convert_hs_to_seconds(groups)
        else:  # m s
            return _convert_ms_to_seconds(groups)
    else:  # single unit
        return _convert_single_unit_to_seconds(pattern, groups)

def _convert_hms_to_seconds(groups):
    """Convert hours, minutes, seconds to total seconds"""
    logging.info("=== FUNCTION START: _convert_hms_to_seconds ===")
    hours, minutes, seconds = map(int, groups)
    return hours * 3600 + minutes * 60 + seconds

def _convert_hm_to_seconds(groups):
    """Convert hours, minutes to total seconds"""
    logging.info("=== FUNCTION START: _convert_hm_to_seconds ===")
    hours, minutes = map(int, groups)
    return hours * 3600 + minutes * 60

def _convert_hs_to_seconds(groups):
    """Convert hours, seconds to total seconds"""
    logging.info("=== FUNCTION START: _convert_hs_to_seconds ===")
    hours, seconds = map(int, groups)
    return hours * 3600 + seconds

def _convert_ms_to_seconds(groups):
    """Convert minutes, seconds to total seconds"""
    logging.info("=== FUNCTION START: _convert_ms_to_seconds ===")
    minutes, seconds = map(int, groups)
    return minutes * 60 + seconds

def _convert_single_unit_to_seconds(pattern, groups):
    """Convert single unit (hours, minutes, or seconds) to total seconds"""
    logging.info("=== FUNCTION START: _convert_single_unit_to_seconds ===")
    value = int(groups[0])
    if "h" in pattern:
        return value * 3600
    elif "m" in pattern:
        return value * 60
    else:  # seconds
        return value

def parse_duration_from_error(error_message):
    """Parse duration like '2h36m57s' from error message and return seconds"""
    logging.info("=== FUNCTION START: parse_duration_from_error ===")
    try:
        # Find matching time pattern
        pattern, groups = _find_time_pattern_match(error_message)
        
        if pattern and groups:
            # Convert matched groups to seconds
            total_seconds = _convert_groups_to_seconds(pattern, groups)
            logging.info(f"Parsed duration: {total_seconds} seconds")
            return total_seconds
        
        logging.warning("No duration pattern found in error message")
        return None

    except Exception as e:
        logging.error(f"Error parsing duration: {e}")
        return None


def set_groq_whisper_rate_limit(reset_seconds):
    """Set Groq Whisper rate limit reset time in config"""
    logging.info("=== FUNCTION START: set_groq_whisper_rate_limit ===")
    try:
        reset_time = datetime.now() + timedelta(seconds=reset_seconds)

        # Load current config
        config = load_config()

        # Add or update rate limit info
        if "rate_limits" not in config:
            config["rate_limits"] = {}

        config["rate_limits"]["groq_whisper_reset_time"] = reset_time.isoformat()
        config["rate_limits"]["groq_whisper_reset_seconds"] = reset_seconds

        # Save updated config
        save_config(config)

        logging.info(
            f"Groq Whisper rate limit set. Reset time: {reset_time.isoformat()}"
        )
        return reset_time

    except Exception as e:
        logging.error(f"Error setting rate limit: {e}")
        return None


def is_groq_whisper_available():
    """Check if Groq Whisper is available (not rate limited)"""
    logging.info("=== FUNCTION START: is_groq_whisper_available ===")
    try:
        config = load_config()

        # Check if rate limit info exists
        if (
            "rate_limits" not in config
            or "groq_whisper_reset_time" not in config["rate_limits"]
        ):
            logging.info("No Groq Whisper rate limit found - service available")
            return True

        reset_time_str = config["rate_limits"]["groq_whisper_reset_time"]
        reset_time = datetime.fromisoformat(
            reset_time_str.replace("Z", "+00:00")
            if "Z" in reset_time_str
            else reset_time_str
        )
        current_time = datetime.now()

        if current_time >= reset_time:
            # Rate limit has expired - clean it up
            logging.info(
                "Groq Whisper rate limit has expired - cleaning up and allowing access"
            )

            # Remove expired rate limit
            config["rate_limits"].pop("groq_whisper_reset_time", None)
            config["rate_limits"].pop("groq_whisper_reset_seconds", None)
            save_config(config)

            return True
        else:
            # Still rate limited
            remaining_time = reset_time - current_time
            logging.warning(
                f"Groq Whisper is still rate limited. Reset in: {remaining_time}"
            )
            return False

    except Exception as e:
        logging.error(f"Error checking Groq Whisper availability: {e}")
        # If we can't check, assume it's available to avoid false negatives
        return True


# Constants
DEFAULT_MODEL = "llama-3.3-70b-versatile"
AI_ANALYSIS_MODEL = "llama-3.1-8b-instant"
UNKNOWN_CHANNEL = "Unknown Channel"
UNKNOWN_DATE = "Okänt datum"
WORST_AUDIO_FORMAT = "worstaudio/worst"

# Initialize Model Manager
model_manager = ModelManager(CONFIG)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
]


def summarize_transcript(transcript_text, title):
    """Summarize long transcript using Gemini 3.1 Flash Lite Preview (Free Tier)"""
    logging.info("=== FUNCTION START: summarize_transcript ===")

    prompt = f"Summarize this YouTube video transcript titled '{title}' into key points (max 300 words):\n\n{transcript_text}"

    #     prompt = f"""Summarize this YouTube video transcript titled '{title}' into a very concise summary (max 300 words):

    # Key elements to capture:
    # - Main topic and central argument
    # - 2-3 key claims or points
    # - Overall tone/sentiment
    # - Any controversial elements

    # Keep it extremely brief!

    # Transcript:
    # {transcript_text}"""

    if GEMINI_API_KEY:
        try:
            from google import genai

            client = genai.Client(api_key=GEMINI_API_KEY)

            # Use 3.1 Flash Lite Preview - highest free tier availability
            response = client.models.generate_content(
                model="gemini-3.1-flash-lite-preview", contents=prompt
            )

            if response and response.text:
                return response.text.strip()
        except Exception as e:
            logging.warning(f"Gemini summarization failed: {e}. Falling back to Groq.")

    # FALLBACK to Groq if Gemini fails
    try:
        chat_completion = model_manager.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model=DEFAULT_MODEL,
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        logging.error(f"Fallback summarization failed: {e}")
        return transcript_text[:1000] + "..."  # Last resort truncation


def get_video_stats(v_id):
    """Fetch likes and dislikes using returnyoutubedislike.com API"""
    logging.info("=== FUNCTION START: get_video_stats ===")
    try:
        api_url = f"https://returnyoutubedislikeapi.com/votes?videoId={v_id}"
        response = requests.get(api_url, timeout=10000)
        if response.status_code == 200:
            data = response.json()
            likes = data.get("likes", 0)
            dislikes = data.get("dislikes", 0)
            views = data.get("viewCount", 0)

            # Calculate engagement ratio
            total_reactions = likes + dislikes
            like_ratio = (likes / total_reactions * 100) if total_reactions > 0 else 0

            return {
                "likes": likes,
                "dislikes": dislikes,
                "views": views,
                "like_ratio": like_ratio,
            }
    except Exception as e:
        logging.warning(f"Failed to fetch video stats from API: {e}")

    return {"likes": 0, "dislikes": 0, "views": 0, "like_ratio": 0}


def _manage_queue_depth(queue_state, fetch_depth):
    """Manage queue depth increment and logging"""
    logging.info("=== FUNCTION START: _manage_queue_depth ===")
    queue_state["fetch_depth"] = min(
        queue_state["fetch_depth"] + 1, min(fetch_depth, 10)
    )
    logging.info(
        f"DATA CHANGE: Incremented fetch_depth to {queue_state['fetch_depth']} (configured max: {fetch_depth})"
    )
    return queue_state


def _is_valid_channel_data(channel_data):
    """Check if channel data is valid dictionary with videos"""
    logging.info("=== FUNCTION START: _is_valid_channel_data ===")
    return isinstance(channel_data, dict) and "videos" in channel_data

def _is_valid_video_entry(video):
    """Check if video entry is valid dictionary with video_id"""
    logging.info("=== FUNCTION START: _is_valid_video_entry ===")
    return isinstance(video, dict) and "video_id" in video

def _is_video_completed(video):
    """Check if video is marked as sent to Discord"""
    logging.info("=== FUNCTION START: _is_video_completed ===")
    return video.get("sentToDiscord")

def _extract_completed_video_id(video):
    """Extract video ID if video is completed"""
    logging.info("=== FUNCTION START: _extract_completed_video_id ===")
    if _is_valid_video_entry(video) and _is_video_completed(video):
        return video["video_id"]
    return None

def _process_channel_videos(channel_data):
    """Process videos from a single channel and return completed IDs"""
    logging.info("=== FUNCTION START: _process_channel_videos ===")
    completed_ids = set()
    
    if not _is_valid_channel_data(channel_data):
        return completed_ids
    
    for video in channel_data["videos"]:
        video_id = _extract_completed_video_id(video)
        if video_id:
            completed_ids.add(video_id)
    
    return completed_ids


def _get_completed_videos():
    """Extract completed video IDs from analysis stats"""
    logging.info("=== FUNCTION START: _get_completed_videos ===")
    analysis_stats = load_analysis_stats()
    completed_videos = set()
    
    for channel_data in analysis_stats.values():
        channel_completed = _process_channel_videos(channel_data)
        completed_videos.update(channel_completed)
    
    return completed_videos


def _setup_browser_context():
    """Initialize Playwright browser and context"""
    logging.info("=== FUNCTION START: _setup_browser_context ===")
    user_agent = random.choice(USER_AGENTS)
    
    browser = None
    context = None
    page = None
    
    return user_agent, browser, context, page


def _handle_youtube_consent(page):
    """Handle YouTube consent page if present"""
    logging.info("=== FUNCTION START: _handle_youtube_consent ===")
    if "Before you continue to YouTube" in page.title():
        try:
            accept_button = (
                page.locator("button")
                .filter(has_text="Accept all")
                .first
            )
            if accept_button.count() > 0:
                accept_button.click()
                page.wait_for_timeout(2000)
                page.wait_for_load_state("networkidle")
        except Exception as e:
            logging.error("Failed to accept consent: " + str(e))


def _extract_video_ids(page, max_videos, completed_videos, channel_name, fetch_depth=1):
    """Extract video IDs from page elements with depth-based pagination"""
    logging.info("=== FUNCTION START: _extract_video_ids ===")
    videos_found = 0
    latest_videos = []
    video_to_channel = {}
    
    # Scroll to load more videos based on fetch_depth
    scroll_count = 0
    target_scroll_depth = fetch_depth
    
    while scroll_count < target_scroll_depth:
        scroll_count += 1
        scroll_distance = scroll_count * 2000  # Scroll 2000px each time
        page.evaluate(f"window.scrollBy(0, {scroll_distance})")
        page.wait_for_timeout(2000)  # Wait for videos to load
    
    video_locators = [
        'ytd-rich-item-renderer a[href*="/watch?v="]',
        'ytd-rich-grid-media a[href*="/watch?v="]',
    ]
    
    for locator_selector in video_locators:
        video_elements = page.locator(locator_selector)
        total_videos = video_elements.count()
        
        # Iterate backwards to get oldest videos first
        max_index = min(total_videos, max_videos * fetch_depth)
        for i in range(max_index - 1, -1, -1):
            if videos_found >= max_videos:
                break
            
            href = video_elements.nth(i).get_attribute("href")
            if href and "v=" in href:
                v_id = href.split("v=")[1].split("&")[0]
                
                # Check if video is already completed
                if v_id not in completed_videos:
                    latest_videos.append(v_id)
                    video_to_channel[v_id] = channel_name  # Map to actual channel username
                    videos_found += 1
                else:
                    logging.info(f"Skipped completed video {v_id} (position {i+1}) from {channel_name}")
    
    return latest_videos, video_to_channel


def _fetch_channel_videos(page, channel, max_videos, completed_videos, fetch_depth=1):
    """Fetch videos from a single channel"""
    logging.info("=== FUNCTION START: _fetch_channel_videos ===")
    try:
        url = f"https://www.youtube.com/@{channel}/videos"
        page.goto(url, timeout=60000)
        page.wait_for_load_state("networkidle")
        
        # Handle consent page
        _handle_youtube_consent(page)
        
        # Scroll to load videos
        page.evaluate("window.scrollBy(0, 1000)")
        page.wait_for_timeout(5000)
        
        # Extract video IDs with filtering
        return _extract_video_ids(page, max_videos, completed_videos, channel, fetch_depth)
        
    except Exception as e:
        logging.error(f"Failed to fetch videos for {channel}: {e}")
        return [], {}


def _filter_new_videos(all_videos, completed_videos):
    """Filter out already completed videos"""
    logging.info("=== FUNCTION START: _filter_new_videos ===")
    return [v_id for v_id in all_videos if v_id not in completed_videos]


def _find_channel_for_video(video_id):
    """Find channel name for a specific video ID"""
    logging.info("=== FUNCTION START: _find_channel_for_video ===")
    analysis_stats = load_analysis_stats()
    
    for channel_key, channel_data in analysis_stats.items():
        if isinstance(channel_data, dict) and "videos" in channel_data:
            for video in channel_data["videos"]:
                if (
                    isinstance(video, dict)
                    and video.get("video_id") == video_id
                ):
                    # Return the actual channel name stored in the video entry
                    return video.get("channel_name", channel_key)
    
    return None


def _should_refill_queue(queue_state):
    """Check if queue needs refilling"""
    logging.info("=== FUNCTION START: _should_refill_queue ===")
    return not queue_state["pending_queue"]


def _get_next_video_from_queue(queue_state):
    """Get next video from queue and update state"""
    logging.info("=== FUNCTION START: _get_next_video_from_queue ===")
    next_video_id = pop_next_video(queue_state)
    
    if next_video_id:
        return next_video_id, queue_state
    else:
        return None, queue_state


def _return_video_result(next_video_id, queue_state, video_to_channel):
    """Format and return video processing result"""
    logging.info("=== FUNCTION START: _return_video_result ===")
    if not next_video_id:
        logging.info("No videos in queue to process")
        return [], {}
    
    save_queue_state(queue_state)
    
    # Find channel mapping
    if video_to_channel and next_video_id in video_to_channel:
        channel_mapping = {next_video_id: video_to_channel[next_video_id]}
    else:
        channel_name = _find_channel_for_video(next_video_id)
        channel_mapping = {next_video_id: channel_name} if channel_name else {}
    
    return [next_video_id], channel_mapping


def fetch_latest_videos(channels, fetch_depth=6):
    """Fetch latest videos from channels using persistent queue system"""
    logging.info("=== FUNCTION START: fetch_latest_videos ===")
    queue_state = load_queue_state()

    # Refill queue if empty
    if _should_refill_queue(queue_state):
        # Manage queue depth
        queue_state = _manage_queue_depth(queue_state, fetch_depth)
        
        # Get completed videos for filtering
        completed_videos = _get_completed_videos()
        
        # Fetch videos from channels
        latest_videos = []
        video_to_channel = {}
        
        user_agent, _, _, _ = _setup_browser_context()
        
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(
                user_agent=user_agent,
                viewport={"width": 1920, "height": 1080},
                locale="en-US",
                extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
            )
            page = context.new_page()
            
            try:
                for channel in channels:
                    channel_videos, channel_mapping = _fetch_channel_videos(
                        page, channel, queue_state["fetch_depth"], completed_videos, queue_state["fetch_depth"]
                    )
                    latest_videos.extend(channel_videos)
                    video_to_channel.update(channel_mapping)
            finally:
                browser.close()
        
        # Add new videos to pending queue
        queue_state = add_to_pending_queue(queue_state, latest_videos)
        save_queue_state(queue_state)
        logging.info(
            f"Queue state: {len(queue_state['pending_queue'])}, {len(queue_state['completed_ids'])} completed"
        )

    # Get all videos from queue (main loop will handle limiting)
    videos_to_process = []
    updated_queue_state = queue_state.copy()
    
    # Get all videos from pending queue
    while queue_state['pending_queue']:
        next_video_id, updated_queue_state = _get_next_video_from_queue(updated_queue_state)
        if next_video_id:
            videos_to_process.append(next_video_id)
        else:
            break
    
    return videos_to_process, updated_queue_state, video_to_channel if 'video_to_channel' in locals() else {}
def is_video_completed(video):
    """Check if video analysis is completed (posted to Discord)"""
    logging.info("=== FUNCTION START: is_video_completed ===")
    return video.get("sentToDiscord") is True


def _extract_video_data(video_entry):
    """Extract video ID and title from entry"""
    logging.info("=== FUNCTION START: _extract_video_data ===")
    v_id = video_entry.get("video_id", "")
    title = video_entry.get("title", "Unknown Title")
    return v_id, title

def _get_video_date(video_entry):
    """Handle video date extraction and formatting"""
    logging.info("=== FUNCTION START: _get_video_date ===")
    video_date = UNKNOWN_DATE
    
    # Use publication_date if available, otherwise fall back to analysis_date
    if (
        "publication_date" in video_entry
        and video_entry["publication_date"] != UNKNOWN_DATE
    ):
        video_date = video_entry["publication_date"]
    elif "analysis_date" in video_entry:
        # Convert from YYYY-MM-DD HH:MM:SS to YYYY-MM-DD
        try:
            from datetime import datetime
            date_obj = datetime.strptime(
                video_entry["analysis_date"], "%Y-%m-%d %H:%M:%S"
            )
            video_date = date_obj.strftime("%Y-%m-%d")
        except (ValueError, TypeError):
            video_date = UNKNOWN_DATE
    
    return video_date

def _get_channel_videos_count(channel_name):
    """Get videos count for channel"""
    logging.info("=== FUNCTION START: _get_channel_videos_count ===")
    analysis_stats = load_analysis_stats()
    
    # Use channel name directly for lookup
    channel_key = channel_name
    
    channel_data = analysis_stats.get(channel_key, {"videos": []})
    return len(channel_data["videos"])

def _determine_video_word_form(videos_count):
    """Determine singular/plural form of video word"""
    logging.info("=== FUNCTION START: _determine_video_word_form ===")
    return "video" if videos_count == 1 else "videor"

def _add_embed_fields(embed, video_entry, channel_name, video_date, title, v_id):
    """Add standard fields to Discord embed"""
    logging.info("=== FUNCTION START: _add_embed_fields ===")
    embed["fields"] = [
        {
            "name": PROMPTS["channel_field"],
            "value": channel_name,
            "inline": True,
        },
        {
            "name": "Like-ratio",
            "value": f"{video_entry.get('video_stats', {}).get('like_ratio', 0):.1f}%",
            "inline": True,
        },
        {"name": "Publicerad", "value": video_date, "inline": True},
        {
            "name": PROMPTS["video_field"],
            "value": f"[{title}](https://www.youtube.com/watch?v={v_id})",
            "inline": False,
        },
    ]

def _add_analysis_description(embed, video_entry):
    """Add AI analysis description if available"""
    logging.info("=== FUNCTION START: _add_analysis_description ===")
    if "analyses" in video_entry and "ai_analysis" in video_entry["analyses"]:
        analysis_output = video_entry["analyses"]["ai_analysis"].get("output", "")
        if analysis_output:
            # Truncate if too long for Discord (2000 char limit per field)
            if len(analysis_output) > 1900:
                analysis_output = analysis_output[:1900] + "..."
            embed["description"] = analysis_output

def _create_embed_footer(videos_count, video_word, channel_name):
    """Create embed footer with video count"""
    logging.info("=== FUNCTION START: _create_embed_footer ===")
    return {
        "text": f"{videos_count} {video_word} från @{channel_name} analyserade"
    }

def _build_discord_embed(video_entry, channel_name, video_date, title, v_id, videos_count, video_word):
    """Build the main Discord embed structure"""
    logging.info("=== FUNCTION START: _build_discord_embed ===")
    embed = {
        "title": PROMPTS["discord_title"].format(title=title),
        "color": get_gradient_color(
            video_entry.get("video_stats", {}).get("like_ratio", 50)
        ),
        "image": {"url": get_thumbnail_url(v_id)},
    }
    
    # Add fields
    _add_embed_fields(embed, video_entry, channel_name, video_date, title, v_id)
    
    # Add analysis description if available
    _add_analysis_description(embed, video_entry)
    
    # Add footer
    embed["footer"] = _create_embed_footer(videos_count, video_word, channel_name)
    
    return embed

def _send_discord_request(embed):
    """Handle the actual Discord API request"""
    logging.info("=== FUNCTION START: _send_discord_request ===")
    payload = {"embeds": [embed]}
    
    # Log the complete payload nicely formatted
    # logging.info("=== DISCORD PAYLOAD ===")
    # logging.info("Payload:\n" + json.dumps(payload, indent=2, ensure_ascii=False))
    # logging.info("=== END DISCORD PAYLOAD ===")
    
    return requests.post(WEBHOOK, json=payload, timeout=30)

def _handle_discord_response(response, video_entry):
    """Process Discord API response"""
    logging.info("=== FUNCTION START: _handle_discord_response ===")
    if response.status_code == 204:
        logging.info(
            f"Successfully posted analysis for video {video_entry.get('video_id')} to Discord"
        )
        return True
    else:
        logging.error(
            f"Failed to post to Discord: HTTP {response.status_code} - {response.text}"
        )
        return False


def send_discord_message(video_entry, channel_name):
    """Send analysis results to Discord webhook"""
    logging.info("=== FUNCTION START: send_discord_message ===")
    if not WEBHOOK:
        logging.warning("Discord webhook not configured - skipping Discord post")
        return False

    try:
        # Extract video data
        v_id, title = _extract_video_data(video_entry)
        
        # Get video date
        video_date = _get_video_date(video_entry)
        
        # Get channel statistics
        videos_count = _get_channel_videos_count(channel_name)
        video_word = _determine_video_word_form(videos_count)
        
        # Build Discord embed
        embed = _build_discord_embed(
            video_entry, channel_name, video_date, title, v_id, videos_count, video_word
        )
        
        # Send to Discord
        response = _send_discord_request(embed)
        
        # Handle response
        return _handle_discord_response(response, video_entry)

    except Exception as e:
        logging.error(f"Error posting to Discord: {e}")
        return False


def load_queue_state():
    """Load queue state from analysis_stats.json with fallback defaults"""
    # logging.info("=== FUNCTION START: load_queue_state ===")
    try:
        analysis_stats = load_analysis_stats()
        queue_state = analysis_stats.get(
            "_queue_state",
            {
                "pending_queue": [],
                "completed_ids": [],
                "fetch_depth": 1,
                "current_processing": None,
            },
        )
        # logging.info(f"Loaded queue state from analysis_stats.json")
        return queue_state
    except FileNotFoundError:
        logging.info("DATA CHANGE: Creating new queue state in analysis_stats.json")
        return {
            "pending_queue": [],
            "completed_ids": [],
            "fetch_depth": 1,
            "current_processing": None,
        }
    except (json.JSONDecodeError, IOError) as e:
        logging.warning(f"Failed to load queue state: {e}. Creating fallback.")
        return {
            "pending_queue": [],
            "completed_ids": [],
            "fetch_depth": 1,
            "current_processing": None,
        }


def save_queue_state(state):
    """Save queue state to analysis_stats.json with retry mechanism"""
    logging.info("=== FUNCTION START: save_queue_state ===")
    try:
        # Load existing analysis stats
        analysis_stats = load_analysis_stats()

        # Update queue state
        analysis_stats["_queue_state"] = state

        # Log what we're saving
        logging.info(f"Saving queue state with {len(state.get('completed_ids', []))} completed IDs")
        logging.info(f"Completed IDs: {state.get('completed_ids', [])}")
        logging.info(f"Pending queue: {state.get('pending_queue', [])}")

        # Save the combined data
        save_analysis_stats(analysis_stats)
        logging.info("Saved queue state to analysis_stats.json")
    except (IOError, OSError) as e:
        logging.error(f"Failed to save queue state: {e}")


def add_to_pending_queue(state, video_ids):
    """Add new videos to pending queue avoiding duplicates"""
    logging.info("=== FUNCTION START: add_to_pending_queue ===")
    added_count = 0
    for v_id in video_ids:
        if v_id not in state["completed_ids"] and v_id not in state["pending_queue"]:
            state["pending_queue"].append(v_id)
            added_count += 1
    logging.info(f"Added {added_count} new videos to pending queue")
    return state


def mark_video_completed(state, video_id):
    """Mark video as completed and remove from processing"""
    logging.info("=== FUNCTION START: mark_video_completed ===")
    state["completed_ids"].append(video_id)
    if video_id in state["pending_queue"]:
        state["pending_queue"].remove(video_id)
    state["current_processing"] = None
    logging.info(f"Marked video {video_id} as completed")


def pop_next_video(state):
    """Get next video from queue (FIFO)"""
    logging.info("=== FUNCTION START: pop_next_video ===")
    if state["pending_queue"]:
        video_id = state["pending_queue"].pop(0)  # FIFO - take first
        state["current_processing"] = video_id
        logging.info(f"Popped video {video_id} from queue for processing")
        return video_id
    return False


def _create_fallback_stats():
    """Create fallback stats structure"""
    logging.info("=== FUNCTION START: _create_fallback_stats ===")
    return {channel: {"videos": []} for channel in CHANNELS}

def _load_stats_file_with_retry():
    """Handle file loading with retry mechanism"""
    logging.info("=== FUNCTION START: _load_stats_file_with_retry ===")
    max_retries = 3
    
    for attempt in range(max_retries):
        try:
            with open(ANALYSIS_STATS_FILE, "r", encoding="utf-8") as f:
                data = json.load(f)
                logging.info(f"Loaded analysis stats from {ANALYSIS_STATS_FILE}")
                return data, None
        except FileNotFoundError:
            logging.info(
                f"DATA CHANGE: Creating new analysis stats file at {ANALYSIS_STATS_FILE}"
            )
            return _create_fallback_stats(), None
        except (json.JSONDecodeError, IOError) as e:
            if attempt == max_retries - 1:
                logging.warning(
                    f"Failed to load analysis stats after {max_retries} attempts: {e}"
                )
                logging.info("DATA CHANGE: Creating fallback analysis stats structure")
                return _create_fallback_stats(), "max_retries_exceeded"
            logging.debug(f"Retry {attempt + 1} for loading analysis stats: {e}")
            time.sleep(0.1 * (attempt + 1))
    
    return _create_fallback_stats(), "unknown_error"

def _validate_channel_structure(data, channel):
    """Check if channel has proper structure"""
    logging.info("=== FUNCTION START: _validate_channel_structure ===")
    channel_key = channel
    return channel_key in data and "videos" in data[channel_key]

def _ensure_channel_exists(data, channel):
    """Ensure channel exists in data"""
    logging.info("=== FUNCTION START: _ensure_channel_exists ===")
    channel_key = channel
    if channel_key not in data:
        logging.info(
            f"DATA CHANGE: Adding new channel '{channel}' to analysis stats (key: '{channel_key}')"
        )
        data[channel_key] = {"videos": []}
        return True
    return False

def _create_migrated_video_entry(old_data):
    """Create migrated video entry"""
    logging.info("=== FUNCTION START: _create_migrated_video_entry ===")
    return {
        "video_id": old_data.get("last_video_id", ""),
        "title": "Migrated Analysis",
        "analysis_date": time.time(),
        "analyses": {
            "comment_review": {
                "input_prompt": old_data.get("last_prompt", ""),
                "output": "[Migrated output]",
                "model": old_data.get("last_model", ""),
                "timestamp": old_data.get("last_checked", time.time()),
            }
        },
    }

def _perform_channel_migration(data, channel):
    """Perform complete channel migration from old to new structure"""
    logging.info("=== FUNCTION START: _perform_channel_migration ===")
    old_data = data[channel]
    old_video_count = len(old_data.get("videos", []))
    
    # Use channel name directly
    channel_key = channel
    
    # Create new structure under channel key
    data[channel_key] = {"videos": []}
    
    # Migrate existing videos if any
    if old_video_count > 0:
        for video in old_data["videos"]:
            # Add channel_name to video if not present
            if "channel_name" not in video:
                video["channel_name"] = channel
            data[channel_key]["videos"].append(video)
        logging.info(
            f"DATA CHANGE: Migrated {old_video_count} videos from '{channel}' to key '{channel_key}'"
        )
    
    # Remove old channel key
    del data[channel]
    
    logging.info(
        f"DATA REPLACEMENT COMPLETE: Migrated channel '{channel}' to '{channel_key}'"
    )

def _process_channel_structure(data, channel):
    """Process and validate channel structure, migrating if needed"""
    logging.info("=== FUNCTION START: _process_channel_structure ===")
    
    # Ensure channel exists
    _ensure_channel_exists(data, channel)
    
    # Check if migration is needed
    if not _validate_channel_structure(data, channel):
        logging.warning(
            f"DATA REPLACEMENT: Migrating old structure for channel '{channel}'"
        )
        _perform_channel_migration(data, channel)

def _migrate_old_channel_keys(data):
    """Function kept for compatibility but no longer needed as we use channel names directly"""
    logging.info("=== FUNCTION START: _migrate_old_channel_keys ===")
    # No migration needed as we use channel names directly
    pass


def load_analysis_stats():
    """Load analysis statistics from JSON file with simple retry mechanism"""
    logging.info("=== FUNCTION START: load_analysis_stats ===")
    data, error = _load_stats_file_with_retry()
    if error:
        logging.info(f"Failed to load analysis stats: {error}")
    
    return data


def save_analysis_stats(stats):
    """Save analysis statistics to JSON file with retry mechanism"""
    logging.info("=== FUNCTION START: save_analysis_stats ===")
    max_retries = 3
    for attempt in range(max_retries):
        try:
            with open(ANALYSIS_STATS_FILE, "w", encoding="utf-8") as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            return
        except (IOError, OSError) as e:
            if attempt == max_retries - 1:
                logging.error(
                    f"Failed to save analysis stats after {max_retries} attempts: {e}"
                )
                return
            logging.debug(f"Retry {attempt + 1} for saving analysis stats: {e}")
            time.sleep(0.1 * (attempt + 1))


def format_timestamp(timestamp):
    """Convert Unix timestamp to human-readable ISO 8601 format"""
    if timestamp is None:
        return None
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d")


def find_or_create_video(
    stats, channel_name, video_id, title, publication_date=UNKNOWN_DATE
):
    """Find existing video or create new entry in the videos array"""
    logging.info("=== FUNCTION START: find_or_create_video ===")
    
    channel_data = stats.get(channel_name, {"videos": []})

    # Find existing video
    for video in channel_data["videos"]:
        if video["video_id"] == video_id:
            return video

    # Create new video entry with human-readable date
    current_time = time.time()
    new_video = {
        "video_id": video_id,
        "title": title,
        "channel_name": channel_name,  # Store actual channel name
        "analysis_date": format_timestamp(current_time),
        "analysis_timestamp": current_time,
        "publication_date": publication_date
        if publication_date != UNKNOWN_DATE
        else format_timestamp(current_time),
        "analyses": {},
    }
    channel_data["videos"].append(new_video)
    logging.info(
        f"DATA CHANGE: Created new video entry for {video_id} from channel '{channel_name}'"
    )
    return new_video


def add_analysis_to_video(video, analysis_type, input_prompt, output, model):
    """Add analysis entry to a video's analyses"""
    logging.info("=== FUNCTION START: add_analysis_to_video ===")
    if "analyses" not in video:
        video["analyses"] = {}

    current_time = time.time()
    video["analyses"][analysis_type] = {
        "input_prompt": input_prompt,
        "output": output,
        "model": model,
        "timestamp": format_timestamp(current_time),
    }
    logging.info(
        f"DATA CHANGE: Added {analysis_type} analysis to video {video.get('video_id', 'unknown')}"
    )


# Constants for browser interactions
SCROLL_SCRIPT = "document.scrollingElement.scrollTop = document.scrollingElement.scrollHeight"

# Constants for date parsing to avoid duplication
SWEDISH_DATE_PATTERN = r"(\d{1,2})\s+(januari|februari|mars|april|maj|juni|juli|augusti|september|oktober|november|december)\s+(\d{4})"
ENGLISH_DATE_PATTERN = r"(\d{1,2})\s+(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s+(\d{4})"

# Swedish months mapping
SWEDISH_MONTHS = {
    "januari": "01",
    "februari": "02",
    "mars": "03",
    "april": "04",
    "maj": "05",
    "juni": "06",
    "juli": "07",
    "augusti": "08",
    "september": "09",
    "oktober": "10",
    "november": "11",
    "december": "12",
}

# English months mapping
ENGLISH_MONTHS = {
    "jan": "01",
    "feb": "02",
    "mar": "03",
    "apr": "04",
    "may": "05",
    "jun": "06",
    "jul": "07",
    "aug": "08",
    "sep": "09",
    "oct": "10",
    "nov": "11",
    "dec": "12",
}

# Private video indicators
PRIVATE_INDICATORS = [
    "This video is private",
    "Private video",
    "Video unavailable",
    "This video is not available",
]

# Members-only content indicators
MEMBERS_INDICATORS = [
    "Join this channel to get access to members-only content like this video",
    "Become a member to watch this video",
    "This video is only available to members",
]


def _is_private_video(page_title):
    """Check if video is private based on title"""
    page_title_lower = page_title.lower()
    return any(
        indicator.lower() == page_title_lower
        or indicator.lower() in page_title_lower
        and "private" in page_title_lower
        for indicator in PRIVATE_INDICATORS
    )


def _is_members_only_content(page_content):
    """Check if content is members-only based on page content"""
    page_content_lower = page_content.lower()
    return any(indicator.lower() in page_content_lower for indicator in MEMBERS_INDICATORS)


def _get_members_only_response():
    """Return standard members-only response tuple"""
    return (
        "MEMBERS_ONLY",
        None,
        None,
        None,
        None,
        None,
        None,
        UNKNOWN_DATE,
        None,
        None,
        None,
    )


def _parse_swedish_date(day, month_name, year):
    """Parse Swedish date format"""
    month_num = SWEDISH_MONTHS.get(month_name.lower(), "01")
    if len(year) == 2:
        year = f"20{year}"
    return f"{year}-{month_num}-{day.zfill(2)}"


def _parse_english_date(month_abbr, day, year):
    """Parse English abbreviation date format"""
    month_num = ENGLISH_MONTHS.get(month_abbr.lower(), "01")
    if len(year) == 2:
        year = f"20{year}"
    return f"{year}-{month_num}-{day.zfill(2)}"


def _parse_date_from_text(date_text):
    """Parse various date formats from text"""
    import re
    
    date_patterns = [
        SWEDISH_DATE_PATTERN,
        ENGLISH_DATE_PATTERN,
        r"(\d{4})-(\d{2})-(\d{2})",
        r"(\d{4})/(\d{2})/(\d{2})",
        r"(\d{1,2})\s+(\d{1,2})\s+(\d{4})",
        r"([A-Za-z]{3})\s+(\d{1,2}),?\s+(\d{4})",
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, date_text.lower())
        if match:
            groups = match.groups()
            if len(groups) == 3:
                # Check if this is English abbreviation format
                if any(month_abbr in date_text.lower() for month_abbr in ENGLISH_MONTHS.keys()):
                    month_abbr, day, year = groups
                    return _parse_english_date(month_abbr, day, year)
                # Check if this is Swedish format
                elif any(month_name in date_text.lower() for month_name in SWEDISH_MONTHS.keys()):
                    day, month_name, year = groups
                    return _parse_swedish_date(day, month_name, year)
                else:  # YYYY-MM-DD format
                    year, month, day = groups
                    return f"{year}-{month}-{day}"
    return None


def _search_dates_in_page_content(page_content):
    """Search for date patterns in page content"""
    import re
    
    content_patterns = [
        r'"publishDate":"(\d{4}-\d{2}-\d{2})"',
        r'"uploadDate":"(\d{4}-\d{2}-\d{2})"',
        SWEDISH_DATE_PATTERN,
    ]
    
    for pattern in content_patterns:
        matches = re.findall(pattern, page_content)
        if matches:
            for match in matches:
                if isinstance(match, tuple):
                    day, month_name, year = match
                    return _parse_swedish_date(day, month_name, year)
                else:
                    return match
    return None


def _extract_video_title(page):
    """Extract video title from page"""
    title_elem = page.locator("h1.ytd-watch-metadata yt-formatted-string")
    return (
        title_elem.text_content().strip()
        if title_elem.count() > 0
        else "Unknown"
    )


def _extract_channel_name(page, v_id):
    """Extract channel name with fallback logic"""
    try:
        channel_selectors = [
            "ytd-channel-name a.yt-simple-endpoint",
            "ytd-video-owner-renderer a.yt-simple-endpoint",
            "ytd-channel-name #channel-name",
            "ytd-video-owner-renderer #channel-name",
            "#owner #channel-name a",
            ".ytd-channel-name a",
        ]

        for selector in channel_selectors:
            try:
                channel_elem = page.locator(selector)
                if channel_elem.count() > 0:
                    channel_text = channel_elem.first.text_content().strip()
                    if channel_text and channel_text != "Unknown":
                        logging.info(
                            f"Extracted channel name '{channel_text}' for video {v_id}"
                        )
                        return channel_text
            except Exception:
                continue

        # Fallback: try to get channel handle from URL
        channel_url_elem = page.locator("ytd-channel-name a.yt-simple-endpoint")
        if channel_url_elem.count() > 0:
            href = channel_url_elem.first.get_attribute("href")
            if href and "@" in href:
                handle = href.split("@")[-1].split("/")[0]
                if handle:
                    channel_name = f"@{handle}"
                    logging.info(
                        f"Extracted channel handle '{channel_name}' from URL for video {v_id}"
                    )
                    return channel_name
    except Exception as e:
        logging.warning(f"Failed to extract channel name for {v_id}: {e}")

    logging.warning(
        f"Could not extract channel name for video {v_id}, will use fallback"
    )
    return None


def _extract_publication_date(page):
    """Extract publication date from page metadata"""
    try:
        date_selectors = [
            "#description-inner yt-formatted-string.ytd-video-secondary-info-renderer",
            "#info-strings yt-formatted-string.ytd-video-secondary-info-renderer",
            ".ytd-video-secondary-info-renderer yt-formatted-string",
            "#info-text",
            ".ytd-video-primary-info-renderer .ytd-simple-timestamp-renderer",
            "ytd-video-view-model-renderer .ytd-simple-timestamp-renderer",
            "ytd-metadata-row-renderer .ytd-simple-timestamp-renderer",
            ".ytd-simple-timestamp-renderer",
            "#meta-contents ytd-video-secondary-info-renderer yt-formatted-string",
            "span.ytd-video-secondary-info-renderer",
        ]

        for selector in date_selectors:
            try:
                date_elem = page.locator(selector)
                if date_elem.count() > 0:
                    date_text = date_elem.first.text_content().strip()
                    parsed_date = _parse_date_from_text(date_text)
                    if parsed_date:
                        logging.info(f"Extracted YouTube publication date: {parsed_date}")
                        return parsed_date
            except Exception as e:
                logging.debug(f"Selector '{selector}' failed: {e}")
                continue

        # If still not found, try searching page content
        logging.info("Searching page content for date patterns...")
        page_content = page.content()
        parsed_date = _search_dates_in_page_content(page_content)
        if parsed_date:
            logging.info(f"Found date in page content: {parsed_date}")
            return parsed_date

    except Exception as e:
        logging.warning(f"Failed to extract publication date: {e}")

    return UNKNOWN_DATE


def _is_video_old_enough(publication_date):
    """Check if video is at least 24 hours old"""
    if publication_date == UNKNOWN_DATE:
        # If we can't determine the date, assume it's old enough
        logging.warning("Could not determine publication date, proceeding with analysis")
        return True
    
    try:
        from datetime import datetime
        today = datetime.now().date()
        pub_date = datetime.strptime(publication_date, "%Y-%m-%d").date()
        
        # Calculate the time difference
        time_diff = today - pub_date
        
        # Check if at least 24 hours have passed (1 day)
        is_old_enough = time_diff.days >= 1
        
        if not is_old_enough:
            logging.info(f"Video too new (published {publication_date}, {time_diff.days} days ago). Skipping analysis.")
        else:
            logging.info(f"Video old enough (published {publication_date}, {time_diff.days} days ago). Proceeding with analysis.")
            
        return is_old_enough
        
    except Exception as e:
        logging.warning(f"Error parsing publication date {publication_date}: {e}. Proceeding with analysis.")
        return True


# def _extract_comment_count(page):
#     """Extract UI comment count from page"""
#     ui_count = 0
#     count_locators = [
#         ("#count .yt-core-attributed-string", "yt-core-attributed-string"),
#         ("#count yt-formatted-string", "yt-formatted-string"),
#         (".ytd-comments-header-renderer #count", "yt-core-attributed-string")
#     ]
# 
#     for selector, fallback_class in count_locators:
#         try:
#             count_element = page.locator(selector).first
#             if count_element.count() > 0:
#                 count_text = count_element.inner_text()
#                 # Extract number from text like "123 comments" or "1.2K comments"
#                 import re
#                 numbers = re.findall(r'[\d,]+', count_text)
#                 if numbers:
#                     number_str = numbers[0].replace(',', '')
#                     if 'K' in count_text.upper():
#                         ui_count = int(float(number_str) * 1000)
#                     else:
#                         ui_count = int(number_str)
#                 break
#         except Exception as e:
#             logging.debug(f"Comment count extraction failed with selector {selector}: {e}")
#             continue
# 
#     return ui_count


def _setup_youtube_browser():
    """Setup Playwright browser and context for YouTube"""
    user_agent = random.choice(USER_AGENTS)
    p = sync_playwright().start()
    browser = p.chromium.launch(headless=True)
    context = browser.new_context(
        user_agent=user_agent,
        viewport={"width": 1920, "height": 1080},
        locale="en-US",
        extra_http_headers={"Accept-Language": "en-US,en;q=0.9"},
    )
    page = context.new_page()
    return browser, page


# def _sort_comments_newest(page):
#     """Sort comments to newest first"""
#     try:
#         page.evaluate("""() => {
#             const btn = document.querySelector('ytd-comments-header-renderer #sort-menu');
#             if (btn) btn.click();
#         """)
#         page.wait_for_timeout(1000)
# 
#         page.evaluate("""() => {
#             const items = document.querySelectorAll('ytd-menu-service-item-renderer');
#             if (items.length > 1) items[1].click();
#         """)
#         page.wait_for_timeout(3000)
#         logging.info("Comments sorted to newest first")
#     except Exception as e:
#         logging.warning(
#             f"Failed to sort comments: {e}. Proceeding with default sort."
#         )


# def _scroll_to_load_comments(page):
#     """Scroll to load all top-level comment threads"""
#     last_thread_count = 0
#     no_change = 0
#     
#     while True:
#         thread_nodes = page.locator("ytd-comment-thread-renderer")
#         current_thread = thread_nodes.count()
#         if current_thread == last_thread_count:
#             no_change += 1
#             if no_change >= 3:
#                 break
#         else:
#             no_change = 0
#             last_thread_count = current_thread
#         page.evaluate(SCROLL_SCRIPT)
#         page.wait_for_timeout(5000)


# def _expand_comment_replies(page):
#     """Expand all comment replies using JavaScript"""
#     max_iterations = 3
#     
#     for i in range(max_iterations):
#         try:
#             clicks_dispatched = page.evaluate("""() => {
#                 const buttons = Array.from(document.querySelectorAll('ytd-button-renderer#more-replies button'));
#                 let count = 0;
#                 for (let btn of buttons) {
#                     if (btn.offsetParent !== null) { 
#                         btn.click();
#                         count++;
#                     }
#                 }
#                 return count;
#             """)
# 
#             if clicks_dispatched == 0:
#                 break
# 
#             page.wait_for_timeout(4000)
#             page.evaluate(SCROLL_SCRIPT)
#             page.wait_for_timeout(2000)
# 
#         except Exception as e:
#             logging.warning(
#                 f"Iteration {i + 1} JS click failed: {str(e).splitlines()[0]}"
#             )
#             break


def _is_restricted_content(video_stats, title):
    """Check if video shows signs of being restricted content"""
    return (
        video_stats["likes"] == 0
        and video_stats["dislikes"] == 0
        and video_stats["like_ratio"] < 0.001
        and title != "Unknown"
    )


def _handle_scrape_error(e, v_id):
    """Handle scraping errors and return appropriate response"""
    logging.error(f"Scrape failed for {v_id}: {e}")
    if "not enough values to unpack" in str(e):
        logging.error(
            "Unpacking error details: Expected 4 values (transcript_text, ai_analysis, ai_model, transcription_model) but got fewer"
        )
        logging.error(
            "This usually means get_transcript_and_analysis() returned wrong number of values"
        )
    return (
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        UNKNOWN_DATE,
        None,
        None,
        None,
    )


def _scrape_video_metadata(page, v_id):
    """Extract all video metadata from page"""
    title = _extract_video_title(page)
    channel_name = _extract_channel_name(page, v_id)
    publication_date = _extract_publication_date(page)
    ui_count = 0  # Comment: Comment count extraction disabled
    
    return {
        "title": title,
        "channel_name": channel_name,
        "publication_date": publication_date,
        "ui_count": ui_count,
        "extracted_channel_name": channel_name
    }


def _perform_deep_scrape(page, v_id, title, channel_name, publication_date, deep_scrape):
    """Perform deep scraping of comments and transcript"""
    ui_count = 0  # Comment: Comment count extraction disabled
    comments = {}  # Comment: Comment extraction disabled
    
    if deep_scrape:
        logging.info(
            f"Starting scrape for '{title}'. Comment analysis disabled."
        )

        # Comment: Comment sorting and extraction disabled
        # _sort_comments_newest(page)
        # _scroll_to_load_comments(page)
        # _expand_comment_replies(page)

        logging.info(
            "Comment extraction disabled - skipping comment analysis."
        )

        # Get transcript and analysis
        transcript_text, ai_analysis, ai_model, transcription_model = (
            get_transcript_and_analysis(
                v_id, title, channel_name, publication_date
            )
        )

        return (
            title,
            channel_name,
            ui_count,
            comments,
            get_video_stats(v_id),
            transcript_text,
            ai_analysis,
            ai_model,
            publication_date,
            channel_name,
            transcription_model,
        )
    
    # Basic response for non-deep scrape
    return (
        title,
        channel_name,
        ui_count,
        {},
        get_video_stats(v_id),
        None,
        None,
        None,
        publication_date,
        channel_name,
        None,
    )


def get_yt_data(v_id, deep_scrape=False):
    """Extract YouTube video data with reduced cognitive complexity"""
    logging.info("=== FUNCTION START: get_yt_data ===")
    
    browser, page = _setup_youtube_browser()
    
    try:
        # Navigate to video page
        page.goto(f"https://www.youtube.com/watch?v={v_id}", timeout=60000)
        page.wait_for_load_state("networkidle")
        page.evaluate("window.scrollBy(0, 800)")

        # Check video accessibility
        page_content = page.content()
        page_title = page.title()
        
        if _is_private_video(page_title):
            logging.warning(f"Private video detected for {v_id}: {page_title}")
            return _get_members_only_response()
            
        if _is_members_only_content(page_content):
            logging.warning(f"Members-only content detected for {v_id}")
            return _get_members_only_response()

        # Wait for comments section
        try:
            page.wait_for_selector("ytd-comments#comments", state="attached", timeout=15000)
            page.wait_for_timeout(3000)
        except TimeoutError:
            logging.warning("Comments section did not attach in time. Video might have comments disabled.")
            return _get_basic_response(None, None, None, get_video_stats(v_id))

        # Extract video metadata
        metadata = _scrape_video_metadata(page, v_id)
        
        # Get video stats for early detection
        video_stats = get_video_stats(v_id)
        
        # Early detection for restricted content
        if _is_restricted_content(video_stats, metadata["title"]):
            logging.warning(f"Early detection: Video {v_id} shows signs of members-only/private content")
            return _get_members_only_response()

        # Perform deep scrape or basic extraction
        return _perform_deep_scrape(
            page, v_id, metadata["title"], metadata["channel_name"], 
            metadata["publication_date"], deep_scrape
        )
        
    except Exception as e:
        return _handle_scrape_error(e, v_id)
    finally:
        browser.close()


def _get_basic_response(title, channel_name, ui_count, video_stats):
    """Return basic response tuple for non-deep scrape"""
    return (
        title,
        channel_name,
        ui_count,
        {},
        video_stats or {"likes": 0, "dislikes": 0, "views": 0, "like_ratio": 0},
        None,
        None,
        None,
        UNKNOWN_DATE,
        channel_name,
        None,
    )


def transcribe_with_assemblyai(audio_filepath, v_id):
    """Transcribe audio using AssemblyAI with automatic polling"""
    logging.info("=== FUNCTION START: transcribe_with_assemblyai ===")
    if not ASSEMBLYAI_API_KEY:
        logging.error("AssemblyAI API key not available")
        # ... (rest of the function remains the same)
        return None

    try:
        logging.info(f"Starting AssemblyAI transcription for {v_id}...")

        # Log AssemblyAI input
        logging.info("=== ASSEMBLYAI API INPUT ===")
        logging.info("Model: universal")
        logging.info("Language: sv")
        logging.info(f"File: {os.path.basename(audio_filepath)}")
        logging.info(f"File size: {os.path.getsize(audio_filepath):,} bytes")
        logging.info("Punctuation: True")
        logging.info("Format text: True")
        logging.info("=== END ASSEMBLYAI API INPUT ===")

        aai.settings.api_key = ASSEMBLYAI_API_KEY
        transcriber = aai.Transcriber()

        # Using your correct, updated parameter
        config = aai.TranscriptionConfig(
            language_code="sv",
            punctuate=True,
            format_text=True,
            speech_models=["universal"],  # Array of fallback models
        )

        # This single call safely handles the upload and the polling loop for us
        transcript = transcriber.transcribe(audio_filepath, config=config)

        if transcript.status == aai.TranscriptStatus.completed:
            # Log AssemblyAI output (truncated for readability)
            logging.info("=== ASSEMBLYAI API OUTPUT ===")
            logging.info("Model: universal")
            logging.info(f"Status: {transcript.status}")
            logging.info(f"Output length: {len(transcript.text)} characters")
            
            # Show first 500 and last 500 characters for debugging
            if len(transcript.text) > 1000:
                first_500 = transcript.text[:500]
                last_500 = transcript.text[-500:]
                truncated_output = f"{first_500}\n...[truncated]...\n{last_500}"
            else:
                truncated_output = transcript.text
            
            logging.info(
                f"Output:\n{json.dumps(truncated_output, indent=2, ensure_ascii=False)}"
            )
            logging.info("=== END ASSEMBLYAI API OUTPUT ===")

            logging.info(
                f"AssemblyAI successful for {v_id} ({len(transcript.text)} chars)"
            )
            return transcript.text
        else:
            logging.error(f"AssemblyAI failed for {v_id}: {transcript.error}")
            return None

    except Exception as e:
        logging.error(f"Unexpected error during AssemblyAI transcription: {str(e)}")
        return None


def transcribe_with_gemini_audio(v_id):
    """Disabled to prevent billing/rate limit issues - using AssemblyAI/Whisper instead"""
    logging.info(f"Gemini audio transcription disabled by config for {v_id}")
    return None


def _check_ffmpeg_availability():
    """Check for FFmpeg availability in expected locations"""
    ffmpeg_available = False
    ffmpeg_dir = None
    
    # Define possible FFmpeg locations in priority order
    current_dir = os.getcwd()
    ffmpeg_locations = [
        current_dir,
        os.path.join(current_dir, "ffmpeg_temp", "ffmpeg-8.0.1-essentials_build", "bin"),
    ]
    
    for location in ffmpeg_locations:
        local_ffmpeg = os.path.join(location, "ffmpeg.exe")
        local_ffprobe = os.path.join(location, "ffprobe.exe")
        
        ffmpeg_exists = os.path.exists(local_ffmpeg)
        ffprobe_exists = os.path.exists(local_ffprobe)
        
        if ffmpeg_exists and ffprobe_exists:
            ffmpeg_available = True
            ffmpeg_dir = location
            break
    
    if not ffmpeg_available:
        logging.warning("FFmpeg files not found in any expected locations")
        logging.info("To fix this issue, you can:")
        logging.info("1. Download FFmpeg from https://ffmpeg.org/download.html")
        logging.info("2. Extract ffmpeg.exe and ffprobe.exe to project root directory")
        logging.info("3. Or ensure they are available in your system PATH")
    
    return ffmpeg_dir, ffmpeg_available


def _get_compression_settings(compression_level):
    """Get compression settings based on compression level"""
    base_settings = {
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "m4a",
            }
        ],
        "postprocessor_args": [
            "-ar", "16000",  # 16kHz sample rate (optimal for Whisper)
            "-ac", "1",  # Mono audio (halves file size)
            "-c:a", "aac",  # Explicit AAC codec for better compression
        ],
    }
    
    if compression_level == "standard":
        base_settings["postprocessor_args"].extend(["-b:a", "16k"])
        base_settings["postprocessors"][0]["preferredquality"] = "16"
    elif compression_level == "balanced":
        base_settings["postprocessor_args"].extend(["-b:a", "8k", "-compression_level", "7"])
        base_settings["postprocessors"][0]["preferredquality"] = "8"
    elif compression_level == "ultra_aggressive":
        base_settings["postprocessor_args"].extend(["-b:a", "4k", "-compression_level", "10"])
        # Remove preferredquality for ultra_aggressive to use default
    
    return base_settings


def _get_ffmpeg_download_options(ffmpeg_available, ffmpeg_dir):
    """Return appropriate download options based on FFmpeg availability"""
    if ffmpeg_available:
        compression_settings = _get_compression_settings("standard")
        return {
            "format": WORST_AUDIO_FORMAT,
            "outtmpl": None,  # Will be set by caller
            "quiet": True,
            "no_warnings": True,
            "extract_audio": True,
            "keepvideo": False,
            "nopart": True,
            "noprogress": True,
            "ffmpeg_location": ffmpeg_dir,
            "postprocessors": compression_settings["postprocessors"],
            "postprocessor_args": compression_settings["postprocessor_args"],
        }
    else:
        # Fallback to basic extraction without compression
        return {
            "format": "bestaudio/best",
            "outtmpl": None,  # Will be set by caller
            "quiet": True,
            "no_warnings": True,
            "extract_audio": True,
            "keepvideo": False,
            "nopart": True,
            "noprogress": True,
        }


def _setup_temp_audio_file():
    """Create secure temporary file for audio download"""
    with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as temp_file:
        return temp_file.name


def _download_audio_file(v_id, audio_filepath, ydl_opts):
    """Download audio file using youtube-dl options"""
    logging.info(f"Downloading audio for {v_id}...")
    ydl_opts["outtmpl"] = audio_filepath  # Set output template
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([f"https://www.youtube.com/watch?v={v_id}"])


def _validate_and_fix_audio_path(audio_filepath):
    """Fix audio file path if yt-dlp changed extension"""
    if not os.path.exists(audio_filepath):
        possible_files = [
            f for f in os.listdir(os.path.dirname(audio_filepath))
            if f.startswith(os.path.basename(audio_filepath))
        ]
        if possible_files:
            return os.path.join(os.path.dirname(audio_filepath), possible_files[0])
    return audio_filepath


def _analyze_file_size(audio_filepath, v_id):
    """Analyze and log file size information"""
    file_size = os.path.getsize(audio_filepath)
    size_mb = file_size / (1024 * 1024)
    size_percentage = (file_size / (25 * 1024 * 1024)) * 100
    
    logging.info(f"=== AUDIO FILE ANALYSIS for {v_id} ===")
    logging.info(f"File size: {file_size:,} bytes ({size_mb:.2f} MB)")
    logging.info(f"Size utilization: {size_percentage:.1f}% of API limit")
    
    return file_size, size_mb


def _cleanup_audio_file(audio_filepath, v_id):
    """Clean up temporary audio file with error handling"""
    if audio_filepath and os.path.exists(audio_filepath):
        try:
            os.remove(audio_filepath)
            logging.info(f"Cleaned up temporary audio file for {v_id}")
        except Exception as cleanup_e:
            logging.warning(f"Failed to cleanup audio file: {cleanup_e}")


def _try_standard_compression(v_id, audio_filepath, ffmpeg_dir):
    """Try standard compression settings"""
    logging.info("Attempting standard compression...")
    compression_settings = _get_compression_settings("standard")
    ydl_opts = _get_ffmpeg_download_options(True, ffmpeg_dir)
    ydl_opts.update({
        "outtmpl": audio_filepath,
        "postprocessors": compression_settings["postprocessors"],
        "postprocessor_args": compression_settings["postprocessor_args"],
    })
    
    try:
        _download_audio_file(v_id, audio_filepath, ydl_opts)
        return True, "Standard compression successful"
    except Exception as e:
        logging.error(f"Standard compression failed: {e}")
        return False, f"Standard compression failed: {e}"


def _try_balanced_compression(v_id, audio_filepath, ffmpeg_dir):
    """Try balanced compression settings"""
    logging.info("Retrying with balanced compression (8kbps, moderate compression)...")
    compression_settings = _get_compression_settings("balanced")
    ydl_opts = _get_ffmpeg_download_options(True, ffmpeg_dir)
    ydl_opts.update({
        "outtmpl": audio_filepath,
        "postprocessors": compression_settings["postprocessors"],
        "postprocessor_args": compression_settings["postprocessor_args"],
    })
    
    try:
        _download_audio_file(v_id, audio_filepath, ydl_opts)
        
        # Check result
        if os.path.exists(audio_filepath):
            file_size = os.path.getsize(audio_filepath)
            size_mb = file_size / (1024 * 1024)
            logging.info(f"Balanced compression result: {file_size:,} bytes ({size_mb:.2f} MB)")
            
            if file_size <= (25 * 1024 * 1024):
                logging.info("Balanced compression successful! Proceeding with transcription.")
                return True, f"Balanced compression successful: {size_mb:.2f} MB"
            else:
                return False, f"Balanced compression still too large: {size_mb:.2f} MB"
        else:
            return False, "Balanced compression failed to produce file"
    except Exception as e:
        logging.error(f"Balanced compression failed: {e}")
        return False, f"Balanced compression failed: {e}"


def _try_ultra_aggressive_compression(v_id, audio_filepath, ffmpeg_dir):
    """Try ultra-aggressive compression settings"""
    logging.info("Retrying with ultra-aggressive compression (4kbps, max compression)...")
    compression_settings = _get_compression_settings("ultra_aggressive")
    ydl_opts = _get_ffmpeg_download_options(True, ffmpeg_dir)
    ydl_opts.update({
        "outtmpl": audio_filepath,
        "postprocessors": compression_settings["postprocessors"],
        "postprocessor_args": compression_settings["postprocessor_args"],
    })
    
    try:
        _download_audio_file(v_id, audio_filepath, ydl_opts)
        
        # Check result
        if os.path.exists(audio_filepath):
            file_size = os.path.getsize(audio_filepath)
            size_mb = file_size / (1024 * 1024)
            logging.info(f"Ultra-aggressive compression result: {file_size:,} bytes ({size_mb:.2f} MB)")
            
            if file_size <= (25 * 1024 * 1024):
                logging.info("Ultra-aggressive compression successful! Proceeding with transcription.")
                return True, f"Ultra-aggressive compression successful: {size_mb:.2f} MB"
            else:
                return False, f"Ultra-aggressive compression still too large: {size_mb:.2f} MB"
        else:
            return False, "Ultra-aggressive compression failed to produce file"
    except Exception as e:
        logging.error(f"Ultra-aggressive compression failed: {e}")
        return False, f"Ultra-aggressive compression failed: {e}"


def _handle_oversized_file(v_id, audio_filepath, ffmpeg_dir, ffmpeg_available):
    """Handle cases where file size exceeds API limit"""
    if not ffmpeg_available:
        logging.error("FFmpeg not available - cannot compress oversized file")
        return None, "TRANSCRIPTION_FAILED", None, None
    
    # Try balanced compression first
    success, _message = _try_balanced_compression(v_id, audio_filepath, ffmpeg_dir)
    if success:
        return _finalize_transcription_attempt(audio_filepath, v_id)
    
    # Try ultra-aggressive as final fallback
    success, _message = _try_ultra_aggressive_compression(v_id, audio_filepath, ffmpeg_dir)
    if success:
        return _finalize_transcription_attempt(audio_filepath, v_id)
    
    # All compression attempts failed
    logging.error("All compression attempts failed. Skipping transcription.")
    return None, "TRANSCRIPTION_FAILED", None, None


def _finalize_transcription_attempt(audio_filepath, v_id):
    """Final check and prepare for transcription after compression"""
    file_size = os.path.getsize(audio_filepath)
    size_mb = file_size / (1024 * 1024)
    return file_size, size_mb


def _try_groq_transcription(audio_filepath, v_id, file_size):
    """Try Groq Whisper transcription service"""
    try:
        logging.info(f"Attempting Groq Whisper transcription for {v_id}...")
        
        # Log Groq input
        logging.info("=== GROQ WHISPER API INPUT ===")
        logging.info("Model: whisper-large-v3-turbo")
        logging.info(f"File: {os.path.basename(audio_filepath)}")
        logging.info(f"File size: {file_size:,} bytes ({file_size / (1024*1024):.2f} MB)")
        logging.info("=== END GROQ WHISPER API INPUT ===")
        
        with open(audio_filepath, "rb") as audio_file:
            transcription = model_manager.client.audio.transcriptions.create(
                file=(audio_filepath, audio_file.read()),
                model="whisper-large-v3-turbo",
                response_format="json",
                language="sv",
                temperature=0.0,
            )
        
        if transcription and transcription.text:
            actual_model = (
                transcription.model
                if hasattr(transcription, "model")
                else "whisper-large-v3-turbo"
            )
            logging.info(f"Groq Whisper transcription successful for {v_id} ({len(transcription.text)} characters)")
            logging.info(f"Transcription service used: {actual_model}")
            return transcription.text, actual_model
        else:
            logging.error(f"Groq Whisper returned empty transcription for {v_id}")
            return None, None
            
    except Exception as e:
        if "413" in str(e) or "too large" in str(e).lower():
            logging.error(f"Audio file too large for Groq API (max 25MB). Skipping video {v_id} as transcription is required.")
            return None, "TRANSCRIPTION_FAILED"
        elif "429" in str(e) or "Too Many Requests" in str(e):
            logging.warning(f"Groq API rate limit hit (429) for {v_id}. Trying fallback services...")
            return None, "RATE_LIMITED"
        else:
            logging.error(f"Transcription API error for video {v_id}. Skipping as transcription is required.")
            return None, "TRANSCRIPTION_FAILED"


def _try_assemblyai_fallback(audio_filepath, v_id):
    """Try AssemblyAI as first fallback service"""
    assemblyai_result = transcribe_with_assemblyai(audio_filepath, v_id)
    if assemblyai_result:
        logging.info(f"Failover successful: AssemblyAI provided transcription for {v_id}")
        return assemblyai_result, "assemblyai"
    else:
        logging.warning(f"AssemblyAI fallback failed for {v_id}. Trying final fallback...")
        return None, None


def _try_gemini_fallback(v_id):
    """Try Gemini as final fallback service"""
    gemini_result = transcribe_with_gemini_audio(v_id)
    if gemini_result:
        logging.info(f"Final fallback successful: Gemini 3.1 Flash Lite Preview provided transcription for {v_id}")
        return gemini_result, "gemini"
    else:
        logging.error(f"All transcription services failed for {v_id}. No transcription available.")
        return None, None


def _handle_all_transcription_failed(v_id):
    """Handle complete transcription failure"""
    logging.error(f"All transcription services failed for {v_id}. No transcription available.")
    return None, "TRANSCRIPTION_FAILED"


def _attempt_transcription(audio_filepath, v_id, file_size):
    """Attempt transcription with fallback services"""
    # Check if Groq Whisper is available
    if not is_groq_whisper_available():
        logging.warning(f"Groq Whisper is rate limited for {v_id}. Skipping to fallback services...")
        
        # Try AssemblyAI as first fallback
        full_text, transcription_model = _try_assemblyai_fallback(audio_filepath, v_id)
        if full_text:
            return full_text, transcription_model
        
        # Try Gemini as final fallback
        full_text, transcription_model = _try_gemini_fallback(v_id)
        if full_text:
            return full_text, transcription_model
        
        # All services failed
        return _handle_all_transcription_failed(v_id)
    
    # Try Groq Whisper transcription
    full_text, transcription_model = _try_groq_transcription(audio_filepath, v_id, file_size)
    if transcription_model == "RATE_LIMITED":
        # Try fallback services when rate limited
        full_text, transcription_model = _try_assemblyai_fallback(audio_filepath, v_id)
        if full_text:
            return full_text, transcription_model
        
        full_text, transcription_model = _try_gemini_fallback(v_id)
        if full_text:
            return full_text, transcription_model
        
        return _handle_all_transcription_failed(v_id)
    
    return full_text, transcription_model


def _prepare_analysis_prompt(title, summarized_transcript):
    """Prepare AI analysis prompt with transcript content"""
    prompt = f"""Du är en klinisk och objektiv medieanalytiker med expertis inom svensk förtalslagstiftning.
        Din uppgift är att leverera en rak och kompakt analys av videon "{title}". Var objektiv men 'hård'.

        INSTRUKTION FÖR FÖRTALSBEDÖMNING:
        Observera att om influencern/kreatören diskuterar händelser som redan är allmänt kända och rapporterade i etablerade nyhetsmedier, minskar sannolikheten för förtal avsevärt. Om påståendena verkar handla om aktuella nyheter, sök online för att bekräfta om detta är allmänt känt och väg in det i din bedömning.
        
        FORMATMALL - FÖLJ EXAKT:
        - BÖRJA DIREKT med analysen, absolut ingen inledningsfras.
        - Svaret MÅSTE bestå av exakt två stycken.
        - Du får ENDAST använda markdown för att fetmarkera startorden (se nedan). Ingen annan formatering, inga listor, inga specialtecken.
        - INGA synliga tankeprocesser, resonemang eller code-blocks. Ge bara det slutgiltiga resultatet.

        DITT SVAR SKA SE UT EXAKT SÅ HÄR:

        **Sammanfattning:** [Skriv max 3 meningar som sammanfattar videons dominerande stämning och huvudsakliga innehåll.]

        **Juridisk bedömning:** Sannolikheten är [hög/måttlig/låg] för förtal. [Följ upp med max 2 meningar som motiverar bedömningen. Ange specifikt om anklagelserna är allmänt kända via nyhetsmedier, eller om det rör sig om nya anklagelser, grova förolämpningar eller drev.]

        DATA FÖR ANALYS:
        Transkription (AI-sammanfattning): 
        {summarized_transcript} 
        """
    
    # Log the formatted prompt for debugging
    print("=== FORMATTED AI ANALYSIS PROMPT ===")
    print(f"Title: {title}")
    print(f"Prompt length: {len(prompt)} characters")
    print(f"Full prompt:\n{prompt}")
    print("=== END FORMATTED AI ANALYSIS PROMPT ===")
    
    return prompt.format(summarized_transcript=summarized_transcript)


def _perform_ai_analysis(v_id, title, summarized_transcript):
    """Perform AI analysis using Groq"""
    try:
        prompt = _prepare_analysis_prompt(title, summarized_transcript)
        
        # Log input for debugging
        print("=== AI ANALYSIS INPUT ===")
        print(f"Video ID: {v_id}")
        print(f"Title: {title}")
        print(f"Prompt length: {len(prompt)} characters")
        print("=== END AI ANALYSIS INPUT ===")
        
        # Call Groq API
        response = model_manager.client.chat.completions.create(
            model=AI_ANALYSIS_MODEL,
            messages=[
                {"role": "system", "content": "Du är en klinisk medieanalytiker."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=4000,
            temperature=0.0,
        )
        
        if response and response.choices:
            analysis = response.choices[0].message.content
            logging.info(f"AI analysis successful for {v_id} ({len(analysis)} characters)")
            return analysis
        else:
            logging.error(f"AI analysis returned empty response for {v_id}")
            return None
            
    except Exception as e:
        logging.error(f"AI analysis failed for {v_id}: {e}")
        return None


def _store_analysis_results(v_id, channel_name, title, publication_date, full_text, ai_analysis, summarized_transcript=None):
    """Store analysis results in analysis stats"""
    analysis_stats = load_analysis_stats()
    
    # Find or create video entry
    video_entry = find_or_create_video(
        analysis_stats, channel_name, v_id, title, publication_date
    )
    
    # Add transcription analysis (store the full transcript)
    transcript_prompt = f"Transkribera och sammanfatta video: {title}"
    add_analysis_to_video(
        video_entry,
        "raw_transcript",
        transcript_prompt,
        full_text if full_text else "No transcription available",
        "whisper-large-v3-turbo",
    )
    
    # Store the summarized transcript if available
    if summarized_transcript:
        add_analysis_to_video(
            video_entry,
            "summarized_transcript",
            transcript_prompt,
            summarized_transcript,
            "whisper-large-v3-turbo",
        )
    
    # Add AI analysis (store the already-generated AI analysis)
    if ai_analysis:
        add_analysis_to_video(
            video_entry,
            "ai_analysis",
            f"AI-analys av video: {title}",
            ai_analysis,
            AI_ANALYSIS_MODEL,
        )


def get_transcript_and_analysis(
    v_id, title, channel_name, publication_date=UNKNOWN_DATE
):
    """Downloads video audio and transcribes it using Groq's Whisper API."""
    logging.info("=== FUNCTION START: get_transcript_and_analysis ===")

    # Constants for file size management
    GROQ_MAX_FILE_SIZE = 25 * 1024 * 1024  # 25MB in bytes
    SAFETY_BUFFER = 2 * 1024 * 1024  # 2MB safety buffer
    MAX_ALLOWED_SIZE = GROQ_MAX_FILE_SIZE - SAFETY_BUFFER

    full_text = None
    transcription_model = None
    audio_filepath = None

    try:
        # Setup and download
        audio_filepath = _setup_temp_audio_file()
        ffmpeg_dir, ffmpeg_available = _check_ffmpeg_availability()
        ydl_opts = _get_ffmpeg_download_options(ffmpeg_available, ffmpeg_dir)
        
        # Download and validate
        _download_audio_file(v_id, audio_filepath, ydl_opts)
        audio_filepath = _validate_and_fix_audio_path(audio_filepath)
        file_size, size_mb = _analyze_file_size(audio_filepath, v_id)
        
        # Handle oversized files
        if file_size > GROQ_MAX_FILE_SIZE:
            return _handle_oversized_file(v_id, audio_filepath, ffmpeg_dir, ffmpeg_available)
        
        elif file_size > MAX_ALLOWED_SIZE:
            logging.warning(f"PRE-FLIGHT CHECK: File size ({size_mb:.2f} MB) exceeds safe threshold ({MAX_ALLOWED_SIZE / (1024 * 1024):.2f} MB)")
            logging.warning("Proceeding with transcription - may risk API rejection")
        else:
            logging.info("PRE-FLIGHT CHECK PASSED: File size within safe limits")

        # Transcription
        full_text, transcription_model = _attempt_transcription(audio_filepath, v_id, file_size)
        
        # AI Analysis
        if full_text and full_text != "TRANSCRIPTION_FAILED":
            print(f"Proceeding with AI analysis for {v_id}...")
            
            # Summarize long transcripts to fit token limits
            print(f"Summarizing transcript for {v_id}...")
            summarized_transcript = summarize_transcript(full_text, title)
            print(f"Transcript summarized for {v_id} (length: {len(summarized_transcript)} characters)")
            
            # Perform AI analysis
            ai_analysis = _perform_ai_analysis(v_id, title, summarized_transcript)
            
            # Store results
            _store_analysis_results(v_id, channel_name, title, publication_date, full_text, ai_analysis, summarized_transcript)
        else:
            ai_analysis = None
        
        return full_text, ai_analysis, transcription_model or "whisper-large-v3-turbo", "whisper-large-v3-turbo"
        
    finally:
        _cleanup_audio_file(audio_filepath, v_id)


# --- MAIN LOGIC ---
def process_single_video(v_id):
    """Process a single video (called from subprocess)"""
    try:
        # Setup logging for subprocess - use basicConfig to ensure logs go to stdout
        import sys
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(message)s',
            stream=sys.stdout  # Ensure logs go to stdout so they can be captured
        )
        
        # Get video data first to extract channel name
        logging.info(f"SUBPROCESS DEBUG: Starting video processing for {v_id}")
        (
            title,
            channel_name,
            ui_count,
            comments,
            video_stats,
            transcript_text,
            ai_analysis,
            ai_model,
            publication_date,
            extracted_channel_name,
            transcription_model,
        ) = get_yt_data(v_id, deep_scrape=True)
        
        logging.info(f"SUBPROCESS DEBUG: Got video data for {v_id}")
        logging.info(f"SUBPROCESS DEBUG: Title: {title}")
        logging.info(f"SUBPROCESS DEBUG: AI analysis result: {ai_analysis is not None}")

        # Check if video is old enough (at least 24 hours)
        if not _is_video_old_enough(publication_date):
            logging.info(f"Skipping video {v_id} - published less than 24 hours ago")
            print(f"PROCESSING_SUCCESS:{v_id}")
            return

        # Normalize channel name to expected identifier
        normalized_channel_name = normalize_channel_name(channel_name)
        if not normalized_channel_name:
            logging.warning(f"Could not normalize channel name '{channel_name}' for video {v_id}")
            print(f"PROCESSING_SUCCESS:{v_id}")
            return
        
        logging.info(f"Normalized channel name '{channel_name}' to '{normalized_channel_name}' for video {v_id}")
        channel_name = normalized_channel_name

        # Handle case where video_to_channel might be empty (from queue processing)
        if channel_name == UNKNOWN_CHANNEL:
            # Try to find channel from analysis stats
            analysis_stats = load_analysis_stats()
            for ch_name, ch_data in analysis_stats.items():
                if isinstance(ch_data, dict) and "videos" in ch_data:
                    for video in ch_data["videos"]:
                        if isinstance(video, dict) and video.get("video_id") == v_id:
                            channel_name = ch_name
                            break
                if channel_name != UNKNOWN_CHANNEL:
                    break

        if title == "MEMBERS_ONLY":
            logging.warning(
                f"Skipping members-only/private video {v_id} - detected during initial fetch (zero engagement but title available)"
            )
            print(f"PROCESSING_SUCCESS:{v_id}")
            return
        else:
            # Process the single video
            logging.info(f"Processing video {v_id} from channel {channel_name}.")

            # Always create video entry and save metadata for any processed video
            analysis_stats = load_analysis_stats()
            video_entry = find_or_create_video(
                analysis_stats, channel_name, v_id, title, publication_date
            )

            # Save video metadata
            video_entry["video_stats"] = video_stats
            video_entry["ui_comment_count"] = ui_count

            # Save basic video data immediately
            save_analysis_stats(analysis_stats)

            # Check for members-only/private content detection (late detection from transcription)
            if transcript_text == "MEMBERS_ONLY":
                logging.warning(
                    f"Skipping members-only/private video {v_id} - content access restricted"
                )
                print(f"PROCESSING_SUCCESS:{v_id}")
                return
            elif title is None:
                logging.warning(f"Skipping video {v_id} due to scraping failure.")
                print(f"PROCESSING_SUCCESS:{v_id}")
                return
            elif transcript_text == "TRANSCRIPTION_FAILED":
                logging.warning(
                    f"Skipping video {v_id} - transcription failed and is required for analysis"
                )
                # Still print completion marker since processing was attempted
                print(f"PROCESSING_SUCCESS:{v_id}")
                return
            else:

                # Save summarized transcript to analyses object if available
                if transcript_text and transcript_text != "TRANSCRIPTION_FAILED":
                    # Summarize transcript for storage
                    summarized_transcript = summarize_transcript(transcript_text, title)
                    logging.info(
                        f"DATA CHANGE: Saved summarized transcript for {v_id} ({len(summarized_transcript)} chars, was {len(transcript_text)} chars)"
                    )

                    # Log summarized transcript content
                    logging.info("=== SUMMARIZED TRANSCRIPT OUTPUT ===")
                    logging.info(f"Video: {title}")
                    logging.info(
                        f"Content preview (first 500 chars): {summarized_transcript[:500]}..."
                    )
                    if len(summarized_transcript) > 500:
                        logging.info(
                            f"Full length: {len(summarized_transcript)} characters"
                        )
                    logging.info("=== END SUMMARIZED TRANSCRIPT OUTPUT ===")

                    # Add transcription analysis to analyses object
                    add_analysis_to_video(
                        video_entry,
                        "raw_transcript",
                        f"Transkribera och sammanfatta video: {title}",
                        summarized_transcript,
                        transcription_model or "whisper-large-v3-turbo",
                    )

                    # Save state after transcription
                    queue_state = load_queue_state()
                    save_queue_state(queue_state)
                else:
                    logging.info(f"No transcript available for {v_id}")

                # Save AI analysis to analyses object if available
                if ai_analysis:
                    logging.info(f"AI analysis available for {v_id}")

                    # Add AI analysis to analyses object with actual model name
                    ai_model_name = ai_model if ai_model else "unknown_model"
                    add_analysis_to_video(
                        video_entry,
                        "ai_analysis",
                        f"AI-analys av video: {title}",
                        ai_analysis,
                        ai_model_name,
                    )

                    # Save state after AI analysis
                    queue_state = load_queue_state()
                    save_queue_state(queue_state)

                # Send Discord message if we have valid analysis data
                has_ai_analysis = bool(ai_analysis)
                has_transcript = bool(transcript_text and transcript_text != "TRANSCRIPTION_FAILED")
                
                logging.info(f"Discord eligibility check for {v_id}: ai_analysis={has_ai_analysis}, transcript_valid={has_transcript}")
                
                if has_ai_analysis and has_transcript:
                    logging.info(f"Sending Discord message for {v_id}")

                    # Send to Discord
                    discord_success = send_discord_message(
                        video_entry, channel_name
                    )
                    if discord_success:
                        logging.info(f"Discord message sent successfully for {v_id}")

                        # Mark video as completed and add timestamp
                        video_entry["sentToDiscord"] = True
                        video_entry["discord_posted_date"] = datetime.now().strftime(
                            "%Y-%m-%d"
                        )
                        logging.info(
                            f"VIDEO COMPLETED: {v_id} successfully posted to Discord at {video_entry['discord_posted_date']}"
                        )

                        # Print success marker for subprocess detection
                        print(f"DISCORD_SUCCESS:{v_id}")

                    else:
                        logging.warning(f"Failed to send Discord message for {v_id}")
                        
                        # Still mark as completed since processing succeeded
                        video_entry["sentToDiscord"] = False
                        logging.info(
                            f"VIDEO COMPLETED: {v_id} processing completed successfully (Discord failed)"
                        )

                        # Print completion marker for subprocess detection
                        print(f"PROCESSING_SUCCESS:{v_id}")
                else:
                    logging.warning(
                        f"Skipping Discord message for {v_id} - missing required data (ai_analysis: {bool(ai_analysis)}, transcript: {bool(transcript_text and transcript_text != 'TRANSCRIPTION_FAILED')})"
                    )

                    # Still print completion marker since processing succeeded
                    print(f"PROCESSING_SUCCESS:{v_id}")

                # Save final analysis stats - this must happen regardless of Discord status
                save_analysis_stats(analysis_stats)

                # Rest of the code remains the same
                logging.info(
                    f"=== VIDEO PROCESSING COMPLETE for {v_id} at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ==="
                )
                
                # Add delay between videos to ensure proper cleanup
                import time
                import gc
                time.sleep(3)
                gc.collect()  # Force garbage collection

    except Exception as e:
        logging.error(f"Error processing single video {v_id}: {e}")
        raise


if __name__ == "__main__":
    import sys
    
    # Check for single video mode
    if len(sys.argv) >= 3 and sys.argv[1] == "--single-video":
        single_video_id = sys.argv[2]
        logging.info(f"Single video mode: processing {single_video_id}")
        
        # Process single video directly
        process_single_video(single_video_id)
        sys.exit(0)
    
    # Setup run logging with sequential numbering
    run_log_file = setup_run_logging()

    logging.info("Starting YouTube video analyzer...")

    # 1. Load queue state and get next video
    queue_state = load_queue_state()

    # Load config for videos per run and fetch depth settings
    config = load_config()
    videos_per_run = config.get("settings", {}).get("videos_per_run", 1)
    fetch_depth = config.get("settings", {}).get("fetch_depth", 6)

    video_ids, queue_state, video_to_channel = fetch_latest_videos(CHANNELS, fetch_depth)

    if not video_ids:
        logging.info("No videos in queue to process.")
    else:
        logging.info(
            f"Processing {videos_per_run} video(s) from queue (depth: {fetch_depth}, pending: {len(queue_state['pending_queue'])})"
        )

        # Load analysis stats for video tracking
        analysis_stats = load_analysis_stats()

        # Get videos to process (limit by config)
        videos_to_process = video_ids[:videos_per_run] if videos_per_run > 0 else video_ids[:1]
        logging.info(
            f"Will process {len(videos_to_process)} video(s): {videos_to_process}"
        )

        # Process each video sequentially using subprocess to avoid Playwright conflicts
        import subprocess
        import sys
        
        processed_count = 0
        discord_messages_sent = 0
        
        for v_id in videos_to_process:
            try:
                logging.info(f"Starting subprocess for video {v_id}")
                
                # Create a separate Python process for each video to avoid Playwright conflicts
                result = subprocess.run([
                    sys.executable, 
                    __file__,  # This script
                    "--single-video", 
                    v_id
                ], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=1200)  # 20 minute timeout
                
                if result.returncode == 0:
                    logging.info(f"Successfully processed video {v_id}")
                    processed_count += 1
                    
                    # Log subprocess output for debugging
                    if result.stdout:
                        logging.info(f"=== SUBPROCESS OUTPUT for {v_id} ===")
                        for line in result.stdout.strip().split('\n'):
                            if line.strip():
                                logging.info(f"SUBPROCESS: {line}")
                        logging.info(f"=== END SUBPROCESS OUTPUT for {v_id} ===")
                    
                    # Check if Discord message was sent by looking for success marker in output
                    if f"DISCORD_SUCCESS:{v_id}" in result.stdout:
                        discord_messages_sent += 1
                        logging.info(f"Discord message sent for {v_id}")
                        
                        # Mark as completed in the main process queue state
                        queue_state = load_queue_state()
                        mark_video_completed(queue_state, v_id)
                        save_queue_state(queue_state)
                    elif f"PROCESSING_SUCCESS:{v_id}" in result.stdout:
                        logging.info(f"Video processing completed successfully for {v_id} (Discord failed)")
                        
                        # Mark as completed in the main process queue state even if Discord failed
                        queue_state = load_queue_state()
                        mark_video_completed(queue_state, v_id)
                        save_queue_state(queue_state)
                    else:
                        logging.warning(f"Discord message was not sent for {v_id}")
                else:
                    logging.error(f"Failed to process video {v_id}: {result.stderr}")
                    
            except subprocess.TimeoutExpired:
                logging.error(f"Timeout processing video {v_id}")
            except Exception as e:
                logging.error(f"Error processing video {v_id}: {e}")
                
            # Brief pause between videos
            import time
            time.sleep(5)

        logging.info(f"Processed {processed_count} out of {len(videos_to_process)} videos successfully")
        logging.info(f"Discord messages sent: {discord_messages_sent}")

        # Final run completion message
        logging.info(
            f"=== RUN COMPLETE at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ==="
        )
        logging.info(f"Run log saved to: {run_log_file}")
