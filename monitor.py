import json
import os
import requests
import time
import random
import sys
import logging
import hashlib
import subprocess
from dotenv import load_dotenv
from groq import Groq
from playwright.sync_api import sync_playwright, TimeoutError
import yt_dlp

# Load environment variables from .env file
load_dotenv()

# Setup logging and encoding
sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
# CHANNELS = ['CarlFredrikAlexanderRask']
CHANNELS = ['CarlFredrikAlexanderRask', 'ANJO1', 'MotVikten', 'Skuldis']
STATE_FILE = "comment_state.json"
ANALYSIS_STATS_FILE = "analysis_stats.json"  # Track video analysis counts per channel
CONFIG_FILE = "config.json"
WEBHOOK = os.getenv("DISCORD_WEBHOOK")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

class ModelManager:
    """Manages AI model selection, fallbacks, and rate limiting"""
    
    def __init__(self, config):
        self.config = config
        self.ai_config = config.get('ai_models', {})
        self.primary_model = self.ai_config.get('primary', 'llama-3.3-70b-versatile')
        self.fallback_models = self.ai_config.get('fallback', [])
        self.available_models = self.ai_config.get('models', {})
        self.client = Groq(api_key=GROQ_API_KEY)
        
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
    
    def try_model_with_fallback(self, prompt, max_retries=3):
        """Try to generate summary with fallback models, prioritizing higher token limits"""
        # Sort models by estimated token capacity (higher limits first)
        models_by_capacity = [
            ("llama-3.3-70b-versatile", 12000),  # Highest capacity
            ("qwen/qwen3-32b", 6000),
            ("llama-3.1-8b-instant", 6000),
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
        
        for attempt, model_name in enumerate(models_to_try):
            try:
                logging.info(f"Attempting to use model: {model_name} (attempt {attempt + 1}/{len(models_to_try)})")
                
                chat_completion = self.client.chat.completions.create(
                    messages=[{"role": "user", "content": prompt}],
                    model=model_name,
                )
                
                summary = chat_completion.choices[0].message.content.strip()
                logging.info(f"Successfully generated summary using model: {model_name}")
                return summary, model_name
                
            except Exception as e:
                error_msg = str(e).lower()
                if "413" in error_msg or "payload too large" in error_msg or "tokens" in error_msg:
                    logging.warning(f"Model {model_name} failed: Request too large ({str(e)[:100]}...). Content may be too long - consider summarization.")
                else:
                    logging.warning(f"Model {model_name} failed: {str(e)}")
                
                if attempt < len(models_to_try) - 1:
                    logging.info(f"Trying next model...")
                    time.sleep(1)  # Brief delay before retry
                else:
                    logging.error("All models failed to generate summary")
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

# Initialize Whisper model for transcription fallback
WHISPER_MODEL = None
def get_whisper_model():
    global WHISPER_MODEL
    if WHISPER_MODEL is None:
        model_name = SETTINGS.get("whisper_model", "base")
        logging.info(f"Loading Whisper model: {model_name}")
        WHISPER_MODEL = whisper.load_model(model_name)
        logging.info(f"Whisper model {model_name} loaded successfully")
    return WHISPER_MODEL

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
    
    # Split into lines and process
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
        
        # Skip lines that look like AI reasoning/thinking
        if any(keyword in line.lower() for keyword in [
            'okay, let\'s start', 'alright, let\'s tackle', 'let me think', 
            'first, i need to', 'next, i should', 'finally', 'in conclusion',
            'step by step', 'breaking this down', 'analysis:', 'thinking:'
        ]):
            continue
        
        cleaned_lines.append(line)
    
    # Rejoin and clean up
    cleaned_text = '\n'.join(cleaned_lines)
    
    return cleaned_text.strip()

def summarize_transcript(transcript_text, title):
    """Summarize long transcript to key points for analysis (saves tokens)"""
    try:
        # Very aggressive summarization for token efficiency
        prompt = f"""Summarize this YouTube video transcript titled '{title}' into a very concise summary (max 300 words):

Key elements to capture:
- Main topic and central argument
- 2-3 key claims or points
- Overall tone/sentiment
- Any controversial elements

Keep it extremely brief!

Transcript excerpt:
{transcript_text[:15000]}..."""  # Limit to 15k chars for even more token efficiency
        
        # Use lightweight model for summarization
        chat_completion = model_manager.client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",  # Lightweight model
        )
        
        summary = chat_completion.choices[0].message.content.strip()
        logging.info(f"Transcript summarized from {len(transcript_text)} to {len(summary)} characters")
        return summary
        
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
    latest_videos = []
    user_agent = random.choice(USER_AGENTS)
    logging.info(f"Selected user agent for fetching videos: {user_agent}")
    
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
                logging.info(f"Fetching latest video for channel: {channel}")
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
                        latest_videos.append(v_id)
                        logging.info(f"Fetched latest video {v_id} for channel {channel}")
                else:
                    # Fallback
                    video_locator = page.locator('ytd-rich-grid-media a[href*="/watch?v="]').first
                    if video_locator.count() > 0:
                        href = video_locator.get_attribute('href')
                        if href and 'v=' in href:
                            v_id = href.split('v=')[1].split('&')[0]
                            latest_videos.append(v_id)
                            logging.info(f"Fetched latest video {v_id} for channel {channel} using fallback")
            except Exception as e:
                logging.error(f"Failed to fetch latest video for {channel}: {e}")
        
        browser.close()
    
    return latest_videos

def generate_persistent_id(author, text):
    raw_str = f"{author}|{text}"
    return hashlib.md5(raw_str.encode('utf-8')).hexdigest()

def load_analysis_stats():
    """Load analysis statistics from JSON file"""
    try:
        with open(ANALYSIS_STATS_FILE, "r", encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        logging.warning(f"Failed to load analysis stats: {e}")
        return {}

def save_analysis_stats(stats):
    """Save analysis statistics to JSON file"""
    try:
        with open(ANALYSIS_STATS_FILE, "w", encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Failed to save analysis stats: {e}")

def get_yt_data(v_id, deep_scrape=False):
    user_agent = random.choice(USER_AGENTS)
    logging.info(f"Selected user agent for scraping comments: {user_agent}")
    
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
            
            try:
                page.wait_for_selector('ytd-comments#comments', state='attached', timeout=15000)
                page.wait_for_timeout(3000)
            except TimeoutError:
                logging.warning("Comments section did not attach in time. Video might have comments disabled.")
                return None, None, None, None

            title_elem = page.locator('h1.ytd-watch-metadata yt-formatted-string')
            title = title_elem.text_content().strip() if title_elem.count() > 0 else 'Unknown'
            
            # Fetch video stats
            video_stats = get_video_stats(v_id)
            
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
                    logging.info("Sorted comments to 'Newest first' (language-independent).")
                except Exception as e:
                    logging.warning(f"Failed to sort comments: {e}. Proceeding with default sort.")

            comments = {}
            if deep_scrape:
                logging.info(f"Starting deep scrape for '{title}'. UI reports ~{ui_count} comments.")
                
                # Phase 1: Load all top-level threads by scrolling
                logging.info("Loading all top-level comment threads...")
                last_thread_count = 0
                no_change = 0
                while True:
                    thread_nodes = page.locator('ytd-comment-thread-renderer')
                    current_thread = thread_nodes.count()
                    logging.info(f"Current top-level threads: {current_thread}")
                    if current_thread == last_thread_count:
                        no_change += 1
                        if no_change >= 3:
                            logging.info(f"Loaded {current_thread} top-level threads.")
                            break
                    else:
                        no_change = 0
                        last_thread_count = current_thread
                    page.evaluate("document.scrollingElement.scrollTop = document.scrollingElement.scrollHeight")
                    logging.info("Scrolled to bottom for thread loading.")
                    page.wait_for_timeout(5000)

                # Phase 2: Expand replies using JavaScript to bypass actionability checks
                logging.info("Expanding nested replies via JavaScript injection...")
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
                        
                        logging.info(f"Iteration {i+1}: Dispatched {clicks_dispatched} clicks via JS.")
                        
                        if clicks_dispatched == 0:
                            logging.info("No more expansion buttons found. Expansion complete.")
                            break
                            
                        # Wait a moment for the requested replies to render in the DOM
                        page.wait_for_timeout(4000)
                        
                        # Scroll down to ensure we trigger any lazy-loaded elements
                        page.evaluate("document.scrollingElement.scrollTop = document.scrollingElement.scrollHeight")
                        page.wait_for_timeout(2000)
                        
                    except Exception as e:
                        logging.warning(f"Iteration {i+1} JS click failed: {str(e).splitlines()[0]}")
                        break

                logging.info("Proceeding to final extraction.")

                # Phase 3: Final scroll to ensure all loaded
                logging.info("Performing final scroll to load any remaining content...")
                page.evaluate("document.scrollingElement.scrollTop = document.scrollingElement.scrollHeight")
                page.wait_for_timeout(5000)

                # Extract all loaded comments
                logging.info("Extracting all loaded comments...")
                author_locs = page.locator('#author-text')
                text_locs = page.locator('#content-text')
                extracted_count = text_locs.count()
                logging.info(f"Found {extracted_count} comment texts for extraction (including replies).")
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
    temp_dir = tempfile.gettempdir()
    audio_filepath = os.path.join(temp_dir, f"{v_id}.m4a")
    
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
            
        logging.info(f"Audio downloaded successfully for {v_id}.")
        
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
            
            os.remove(audio_filepath)
            audio_filepath = trimmed_filepath
            logging.info("Audio trimmed successfully using yt-dlp.")
        except Exception as ydl_trim_e:
            logging.warning(f"Audio trimming failed: {ydl_trim_e}. Proceeding with full audio - may hit rate limits.")
        
        logging.info(f"Sending audio to Groq Whisper API for transcription...")
        
        # Using the Groq client you already initialized in ModelManager
        with open(audio_filepath, "rb") as file:
            transcription = model_manager.client.audio.transcriptions.create(
                file=(os.path.basename(audio_filepath), file.read()),
                model="whisper-large-v2",  # More generous rate limits than v3-turbo
                response_format="text",
                language="sv" # Forcing Swedish context for channels like Anjo/Rask. Remove this line for auto-detect.
            )
            
        full_text = transcription
        logging.info(f"Audio successfully transcribed! ({len(full_text)} characters)")
        
    except Exception as e:
        error_msg = str(e).lower()
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
        
        analysis, used_model = model_manager.try_model_with_fallback(prompt)
        
        if analysis:
            cleaned_analysis = clean_ai_output(analysis)
            logging.info(f"AI analysis completed for {v_id} using model '{used_model}'.")
            return full_text, cleaned_analysis
        else:
            logging.warning(f"AI analysis failed for {v_id} with all available models.")
            return full_text, None
            
    except Exception as e:
        logging.error(f"Groq analysis failed for {v_id}: {e}")
        return full_text, None

def analyze_video_comprehensive(v_id, title, comments_dict, video_stats, transcript_text=None):
    """Unified analysis combining transcript content and comment sentiment in one AI call."""
    if not GROQ_API_KEY:
        logging.warning("No GROQ_API_KEY found. Skipping comprehensive AI analysis.")
        return None

    logging.info(f"Generating comprehensive AI analysis for video: {title}")
    
    # Prepare transcript section
    transcript_section = ""
    if transcript_text:
        transcript_section = f"**VIDEO TRANSCRIPT:**\n{transcript_text[:3000]}"  # Limit transcript length
    else:
        transcript_section = "**VIDEO TRANSCRIPT:**\n(No transcript available - captions disabled)"
    
    # Prepare video stats section
    stats_section = f"**VIDEO ENGAGEMENT:**\n{video_stats['likes']:,} likes, {video_stats['dislikes']:,} dislikes ({video_stats['like_ratio']:.1f}% like ratio)"
    
    # Prepare comments section
    comments_section = ""
    if comments_dict:
        texts = [c['t'] for c in comments_dict.values() if not c.get('deleted', False)]
        if texts:
            max_comments = SETTINGS.get("max_comments", 150)
            sample_texts = texts[:max_comments]
            comments_string = "\n".join([f"- {t}" for t in sample_texts])
            comments_section = f"**AUDIENCE COMMENTS:**\n{comments_string}"
        else:
            comments_section = "**AUDIENCE COMMENTS:**\n(No comments available)"
    else:
        comments_section = "**AUDIENCE COMMENTS:**\n(No comments available)"
    
    # Create comprehensive prompt
    prompt = f"""Analyze this YouTube video comprehensively:

{transcript_section}

{stats_section}

{comments_section}

**ANALYSIS REQUEST:**
Provide a unified analysis covering:

**CONTENT SUMMARY:** Key topics, arguments, and narrative structure from the video transcript
**AUDIENCE REACTION:** Comment sentiment, dominant themes, and public reception patterns  
**CONTENT-RECEPTION ALIGNMENT:** How well the video content aligns with audience reactions
**LEGAL ASSESSMENT:** Defamation risk assessment (always start with 'Sannolikheten är [hög/låg] för förtal')
**MONITORING INSIGHTS:** Suggested tags and themes for monitoring purposes

**FORMATTING:** 
- Structure clearly with bold headings
- Be concise but comprehensive
- Only mention like ratio if below 90%
- Add a line break (\n) between **Juridisk bedömning:** and **Sammanfattning:** sections
- Provide direct analysis without thinking blocks or meta-commentary
"""
    
    try:
        # Use ModelManager for intelligent model selection and fallback
        analysis, used_model = model_manager.try_model_with_fallback(prompt)
        
        if not analysis:
            logging.error("Failed to generate comprehensive AI analysis with all available models")
            return None
        
        # Clean the AI output to remove any unwanted formatting
        cleaned_analysis = clean_ai_output(analysis)
        
        logging.info(f"Comprehensive AI analysis completed using model '{used_model}'")
        
        # Generate embed color based on like ratio gradient
        embed_color = get_gradient_color(video_stats['like_ratio'])
        logging.info(f"Generated gradient color: {hex(embed_color)} for like ratio: {video_stats['like_ratio']:.1f}%")
        
        # Send comprehensive analysis to Discord
        if WEBHOOK:
            # Find channel name from video URL by checking which channel list contains this video ID
            channel_name = "Unknown Channel"
            for channel in CHANNELS:
                # This is a simplified approach - in a real implementation you might want to 
                # store channel info when fetching videos
                if v_id in [vid for vid in fetch_latest_videos([channel])]:
                    channel_name = channel
                    break
            
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
                    "description": cleaned_analysis
                }]
            }
            requests.post(WEBHOOK, json=payload)
            
    except Exception as e:
        logging.error(f"Failed to generate comprehensive AI analysis: {e}")

def summarize_comments_with_ai(title, comments_dict, v_id, video_stats):
    if not GROQ_API_KEY:
        logging.warning("No GROQ_API_KEY found. Skipping AI summary.")
        return

    logging.info(f"Generating AI summary for video: {title}")
    
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
    
    # Use prompt from configuration
    prompt = PROMPTS["ai_summary"].format(
        title=title,
        comments_string=comments_string + stats_info
    )

    try:
        # Use ModelManager for intelligent model selection and fallback
        summary, used_model = model_manager.try_model_with_fallback(prompt)
        
        if not summary:
            logging.error("Failed to generate AI summary with all available models")
            return
        
        # Clean the AI output to remove any unwanted formatting
        cleaned_summary = clean_ai_output(summary)
        
        logging.info(f"AI Summary generated using model '{used_model}':\n{cleaned_summary}")
        
        # Generate embed color based on like ratio gradient
        embed_color = get_gradient_color(video_stats['like_ratio'])
        logging.info(f"Generated gradient color: {hex(embed_color)} for like ratio: {video_stats['like_ratio']:.1f}%")
        
        # Determine channel name (used for tracking even without webhook)
        channel_name = "Unknown Channel"
        for channel in CHANNELS:
            # This is a simplified approach - in a real implementation you might want to 
            # store channel info when fetching videos
            try:
                if v_id in [vid for vid in fetch_latest_videos([channel])]:
                    channel_name = channel
                    break
            except Exception:
                continue
        
        # Track analysis count and prompt
        import time
        current_timestamp = time.time()
        
        analysis_stats = load_analysis_stats()
        if channel_name not in analysis_stats:
            analysis_stats[channel_name] = {
                "videos_analyzed": 0, 
                "last_model": used_model, 
                "last_prompt": prompt,
                "last_video_id": v_id,
                "last_checked": current_timestamp
            }
        analysis_stats[channel_name]["videos_analyzed"] += 1
        analysis_stats[channel_name]["last_model"] = used_model
        analysis_stats[channel_name]["last_prompt"] = prompt
        analysis_stats[channel_name]["last_video_id"] = v_id
        analysis_stats[channel_name]["last_checked"] = current_timestamp
        save_analysis_stats(analysis_stats)
        
        # Create footer with analysis count and model
        videos_analyzed = analysis_stats[channel_name]["videos_analyzed"]
        footer_text = f"**{videos_analyzed}** videos from @{channel_name} has been analysed. This analysis was made using __{used_model}__"
        
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
    logging.info("Starting merged YouTube monitor...")
    
    # 1. Fetch latest videos directly into memory
    video_ids = fetch_latest_videos(CHANNELS)
    if not video_ids:
        logging.warning("No videos fetched. Exiting.")
        sys.exit()
    
    logging.info(f"Monitoring videos: {video_ids}")
    
    # 2. Load historical state
    history = {}
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE, "r", encoding='utf-8') as f:
            history = json.load(f)
        logging.info("Loaded existing comment state.")

    # 3. Process each video
    for v_id in video_ids:
        logging.info(f"Processing video {v_id}.")
        _, current_comments, title, video_stats, transcript_text, ai_analysis = get_yt_data(v_id, deep_scrape=True)
        
        if current_comments is None:
            logging.warning(f"Skipping video {v_id} due to scraping failure.")
            continue
        
        # Log transcript and analysis results
        if transcript_text:
            logging.info(f"Transcript available for {v_id} ({len(transcript_text)} characters)")
        else:
            logging.info(f"No transcript available for {v_id}")
            
        if ai_analysis:
            logging.info(f"Video analysis for {v_id}: {ai_analysis[:200]}...")
        else:
            logging.info(f"No AI analysis available for {v_id}")
        
        old_state = history.get(v_id, {"count": 0, "comments": {}})
        updated_comments = {}
        deletions = []
        
        # Check existing comments for deletions
        for c_id, comment_data in old_state["comments"].items():
            if c_id in current_comments:
                updated_comments[c_id] = comment_data.copy()
                updated_comments[c_id]['lastSeen'] = int(time.time())
                updated_comments[c_id]['notFoundCounter'] = 0
            else:
                updated_comments[c_id] = comment_data.copy()
                updated_comments[c_id]['notFoundCounter'] = comment_data.get('notFoundCounter', 0) + 1
                
                if updated_comments[c_id]['notFoundCounter'] >= 3 and not comment_data.get('deleted', False):
                    updated_comments[c_id]['deleted'] = True
                    deletions.append(updated_comments[c_id])
        
        # Add new comments
        for c_id, comment_data in current_comments.items():
            if c_id not in updated_comments:
                updated_comments[c_id] = comment_data.copy()
        
        # Fire deletion webhooks - DISABLED
        # if deletions:
        #     total_tracked = len([c for c in updated_comments.values() if not c.get('deleted', False)])
        #     perc = (len(deletions) / max(total_tracked + len(deletions), 1)) * 100
        #     for d in deletions:
        #         send_deletion_alert(d['a'], d['t'], v_id, d.get('ts_posted', time.time()), time.time(), perc, title)

        # 4. Trigger the AI to summarize the comment section
        # We only pass the current live comments to get the real-time sentiment
        summarize_comments_with_ai(title, current_comments, v_id, video_stats)
        
        # Save history
        history[v_id] = {
            "count": len(current_comments),
            "comments": updated_comments,
            "title": title,
            "transcript": transcript_text,
            "analysis": ai_analysis,
            "last_checked": time.time()
        }

    # 5. Save state to disk
    with open(STATE_FILE, "w", encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    logging.info("Monitoring complete and state saved.")