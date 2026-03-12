import json
import os
import requests
import time
import random
import sys
import logging
import hashlib
from dotenv import load_dotenv
from groq import Groq
from playwright.sync_api import sync_playwright, TimeoutError

# Load environment variables from .env file
load_dotenv()

# Setup logging and encoding
sys.stdout.reconfigure(encoding='utf-8')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Configuration
CHANNELS = ['CarlFredrikAlexanderRask', 'ANJO1', 'MotVikten', 'Skuldis']
STATE_FILE = "comment_state.json"
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
        """Try to generate summary with fallback models"""
        models_to_try = [self.primary_model] + self.fallback_models
        
        # Add legacy fallback if not in list
        if "llama-3.3-70b-versatile" not in models_to_try:
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
COLORS = CONFIG.get("colors", {
    "positive": "0x00FF00",
    "neutral": "0x3498DB", 
    "negative": "0xFF0000"
})

# Initialize Model Manager
model_manager = ModelManager(CONFIG)

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
]

def generate_persistent_id(author, text):
    raw_str = f"{author}|{text}"
    return hashlib.md5(raw_str.encode('utf-8')).hexdigest()

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

            comments = {}
            if deep_scrape:
                logging.info(f"Starting deep scrape for '{title}'.")
                
                # Scroll to load all threads
                last_thread_count, no_change = 0, 0
                while True:
                    current_thread = page.locator('ytd-comment-thread-renderer').count()
                    if current_thread == last_thread_count:
                        no_change += 1
                        if no_change >= 3: break
                    else:
                        no_change, last_thread_count = 0, current_thread
                    page.evaluate("document.scrollingElement.scrollTop = document.scrollingElement.scrollHeight")
                    page.wait_for_timeout(5000)

                # Final extraction
                author_locs = page.locator('#author-text')
                text_locs = page.locator('#content-text')
                extracted_count = text_locs.count()
                
                for i in range(extracted_count):
                    try:
                        author = author_locs.nth(i).text_content().strip()
                        text = text_locs.nth(i).text_content().strip()
                        c_id = generate_persistent_id(author, text)
                        if c_id not in comments:
                            comments[c_id] = {
                                'a': author, 't': text, 'ts_posted': int(time.time()),
                                'lastSeen': int(time.time()), 'deleted': False, 'notFoundCounter': 0
                            }
                    except Exception:
                        pass
                
                if ui_count == 0 and len(comments) > 0:
                    ui_count = len(comments)
                    
            return ui_count, comments, title, video_stats
            
        except Exception as e:
            logging.error(f"Scrape failed for {v_id}: {e}")
            return None, None, None, None
        finally:
            browser.close()

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
            
        logging.info(f"AI Summary generated using model '{used_model}':\n{summary}")
        
        # Generate embed color based on like ratio gradient
        embed_color = get_gradient_color(video_stats['like_ratio'])
        logging.info(f"Generated gradient color: {hex(embed_color)} for like ratio: {video_stats['like_ratio']:.1f}%")
        
        # Send summary to Discord
        if WEBHOOK:
            from datetime import datetime
            date_format = SETTINGS.get("date_format", "%Y-%m-%d")
            current_date = datetime.now().strftime(date_format)
            
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
                    "description": summary
                }]
            }
            requests.post(WEBHOOK, json=payload)
    except Exception as e:
        logging.error(f"Failed to generate AI summary: {e}")

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
        _, current_comments, title, video_stats = get_yt_data(v_id, deep_scrape=True)
        
        if current_comments is None:
            logging.warning(f"Skipping video {v_id} due to scraping failure.")
            continue
        
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
            "last_checked": time.time()
        }

    # 5. Save state to disk
    with open(STATE_FILE, "w", encoding='utf-8') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    logging.info("Monitoring complete and state saved.")
