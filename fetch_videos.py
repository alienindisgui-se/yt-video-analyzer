#!/usr/bin/env python3
"""
Separate video fetching script to avoid Playwright async conflicts.
This script fetches latest videos from channels and saves to a cache file.
"""

import json
import os
import sys
import time
import logging
from playwright.sync_api import sync_playwright, TimeoutError

# Configuration
CHANNELS = ['CarlFredrikAlexanderRask', 'ANJO1', 'MotVikten', 'Skuldis']
CACHE_FILE = "video_cache.json"
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
]

def setup_logging():
    """Setup logging configuration"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_cache():
    """Load existing video cache"""
    try:
        if os.path.exists(CACHE_FILE):
            with open(CACHE_FILE, 'r', encoding='utf-8') as f:
                return json.load(f)
    except Exception as e:
        logging.warning(f"Failed to load cache: {e}")
    return {}

def save_cache(cache_data):
    """Save video cache to file"""
    try:
        with open(CACHE_FILE, 'w', encoding='utf-8') as f:
            json.dump(cache_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        logging.error(f"Failed to save cache: {e}")

def fetch_latest_videos_sync():
    """Fetch latest videos using Playwright sync API"""
    latest_videos = []
    user_agent = USER_AGENTS[0]  # Use first user agent
    
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context(
            user_agent=user_agent,
            viewport={'width': 1920, 'height': 1080},
            locale='en-US',
            extra_http_headers={'Accept-Language': 'en-US,en;q=0.9'}
        )
        page = context.new_page()
        
        for channel in CHANNELS:
            try:
                url = f"https://www.youtube.com/@{channel}/videos"
                logging.info(f"Fetching videos for channel: {channel}")
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
                
                # Find first video link
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
                    else:
                        logging.warning(f"No videos found for channel {channel}")
                        
            except Exception as e:
                logging.error(f"Failed to fetch latest video for {channel}: {e}")
        
        browser.close()
    
    return latest_videos

def main():
    """Main function to fetch and cache videos"""
    setup_logging()
    
    logging.info("Starting video fetcher...")
    
    # Load existing cache
    cache_data = load_cache()
    
    # Fetch fresh data
    fresh_videos = fetch_latest_videos_sync()
    
    # Update cache with fresh data and timestamp
    cache_data.update({
        "videos": fresh_videos,
        "last_updated": time.time()
    })
    
    # Save updated cache
    save_cache(cache_data)
    
    logging.info(f"Successfully fetched and cached {len(fresh_videos)} videos")
    print(json.dumps({
        "videos": fresh_videos,
        "last_updated": cache_data.get("last_updated", time.time())
    }))

if __name__ == "__main__":
    main()
