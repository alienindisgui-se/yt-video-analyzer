# YT Comment Deletion Tracker

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A stealthy, automated monitor for tracking deleted comments on YouTube videos. Uses advanced techniques to avoid detection while reliably detecting and alerting on comment deletions.

## 📋 Description

This Python-based tool continuously monitors a list of YouTube videos for comment deletions. When a comment that was previously recorded disappears, it sends a detailed Discord notification with the deleted comment's information. Designed for researchers, content creators, and anyone interested in tracking comment moderation patterns.

## ✨ Features

- **Stealth Operation**: Randomized user agents, delays, and request patterns to avoid YouTube's detection systems
- **Robust Error Handling**: Gracefully handles network issues, invalid video IDs, and API failures
- **Intelligent Monitoring**: Checks comment counts first, then performs deep comparison only when necessary
- **Rich Notifications**: Discord embeds with dynamic colors based on deletion severity, timestamps, and direct video links
- **State Persistence**: Maintains a JSON-based state file to track comment history across runs
- **Batch Processing**: Processes videos in configurable batches with least-recently-checked prioritization
- **Automated Execution**: Designed for periodic execution via GitHub Actions or cron jobs

## 🛠️ Prerequisites

- Python 3.7+
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) (YouTube downloader)
- [requests](https://pypi.org/project/requests/) (HTTP library)

## 📦 Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/alienindisgui-se/yt-comment-deletion-tracker.git
   cd yt-comment-deletion-tracker
   ```

2. **Install dependencies:**
   ```bash
   pip install yt-dlp requests
   ```

## ⚙️ Configuration

### 1. Discord Webhook
Create a Discord webhook for notifications:
1. Go to your Discord server settings
2. Navigate to Integrations > Webhooks
3. Create a new webhook and copy the URL

Set the webhook URL as an environment variable:
```bash
export DISCORD_WEBHOOK="https://discord.com/api/webhooks/YOUR_WEBHOOK_ID/YOUR_WEBHOOK_TOKEN"
```

Or create a `.env` file in the project root:
```
DISCORD_WEBHOOK=https://discord.com/api/webhooks/YOUR_WEBHOOK_ID/YOUR_WEBHOOK_TOKEN
```

### 2. Video List
Edit `videos.json` to include the YouTube video IDs you want to monitor:
```json
[
  "VIDEO_ID_1",
  "VIDEO_ID_2",
  "VIDEO_ID_3"
]
```

- Video IDs must be exactly 11 characters
- Invalid IDs are automatically skipped with warnings

### 3. YouTube Bot Bypass (Automatic in CI)
The GitHub Actions workflow automatically handles YouTube's bot detection by:

- Downloading and running `bgutil-pot` (a Proof-of-origin token provider) to generate guest PO tokens on-demand.
- Configuring yt-dlp to use these guest tokens without requiring personal YouTube account data.

This approach is safer as it uses anonymous guest sessions that can be refreshed per run, eliminating the need for cookies or personal tokens.

For local testing, you may occasionally encounter bot detection. If needed, you can run the same bgutil-pot tool locally or adjust request frequencies.

#### Local Development Setup
To run the PO token provider locally for development or testing:

1. **Download the binary:**
   ```bash
   # For Linux x86_64
   curl -L https://github.com/jim60105/bgutil-ytdlp-pot-provider-rs/releases/download/v0.7.2/bgutil-pot-linux-x86_64 -o bgutil-pot
   chmod +x bgutil-pot

   # For other platforms, check the releases page for available binaries
   ```

2. **Run the server:**
   ```bash
   ./bgutil-pot server --host 127.0.0.1 --port 4416
   ```

3. **The monitor script will automatically use the local server** when `po_token: ['web+http://127.0.0.1:4416']` is configured.

No additional GitHub secrets are required beyond `DISCORD_WEBHOOK`.

## 🚀 Usage

### Manual Execution
Run the script manually:
```bash
python monitor.py
```

### Automated Execution with GitHub Actions
1. Add your repository secrets in GitHub:
   - Go to Settings > Secrets and variables > Actions
   - Add `DISCORD_WEBHOOK` with your webhook URL

2. Create a GitHub Actions workflow (`.github/workflows/monitor.yml`):
   ```yaml
   name: Monitor Comments

   on:
     schedule:
       - cron: '*/20 * * * *'  # Run every 20 minutes
     workflow_dispatch:       # Allow manual trigger

   jobs:
     monitor:
       runs-on: ubuntu-latest
       steps:
         - uses: actions/checkout@v3
         - uses: actions/setup-python@v4
           with:
             python-version: '3.9'
         - name: Install dependencies
           run: pip install yt-dlp requests
         - name: Run monitor
           env:
             DISCORD_WEBHOOK: ${{ secrets.DISCORD_WEBHOOK }}
           run: python monitor.py
   ```

### First Run
On first execution, the script will create necessary files and exit. Add your video IDs to `videos.json` and run again.

## 🔧 How It Works

1. **Initialization**: Loads video list and previous state from JSON files
2. **Batch Selection**: Selects videos to check based on last checked time (prioritizing older checks)
3. **Comment Count Check**: Fetches current comment count for each video using yt-dlp
4. **Deep Scan**: If count changed, fetches full comment list and compares with stored state
5. **Deletion Detection**: Identifies comments present in state but missing from current fetch
6. **Notification**: Sends Discord embed with deletion details, colored by severity
7. **State Update**: Saves updated comment state and last checked timestamps

### Stealth Features
- **Randomized Delays**: 10-30 seconds between requests
- **User Agent Rotation**: Cycles through 40+ realistic browser user agents
- **Error Recovery**: Continues processing on individual failures
- **Rate Limiting**: Built-in delays to mimic human behavior

### Notification Details
Discord notifications include:
- 🚨 Alert emoji and title
- Author and content of deleted comment
- Timestamps for posting and deletion
- Percentage of deleted comments with bold formatting
- Direct link to video
- Dynamic embed color based on deletion rate:
  - 0-25%: Yellow
  - 25-50%: Orange-yellow
  - 50-75%: Orange-red
  - 75-100%: Red

## 📁 File Structure

```
yt-comment-deletion-tracker/
├── monitor.py              # Main monitoring script
├── videos.json             # List of video IDs to monitor
├── comment_state.json      # Persistent state (auto-generated)
├── .env                    # Environment variables (optional)
├── .gitignore              # Git ignore rules
├── LICENSE                 # MIT License
└── README.md               # This file
```

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Setup
```bash
# Install development dependencies (if any)
pip install -r requirements-dev.txt  # Create if needed

# Run tests (add if you create them)
pytest

# Format code
black monitor.py
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This tool is for educational and research purposes. Respect YouTube's Terms of Service and Discord's guidelines. Use responsibly and avoid overloading their servers with excessive requests.
