# YouTube Video Analyzer

![AI Assisted](https://img.shields.io/badge/AI%20Assisted-purple?style=for-the-badge)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Repo Size](https://img.shields.io/github/repo-size/alienindisgui-se/yt-video-analyzer?style=for-the-badge&color=blue)
![License](https://img.shields.io/github/license/alienindisgui-se/yt-video-analyzer?style=for-the-badge&color=green)

![CI](https://img.shields.io/github/actions/workflow/status/alienindisgui-se/yt-video-analyzer/monitor.yml?label=CI&logo=github&style=for-the-badge&color=0099FF) ![Every 30min](https://img.shields.io/badge/Schedule-Every%2030min-blue?style=for-the-badge&logo=github)

A sophisticated Python-based system for automated YouTube video analysis with AI-powered content analysis, transcription, legal assessment, and Discord notifications.

## 🚀 Features

- **🤖 AI-Powered Analysis**: Advanced video content analysis using multiple AI models (Groq, Gemini, AssemblyAI)
- **📝 Automated Transcription**: Video transcription using OpenAI Whisper and AssemblyAI with fallback options
- **⚖️ Legal Assessment**: Automatic legal risk assessment for potential defamation cases in Swedish
- **💬 Discord Notifications**: Rich embed reports with analysis results and video statistics
- **~~🎯 Smart Comment Analysis~~**: ~~Processes up to 200 comments per video for comprehensive sentiment analysis~~
- **🔄 Queue Management**: Intelligent video queue system with deduplication and state persistence
- **🌐 Multi-Model Support**: Primary/fallback model architecture with multiple AI providers
- **📊 Analytics Tracking**: Comprehensive video statistics including like ratios, view counts, and engagement metrics
- **🔒 Security-First**: Rate limiting, error handling, and secure API key management
- **⚡ High-Frequency Monitoring**: Runs every 30 minutes for timely content analysis
- **🔧 Improved Configuration**: Updated workflow name and gitignore for better project organization

## 🏗️ Architecture

```
yt-video-analyzer/
├── monitor.py                      # Main analyzer script
├── config.json                     # AI models, prompts, and settings configuration
├── analysis_stats.json              # Historical analysis data and queue state
├── requirements.txt                # Python dependencies
├── .github/workflows/
│   └── monitor.yml                 # GitHub Actions automation (every 30 min)
└── README.md                       # This file
```

## 🛠️ Setup Instructions

### Prerequisites

- Python 3.12 or higher
- FFmpeg for video processing
- API keys for Groq, Gemini, and AssemblyAI
- Discord webhook URL for notifications
- GitHub repository with Actions enabled

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/alienindisgui-se/yt-video-analyzer.git
   cd yt-video-analyzer
   ```

2. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Playwright browsers**
   ```bash
   playwright install chromium
   ```

4. **Install FFmpeg** (Ubuntu/Debian)
   ```bash
   sudo apt update
   sudo apt install -y ffmpeg
   ```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file or set GitHub Secrets with the following variables:

```bash
DISCORD_WEBHOOK=your_discord_webhook_url
GROQ_API_KEY=your_groq_api_key
GEMINI_API_KEY=your_gemini_api_key
ASSEMBLYAI_API_KEY=your_assemblyai_api_key
CHANNELS_LIST=channel1,channel2,channel3  # Comma-separated YouTube channel names
```

### Configuration Files

#### `config.json`

Main configuration file containing:

- **AI Models**: Primary and fallback model configuration
- **Prompts**: Swedish analysis prompts for legal assessment
- **Settings**: Video processing limits and analysis configuration
- **Rate Limits**: API rate limiting configuration

Example configuration:
```json
{
  "language": "sv",
  "ai_models": {
    "primary": "qwen/qwen3-32b",
    "fallback": ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]
  },
  "settings": {
    "videos_per_run": 4,
    "~~max_comments~~": ~~200~~,  # Comment analysis disabled
    "gemini_model": "gemini-3.1-flash-lite-preview",
    "fetch_depth": 3
  }
}
```

## 📊 Data Structure

### `analysis_stats.json`

Stores historical analysis data and queue state:

```json
{
  "channel_name": {
    "videos": [
      {
        "video_id": "abc123",
        "title": "Video Title",
        "analysis_date": "2026-03-19",
        "analyses": {
          "raw_transcript": {...},
          "ai_analysis": {...}
        },
        "video_stats": {...},
        "sentToDiscord": true
      }
    ]
  },
  "_queue_state": {
    "pending_queue": [],
    "completed_ids": ["abc123"],
    "fetch_depth": 3,
    "current_processing": null
  }
}
```

## 🤖 GitHub Actions Automation

### Scheduled Workflows

- **Frequency**: Every 30 minutes (`*/30 * * * *`)
- **Manual Trigger**: Available via `workflow_dispatch`
- **Timeout**: 30 minutes per run
- **Concurrency**: Prevents overlapping runs

### Required GitHub Secrets

- `DISCORD_WEBHOOK`: Discord webhook URL for notifications
- `GROQ_API_KEY`: Groq API key for primary AI model
- `GEMINI_API_KEY`: Google Gemini API key for fallback
- `ASSEMBLYAI_API_KEY`: AssemblyAI API key for transcription
- `CHANNELS_LIST`: Comma-separated list of YouTube channels (e.g., `techchannel,gamingchannel,newschannel,reviewchannel`)

### Manual Execution

You can manually trigger the analysis workflow:
1. Go to Actions tab in GitHub
2. Select "YouTube Video Analysis" workflow
3. Click "Run workflow"

## 🧠 AI Analysis Features

### Multi-Model Architecture

- **Primary Model**: Groq (qwen/qwen3-32b) for main analysis
- **Fallback Models**: Llama variants for redundancy
- **Transcription**: OpenAI Whisper with AssemblyAI fallback
- **Legal Analysis**: Specialized prompts for Swedish defamation assessment

### Analysis Process

1. **Video Discovery**: Automated YouTube video detection
2. **Transcription**: Audio-to-text conversion
3. **~~Comment Analysis~~**: ~~Sentiment and content analysis of comments~~
4. **Legal Assessment**: Defamation risk evaluation
5. **Report Generation**: Comprehensive analysis summary
6. **Discord Notification**: Rich embed with results

## 💬 Discord Integration

### Notification Format

- **Rich Embeds**: Structured analysis reports
- **Video Statistics**: Like ratios, view counts, engagement
- **Legal Assessment**: Defamation risk classification
- **Analysis Summary**: AI-generated content insights
- **Swedish Language**: Localized analysis and reporting

### Example Notification

```
🤖 AI-Analys: [Video Title]

Kanal: [Channel Name]     Like-ratio: [XX.X]%
Publicerad: [YYYY-MM-DD]   
Video: [Title with YouTube link]

[AI Analysis Description]
```

## 🔒 Security Considerations

### Rate Limiting

- Configurable API rate limits
- Intelligent retry mechanisms
- Error handling and fallback strategies

### API Security

- Secure API key storage via GitHub Secrets
- Environment-based configuration
- No hardcoded credentials in source code

## 📱 API Method

The system uses multiple APIs and services:

- **Playwright**: Web scraping for YouTube video metadata and content
- **yt-dlp**: Audio download from YouTube videos
- **ReturnYouTubeDislikeAPI**: Video statistics (likes, dislikes, views)
- **Groq API**: Primary AI analysis
- **Gemini API**: Fallback AI processing and transcript summarization
- **AssemblyAI**: Audio transcription services
- **Discord API**: Webhook notifications

## 🚀 Usage

### Local Development

1. **Set up environment variables**:
   ```bash
   # Create or edit the .env file with your API keys and channels
   CHANNELS_LIST=techchannel,gamingchannel,newschannel,reviewchannel
   ```

2. **Run the analyzer**:
   ```bash
   python monitor.py
   ```

### Channel Configuration

The system supports dynamic channel configuration through environment variables:

**Local Development**: Set `CHANNELS_LIST` in your `.env` file
**GitHub Actions**: Set `CHANNELS_LIST` repository secret

**Format**: Comma-separated channel names
```bash
channel1,channel2,channel3
```
**Benefits**:
- No code changes needed to update channels
- Easy testing with different channel combinations
- Secure management via GitHub secrets
- Automatic fallback to default channels if not configured

## 🔧 Troubleshooting

### Common Issues

1. **Missing API Keys**: Ensure all required environment variables are set
2. **FFmpeg Not Found**: Install FFmpeg for video processing
3. **Rate Limiting**: Check API quota and rate limit configurations
4. **Browser Issues**: Ensure Playwright browsers are installed correctly

### Debug Mode

Enable detailed logging:
```python
logging.basicConfig(level=logging.DEBUG)
```

### Queue Management

Reset the processing queue:
```bash
# Clear completed_ids in analysis_stats.json
# Set current_processing to null
# Clear pending_queue if needed
```

## 📈 Performance Metrics

- **Processing Speed**: ~4 videos per run
- **~~Comment Analysis~~**: ~~Up to 200 comments per video~~  # ~~Comment analysis disabled~~
- **Transcription Accuracy**: High-quality audio processing
- **Legal Analysis**: Specialized Swedish legal assessment
- **Notification Latency**: Real-time Discord alerts

## 🔄 Data Management

### Automatic Cleanup

- Queue state persistence
- Completed video tracking
- Automatic deduplication
- Historical data retention

### Manual Operations

- Clear analysis history
- Reset queue state
- Export analysis data
- Backup configuration

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## 📞 Support

For issues and questions:
- Create an issue in the GitHub repository
- Check the troubleshooting section
- Review the configuration documentation

---

**Built with ❤️ using Python, AI models, and modern web scraping technologies**