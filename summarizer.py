import streamlit as st
import os
import json
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Any
import requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api._errors import TranscriptsDisabled, NoTranscriptFound
import yt_dlp
import openai

# App title and config
st.set_page_config(page_title="YouTube Video Summarizer", layout="wide")
st.title("🎥 YouTube Video Summarizer")
st.caption("Summarize any YouTube video using AI (DeepSeek or OpenAI)")

# Constants
CONFIG_FILE = "config.json"
OUTPUT_DIR = "summaries"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@st.cache_data(show_spinner=False)
def init_config():
    """Initialize or load config file"""
    if not os.path.exists(CONFIG_FILE):
        return {
            "deepseek_api_key": "",
            "openai_api_key": "",
            "default_model": "deepseek",
            "allow_paid_transcription": False,
            "whisper_model": "base"
        }
    with open(CONFIG_FILE, 'r') as f:
        return json.load(f)

@dataclass
class SummaryResult:
    video_id: str
    title: str
    transcript: str
    summary: str
    model_used: str
    cost: Optional[float] = None
    error: Optional[str] = None

class VideoSummarizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
    def get_video_title(self, video_id: str) -> str:
        """Get YouTube video title using yt-dlp"""
        ydl_opts = {'quiet': True, 'extract_flat': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(f"https://www.youtube.com/watch?v={video_id}", download=False)
            return info.get('title', 'Untitled Video')

    def get_transcript(self, video_id: str, use_paid: bool = False) -> str:
        """Get transcript with free/paid options"""
        try:
            # Try free YouTube transcript first
            transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=['en'])
            return " ".join([entry['text'] for entry in transcript])
        except (TranscriptsDisabled, NoTranscriptFound) as e:
            if not use_paid:
                raise ValueError("No free transcript available. Enable paid transcription in settings.")
            return self._transcribe_with_whisper(video_id)

    def _transcribe_with_whisper(self, video_id: str) -> str:
        """Fallback method using Whisper for transcription"""
        try:
            import whisper
        except ImportError:
            raise ImportError("Whisper not installed. Run 'pip install openai-whisper'")
        
        with st.spinner("Downloading audio and transcribing with Whisper..."):
            # Download audio
            audio_file = f"temp_{video_id}.mp3"
            ydl_opts = {
                'format': 'bestaudio/best',
                'postprocessors': [{'key': 'FFmpegExtractAudio', 'preferredcodec': 'mp3'}],
                'outtmpl': f'temp_{video_id}',
                'quiet': True,
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
            
            # Transcribe with Whisper
            model = whisper.load_model(self.config.get("whisper_model", "base"))
            result = model.transcribe(audio_file)
            
            # Clean up
            if os.path.exists(audio_file):
                os.remove(audio_file)
            
            return result["text"]

    def summarize_with_deepseek(self, text: str, custom_prompt: Optional[str] = None) -> str:
        """Summarize text using DeepSeek API"""
        if not self.config.get("deepseek_api_key"):
            raise ValueError("DeepSeek API key not configured")
        
        headers = {
            "Authorization": f"Bearer {self.config['deepseek_api_key']}",
            "Content-Type": "application/json",
        }
        
        prompt = custom_prompt or """Provide a comprehensive summary of this video transcript. 
        Include key points, main arguments, and important details. Structure with clear sections. 
        Aim for about 15-20% of the original length."""
        
        payload = {
            "model": "deepseek-chat",
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ],
            "temperature": 0.7,
            "max_tokens": 2000
        }
        
        response = requests.post(
            "https://api.deepseek.com/v1/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise ValueError(f"DeepSeek API error: {response.text}")
        
        return response.json()["choices"][0]["message"]["content"]

    def summarize_with_openai(self, text: str, custom_prompt: Optional[str] = None) -> str:
        """Summarize text using OpenAI API"""
        if not self.config.get("openai_api_key"):
            raise ValueError("OpenAI API key not configured")
        
        prompt = custom_prompt or """Provide a comprehensive summary of this video transcript. 
        Include key points, main arguments, and important details. Structure with clear sections. 
        Aim for about 15-20% of the original length."""
        
        response = openai.ChatCompletion.create(
            model="gpt-4-turbo-preview",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": text}
            ],
            temperature=0.7,
            max_tokens=2000,
            api_key=self.config["openai_api_key"]
        )
        
        return response.choices[0].message.content

    def summarize_video(self, video_url: str, model_choice: str, use_paid_transcription: bool, custom_prompt: Optional[str] = None) -> SummaryResult:
        """Main method to summarize a YouTube video"""
        video_id = self._extract_video_id(video_url)
        if not video_id:
            return SummaryResult(
                video_id="",
                title="",
                transcript="",
                summary="",
                model_used="",
                error="Invalid YouTube URL"
            )
        
        try:
            title = self.get_video_title(video_id)
            transcript = self.get_transcript(video_id, use_paid_transcription)
            
            if model_choice == "deepseek":
                summary = self.summarize_with_deepseek(transcript, custom_prompt)
                model_used = "DeepSeek Chat"
            else:
                summary = self.summarize_with_openai(transcript, custom_prompt)
                model_used = "OpenAI GPT-4"
            
            return SummaryResult(
                video_id=video_id,
                title=title,
                transcript=transcript,
                summary=summary,
                model_used=model_used
            )
            
        except Exception as e:
            return SummaryResult(
                video_id=video_id if 'video_id' in locals() else "",
                title=title if 'title' in locals() else "",
                transcript="",
                summary="",
                model_used="",
                error=str(e)
            )

    def _extract_video_id(self, url: str) -> Optional[str]:
        """Extract YouTube video ID from URL"""
        import re
        patterns = [
            r'(?:https?:\/\/)?(?:www\.)?youtu\.be\/([a-zA-Z0-9_-]+)',
            r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/watch\?v=([a-zA-Z0-9_-]+)',
            r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/embed\/([a-zA-Z0-9_-]+)',
            r'(?:https?:\/\/)?(?:www\.)?youtube\.com\/v\/([a-zA-Z0-9_-]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                return match.group(1)
        return None

def save_summary(result: SummaryResult):
    """Save the summary result to a JSON file"""
    filename = f"{OUTPUT_DIR}/{result.video_id}.json"
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(asdict(result), f, indent=2, ensure_ascii=False)
    return filename

def settings_section(config):
    """Settings sidebar for API configuration"""
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # API Selection
        model_choice = st.radio(
            "Default AI Model",
            ["deepseek", "openai"],
            index=0 if config["default_model"] == "deepseek" else 1,
            help="Which AI API to use by default"
        )
        config["default_model"] = model_choice
        
        # API Keys
        st.subheader("API Keys")
        config["deepseek_api_key"] = st.text_input(
            "DeepSeek API Key",
            value=config.get("deepseek_api_key", ""),
            type="password"
        )
        config["openai_api_key"] = st.text_input(
            "OpenAI API Key",
            value=config.get("openai_api_key", ""),
            type="password"
        )
        
        # Advanced Options
        with st.expander("Advanced Options"):
            config["allow_paid_transcription"] = st.checkbox(
                "Allow paid transcription fallback",
                value=config.get("allow_paid_transcription", False),
                help="Use Whisper when free transcript isn't available"
            )
            config["whisper_model"] = st.selectbox(
                "Whisper Model",
                ["tiny", "base", "small", "medium", "large"],
                index=["tiny", "base", "small", "medium", "large"].index(
                    config.get("whisper_model", "base")
                ),
                help="Larger models are more accurate but slower"
            )
        
        # Save config
        if st.button("Save Settings"):
            with open(CONFIG_FILE, 'w') as f:
                json.dump(config, f)
            st.success("Settings saved!")

def main():
    config = init_config()
    summarizer = VideoSummarizer(config)
    
    # Settings sidebar
    settings_section(config)
    
    # Main content
    st.header("📺 Summarize a YouTube Video")
    
    col1, col2 = st.columns(2)
    with col1:
        video_url = st.text_input(
            "YouTube Video URL",
            placeholder="https://www.youtube.com/watch?v=...",
            help="Paste the full URL of the YouTube video"
        )
        
        custom_prompt = st.text_area(
            "Custom Prompt (Optional)",
            height=100,
            help="Customize how the AI should summarize the content"
        )
    
    with col2:
        model_choice = st.radio(
            "AI Model to Use",
            ["deepseek", "openai"],
            index=0 if config["default_model"] == "deepseek" else 1,
            horizontal=True
        )
        
        use_paid = st.checkbox(
            "Allow paid transcription fallback",
            value=config["allow_paid_transcription"],
            help="Use Whisper if free transcript isn't available"
        )
        
        process_btn = st.button(
            "Summarize Video",
            type="primary",
            use_container_width=True
        )
    
    if process_btn and video_url:
        with st.spinner("Processing video..."):
            result = summarizer.summarize_video(
                video_url,
                model_choice,
                use_paid,
                custom_prompt if custom_prompt else None
            )
        
        if result.error:
            st.error(f"Error: {result.error}")
        else:
            st.success("Summary generated successfully!")
            
            # Display results
            st.subheader(result.title)
            st.caption(f"Summarized with {result.model_used}")
            
            col1, col2 = st.columns([1, 1], gap="large")
            
            with col1:
                with st.expander("📝 Summary", expanded=True):
                    st.write(result.summary)
                
                # Save button
                if st.button("Save Summary"):
                    filename = save_summary(result)
                    st.toast(f"Summary saved to {filename}")
            
            with col2:
                with st.expander("📊 Statistics"):
                    st.metric("Original Length", f"{len(result.transcript):,} characters")
                    st.metric("Summary Length", f"{len(result.summary):,} characters")
                    reduction = 100 - (len(result.summary)/len(result.transcript)*100)
                    st.metric("Reduction", f"{reduction:.1f}%")
                
                with st.expander("📜 Full Transcript"):
                    st.text_area(
                        "Transcript",
                        value=result.transcript,
                        height=300,
                        label_visibility="collapsed"
                    )

if __name__ == "__main__":
    main()