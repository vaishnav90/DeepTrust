from app.services.audio_analyzer import AudioAnalyzer
from app.services.image_analyzer import ImageAnalyzer
from app.services.video_analyzer import VideoAnalyzer

class Analyzer:
    
    audio_analyzer = None
    image_analyzer = None
    video_analyzer = None
    
    def __init__(self):
        self.audio_analyzer = AudioAnalyzer()
        self.image_analyzer = ImageAnalyzer()
        self.video_analyzer = VideoAnalyzer()
            
    def analyze_audio(self, audio_data: bytes, filename: str | None = None, content_type: str | None = None):
        
        result = self.audio_analyzer.analyze_audio(audio_data, filename=filename, content_type=content_type)
        
        return result
    
    def analyze_image(self, image_data: bytes, filename: str | None = None, content_type: str | None = None):
        result = self.image_analyzer.analyze_image(image_data, filename=filename, content_type=content_type)
        return result
    
    def analyze_video(self, video_data: bytes, filename: str | None = None, *, seconds: int = 10, frames: int = 10):
        result = self.video_analyzer.analyze_video(
            video_data,
            filename=filename,
            seconds=seconds,
            frames=frames,
        )
        return result