import os
import cv2
import imageio
import tempfile

def create_gif_preview(video_path, duration=2.0, fps=10, resize_width=320):
    """
    Extracts the first few seconds of a video and saves it as a GIF.
    Returns the path to the temporary GIF file.
    """
    if not os.path.exists(video_path):
        return None

    try:
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        max_frames = duration * fps
        step = int(30 / fps) # Assuming 30fps source, skip frames to match target fps

        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Capture every n-th frame
            if frame_count % step == 0:
                # Resize to reduce memory/file size
                h, w, _ = frame.shape
                aspect_ratio = h / w
                new_h = int(resize_width * aspect_ratio)
                frame = cv2.resize(frame, (resize_width, new_h))
                
                # Convert BGR (OpenCV) to RGB (ImageIO)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
            
            frame_count += 1
        
        cap.release()

        if not frames:
            return None

      
        filename = f"preview_{abs(hash(video_path))}.gif"
        output_path = os.path.join(tempfile.gettempdir(), filename)
        
        # Only create if it doesn't exist (Caching mechanism)
        if not os.path.exists(output_path):
            imageio.mimsave(output_path, frames, fps=fps, loop=0)
            
        return output_path

    except Exception as e:
        print(f"Error creating GIF for {video_path}: {e}")
        return None
