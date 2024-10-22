from moviepy.video.io.VideoFileClip import VideoFileClip

# Specify the input and output video file paths
input_video_path = r"asset\20 sec.mp4"
output_video_path = r"asset\output_video.mp4"

# Set the start and end times for the clip (in seconds)
start_time = 0  # Start at 0 seconds
end_time = 10    # End at 10 seconds

# Load the video
clip = VideoFileClip(input_video_path).subclip(start_time, end_time)

# Save the extracted portion to a new file
clip.write_videofile(output_video_path, codec="mpeg4")