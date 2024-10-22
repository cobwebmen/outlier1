from moviepy.video.io.VideoFileClip import VideoFileClip

# Video dosyasının yolu
video_path = "assets\input.mp4"  # Buraya kendi video dosyanızın yolunu yazın
output_path = "assets\output.mp4"  # Çıkış dosyasının adı

# Video klibini yükle
with VideoFileClip(video_path) as video:
    # İlk 10 saniyeyi al
    first_10_seconds = video.subclip(0, 10)
    # Yeni dosyayı kaydet
    first_10_seconds.write_videofile(output_path, codec="libx264")