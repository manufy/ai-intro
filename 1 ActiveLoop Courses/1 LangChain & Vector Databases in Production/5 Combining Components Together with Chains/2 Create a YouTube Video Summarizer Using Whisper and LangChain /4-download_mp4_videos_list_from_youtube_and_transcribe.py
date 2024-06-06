import yt_dlp

# download list of yt urls mp4

def download_mp4_from_youtube(urls, job_id):
    # This will hold the titles and authors of each downloaded video
    video_info = []

    for i, url in enumerate(urls):
        # Set the options for the download
        file_temp = f'./{job_id}_{i}.mp4'
        ydl_opts = {
            'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]',
            'outtmpl': file_temp,
            'quiet': True,
        }

        # Download the video file
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            result = ydl.extract_info(url, download=True)
            title = result.get('title', "")
            author = result.get('uploader', "")

        # Add the title and author to our list
        video_info.append((file_temp, title, author))

    return video_info

urls=["https://www.youtube.com/watch?v=mBjPyte2ZZo&t=78s",
    "https://www.youtube.com/watch?v=cjs7QKJNVYM",]
vides_details = download_mp4_from_youtube(urls, 1)

# trasncribe the videos and save to text file

import whisper


# load the model
model = whisper.load_model("base")

# iterate through each video and transcribe
results = []
for video in vides_details:
    result = model.transcribe(video[0])
    results.append( result['text'] )
    print(f"Transcription for {video[0]}:\n{result['text']}\n")

with open ('text.txt', 'w') as file:  
    file.write(results['text'])
    
    
    
    