from pytube import YouTube
from KVCOMM.utils.const import KVCOMM_ROOT
from KVCOMM.utils.log import logger

def Youtube(url, has_subtitles):

    video_id=url.split('v=')[-1].split('&')[0]

    youtube = YouTube(url)

    video_stream = youtube.streams.filter(progressive=True, file_extension='mp4').order_by('resolution').desc().first()
    if has_subtitles:

        logger.info('Downloading video')
        video_stream.download(output_path="{KVCOMM_ROOT}/workspace",filename=f"{video_id}.mp4")
        logger.info('Video downloaded successfully')
        return f"{KVCOMM_ROOT}/workspace/{video_id}.mp4"
    else:
        return video_stream.url 
