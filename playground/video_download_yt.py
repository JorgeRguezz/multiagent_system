import os
import re
import yt_dlp


VIDEO_URL = "https://www.youtube.com/watch?v=skeg9D2N8uA"
OUTPUT_DIR = "downloads/queue"
USE_BROWSER_COOKIES = True
BROWSER_FOR_COOKIES = "chrome"
COOKIES_PATH = "/home/gatv-projects/Desktop/project/chatbot_system/cookies.txt"


def normalize_filename(filename: str) -> str:
    problematic_chars = r'[<>:"/\\|?*\x00-\x1f! ]'
    normalized_name = re.sub(problematic_chars, "_", filename)
    normalized_name = re.sub(r"_{2,}", "_", normalized_name)
    normalized_name = normalized_name.strip(" _")
    return normalized_name or "untitled_video"


def build_base_opts(use_browser_cookies: bool = True) -> dict:
    opts = {
        "format": "bv*+ba/b",
        "merge_output_format": "mp4",
        "noplaylist": True,
        "quiet": False,
        "no_warnings": False,
    }

    # Use only one cookie source
    if use_browser_cookies:
        opts["cookies_from_browser"] = (BROWSER_FOR_COOKIES,)
    else:
        opts["cookiefile"] = COOKIES_PATH

    return opts


def download_youtube_video(url: str, output_dir: str = OUTPUT_DIR) -> str:
    os.makedirs(output_dir, exist_ok=True)

    base_opts = build_base_opts(use_browser_cookies=USE_BROWSER_COOKIES)

    # First: extract info
    with yt_dlp.YoutubeDL(base_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        original_title = info.get("title", "untitled_video")
        normalized_title = normalize_filename(original_title)
        ext = info.get("ext", "mp4")

    print(f"Normalized filename: {normalized_title}")

    # Second: download with final outtmpl already set
    download_opts = {
        **base_opts,
        "outtmpl": os.path.join(output_dir, f"{normalized_title}.%(ext)s"),
    }

    with yt_dlp.YoutubeDL(download_opts) as ydl:
        ydl.download([url])

    # Expected final path
    final_path = os.path.abspath(os.path.join(output_dir, f"{normalized_title}.mp4"))
    if os.path.exists(final_path):
        return final_path

    # Fallback in case yt-dlp used original extension/container
    fallback_path = os.path.abspath(os.path.join(output_dir, f"{normalized_title}.{ext}"))
    return fallback_path


def main() -> None:
    if not VIDEO_URL:
        raise ValueError("Set VIDEO_URL in this file before running the script.")

    downloaded_path = download_youtube_video(VIDEO_URL)
    print(f"Video downloaded to: {downloaded_path}")


if __name__ == "__main__":
    main()