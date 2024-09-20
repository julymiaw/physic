import os
from moviepy.editor import VideoFileClip


def list_mp4_files(directory):
    return [f for f in os.listdir(directory) if f.endswith(".mp4")]


def convert_mp4_to_gif(mp4_path, gif_path, resize_factor=0.5, duration_per_segment=10):
    clip = VideoFileClip(mp4_path)
    # 降低分辨率
    clip = clip.resize(resize_factor)
    # 分段处理
    segments = []
    for start in range(0, int(clip.duration), duration_per_segment):
        end = min(start + duration_per_segment, clip.duration)
        segment = clip.subclip(start, end)
        segment_gif_path = (
            f"{os.path.splitext(gif_path)[0]}_part{start // duration_per_segment}.gif"
        )
        segment.write_gif(segment_gif_path)
        segments.append(segment_gif_path)

    # 合并GIF段
    with open(gif_path, "wb") as outfile:
        for segment_gif_path in segments:
            with open(segment_gif_path, "rb") as infile:
                outfile.write(infile.read())
            os.remove(segment_gif_path)  # 删除临时文件


def main():
    directory = "/home/july/physic/test"
    mp4_files = list_mp4_files(directory)

    if not mp4_files:
        print("No MP4 files found in the directory.")
        return

    print("Select an MP4 file to convert to GIF:")
    for idx, file in enumerate(mp4_files):
        print(f"{idx + 1}. {file}")

    choice = int(input("Enter the number of the file: ")) - 1

    if choice < 0 or choice >= len(mp4_files):
        print("Invalid choice.")
        return

    mp4_path = os.path.join(directory, mp4_files[choice])
    gif_path = os.path.splitext(mp4_path)[0] + ".gif"

    print(f"Converting {mp4_files[choice]} to {gif_path}...")
    convert_mp4_to_gif(mp4_path, gif_path)
    print("Conversion complete.")


if __name__ == "__main__":
    main()
