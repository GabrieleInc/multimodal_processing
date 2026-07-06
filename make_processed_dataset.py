import os
from pathlib import Path
import pandas as pd
import shutil
import processin_functions as process
from config import DURATION, WIDTH, HEIGHT, FRAMERATE, NBFRAMES, SAMPLERATE_DNN, BITRATE_DNN


source_path =  "../data/videos" # VGGSound videos
source_csv_path = "../data/vggsound/vggsound.csv" # VGGSound CSV from https://github.com/hche11/VGGSound/blob/master/data/vggsound.csv

# All these directories are created if they don't exist
root_path = "../data/VGGSound_processed"

raw_video_path = "videos_raw"
raw_audio_path = "audios_raw"
video_error_csv_path = "raw_video_errors.csv"
audio_error_csv_path = "raw_audio_errors.csv"
multimodal_dataset_csv_path_unfiltered = "dataset_unfiltered.csv"
video_metrics_path_unprocessed = "video_metrics_unprocessed.csv"
audio_metrics_path_unprocessed = "audio_metrics_unprocessed.csv"

video_path = "videos"
video_path_frames = "videos_frames"
audio_path = "audios"
audio_path_mono = "audios_mono"
audio_path_stereo = "audios_stereo"
video_csv_path = "videos.csv"
audio_csv_path = "audios.csv"
multimodal_dataset_csv_path = "dataset.csv"
video_metrics_path = "video_metrics.csv"
audio_metrics_path = "audio_metrics.csv"

merged_video_path = "merged_videos"  # Final merged videos

raw_video_path = os.path.join(root_path, raw_video_path)
raw_audio_path = os.path.join(root_path, raw_audio_path)
video_error_csv_path = os.path.join(root_path, video_error_csv_path)
audio_error_csv_path = os.path.join(root_path, audio_error_csv_path)
multimodal_dataset_csv_path_unfiltered = os.path.join(root_path, multimodal_dataset_csv_path_unfiltered)
video_metrics_path_unprocessed = os.path.join(root_path, video_metrics_path_unprocessed)
video_csv_path = os.path.join(root_path, video_csv_path)
video_path = os.path.join(root_path, video_path)
video_path_frames = os.path.join(root_path, video_path_frames)
audio_metrics_path_unprocessed = os.path.join(root_path, audio_metrics_path_unprocessed)
audio_csv_path = os.path.join(root_path, audio_csv_path)
audio_path = os.path.join(root_path, audio_path)
audio_path_mono = os.path.join(root_path, audio_path_mono)
audio_path_stereo = os.path.join(root_path, audio_path_stereo)
multimodal_dataset_csv_path = os.path.join(root_path, multimodal_dataset_csv_path)
video_metrics_path = os.path.join(root_path, video_metrics_path)
audio_metrics_path = os.path.join(root_path, audio_metrics_path)
merged_video_path = os.path.join(root_path, merged_video_path)

# Ensure target directories exist
for path in [root_path,
             raw_video_path, raw_audio_path,
             video_path, video_path_frames,
             audio_path, audio_path_mono, audio_path_stereo,
             merged_video_path,
             ]:
        os.makedirs(path, exist_ok=True)

# %%


# Load VGGSound CSV (without header)
df = pd.read_csv(source_csv_path, header=None)
df.columns = ["video_id", "label_code", "label", "split"]
# Ensure label_code is a string
df["label_code"] = df["label_code"].astype(str)
# Create unique_id: video_id + "_" + label_code padded to 6 digits
df.insert(
    0,
    "unique_id",
    df["video_id"].astype(str) + "_" + df["label_code"].str.zfill(6)
)
df.to_csv(multimodal_dataset_csv_path_unfiltered, index=False)


### Extract video and audio from merged files
process.extract_video_audio_from_DIRECTORY(raw_video_path, raw_audio_path, source_path, video_error_csv_path, audio_error_csv_path)

# %%
### Extract video meta data
process.extract_metadata(raw_video_path, video_metrics_path_unprocessed, "video")

### Filter raw video quality
df_video = pd.read_csv(video_metrics_path_unprocessed)
# Filter
df_video = df_video[df_video["Duration"] >= DURATION]
df_video = df_video[(df_video["Framerate"] >= FRAMERATE) & (df_video["Nb_frames"] >= NBFRAMES)]
df_video = df_video[(df_video["Width"] >= WIDTH) & (df_video["Height"] >= HEIGHT)]
# Save
df_video.to_csv(video_csv_path, index=False)

### Equalise frame rate, duration, size
# For human viewing and to merge the videos again at the end:
process.reencode_files(video_csv_path, raw_video_path, video_path, "mp4", "merger") # same as "human"
# For DNN - adapt config.py (# video encoding):
process.reencode_files(video_csv_path, raw_video_path, video_path_frames, "mp4", "dnn")

# %%
### Extract audio meta data
process.extract_metadata(raw_audio_path, audio_metrics_path_unprocessed, "audio")

### Filter raw audio quality
df_audio = pd.read_csv(audio_metrics_path_unprocessed)
# Filter
df_audio = df_audio[df_audio["Duration"] >= DURATION]
df_audio = df_audio[df_audio["Integrated_Loudness"] >= -99] # Filter out audios with no sound
# Save
df_audio.to_csv(audio_csv_path, index=False)

### Equalise sample rate, duration, channels
# To merge the videos again at the end:
process.reencode_files(audio_csv_path, raw_audio_path, audio_path, "m4a", "merger")
# For DNN - adapt config.py (# audio encoding):
process.reencode_files(audio_csv_path, raw_audio_path, audio_path_mono,"m4a", "dnn")
# For human listening:
#process.reencode_files(audio_csv_path, raw_audio_path, audio_path_stereo, "m4a", "human")

# %%
### Discard the files that were filtered out
def filter_files(video_csv_p, audio_csv_p, multimodal_dataset_csv_p_unfiltered, multimodal_dataset_csv_p, root_p,
                     video_frames_p):
    # === 1. Load CSVs ===
    df_video = pd.read_csv(video_csv_p)  # contains "File" column with .mp4
    df_audio = pd.read_csv(audio_csv_p)  # contains "File" column with .m4a
    id_df = pd.read_csv(multimodal_dataset_csv_p_unfiltered)  # contains "unique_id" column (no suffix)

    # === 2. Extract stems from video/audio ===
    video_ids = df_video['File'].apply(lambda x: Path(x).stem)
    audio_ids = df_audio['File'].apply(lambda x: Path(x).stem)

    # === 3. Combine all known IDs ===
    intersection_ids = set(video_ids) & set(audio_ids)

    # === 4. Find missing ones ===
    all_ids = set(id_df['unique_id'])
    missing_ids = all_ids - intersection_ids

    print(f"\nMissing IDs after filtering ({len(missing_ids)})")
    #for id in sorted(missing_ids):
    #    print(id)

    # === 5. Remove missing from CSV and save ===
    id_df = id_df[~id_df['unique_id'].isin(missing_ids)]
    id_df.to_csv(multimodal_dataset_csv_p, index=False)

    # === 6. Delete any matching .mp4, .m4a, or .wav files ===
    folder_to_clean = Path(root_p)

    def should_delete(file_path):
        return file_path.stem in missing_ids and file_path.suffix.lower() in ['.mp4', '.m4a', '.wav']

    deleted_files = []

    for filepath in folder_to_clean.rglob("*"):
        if filepath.is_file() and should_delete(filepath):
            try:
                filepath.unlink()
                deleted_files.append(str(filepath))
            except Exception as e:
                print(f"Could not delete {filepath}: {e}")

    print(f"Deleted {len(deleted_files)} files associated with missing IDs")

    # === 7. Delete frame folders for missing IDs ===
    frames_root = Path(video_frames_p)
    deleted_dirs = []

    for clip_dir in frames_root.iterdir():
        if clip_dir.is_dir() and clip_dir.name in missing_ids:
            try:
                shutil.rmtree(clip_dir)
                deleted_dirs.append(str(clip_dir))
            except Exception as e:
                print(f"Could not delete {clip_dir}: {e}")

    print(f"Deleted {len(deleted_dirs)} frame folders associated with missing IDs")

# %%
filter_files(video_csv_path, audio_csv_path, multimodal_dataset_csv_path_unfiltered, multimodal_dataset_csv_path, root_path, video_path_frames)
os.remove(multimodal_dataset_csv_path_unfiltered) # it was a copy of the VGGSound CSV, not needed anymore

# %%
### Extract metadata of preprocessed and selected files
process.extract_metadata(video_path, video_metrics_path, "video")
process.extract_metadata(audio_path_mono, audio_metrics_path, "audio")


# %%
### Get rid of files where preprocessing unsuccessful
df_video = pd.read_csv(video_metrics_path)
#df_video = df_video[df_video["Duration"] == DURATION]
#df_video = df_video[df_video["Nb_frames"] == NBFRAMES]
df_video.to_csv(video_metrics_path, index=False)

df_audio = pd.read_csv(audio_metrics_path)
df_audio = df_audio[df_audio["Duration"] == DURATION]
df_audio = df_audio[df_audio["Sample_Rate"] == SAMPLERATE_DNN]
df_audio = df_audio[df_audio["Bit_Rate"] == BITRATE_DNN]
df_audio.to_csv(audio_metrics_path, index=False)

filter_files(video_metrics_path, audio_metrics_path, multimodal_dataset_csv_path, multimodal_dataset_csv_path, root_path, video_path_frames)

# %%
### Merge video and audio
process.merge_video_audio(video_path, audio_path, merged_video_path)
