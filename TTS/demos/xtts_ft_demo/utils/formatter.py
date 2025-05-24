import gc
import os

import pandas
import torch
import torchaudio
from faster_whisper import WhisperModel
from tqdm import tqdm

# torch.set_num_threads(1)
from TTS.tts.layers.xtts.tokenizer import multilingual_cleaners

torch.set_num_threads(16)

audio_types = (".wav", ".mp3", ".flac")


def list_audios(basePath, contains=None):
    # return the set of files that are valid
    return list_files(basePath, validExts=audio_types, contains=contains)


def list_files(basePath, validExts=None, contains=None):
    # loop over the directory structure
    for rootDir, dirNames, filenames in os.walk(basePath):
        # loop over the filenames in the current directory
        for filename in filenames:
            # if the contains string is not none and the filename does not contain
            # the supplied string, then ignore the file
            if contains is not None and filename.find(contains) == -1:
                continue

            # determine the file extension of the current file
            ext = filename[filename.rfind(".") :].lower()

            # check to see if the file is an audio and should be processed
            if validExts is None or ext.endswith(validExts):
                # construct the path to the audio and yield it
                audioPath = os.path.join(rootDir, filename)
                yield audioPath


def format_audio_list(
    audio_files,
    target_language="en",
    out_path=None,
    buffer=0.2,
    eval_percentage=0.15,
    speaker_name="coqui",
    gradio_progress=None,
):
    audio_total_size = 0
    os.makedirs(out_path, exist_ok=True)

    # Check if audio_files is empty
    if not audio_files:
        raise ValueError("No audio files found in the provided directory.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Loading Whisper Model!")
    asr_model = WhisperModel("large-v2", device=device, compute_type="float16")

    metadata = {"audio_file": [], "text": [], "speaker_name": []}

    if gradio_progress is not None:
        tqdm_object = gradio_progress.tqdm(audio_files, desc="Formatting...")
    else:
        tqdm_object = tqdm(audio_files)

    for audio_path in tqdm_object:
        try:
            wav, sr = torchaudio.load(audio_path)
            if wav.size(0) != 1:
                wav = torch.mean(wav, dim=0, keepdim=True)

            wav = wav.squeeze()
            audio_total_size += wav.size(-1) / sr

            # Fix: Force immediate transcription to avoid tqdm issues
            segments, _ = asr_model.transcribe(audio_path, word_timestamps=True, language=target_language)
            segments = list(segments)  # Convert generator to list immediately

            i = 0
            sentence = ""
            sentence_start = None
            first_word = True
            words_list = []

            for segment in segments:
                words_list.extend(list(segment.words))

            for word_idx, word in enumerate(words_list):
                if first_word:
                    sentence_start = max(word.start - buffer, 0) if word_idx == 0 else max(
                        word.start - buffer, (words_list[word_idx - 1].end + word.start) / 2
                    )
                    sentence = word.word
                    first_word = False
                else:
                    sentence += word.word

                if word.word[-1] in ["!", ".", "?"]:
                    sentence = sentence[1:]
                    sentence = multilingual_cleaners(sentence, target_language)
                    audio_file_name = os.path.splitext(os.path.basename(audio_path))[0]
                    audio_file = f"wavs/{audio_file_name}_{str(i).zfill(8)}.wav"

                    next_word_start = words_list[word_idx + 1].start if word_idx + 1 < len(words_list) else (wav.shape[0] - 1) / sr
                    word_end = min((word.end + next_word_start) / 2, word.end + buffer)

                    abs_path = os.path.join(out_path, audio_file)
                    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
                    i += 1
                    first_word = True

                    audio = wav[int(sr * sentence_start) : int(sr * word_end)].unsqueeze(0)
                    if audio.size(-1) >= sr / 3:
                        torchaudio.save(abs_path, audio, sr)
                        metadata["audio_file"].append(audio_file)
                        metadata["text"].append(sentence)
                        metadata["speaker_name"].append(speaker_name)

        except Exception as e:
            print(f"Error processing {audio_path}: {str(e)}")
            continue

    if not metadata["audio_file"]:
        raise RuntimeError("No valid audio segments were processed. Check input files and Whisper output.")

    df = pandas.DataFrame(metadata).sample(frac=1)
    num_val_samples = int(len(df) * eval_percentage)
    df_eval = df[:num_val_samples].sort_values("audio_file")
    df_train = df[num_val_samples:].sort_values("audio_file")

    train_metadata_path = os.path.join(out_path, "metadata_train.csv")
    eval_metadata_path = os.path.join(out_path, "metadata_eval.csv")
    df_train.to_csv(train_metadata_path, sep="|", index=False)
    df_eval.to_csv(eval_metadata_path, sep="|", index=False)

    del asr_model
    gc.collect()
    return train_metadata_path, eval_metadata_path, audio_total_size
