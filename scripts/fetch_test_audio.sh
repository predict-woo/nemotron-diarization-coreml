#!/bin/sh
# Downloads the test recordings (hf-internal-testing/dummy-audio-samples) and resamples them to 16 kHz mono
# float32 WAV in data/16k, so that resampling is not part of what the equivalence tests compare.
set -eu
cd "$(dirname "$0")/.."
mkdir -p data/raw data/16k
base=https://huggingface.co/datasets/hf-internal-testing/dummy-audio-samples/resolve/main
for f in diarization_example.mp3 two_speakers.wav dude_where_is_my_car.wav en-Alice_woman.wav en-Carter_man.wav \
         en-Frank_man.wav Ch6Ae9DT6Ko_00-04-03_00-04-31.wav macron.wav obama.mp3 librispeech_mr_quilter.wav \
         fleurs_en_us_test_0346.wav mandarin_voxcpm_zh.wav; do
    [ -f "data/raw/$f" ] || curl -sfL -o "data/raw/$f" "$base/$f"
    ffmpeg -v error -y -i "data/raw/$f" -ac 1 -ar 16000 -c:a pcm_f32le "data/16k/${f%.*}.wav"
done
echo "test audio in data/16k"
