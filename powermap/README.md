# powermap

Tells from sound whether the grid power (NEPA) is on:

- **NEPA alarm** heard: the light just came on
- **generator** running: there is no light

It uses YAMNet, frozen, with a small trained head on top that outputs one score for
`alarm` and one for `gen`. Negative audio (everything else) should score low on both.

## Layout

```
powermap/
  detector.py   the model: YAMNet (imported from ../pytorch_dir) + the head
  training/     data.py (reads the csvs and wav audio), train.py
  serving/      listen.py + listen.html, the microphone interface
  saved/        detector.pt, written by training
```

## Data

- Labels: `../dataset/checked_data/{alarm,gen,negative}.csv`, rows of
  `YTID,title,start_seconds,end_seconds`. The csv a row is in is its label.
- Audio: 16 kHz wav files named `<YTID>_<title>.wav`, by default in
  `~/Documents/Research/Learning/agent-research/power_map/ml/data/raw` (`--raw-dir` to change).
  Rows without audio are skipped and listed.

## Train

```
.venv/bin/python powermap/training/train.py
```

This runs 5-fold cross-validation by video (a video is never in both train and test),
prints the scores next to YAMNet's own untrained classes, then trains on all videos and
saves `saved/detector.pt` with the thresholds.

## Listen

```
.venv/bin/python powermap/serving/listen.py
```

Then open http://localhost:8000, press **Start listening** and allow the microphone.
The page sends audio every 0.48 s and shows:

- **Light on** when the alarm sounds
- **No light** when a generator starts
- otherwise the last state, with when and why it changed

## Known issues

- Alarm data is small (5 videos), and the head is still worse than untrained YAMNet on alarm.
- The head has never heard alarm and generator together, so it misses the generator while
  the alarm sounds. The fix is to train on mixed patches.
- A neighbour's generator also reads as "no light".
