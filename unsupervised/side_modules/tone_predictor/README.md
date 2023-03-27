## Usage 

Extract the audio tone from a single segment
```
python extract_audio_tone.py
```
<figure>
  <figcaption>Expected tone plot</figcaption>
  <img src="output/tone_plot.png" width="400">
</figure>

Bar plot for a sampled balanced set of audio segments regarding the tone features
```
python full_extract_audio_tone.py
```

<figure>
  <figcaption>Expected output using tonnetz features</figcaption>
  <img src="output/tonnetz.png" width="400">
</figure>

<figure>
  <figcaption>Expected output using chroma features</figcaption>
  <img src="output/chromaa.png" width="400">
</figure>

<figure>
<figcaption>Expected output using spectral_contrast features</figcaption>
  <img src="output/spectral_contrast.png" width="400">
</figure>

<figure>
    <figcaption>Expected output using mfcc features</figcaption>
  <img src="output/mfcc.png" width="400">
</figure>

Visualize the types of features (audio emotional features/tone features)
```
sh visualize_features.sh
```
<figure>
    <figcaption>Expected output for CP segment</figcaption>
  <img src="output/M01003ME4_0012.png" width="400">
</figure>

<figure>
    <figcaption>Expected output for non-CP segment</figcaption>
  <img src="output/M01003JSD_0015.png" width="400">
</figure>
