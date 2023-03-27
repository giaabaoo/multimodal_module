## Usage 

Extract the audio tone from a single segment
```
python extract_audio_tone.py
```
![Expected output](output/tone_plot.png)

Bar plot for a sampled balanced set of audio segments regarding the tone features
```
python full_extract_audio_tone.py
```

<figure>
  <img src="output/tonnetz.png" width="400">
  <figcaption>Expected output using tonnetz features</figcaption>
</figure>

<figure>
  <img src="output/chroma.png" width="400">
  <figcaption>Expected output using chroma features</figcaption>
</figure>

<figure>
  <img src="output/spectral_contrast.png" width="400">
  <figcaption>Expected output using spectral_contrast features</figcaption>
</figure>

<figure>
  <img src="output/mfcc.png" width="400">
  <figcaption>Expected output using mfcc features</figcaption>
</figure>

Visualize the types of features (audio emotional features/tone features)
```
sh visualize_features.sh
```
![Expected output for CP segment](output/M01003ME4_0012.png)
![Expected output for non-CP segment](output/M01003JSD_0015.png)

