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
![Expected output using tonnetz features](tonnetz.png)
![Expected output using chroma features](output/chroma.png)
![Expected output using spectral_contrast features](output/spectral_contrast.png)
![Expected output using mfcc features](output/mfcc.png)


Visualize the types of features (audio emotional features/tone features)

```
sh visualize_features.sh
```
![Expected output for CP segment](output/M01003ME4_0012.png)
![Expected output for non-CP segment](output/M01003JSD_0015.png)

