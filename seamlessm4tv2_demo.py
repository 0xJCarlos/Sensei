#Install the following dependencies first!
# pip install --quiet git+https://github.com/huggingface/transformers sentencepiece datasets

from transformers import AutoProcessor, SeamlessM4Tv2Model
from datasets import load_dataset

processor = AutoProcessor.from_pretrained("facebook/seamless-m4t-v2-large")
model = SeamlessM4Tv2Model.from_pretrained("facebook/seamless-m4t-v2-large")

#Modifiable variables :)
texto = input("Ingresa un texto a ser traducido a Japonés")
src_lang="spa"
tgt_lang="jpn"

#Preparing the text to be translated
text_inputs = processor(text=texto, src_lang=src_lang, return_tensors="pt")

#Generating speech
audio_array_from_text = model.generate(**text_inputs, tgt_lang=tgt_lang)[0].cpu().numpy().squeeze()

#Saving the audi
import scipy 
sample_rate = model.config.sampling_rate
scipy.io.wavfile.write("esp_to_jap_m4tv2.wav", rate=sample_rate, data=audio_array_from_text)

#Generating text
output_tokens = model.generate(**text_inputs, tgt_lang=tgt_lang, generate_speech=False)
translated_text = processor.decode(output_tokens[0].tolist()[0], skip_special_tokens=True)
print(f"Texto traducido: {translated_text}")