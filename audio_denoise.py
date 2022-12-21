from speechbrain.pretrained import SepformerSeparation as separator
import torchaudio

model = separator.from_hparams(source="speechbrain/sepformer-wham16k-enhancement", savedir='pretrained_models/sepformer-wham16k-enhancement')

# for custom file, change path
est_sources = model.separate_file(path='static/admin/noisy.wav') 

mono_audio = (est_sources[:, :, 0].detach().cpu())[0].numpy()
