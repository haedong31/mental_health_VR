from pathlib2 import Path
from pydub import AudioSegment

def multiple_mp3_to_wav(source_dir, out_dir):
    source_dir = Path(source_dir)
    out_dir = Path(out_dir)
    Path.mkdir(out_dir, exist_ok=True)
    
    src_paths =  source_dir.glob('*.mp3')
    for src in src_paths:
        fname = src.stem
        fname = fname + '.wav'
        
        sound = AudioSegment.from_mp3(src)
        sound.export(out_dir/fname, format="wav")

##### convert mp3 to wav -----
multiple_mp3_to_wav("./data/audio/control_mp3", './data/audio/control_wav')
multiple_mp3_to_wav("./data/audio/dementia_mp3", './data/audio/dementia_wav')
