import librosa

def load_LaionCLAP_audio():
    from fadtk import CLAPLaionModel
    clap_type = "audio"
    model = CLAPLaionModel(type=clap_type)
    model.load_model()
    return model

def load_LaionCLAP_music():
    from fadtk import CLAPLaionModel
    clap_type = "music"
    model = CLAPLaionModel(type=clap_type)
    model.load_model()
    return model

def load_MSCLAP():
    from fadtk import CLAPModel
    model = CLAPModel(type='2023')
    model.load_model()
    return model

def load_MERT_v1_95M():
    from fadtk import MERTModel
    model = MERTModel(size='v1-95M')
    model.load_model()
    return model

def load_MERT_v1_330M():
    from fadtk import MERTModel
    model = MERTModel(size='v1-330M')
    model.load_model()
    return model

def load_MERT_v0_public():
    from fadtk import MERTModel
    model = MERTModel(size='v0-public')
    model.load_model()
    return model

def load_VGGish():
    from fadtk import VGGishModel
    model = VGGishModel()
    model.load_model()
    return model

def _load_model(model_name):
    match model_name:
        case "LaionCLAP_audio":
            model = load_LaionCLAP_audio()

        case "LaionCLAP_music":
            model = load_LaionCLAP_music()

        case "MSCLAP":
            model = load_MSCLAP()

        case "MERT_v1-95M":
            model = load_MERT_v1_95M()

        case "MERT_v1-330M":
            model = load_MERT_v1_330M()

        case "MERT_v0-public":
            model = load_MERT_v0_public()

        case "VGGish":
            model = load_VGGish()
        
        case _:
            raise ValueError(f"Unknown model name: {model_name}")

    return model

def _load_audio(model, audio_path):
    """Helper to load audio file using librosa."""
    return librosa.load(audio_path, sr=model.sr, mono=True)[0]