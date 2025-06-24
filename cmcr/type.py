from types import SimpleNamespace

ModalityType = SimpleNamespace(
    VISION="vision",
    TEXT="text",
    ENG_TEXT="eng_text",
    KOR_TEXT="kor_text",
    AUDIO="audio",
    PC="pointcloud",
)

MCRType = SimpleNamespace(
    CLIP="clip",
    CLAP="clap",
    ULIP="ulip"
)