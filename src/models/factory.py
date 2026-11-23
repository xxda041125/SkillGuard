from .detective import ToxicSkillDetective


def build_model(cfg):
    return ToxicSkillDetective(cfg)
