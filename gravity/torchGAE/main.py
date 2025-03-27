from util.conf_util import init_config
from trainer import Trainer
from inferencer import Inferencer

if __name__ == "__main__":
    config = init_config("res/config.dev.yaml")

    trainer = Trainer(config)
    trainer.train()
    trainer.save_tensor_as_numpy(tensor=trainer.node_emb, path="artifact", name="emb.npy")
    
    inferencer = Inferencer(config)
    inferencer.topK()
    
    
