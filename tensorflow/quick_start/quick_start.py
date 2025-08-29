from trainer import Trainer

def run(args):
    r"""
        Args:
            args (ArgumentParser)
    """

    trainer = Trainer(args.config)
    trainer.train()
