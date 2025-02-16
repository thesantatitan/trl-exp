from transformers.integrations import WandbCallback
import pandas as pd
from .rewards import SVGRewardFunction

class WandbPredictionProgressCallback(WandbCallback):
    """Custom WandbCallback to log model predictions during training.

    This callback logs model predictions and labels to a wandb.Table at each
    logging step during training. It allows to visualize the
    model predictions as the training progresses.

    Attributes:
        trainer (Trainer): The Hugging Face Trainer instance.
        tokenizer (AutoTokenizer): The tokenizer associated with the model.
        sample_dataset (Dataset): A subset of the validation dataset
          for generating predictions.
        num_samples (int, optional): Number of samples to select from
          the validation dataset for generating predictions. Defaults to 100.
        freq (int, optional): Frequency of logging. Defaults to 2.
    """

    def __init__(self, reward_func):
        """Initializes the WandbPredictionProgressCallback instance.

        Args:
            reward_func (SVGRewardFunction): The reward function used to
              evaluate the model's predictions.
        """
        super().__init__()
        self.reward_func = reward_func

    def on_predict(self, args, state, control, metrics, **kwargs):
        for k,v in self.reward_func.rewards.items():
            metrics[f"reward/{k}"] = v/len(self.reward_func.count_since_logged)
            self.reward_func.rewards[k] = 0.0
        self.reward_func.count_since_logged = 0
        super().on_predict(args, state, control, metrics, **kwargs)
