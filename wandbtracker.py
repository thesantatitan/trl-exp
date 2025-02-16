from transformers.integrations import WandbCallback
import pandas as pd
from rewards import SVGRewardFunction

class WandbPredictionProgressCallback(WandbCallback):
    """Custom WandbCallback to log model predictions during training.

    This callback logs model predictions and labels to a wandb.Table at each
    logging step during training. It allows to visualize the
    model predictions as the training progresses.

    Attributes:
        reward_func (SVGRewardFunction): The reward function used to
          evaluate the model's predictions.
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
