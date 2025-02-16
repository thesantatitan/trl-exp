from transformers.integrations import WandbCallback
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[logging.StreamHandler()]
)


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
        logging.info("called on_predict")
        super().on_predict(args, state, control, metrics, **kwargs)

    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        logging.info("called on_log")
        super().on_log(args, state, control, model=model, logs=logs, **kwargs)
