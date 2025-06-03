from transformers import TrainerCallback

class RealTimeLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "train_loss" in logs:
            print(f"\r Step {state.global_step}: Loss = {logs['train_loss']:.4f}", end="", flush=True)

    def on_evaluate(self, args, state, control, logs=None, **kwargs):
        if logs:
            print()