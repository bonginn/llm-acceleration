import torch
import torch.nn as nn
from tqdm import tqdm
from datasets import load_dataset
from trl import GKDConfig, GKDTrainer, SFTTrainer

class CustomGKDTrainer(GKDTrainer):
    def evaluate_ppl(self, model, tokenizer, device=None):
        if device is None:
            device = next(model.parameters()).device

        test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        test_enc = tokenizer("\n\n".join(test_dataset["text"]), return_tensors="pt")
        model.seqlen = 2048
        test_enc = test_enc.input_ids.to(device)

        nsamples = test_enc.numel() // model.seqlen
        nlls = []

        model.eval()
        for i in tqdm(range(nsamples), desc="Computing PPL"):
            batch = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)]

            with torch.no_grad():
                lm_logits = model(batch).logits

            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]

            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            neg_log_likelihood = loss.float() * model.seqlen
            nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
        return ppl.item()

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)

        student_ppl = self.evaluate_ppl(self.model, self.processing_class)
        metrics[f"{metric_key_prefix}_ppl"] = student_ppl

        teacher_ppl = self.evaluate_ppl(self.teacher_model, self.processing_class)
        metrics[f"{metric_key_prefix}_teacher_ppl"] = teacher_ppl

        print(f"Step {self.state.global_step}: Student PPL = {student_ppl:.2f}, Teacher PPL = {teacher_ppl:.2f}")

        return metrics
    
class CustomSFTTrainer(SFTTrainer):
    def evaluate_ppl(self, model, tokenizer):

        test_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        test_enc = tokenizer("\n\n".join(test_dataset["text"]), return_tensors="pt")
        model.seqlen = 2048
        test_enc = test_enc.input_ids.to(model.device)

        nsamples = test_enc.numel() // model.seqlen
        nlls = []

        model.eval()
        for i in tqdm(range(nsamples), desc="Computing PPL"):
            batch = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)]
            with torch.no_grad():
                lm_logits = model(batch).logits
            shift_logits = lm_logits[:, :-1, :].contiguous().float()
            shift_labels = test_enc[:, (i * model.seqlen):((i + 1) * model.seqlen)][:, 1:]
            loss_fct = torch.nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            neg_log_likelihood = loss.float() * model.seqlen
            nlls.append(neg_log_likelihood)
        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * model.seqlen))
        return ppl.item()

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        metrics = super().evaluate(eval_dataset, ignore_keys, metric_key_prefix)
        ppl = self.evaluate_ppl(self.model, self.processing_class)
        metrics[f"{metric_key_prefix}_ppl"] = ppl
        print(f"PPL: {ppl:.2f}")
        return metrics