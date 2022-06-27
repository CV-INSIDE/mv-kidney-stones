import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
# from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning.callbacks import EarlyStopping
import torch
from sklearn.metrics import classification_report, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random


class BaseModel(pl.LightningModule):
    """
    Base model where all the methods required by pytorch lightning are defined.
    """

    def __init__(self, hparams={}, index_to_label=None, seed=None):
        if seed != None:
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            pl.seed_everything(seed)
            # os.environ['PYTHONHASHSEED'] = str(seed)

        super(BaseModel, self).__init__()

        self.save_hyperparameters(hparams)
        self.index_to_label = index_to_label
        self.learning_rate = hparams["lr"]
        self.last_classification_report = None

    def set_class_indices(self, index_to_label):
        self.index_to_label = index_to_label

    def training_step(self, batch, batch_idx):
        batch1, batch2 = batch['view1'], batch['view2']
        # Information of view 1
        inputs1, targets1 = batch1
        # Information of view 2
        inputs2, targets2 = batch2
        # assert targets1 == targets2
        predictions = self(inputs1, inputs2)
        loss = self.loss_fn(predictions, targets1)
        _, preds = torch.max(predictions, 1)
        acc = torch.sum(preds == targets1.data) / (targets1.shape[0] * 1.0)
        # acc = self.multi_acc(y_val_pred, y)
        # self.logger('train_loss', loss)
        return {
            'loss': loss,
            'accuracy': acc
        }

    def on_fit_start(self):
        self.accuracy_history = {"train": [], "val": []}
        self.loss_history = {"train": [], "val": []}

    def on_fit_end(self):
        if len(self.accuracy_history["train"]) < len(self.accuracy_history["val"]):
            self.accuracy_history["val"] = self.accuracy_history["val"][:len(self.accuracy_history["train"])]

        if len(self.loss_history["train"]) < len(self.loss_history["val"]):
            self.loss_history["val"] = self.loss_history["val"][:len(self.loss_history["train"])]

    def test_step(self, batch, batch_idx):
        batch1, batch2 = batch['view1'], batch['view2']
        # Information of view 1
        inputs1, targets1 = batch1
        # Information of view 2
        inputs2, targets2 = batch2
        predictions = self(inputs1, inputs2)
        predictions = torch.log_softmax(predictions[0], dim=0)
        pred, pred_index = torch.max(predictions, dim=0)
        return {"real": targets1, "pred": pred_index, "correct": int(targets1 == pred_index)}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
        acc = sum([x["accuracy"] for x in outputs])
        avg_acc = acc / len(outputs)
        self.accuracy_history["train"].append(avg_acc.item())
        self.loss_history["train"].append(avg_loss.item())
        # logs = {'Train Loss': avg_loss, 'Train Accuracy': acc/len(outputs), 'step': self.current_epoch}
        self.logger.experiment.add_scalars("Loss", {"train_loss": avg_loss}, global_step=self.current_epoch)
        self.logger.experiment.add_scalars("Accuracy", {"train_acc": avg_acc}, global_step=self.current_epoch)
        # return {
        #  'loss': avg_loss,
        # 'log': logs
        # }

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_acc'].float() for x in outputs]).mean()
        self.accuracy_history["val"].append(avg_acc.item())
        self.loss_history["val"].append(avg_loss.item())
        self.logger.experiment.add_scalars("Loss", {"val_loss": avg_loss}, global_step=self.current_epoch)
        self.logger.experiment.add_scalars("Accuracy", {"val_acc": avg_acc}, global_step=self.current_epoch)

        logs = {'val_loss': avg_loss, 'val_acc': avg_acc}
        return {'progress_bar': logs}

    def test_y(self):
        return np.array(self.last_test_y)

    def test_y_pred(self):
        return np.array(self.last_test_y_pred)

    def test_epoch_end(self, outputs):
        real = [x["real"].item() for x in outputs]
        pred = [x["pred"].item() for x in outputs]
        correct = sum([x["correct"] for x in outputs])
        self.last_test_y = real
        self.last_test_y_pred = pred
        labels = None
        if self.index_to_label:
            labels = []
            for k in self.index_to_label:
                labels.append(self.index_to_label[k])

            cf = classification_report(real, pred, target_names=labels)
        else:
            cf = classification_report(real, pred)

        print(cf)

        cm = confusion_matrix(real, pred)
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        self.last_confusion_matrix = pd.DataFrame(cm)
        if self.index_to_label:
            self.last_confusion_matrix = pd.DataFrame(cm).rename(columns=self.index_to_label, index=self.index_to_label)
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(self.last_confusion_matrix, annot=True, ax=ax, cmap="Blues")

        self.last_classification_report = classification_report(real, pred)

        # result = pl.EvalResult()
        # result.log('accuracy', round(correct/len(outputs), 2))
        # return result
        return {'accuracy', round(correct / len(outputs), 2)}

    def teardown(self, stage):
        self.logger.experiment.flush()
        if stage == 'test' and self.last_classification_report:
            classes = ""

            if self.index_to_label:
                for k in self.index_to_label:
                    classes = classes + str(k) + "=> " + self.index_to_label[k] + "  \n"
                self.logger.experiment.add_text("Index class to label", classes)

            self.logger.experiment.add_text('Classification report', self.last_classification_report)
            # hparams['val_percentage'] = self.train_batch_size
            # print("datam", self._datamodule)
            # self.params['train_batch_size'] = self.datamodule().train_batch_size
            # self.logger.experiment.add_hparams({'val_percentage', self.datamodule().val_percentage})

    def validation_step(self, batch, batch_idx):
        # Split batch in views
        batch1, batch2 = batch['view1'], batch['view2']
        # Information of view 1
        inputs1, targets1 = batch1
        # Information of view 2
        inputs2, targets2 = batch2
        # assert targets1 == targets2
        predictions = self(inputs1, inputs2)
        val_loss = self.loss_fn(predictions, targets1)
        _, preds = torch.max(predictions, 1)
        acc = torch.sum(preds == targets1.data) / (targets1.shape[0] * 1.0)
        # acc = self.multi_acc(y_val_pred, y)
        # self.logger('val_loss', val_loss)
        return {'val_loss': val_loss, 'val_acc': acc}

    """
    Returns the model accuracy comparing y predicted vs y expected.
    """

    def multi_acc(self, y_pred, y_test):
        y_pred_softmax = torch.log_softmax(y_pred, dim=1)
        _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
        correct_pred = (y_pred_tags == y_test).float()
        acc = correct_pred.sum() / len(correct_pred)
        acc = acc * 100
        return acc

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def get_accuracy_history(self):
        return self.accuracy_history

    def get_loss_history(self):
        return self.loss_history

