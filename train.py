import numpy as np
import os
import torch
from torch import nn
from torch import optim
from sklearn import metrics
import store_result
from typing import Dict,Tuple, Optional, Any
from dataclasses import dataclass

from utils import save_pickle

@dataclass
class MetricsResult:
    accuracy: float
    auc: float
    loss: float
    f1_score: float
    auprc: float
    sensitivity: Optional[float] = None
    specificity: Optional[float] = None
    ppv: Optional[float] = None
    npv: Optional[float] = None

class ModelTrainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module = nn.BCELoss(),
        device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ):
        self.model = model
        self.criterion = criterion
        self.device = device
        
    def _process_batch(
        self,
        inputs: torch.Tensor,
        labels: torch.Tensor,
        demography: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Process a single batch of data."""
        inputs = inputs.to(self.device, dtype=torch.float)
        labels = labels.to(self.device, dtype=torch.float)
        
        if demography is not None:
            output = self.model(inputs, demography)
        else:
            output = self.model(inputs)
            
        output = output.to(self.device, dtype=torch.float)
        output = torch.reshape(output, (output.shape[0], 1))
        labels = torch.reshape(labels, (labels.shape[0], 1))
        
        return output, labels, labels.detach().cpu().numpy()

    def _prepare_demography(
        self,
        minibatch_idx: torch.Tensor,
        demography_data: Dict
    ) -> torch.Tensor:
        """Prepare demography data for the current batch."""
        minibatch_demography = {key: demography_data[key] for key in minibatch_idx.cpu().numpy()}
        demography = [list(minibatch_demography[key]) for key in minibatch_demography.keys()]
        return torch.FloatTensor(demography).to(self.device, dtype=torch.float)

    def _calculate_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray
    ) -> Tuple[MetricsResult, Dict[str, float], float]:
        """Calculate various classification metrics."""
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_scores, pos_label=1)
        
        # Find optimal threshold using Youden's Index
        J = tpr - fpr
        best_idx = np.argmax(J)
        best_threshold = thresholds[best_idx]
        
        # Calculate binary predictions using best threshold
        y_pred_binary = (y_scores > best_threshold).astype(int)
        
        # Calculate confusion matrix
        cf_mat = metrics.confusion_matrix(y_true, y_pred_binary, labels=[0, 1])
        tn, fp, fn, tp = cf_mat.ravel()
        
        # Calculate metrics
        sensitivity = round(tp / (tp + fn), 3)
        specificity = round(tn / (fp + tn), 3)
        ppv = round(tp / (tp + fp), 3)
        npv = round(tn / (fn + tn), 3)
        
        metrics_result = MetricsResult(
            accuracy=round(metrics.accuracy_score(y_true, y_pred_binary), 3),
            auc=round(metrics.auc(fpr, tpr), 3),
            loss=round(np.mean(y_pred), 3),
            f1_score=round(metrics.f1_score(y_true, y_pred_binary), 3),
            auprc=round(metrics.average_precision_score(y_true, y_scores), 3),
            sensitivity=sensitivity,
            specificity=specificity,
            ppv=ppv,
            npv=npv
        )
        
        thresholds_dict = {
            'best_thresh': best_threshold,
            'best_thresh_idx': best_idx
        }
        
        return metrics_result, thresholds_dict, best_threshold

    def _save_results(
        self,
        metrics_result: MetricsResult,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_scores: np.ndarray,
        result_path: str,
        phase: str,
        line_str: str
    ) -> None:
        """Save training/validation results."""
        # Save predictions
        save_pickle(y_pred, os.path.join(result_path, f'total_pred_label ({phase}).pkl'))
        save_pickle(y_true, os.path.join(result_path, f'total_truth ({phase}).pkl'))
        save_pickle(y_scores, os.path.join(result_path, f'total_prediction ({phase}).pkl'))
        
        # Save classification report
        classification_results = metrics.classification_report(
            y_true, y_pred, target_names=['0', '1'], output_dict=True
        )
        store_result.save_to_csv(
            classification_results,
            os.path.join(result_path, f'[{phase}] result')
        )
        
        # Create visualization plots
        fpr, tpr, _ = metrics.roc_curve(y_true, y_scores, pos_label=1)
        precision, recall, _ = metrics.precision_recall_curve(y_true, y_scores)
        
        store_result.make_roc_curve(
            fpr, tpr, metrics_result.auc,
            metrics_result.sensitivity, metrics_result.specificity,
            result_path, title=phase, line_str=line_str
        )
        store_result.make_prc_curve(
            recall, precision, metrics_result.auprc,
            result_path, title=phase, line_str=line_str
        )
        store_result.make_confusion_matrix(
            y_true, y_pred, result_path, title=phase
        )

    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        optimizer: torch.optim.Optimizer,
        train_demography: Optional[Dict] = None
    ) -> MetricsResult:
        """Train for one epoch."""
        self.model.train()
        predictions, targets, losses = [], [], []
        
        for data in train_loader:
            inputs, labels, minibatch_idx = data
            
            if train_demography:
                demography = self._prepare_demography(minibatch_idx, train_demography)
            else:
                demography = None
                
            output, labels, truth = self._process_batch(inputs, labels, demography)
            
            loss = self.criterion(output, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            predictions.extend(output.detach().cpu().numpy())
            targets.extend(truth)
            losses.append(loss.item())
            
        metrics_result, _, _ = self._calculate_metrics(
            np.array(targets),
            np.array(losses),
            np.array(predictions)
        )
        
        return metrics_result

    def validate(
        self,
        valid_loader: torch.utils.data.DataLoader,
        result_path: str,
        line_str: str,
        valid_demography: Optional[Dict] = None
    ) -> Tuple[MetricsResult, Dict[str, float]]:
        """Validate the model."""
        self.model.eval()
        predictions, targets, losses = [], [], []
        
        with torch.no_grad():
            for data in valid_loader:
                inputs, labels, minibatch_idx = data
                
                if valid_demography:
                    demography = self._prepare_demography(minibatch_idx, valid_demography)
                else:
                    demography = None
                    
                output, labels, truth = self._process_batch(inputs, labels, demography)
                loss = self.criterion(output, labels)
                
                predictions.extend(output.detach().cpu().numpy())
                targets.extend(truth)
                losses.append(loss.item())
        
        predictions = np.array(predictions)
        targets = np.array(targets)
        
        metrics_result, thresholds_dict, _ = self._calculate_metrics(
            targets, losses, predictions
        )
        
        self._save_results(
            metrics_result, targets, predictions, predictions,
            result_path, 'valid', line_str
        )
        
        return metrics_result, thresholds_dict

def run_train_valid(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    valid_loader: torch.utils.data.DataLoader,
    counts: np.ndarray,
    epoch: int,
    parameter_dict: Dict[str, Any],
    best_dict: Dict[str, float],
    train_demography: Optional[Dict] = None,
    valid_demography: Optional[Dict] = None
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """Main training and validation function."""
    trainer = ModelTrainer(model, device=parameter_dict['device'])
    optimizer = optim.Adam(
        model.parameters(),
        lr=parameter_dict['learning_rate'],
        weight_decay=parameter_dict['weight_decay']
    )
    
    # Train epoch
    train_metrics = trainer.train_epoch(train_loader, optimizer, train_demography)
    
    # Validate
    valid_metrics, thresholds = trainer.validate(
        valid_loader,
        parameter_dict['RESULT_PATH'],
        parameter_dict['line_str'],
        valid_demography
    )
    
    # Update best metrics if needed
    if valid_metrics.auc > best_dict['BEST_AUC_v']:
        best_dict.update({
            'BEST_AUC_v': valid_metrics.auc,
            'BEST_ACC_v': valid_metrics.accuracy,
            'BEST_LOSS_v': valid_metrics.loss,
            'BEST_F1_v': valid_metrics.f1_score,
            'BEST_AUPRC_v': valid_metrics.auprc,
            'BEST_sens': valid_metrics.sensitivity,
            'BEST_spec': valid_metrics.specificity,
            'BEST_ppv': valid_metrics.ppv,
            'BEST_npv': valid_metrics.npv
        })
        
        # Save best model
        if parameter_dict['RESULT_PATH'].split('/')[-1] == '5':
            store_result.save_checkpoint(
                os.path.join(parameter_dict['RESULT_PATH'], 'model(best_auc).pt'),
                model, optimizer, valid_metrics.loss
            )
            save_pickle(
                thresholds,
                os.path.join(parameter_dict['RESULT_PATH'], 'roc_threshold.pkl')
            )
    
    return (
        train_metrics.__dict__,
        valid_metrics.__dict__,
        best_dict
    )

def run_test(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    num_features: int,
    device: torch.device,
    what_model: str,
    valid_best_threshold: float,
    line_str: str,
    result_path: str,
    test_demography: Optional[Dict] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Run model testing and evaluation
    
    Args:
        model: The neural network model
        test_loader: DataLoader for test data
        num_features: Number of features to use
        device: Device to run the model on
        what_model: Model type identifier
        valid_best_threshold: Best threshold from validation
        line_str: String identifier for plotting
        result_path: Path to save results
        test_demography: Optional demographic data
        
    Returns:
        Tuple containing predictions, ground truth, probabilities, and metrics
    """
    trainer = ModelTrainer(model, device=device)
    predictions, targets, probabilities = [], [], []
    
    model.eval()
    with torch.no_grad():
        for data in test_loader:
            inputs, labels, minibatch_idx = data
            
            if test_demography:
                demography = trainer._prepare_demography(minibatch_idx, test_demography)
            else:
                demography = None
                
            output, labels, truth = trainer._process_batch(inputs, labels, demography)
            loss = trainer.criterion(output, labels)
            
            probabilities.extend(output.detach().cpu().numpy())
            predictions.extend(output.detach().cpu().numpy())
            targets.extend(truth)
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    probabilities = np.array(probabilities)
    
    # Calculate metrics using validation threshold
    pred_labels = (predictions > valid_best_threshold).astype(int)
    
    # Calculate ROC curve metrics
    fpr, tpr, thresholds = metrics.roc_curve(targets, probabilities, pos_label=1)
    threshold_idx = (np.abs(thresholds - valid_best_threshold)).argmin()
    
    # Calculate confusion matrix and derived metrics
    cf_mat = metrics.confusion_matrix(targets, pred_labels, labels=[0, 1])
    tn, fp, fn, tp = cf_mat.ravel()
    
    sensitivity = round(tp / (tp + fn), 3)
    specificity = round(tn / (fp + tn), 3)
    ppv = round(tp / (tp + fp), 3)
    npv = round(tn / (fn + tn), 3)
    
    # Calculate final metrics
    final_metrics = {
        'FINAL_AUC': round(metrics.auc(fpr, tpr), 3),
        'FINAL_ACC': round(metrics.accuracy_score(targets, pred_labels), 3),
        'FINAL_LOSS': round(np.mean(predictions), 3),
        'FINAL_F1_SCORE': round(metrics.f1_score(targets, pred_labels), 3),
        'FINAL_SENS': sensitivity,
        'FINAL_SPEC': specificity,
        'FINAL_PPV': ppv,
        'FINAL_NPV': npv,
        'FINAL_AUPRC': round(metrics.average_precision_score(targets, predictions), 3)
    }
    
    # Save results
    save_pickle(pred_labels, os.path.join(result_path, 'total_pred_label (test).pkl'))
    save_pickle(targets, os.path.join(result_path, 'total_truth (test).pkl'))
    save_pickle(predictions, os.path.join(result_path, 'total_prediction (test).pkl'))
    
    # Save metrics to file
    metrics_str = (f'[test] F1{final_metrics["FINAL_F1_SCORE"]} '
                  f'AUPRC{final_metrics["FINAL_AUPRC"]} '
                  f'AUC{final_metrics["FINAL_AUC"]} '
                  f'ACC{final_metrics["FINAL_ACC"]} '
                  f'LOSS{final_metrics["FINAL_LOSS"]} '
                  f'SENS{final_metrics["FINAL_SENS"]} '
                  f'SPEC{final_metrics["FINAL_SPEC"]} '
                  f'PPV{final_metrics["FINAL_PPV"]} '
                  f'NPV{final_metrics["FINAL_NPV"]}')
    
    with open(os.path.join(result_path, f'{metrics_str}.txt'), 'w') as f:
        pass
    
    # Generate plots
    store_result.make_confusion_matrix(
        targets, pred_labels, result_path, title='test'
    )
    store_result.make_roc_curve(
        fpr, tpr, final_metrics['FINAL_AUC'],
        threshold_idx, valid_best_threshold,
        final_metrics['FINAL_SENS'],
        final_metrics['FINAL_SPEC'],
        result_path, title='test',
        line_str=line_str
    )
    
    precision, recall, _ = metrics.precision_recall_curve(targets, predictions)
    store_result.make_prc_curve(
        recall, precision,
        final_metrics['FINAL_AUPRC'],
        result_path, title='test',
        line_str=line_str
    )
    
    return pred_labels, targets, probabilities, final_metrics