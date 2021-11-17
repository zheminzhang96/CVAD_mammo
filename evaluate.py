import logging
import torch
from numpy import sqrt, argmax
from sklearn.metrics import auc, roc_curve

def get_fpr_tpr_auc(Y_label, Y_preds): 
    fpr, tpr, thresholds = roc_curve(Y_label, Y_preds)
    aucscore = auc(fpr, tpr)
    gmeans = sqrt(tpr * (1-fpr))
    ix = argmax(gmeans)
    logger = logging.getLogger()
    logger.info('Best Threshold=%f, G-Mean=%.3f, FPR=%.3f, TPR=%.3f, AUC=%.3f' % (thresholds[ix], gmeans[ix], fpr[ix], tpr[ix], aucscore))
    return fpr, tpr, aucscore


def cvad_evaluate(embnet, cls_model, recon_loss, cls_loss, test_dataloader, device):
    logger = logging.getLogger()
    logger.info("----------- CVAD evaluating------------")
    Targets = []
    anomaly_score = []

    with torch.set_grad_enabled(False):    
        for idx, inputs in enumerate(test_dataloader):
        
            images, targets = inputs
            images = images.to(device)

            for i in range(0, images.shape[0]):
                recon_x, mu, logvar, mu2, logvar2 = embnet(images[i].unsqueeze(0))
                outputs = cls_model(images[i].unsqueeze(0))
                cvae_loss = recon_loss(recon_x, images[i].unsqueeze(0), mu, logvar, mu2, logvar2)

                if not np.isnan(cvae_loss.item()+outputs.detach().cpu().numpy()[0][0]) and not np.isinf(cvae_loss.item()+outputs.detach().cpu().numpy()[0][0]):
                    anomaly_score.append(cvae_loss.item()+outputs.detach().cpu().numpy()[0][0])
                    Targets.append(targets[i].detach().cpu().numpy())
            
    Y_label = np.array(np.vstack(Targets).squeeze(1),dtype=int).tolist() 
    Y_preds = []
    for s in anomaly_score:
        Y_preds.append((s-np.min(np.array(anomaly_score)))/(np.max(np.array(anomaly_score))-np.min(np.array(anomaly_score))))
    aucscore = None
    fpr, tpr, aucscore = get_fpr_tpr_auc(Y_label, Y_preds) 
    return fpr, tpr, aucscore  