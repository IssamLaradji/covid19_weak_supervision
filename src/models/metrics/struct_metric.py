from collections import defaultdict

from scipy import spatial
import numpy as np
import torch

def compute_struct_metric(pred_mask, masks):
    masks = masks.float().squeeze()
    pred_mask = torch.from_numpy(pred_mask).float().squeeze()
    y = masks.mean();

    if (y==0):
        x = pred_mask.mean();
        Q = 1.0 - x
    elif (y==1):
        x = pred_mask.mean();
        Q = x
    else:
        alpha = 0.5
        Q = alpha*S_object(pred_mask,masks)+(1-alpha)*S_region(pred_mask, masks);
        if (Q<0):
            Q=0;
    return Q 
    
def S_region(prediction, GT):
    X, Y = centroid(GT);

    # divide GT into 4 regions
    [GT_1,GT_2,GT_3,GT_4,w1,w2,w3,w4] = divideGT(GT,X,Y);

    # Divede prediction into 4 regions
    [prediction_1,prediction_2,prediction_3,prediction_4] = Divideprediction(prediction,X,Y);

    # Compute the ssim score for each regions
    Q1 = ssim(prediction_1,GT_1);
    Q2 = ssim(prediction_2,GT_2);
    Q3 = ssim(prediction_3,GT_3);
    Q4 = ssim(prediction_4,GT_4);

    # Sum the 4 scores
    Q = w1 * Q1 + w2 * Q2 + w3 * Q3 + w4 * Q4;

    return Q

def centroid(GT):
    """
    % Centroid Compute the centroid of the GT
    % Usage:
    %   [X,Y] = Centroid(GT)
    % Input:
    %   GT - Binary ground truth. Type: logical.
    % Output:
    %   [X,Y] - The coordinates of centroid.
    """
    [rows,cols] = GT.shape;

    if(GT.sum()==0):
        X = round(cols / 2);
        Y = round(rows / 2);
    else:
        total = GT.sum();
        i=np.arange(cols)
        j=np.arange(rows)
        X=torch.round(torch.sum(torch.sum(GT,0)*i)/total)
        Y=torch.round(torch.sum(torch.sum(GT,1)*j)/total);

    return int(X), int(Y)
    
def divideGT(GT,X,Y):
    # divide the GT into 4 regions according to the centroid of the GT and return the weights
    # LT - left top;
    # RT - right top;
    # LB - left bottom;
    # RB - right bottom;

    # width and height of the GT
    [hei,wid] = GT.shape;
    area = wid * hei;

    #copy the 4 regions 
    LT = GT[:Y,:X];
    RT = GT[Y,X:];
    LB = GT[Y:,:X];
    RB = GT[Y:,X:];

    # The different weight (each block proportional to the GT foreground region).
    w1 = (X*Y)/area;
    w2 = ((wid-X)*Y)/area;
    w3 = (X*(hei-Y))/area;
    w4 = 1.0 - w1 - w2 - w3;

    return LT,RT,LB,RB,w1,w2,w3,w4

    # Divide the prediction into 4 regions according to the centroid of the GT 
def Divideprediction(prediction, X, Y):

    # width and height of the prediction
    [hei,wid] = prediction.shape;

    # copy the 4 regions 
    LT = prediction[:Y,:X];
    RT = prediction[:Y,X:];
    LB = prediction[Y:,:X];
    RB = prediction[Y:,X:];

    return [LT,RT,LB,RB]

def ssim(prediction, GT):
    dGT = GT;

    [hei,wid] = prediction.shape;
    N = wid*hei;

    #Compute the mean of SM,GT
    x = prediction.mean();
    y = dGT.mean();

    #Compute the variance of SM,GT
    eps = 1e-8
    sigma_x2 = ((prediction - x)**2).sum()/(N - 1 + eps);
    sigma_y2 = ((dGT - y)**2).sum()/(N - 1 + eps);      

    #Compute the covariance between SM and GT
    sigma_xy = ((prediction - x)*(dGT - y)).sum()/(N - 1 + eps);

    alpha = 4 * x * y * sigma_xy;
    beta = (x**2 + y**2)*(sigma_x2 + sigma_y2);

    if(alpha != 0):
        Q = alpha/(beta + eps);
    elif(alpha == 0 and beta == 0):
        Q = 1.0;
    else:
        Q = 0;

    return Q

def S_object(prediction, GT):
    # compute the similarity of the foreground in the object level
    prediction_fg = prediction.clone();
    prediction_fg[~GT.bool()]=0;
    O_FG = Object(prediction_fg,GT);

    # compute the similarity of the background
    prediction_bg = 1.0 - prediction.clone();
    prediction_bg[GT.bool()] = 0;
    O_BG = Object(prediction_bg,~GT.bool());

    # combine the foreground measure and background measure together
    u = GT.mean();
    Q = u * O_FG + (1 - u) * O_BG;

    return Q

def Object(prediction, GT):
    # compute the mean of the foreground or background in prediction
    x = prediction[GT.bool()].mean();

    # compute the standard deviations of the foreground or background in prediction
    sigma_x = prediction[GT.bool()].std()

    score = 2.0 * x/(x**2 + 1.0 + sigma_x + 1e-8);

    return score
