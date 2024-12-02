'''
    Please read the following paper for more information:\
    M. Heydarian, T. Doyle, and R. Samavi, MLCM: Multi-Label Confusion Matrix, 
    IEEE Access, Feb. 2022, DOI: 10.1109/ACCESS.2022.3151048
'''

import numpy as np
import plotly.graph_objects as go

def cm(label_true,label_pred,print_note=True):
    '''
    Computes the "Multi-Lable Confusion Matrix" (MLCM). 
    MLCM satisfies the requirements of a 2-dimensional confusion matrix.
    Please read the following paper for more information:\
    M. Heydarian, T. Doyle, and R. Samavi, MLCM: Multi-Label Confusion Matrix,
    IEEE Access, Feb. 2022, DOI: 10.1109/ACCESS.2022.3151048
    
    Parameters
    ----------
    label_true : {arraylike, sparse matrix} of shape (num_instance,num_classes)
        Assigned (True) labels in one-hot-encoding format.
        
    label_pred : {arraylike, sparse matrix} of shape (num_instance,num_classes)
        Predicted labels in one-hot-encoding format.
        
    print_note : bool, default=True
        If true, shows a note on the dimension of the confusion matrix.

    Returns
    -------
    conf_mat: multi-label confusion matrix (MLCM)
        ndarray of shape (num_classes+1, num_classes+1).
        Rows represent True labels and columns represent Predicted labels.
        The last row is for "No True Label" assigned (NTL).
        The last column is for "No Predicted Label" found (NPL).
        
    normal_conf_mat: normalized multi-label confusion matrix (normalizd MLCM)
        Numbers show the percentage.
        
    Notes
    -----
    Implemented by Mohammadreza Heydarian, at BioMedic.AI (McMaster University)
    Aug 13, 2020; Modified: Feb 8, 2022.
    '''

    num_classes = len(label_pred[0])  # number of all classes
    num_instances = len(label_pred)  # number of instances (input) 
    # initializing the confusion matrix
    conf_mat = np.zeros((num_classes+1,num_classes+1), dtype=np.int64) 

    for i in range(num_instances): 

        num_of_true_labels = np.sum(label_true[i])
        num_of_pred_labels = np.sum(label_pred[i])

        if num_of_true_labels == 0: 
            if num_of_pred_labels == 0: 
                conf_mat[num_classes][num_classes] += 1 
            else:
                for k in range(num_classes):
                    if label_pred[i][k] == 1:  
                        conf_mat[num_classes][k] += 1  # NTL 


        elif num_of_true_labels == 1:  
            for j in range(num_classes): 
                if label_true[i][j] == 1:  
                    if num_of_pred_labels == 0: 
                        conf_mat[j][num_classes] += 1  # NPL 
                    else: 
                        for k in range(num_classes): 
                            if label_pred[i][k] == 1:  
                                conf_mat[j][k] += 1 

        else: 
            if num_of_pred_labels == 0: 
                for j in range(num_classes): 
                    if label_true[i][j] == 1: 
                        conf_mat[j][num_classes] += 1  # NPL               
            else: 
                true_checked = np.zeros((num_classes,1), dtype=np.int64) 
                pred_checked = np.zeros((num_classes,1), dtype=np.int64) 
                # Check for correct prediction
                for j in range(num_classes): 
                    if label_true[i][j] == 1: 
                        if label_pred[i][j] == 1: 
                            conf_mat[j][j] += 1 
                            true_checked[j] = 1 
                            pred_checked[j] = 1  
                # check for incorrect prediction(s)
                for k in range(num_classes): 
                    if (label_pred[i][k] == 1) and (pred_checked[k] != 1): 
                        for j in range(num_classes):
                            if (label_true[i][j] == 1)and(true_checked[j]!=1):
                                conf_mat[j][k] += 1 
                                pred_checked[k] = 1 
                                true_checked[j] = 1 
                # check for incorrect prediction(s) while all True labels were
                # predicted correctly
                for k in range(num_classes):
                    if (label_pred[i][k] == 1) and (pred_checked[k] != 1): 
                        for j in range(num_classes): 
                            if (label_true[i][j] == 1): 
                                conf_mat[j][k] += 1 
                                pred_checked[k] = 1 
                                true_checked[j] = 1 
                # check for cases with True label(s) and no predicted label
                for k in range(num_classes):
                    if (label_true[i][k] == 1) and (true_checked[k] != 1): 
                        conf_mat[k][num_classes] += 1  # NPL               

    # calculating the normal confusion matrix
    divide = conf_mat.sum(axis=1, dtype='int64') 
    for indx in range(len(divide)):
        if divide[indx] == 0:  # To avoid division by zero
            divide[indx] = 1

    normal_conf_mat = np.zeros((len(divide),len(divide)), dtype=np.float64)
    for i in range (len(divide)):
        for j in range (len(divide)):
            normal_conf_mat[i][j] = round((float(conf_mat[i][j]) / divide[i]) \
                                          *100)

    if print_note:
        print('MLCM has one extra row (NTL) and one extra column (NPL).\
        \nPlease read the following paper for more information:\n\
        Heydarian et al., MLCM: Multi-Label Confusion Matrix, IEEE Access,2022\
        \nTo skip this message, please add parameter "print_note=False"\n\
        e.g., conf_mat,normal_conf_mat = mlcm.cm(label_true,label_pred,False)')

    return conf_mat, normal_conf_mat


def matrix_to_heatmap(matrix, cmap='OrRd', colorbar_label='Value', save_path=None, labels=None, annotate=True):
    """
    Converts a numpy matrix to a heatmap with annotations and tick labels using Plotly.

    Parameters:
    matrix (numpy.ndarray): The matrix to be converted to a heatmap.
    cmap (str): The colormap to use for the heatmap. Default is 'OrRd'.
    colorbar_label (str): The label for the colorbar. Default is 'Value'.
    title (str): The title for the heatmap. Default is 'Confusion Matrix'.
    save_path (str): The path to save the heatmap image. If None, the heatmap will be shown but not saved.
    labels (list): The labels for the heatmap axes. Default is None.
    annotate (bool): Whether to annotate cells with values. Default is True.

    Returns:
    fig (plotly.graph_objects.Figure): The generated plotly figure.
    """
    labels = list(labels)
    # Prepare labels for the axes
    # labels = labels if labels is not None else list(range(matrix.shape[0]))

    # Create text annotations for the heatmap
    annotations = np.round(matrix, 2).astype(str) if annotate else None

    # Create heatmap using plotly
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=labels + ['NPL'],  # X-axis labels
            y=labels + ['NTL'],  # Y-axis labels
            colorscale=cmap,
            text=annotations,
            hoverinfo="z",  # Show values on hover
            showscale=True,
            colorbar=dict(title=colorbar_label),
            texttemplate="%{text}" if annotate else None,  # Display annotations
            # zmin=matrix.min(),
            # zmax=matrix.max()
        )
    )

    # Set layout options
    fig.update_layout(
        xaxis=dict(title='Predicted', tickvals=list(range(len(labels) + 1)), ticktext=labels + ['NPL']),
        yaxis=dict(title='Truth', tickvals=list(range(len(labels) + 1)), ticktext=labels + ['NTL']),
        autosize=False,
        width=600,
        height=500
    )


    return fig
    