from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np

def calculate_metrics(pred, target):
    """
    Calculate and return R^2, NRMSE, MAE for given prediction and target values.

    Parameters:
    pred (array-like): The predicted values.
    target (array-like): The true target values.

    Returns:
    tuple: A tuple containing R^2, NRMSE, and MAE.
    """

    # Calculate R^2 (coefficient of determination)
    r2 = r2_score(target, pred)

    # Calculate RMSE (root mean squared error) from MSE
    rmse = np.sqrt(mean_squared_error(target, pred))


    # Calculate NRMSE (normalized root mean squared error)
    # Normalize RMSE by dividing it by the range of the target values (max - min)
    range_target = np.max(target) - np.min(target)
    if range_target == 0:
        raise ValueError("The range of target values cannot be zero")
    nrmse = rmse / range_target

    # Calculate MAE (mean absolute error)
    mae = mean_absolute_error(target, pred)

    return r2, nrmse, mae

# Example usage
if __name__ == "__main__":
    # Assume that pred and target are numpy arrays or similar sequence types
    pred = np.array([[2.5, 3.5], [4.5, 1.5]])
    target = np.array([[3, 4], [5, 2]])

    r2, nrmse, mae = calculate_metrics(pred, target)
    print(f"R^2: {r2}, NRMSE: {nrmse}, MAE: {mae}")